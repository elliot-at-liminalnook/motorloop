#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Assemble the curated whole-project documentation site.

Source documentation stays beside the code and evidence it explains. This
script mirrors the maintained entry pages into site-src/, adds generated timing
diagrams, and regenerates the contract navigation. Run scripts/check_docs.py
before this script; MkDocs then renders site-src/ in strict mode.
"""

from __future__ import annotations

import shutil
import posixpath
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "site-src"
CONTRACTS = ROOT / "rtl" / "contracts"
assert DOCS.name == "site-src", "refusing to manage any directory but site-src/"

SITE_DOCUMENTS = (
    "notes/README.md",
    "notes/getting-started.md",
    "notes/current-status.md",
    "notes/system-architecture.md",
    "notes/reader-paths.md",
    "notes/glossary.md",
    "notes/documentation-guide.md",
    "notes/document-catalog.md",
    "notes/archive/README.md",
    "notes/reproduce.md",
    "notes/blender-agent-workflow.md",
    "notes/pre-gpu-test-entrypoint.md",
    "notes/training-ladder-runbook.md",
    "notes/architecture.md",
    "notes/verification-plan.md",
    "notes/status-matrix-generated.md",
    "notes/robot-hardware-contract.md",
    "notes/runpod-warp-validation-2026-07-10.md",
    "notes/locomotion-status.md",
    "notes/training-uplift-results.md",
    "notes/rl-verification-playbook.md",
    "notes/open-questions.md",
    "notes/hardware-bringup-notes.md",
    "notes/docs-digest.md",
    "notes/ethos.md",
    "sim/README.md",
    "formal/README.md",
    "formal/proof_report.md",
    "synth/synth_report.md",
)

TIMING = """<!-- SPDX-License-Identifier: MIT -->
# Timing diagrams

> **Document status:** Generated · **Source:** RTL contracts and formal timing properties

WaveDrom views of timing guaranteed by the named contract or proof.

## PWM dead-time handoff (`pwm_generator`)

A complementary gate asserts only after its partner has been off for at least
`DEAD_CYCLES` (proof: `pwm_deadtime`).

```wavedrom
{ "signal": [
  {"name": "clk",        "wave": "p........"},
  {"name": "gate_low",   "wave": "10......."},
  {"name": "off_time_l", "wave": "=2222222.", "data": ["0","1","..","DEAD",">=","",""]},
  {"name": "gate_high",  "wave": "0.....1.."}
], "head": {"text": "high rises only after low off >= DEAD"} }
```

## SPI mode-1 frame (`spi_drv_master`)

Sixteen bits, CPOL=0/CPHA=1: MOSI changes on the leading edge and MISO is
sampled on the trailing edge.

```wavedrom
{ "signal": [
  {"name": "ncs",  "wave": "10.....1"},
  {"name": "sclk", "wave": "0.1010.0"},
  {"name": "mosi", "wave": "x=.=.=.x", "data": ["b15","b14","b13"]},
  {"name": "miso", "wave": "x=.=.=.x", "data": ["d15","d14","d13"]}
] }
```

## AXI-Lite write handshake (`axil_regfile`)

`VALID` holds until `READY`; the write response is `OKAY`.

```wavedrom
{ "signal": [
  {"name": "clk",     "wave": "p....."},
  {"name": "awvalid", "wave": "01.0.."},
  {"name": "awready", "wave": "0.10.."},
  {"name": "wvalid",  "wave": "01.0.."},
  {"name": "wready",  "wave": "0.10.."},
  {"name": "bvalid",  "wave": "0.1.0."},
  {"name": "bready",  "wave": "1....."}
] }
```

## ADS9224R conversion (`ads9224r_master`)

One `CONVST` samples both channels; data is read after the ready interval.

```wavedrom
{ "signal": [
  {"name": "convst", "wave": "010....."},
  {"name": "ready",  "wave": "0..1..0."},
  {"name": "cs",     "wave": "1...0..1"},
  {"name": "sclk",   "wave": "0....10."}
] }
```
"""

WAVEDROM_JS = """// SPDX-License-Identifier: MIT
window.addEventListener("DOMContentLoaded", function () {
  if (!window.WaveDrom) return;
  var blocks = document.querySelectorAll("code.wavedrom, .wavedrom > code");
  blocks.forEach(function (el, i) {
    try {
      var src = JSON.parse(el.textContent);
      var div = document.createElement("div");
      div.id = "WaveDrom_Display_" + i;
      (el.closest("pre") || el).replaceWith(div);
      WaveDrom.RenderWaveForm(i, src, "WaveDrom_Display_");
    } catch (e) { /* keep source visible when a diagram is malformed */ }
  });
});
"""

MERMAID_JS = """// SPDX-License-Identifier: MIT
window.addEventListener("DOMContentLoaded", function () {
  if (window.mermaid) {
    window.mermaid.initialize({startOnLoad: true, securityLevel: "strict"});
  }
});
"""

MARKDOWN_LINK_RE = re.compile(
    r"(?P<prefix>!?\[[^\]]*\]\()"
    r"(?P<target><[^>]+>|[^)\s]+)"
    r"(?P<suffix>(?:\s+[\"'][^)]*[\"'])?\))"
)
GITHUB = "https://github.com/elliot-at-liminalnook/motorloop"
IMAGE_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"}


def tracked_under(prefix: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", prefix],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return [line for line in result.stdout.splitlines() if (ROOT / line).is_file()]


def rewrite_links(text: str, source: str, destination: str, site_map: dict[str, str]) -> str:
    """Keep curated links inside the site; send other valid repo links to GitHub."""

    def replace(match: re.Match[str]) -> str:
        raw_target = match.group("target")
        bracketed = raw_target.startswith("<") and raw_target.endswith(">")
        target = raw_target[1:-1] if bracketed else raw_target
        if not target or target.startswith(("http://", "https://", "mailto:", "#", "data:")):
            return match.group(0)

        path_part, separator, fragment = target.partition("#")
        decoded = unquote(path_part)
        source_parent = posixpath.dirname(source)
        repo_target = posixpath.normpath(posixpath.join(source_parent, decoded))
        if repo_target.startswith("../"):
            return match.group(0)

        mapped = site_map.get(repo_target)
        if mapped is None and (ROOT / repo_target).is_dir():
            mapped = site_map.get(posixpath.join(repo_target, "README.md"))

        if mapped is not None:
            destination_parent = posixpath.dirname(destination) or "."
            new_target = posixpath.relpath(mapped, destination_parent)
        elif (ROOT / repo_target).exists():
            if match.group("prefix").startswith("!") and Path(repo_target).suffix.lower() in IMAGE_SUFFIXES:
                new_target = f"https://raw.githubusercontent.com/elliot-at-liminalnook/motorloop/main/{repo_target}"
            else:
                view = "tree" if (ROOT / repo_target).is_dir() else "blob"
                new_target = f"{GITHUB}/{view}/main/{repo_target}"
        else:
            return match.group(0)

        if separator:
            new_target += f"#{fragment}"
        if bracketed:
            new_target = f"<{new_target}>"
        return f"{match.group('prefix')}{new_target}{match.group('suffix')}"

    return MARKDOWN_LINK_RE.sub(replace, text)


def copy_source(
    relative: str,
    destination: str | None,
    site_map: dict[str, str],
) -> None:
    src = ROOT / relative
    dst = DOCS / (destination or relative)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".md":
        dst.write_text(rewrite_links(src.read_text(), relative, destination or relative, site_map))
    else:
        shutil.copy2(src, dst)


def main() -> None:
    if DOCS.exists():
        shutil.rmtree(DOCS)
    (DOCS / "js").mkdir(parents=True)

    contracts = sorted(CONTRACTS.glob("*.md"))
    figure_assets = [
        asset for asset in tracked_under("figures")
        if Path(asset).suffix.lower() != ".md"
    ]
    site_map = {"README.md": "index.md"}
    site_map.update({relative: relative for relative in SITE_DOCUMENTS})
    site_map.update(
        {
            contract.relative_to(ROOT).as_posix(): contract.relative_to(ROOT).as_posix()
            for contract in contracts
        }
    )
    site_map.update({asset: asset for asset in figure_assets})

    copy_source("README.md", "index.md", site_map)
    for relative in SITE_DOCUMENTS:
        copy_source(relative, None, site_map)

    for contract in contracts:
        relative = contract.relative_to(ROOT).as_posix()
        copy_source(relative, None, site_map)

    for asset in figure_assets:
        copy_source(asset, None, site_map)
    (DOCS / "timing.md").write_text(TIMING)
    (DOCS / "js/wavedrom-init.js").write_text(WAVEDROM_JS)
    (DOCS / "js/mermaid-init.js").write_text(MERMAID_JS)

    update_nav(contracts)
    print(
        "assembled site-src/: project landing, "
        f"{len(SITE_DOCUMENTS)} curated documents, timing, and "
        f"{len(contracts)} RTL contracts, and {len(figure_assets)} media assets"
    )


def update_nav(contracts: list[Path]) -> None:
    """Regenerate the final Contracts section in mkdocs.yml."""
    config = ROOT / "mkdocs.yml"
    text = config.read_text()
    marker = "  - Contracts:"
    if marker not in text:
        raise RuntimeError("mkdocs.yml must end with a Contracts nav section")
    head = text[: text.index(marker)]
    lines = [marker]
    lines += [
        f"      - {contract.stem}: rtl/contracts/{contract.name}"
        for contract in contracts
    ]
    config.write_text(head + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
