#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Assemble the docs site (robotics-ip-checklist stage 7): gather the per-block
contracts, the status matrix, and the verification plan, and emit a timing page
with WaveDrom diagrams authored from the proven/contracted timing. mkdocs then
renders it (mkdocs.yml). The WaveDrom JSON is the timing-doc source.

  scripts/gen_docs.py            # writes site-src/
  mkdocs build                   # (docs venv) renders the site
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The mkdocs source dir. NOT "docs/" - that holds the (untracked) vendor
# datasheets; this generator rmtree's its target, so it must never point there.
DOCS = ROOT / "site-src"
CONTRACTS = ROOT / "rtl" / "contracts"
assert DOCS.name == "site-src", "refusing to manage any dir but site-src/"

INDEX = """<!-- SPDX-License-Identifier: MIT -->
# motorloop — robotics HDL blocks

Reusable, verified motor-control IP. Each block ships with a **contract**
(interface, timing, parameters), a **proof or declared sim-only** status, block
**tests**, **timing diagrams**, and **bus wrappers** (AXI-Lite / AXI-Stream /
Wishbone).

- **[Status matrix](status-matrix.md)** — proven / simulated / fit, per block.
- **[Verification plan](verification-plan.md)** — requirement → proof/test map.
- **[Timing diagrams](timing.md)** — WaveDrom views of the proven timing.
- **Contracts** — one per reusable block (see the nav).

Consume with `fusesoc run motorloop:ip:<block>` or `bender`. Pulled from one
pinned toolchain (`toolchain.lock`); see `notes/reproduce.md`.
"""

TIMING = """<!-- SPDX-License-Identifier: MIT -->
# Timing diagrams

WaveDrom views of the timing each block's contract + proof guarantee.

## PWM dead-time handoff (`pwm_generator`)

A complementary gate asserts only after its partner has been off >= DEAD_CYCLES
(proven: `pwm_deadtime`).

```wavedrom
{ "signal": [
  {"name": "clk",        "wave": "p........"},
  {"name": "gate_low",   "wave": "10......."},
  {"name": "off_time_l", "wave": "=2222222.", "data": ["0","1","..","DEAD",">=","",""]},
  {"name": "gate_high",  "wave": "0.....1.."}
], "head": {"text": "high rises only after low off >= DEAD"} }
```

## SPI mode-1 frame (`spi_drv_master`)

16-bit, CPOL=0/CPHA=1: MOSI launched on the rising edge, MISO sampled on the
trailing edge (proven framing on the slave-facing masters).

```wavedrom
{ "signal": [
  {"name": "ncs",  "wave": "10.....1"},
  {"name": "sclk", "wave": "0.1010.0"},
  {"name": "mosi", "wave": "x=.=.=.x", "data": ["b15","b14","b13"]},
  {"name": "miso", "wave": "x=.=.=.x", "data": ["d15","d14","d13"]}
] }
```

## AXI-Lite write handshake (`axil_regfile`)

VALID holds until READY; the write response is OKAY (proven: `axil_regfile`).

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

One CONVST samples both channels; READY after tDRDY, then the 16-bit read
(proven framing).

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
// Render ```wavedrom code blocks client-side via WaveDrom.
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
    } catch (e) { /* leave the source block as-is on parse error */ }
  });
});
"""


def main():
    if DOCS.exists():
        shutil.rmtree(DOCS)
    (DOCS / "contracts").mkdir(parents=True)
    (DOCS / "js").mkdir(parents=True)
    (DOCS / "js" / "wavedrom-init.js").write_text(WAVEDROM_JS)
    (DOCS / "index.md").write_text(INDEX)
    (DOCS / "timing.md").write_text(TIMING)
    for src in ("status-matrix.md", "verification-plan.md"):
        shutil.copy(ROOT / "notes" / src, DOCS / src)
    contracts = sorted(CONTRACTS.glob("*.md"))
    for c in contracts:
        shutil.copy(c, DOCS / "contracts" / c.name)
    print(f"assembled site-src/: index, timing, 2 references, "
          f"{len(contracts)} contracts")


if __name__ == "__main__":
    main()
