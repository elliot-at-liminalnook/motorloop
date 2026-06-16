#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Scaffold a per-block contract (datasheet) for every reusable core that lacks
one (tier2-adoption-checklist §4). Auto-fills the mechanical 80% - the interface
table (parsed from the RTL port list), the parameter table, and the
proof-or-sim-only status (from formal/manifest.toml + formal/sim_only.toml, the
single source) - and leaves `TODO:` markers for the maintainer-written prose
(claim, assumptions, timing, reuse). Mirrors the structure of the two existing
contracts (rtl/contracts/{pwm_generator,foc_core}.md).

  python3 scripts/gen_contract_stubs.py            # stub every missing contract
  python3 scripts/gen_contract_stubs.py --force    # also overwrite existing

Existing contracts are left alone unless --force.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS = ROOT / "rtl" / "contracts"
SYSTEM = {"motorloop"}


def rtl_path(block: str) -> Path:
    p = ROOT / "rtl" / f"{block}.v"
    return p if p.exists() else ROOT / "rtl" / "bus" / f"{block}.v"


def _port_group(src: str, module: str) -> str:
    """The ( ... ) port-list text of `module`, skipping an optional #( ... )."""
    i = src.index(f"module {module}")
    j = src.index("(", i)
    if "#" in src[i:j]:
        k = src.index("(", src.index("#", i))
        depth = 0
        while True:
            depth += (src[k] == "(") - (src[k] == ")")
            if depth == 0:
                break
            k += 1
        j = src.index("(", k + 1)
    depth, k = 0, j
    while True:
        depth += (src[k] == "(") - (src[k] == ")")
        if depth == 0:
            break
        k += 1
    return src[j + 1:k]


def ports(block: str, module: str):
    body = _port_group(rtl_path(block).read_text(), module)
    out = []
    for m in re.finditer(
            r"\b(input|output)\s+(?:wire|reg)\s+(signed\s+)?"
            r"(?:\[([^\]]+)\]\s+)?(\w+)", body):
        d, sgn, msb, name = m.group(1), bool(m.group(2)), m.group(3), m.group(4)
        d = "in" if d == "input" else "out"
        width = "1" if not msb else f"{int(msb.split(':')[0]) + 1}" \
            if msb.split(':')[0].isdigit() and msb.split(':')[1].strip() == '0' \
            else f"[{msb}]"
        out.append((name, d, width, "yes" if sgn else "—"))
    return out


def params(block: str):
    src = rtl_path(block).read_text()
    grp = _port_group(src, block)
    head = src[:src.index(grp)] if grp in src else src
    out = []
    for m in re.finditer(
            r"\bparameter\s+(?:integer\s+|signed\s+)*(?:\[[^\]]+\]\s+)?"
            r"(\w+)\s*=\s*([^,)\n]+)", head):
        out.append((m.group(1), m.group(2).strip()))
    return out


def description(block: str) -> str:
    txt = (ROOT / f"{block}.core").read_text()
    m = re.search(r'description:\s*"([^"]*)"', txt)
    return m.group(1) if m else block


def proof_status(block: str) -> str:
    manifest = tomllib.loads((ROOT / "formal" / "manifest.toml").read_text())
    for c in manifest.get("check", []):
        if block in (c.get("module"), c.get("id")):
            exp = c.get("expected", "PROVEN")
            checker = c.get("checker", "")
            props = "; ".join(c.get("properties", []))
            return (f"- **{exp}** (`formal/manifest.toml`, `{checker}`): "
                    f"{props or 'TODO: state the proven property'}.\n"
                    f"- **Method:** {c.get('method', 'prove')}, "
                    f"`engine {c.get('engine', 'smtbmc boolector')}`.")
    sim = tomllib.loads((ROOT / "formal" / "sim_only.toml").read_text())
    if block in sim:
        return (f"- **Sim-only** (`formal/sim_only.toml`): {sim[block]['reason']}\n"
                f"- Covered by the simulation tiers, not a machine proof.")
    return "- TODO: declare proof or sim-only status."


CONTRACT = """<!-- SPDX-License-Identifier: MIT -->
# `{name}` — {desc}

TODO: one-paragraph claim — what this block guarantees and where it fits.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
{iface}

## Clocking & reset

- **Clock domains:** TODO (single `clk` / combinational / handshake).
- **Reset:** TODO (async active-low `rst_n` behaviour).
- **Latency:** TODO (combinational / N-cycle / start→done).

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
{params}

## Formal contract

{status}
- **Assumptions:** TODO (clock domain, input ranges, reset).

## Synthesis fit

- **Device:** ECP5. TODO (standalone Fmax via `synth/fmax_module.py {name}`; size).

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** TODO (none / child modules).
- **Pull it:** `fusesoc run motorloop:ip:{name}` (core at repo root).
"""


def gen(block: str) -> str:
    iface = "\n".join(
        f"| `{n}` | {d} | {w} | {s} | TODO | TODO |" for n, d, w, s in ports(block, block))
    ps = params(block)
    ptab = "\n".join(f"| `{n}` | `{v}` | TODO |" for n, v in ps) or \
        "| _(none)_ | | |"
    return CONTRACT.format(name=block, desc=description(block), iface=iface,
                           params=ptab, status=proof_status(block))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    blocks = sorted(c.stem for c in ROOT.glob("*.core") if c.stem not in SYSTEM)
    wrote, skipped = [], []
    for b in blocks:
        out = CONTRACTS / f"{b}.md"
        # Never clobber a FINISHED contract (one with no TODO markers), even with
        # --force; --force only re-stubs contracts that are still scaffolds.
        if out.exists() and (not args.force or "TODO:" not in out.read_text()):
            skipped.append(b)
            continue
        out.write_text(gen(b))
        wrote.append(b)
    print(f"wrote {len(wrote)} stubs: {' '.join(wrote)}")
    if skipped:
        print(f"skipped {len(skipped)} existing: {' '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
