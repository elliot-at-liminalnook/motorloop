#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Multi-vendor resource portability (release-and-portability-checklist §4.1).
yosys ships per-vendor synthesis passes, so the controller maps to Xilinx /
Intel / Gowin / Lattice **with no vendor tools** - proving the deliberate
Verilog-2005 portability and giving ballpark resources. A pass *erroring* surfaces
non-portable RTL (the high-value check). These are pre-place&route ESTIMATES;
authoritative Fmax/utilisation come from the vendor flows (synth/vivado,
synth/quartus) or the open Gowin P&R (synth/run_gowin.py).

  source ~/oss-cad-suite/environment
  python3 synth/portability.py            # write synth/portability_report.md
  python3 synth/portability.py --check     # exit 1 if any family fails to map
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (label, yosys synth pass, top). controller_top is the vendor-neutral system top
# (board_top.v is the ECP5-specific wrapper, excluded).
TARGETS = [
    ("Lattice ECP5",   "synth_ecp5",      "controller_top"),
    ("Xilinx 7-series", "synth_xilinx",    "controller_top"),
    ("Intel Cyclone",  "synth_intel_alm", "controller_top"),
    ("Gowin GW5A",     "synth_gowin",     "controller_top"),
]


def rtl_files() -> list[str]:
    """The controller_top RTL set, parsed from synth_ecp5.ys (one source)."""
    ys = (ROOT / "synth" / "synth_ecp5.ys").read_text()
    files = re.findall(r"(rtl/\S+\.v)\b", ys)
    seen: set[str] = set()
    return [f for f in files if not (f in seen or seen.add(f))]


def _sum(stat: str, *needles: str) -> int:
    n = 0
    for ln in stat.splitlines():
        m = re.match(r"\s*(\d+)\s+(\S+)", ln)
        if m and any(k in m.group(2).upper() for k in needles):
            n += int(m.group(1))
    return n


def run(label: str, synth_pass: str, top: str) -> dict:
    reads = "\n".join(f"read_verilog -I rtl -I rtl/gen {f}" for f in rtl_files())
    # `flatten` (the pass, portable across synth_*) collapses to one module so
    # `stat` is the flat top - otherwise per-module counts double-count.
    ys = f"{reads}\nhierarchy -top {top}\n{synth_pass} -top {top}\nflatten\nstat\n"
    p = subprocess.run(["yosys", "-p", ys], cwd=ROOT, capture_output=True, text=True)
    out = p.stdout + p.stderr
    ok = p.returncode == 0
    (ROOT / "synth" / "work").mkdir(parents=True, exist_ok=True)
    log = ROOT / "synth" / "work" / f"portability_{label.split()[0].lower()}.log"
    log.write_text(out)
    # Report only what's vendor-comparable + reliable: maps (rc), total mapped
    # cells (flat top), and LUTs (the design is LUT-dominated; consistent across
    # families). The full per-primitive breakdown (FFs/DSP/BRAM, named per vendor)
    # is in the .log - those names aren't comparable across vendors.
    cm = re.findall(r"Number of cells:\s+(\d+)", out)
    return {
        "label": label, "ok": ok, "log": log.name,
        "lut": _sum(out, "LUT"),
        "cells": int(cm[-1]) if cm else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any family fails to map")
    args = ap.parse_args()
    rows = [run(*t) for t in TARGETS]

    lines = [
        # REUSE-IgnoreStart (SPDX tag emitted INTO the generated report)
        "<!-- SPDX-License-Identifier: MIT -->",
        # REUSE-IgnoreEnd
        "# Multi-vendor portability (resource estimates)",
        "",
        "`controller_top` mapped to each FPGA family with **yosys vendor backends "
        "and no vendor tools** (release-and-portability §4.1) - proof the "
        "Verilog-2005 RTL ports off Lattice. **Pre-place&route estimates**; "
        "authoritative Fmax/utilisation: `synth/synth_report.md` (ECP5, real P&R), "
        "`synth/run_gowin.py` (Gowin, open P&R), `synth/vivado/` + `synth/quartus/` "
        "(vendor, where licensed).",
        "",
        "| Family | Maps | LUTs | Flow | Full breakdown |",
        "| --- | --- | --- | --- | --- |",
    ]
    all_ok = True
    for r in rows:
        all_ok &= r["ok"]
        maps = "yes ✓" if r["ok"] else "**NO** ✗"
        lines.append(f"| {r['label']} | {maps} | {r['lut']} | "
                     f"yosys synth (open-estimate) | `synth/work/{r['log']}` |")
    lines += ["",
              f"**Result:** the RTL maps to {sum(r['ok'] for r in rows)}/{len(rows)} "
              "FPGA families." + ("" if all_ok else " SEE FAILURES."),
              "",
              "_LUTs are the dominant, roughly-comparable resource (the design is "
              "LUT-heavy ~13k on every family). Mapped-cell totals and the full "
              "per-vendor FF/DSP/BRAM breakdown (primitive names differ per vendor) "
              "are in the linked logs. These are pre-P&R estimates - real timing is "
              "in the P&R/vendor flows._"]
    (ROOT / "synth" / "portability_report.md").write_text("\n".join(lines))
    for r in rows:
        print(f"[portability] {'OK  ' if r['ok'] else 'FAIL'} {r['label']:16s} "
              f"maps={r['ok']} LUT={r['lut']} cells={r['cells']}")
    print("[portability] wrote synth/portability_report.md")
    if args.check and not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
