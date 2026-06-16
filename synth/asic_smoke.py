#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ASIC-readiness smoke (robotics-ip-checklist stage 10).

For each reusable block, drive yosys to gate-level synthesis and assert it is
ASIC-clean: it maps to gates, infers **no latches**, and has **no combinational
loops / multiple drivers / undriven-used nets** (`check -assert`). Reports the
gate/cell count per block. The block list + file sets come from the FuseSoC
`.core` files (one source - same as the FPGA flow and Bender).

HONEST FRAMING: this is a *synthesis* smoke - "the RTL is ASIC-synthesizable and
structurally clean at the gate level." It is **not** a tapeout and **not** P&R
timing/area on a real PDK. Full OpenLane (sky130) place&route - real cell area
and post-route timing - needs OpenROAD + the PDK; see `synth/openlane/README.md`
for the wired flow (the FPGA flow already reports a real post-route Fmax of
41 MHz for the system, robotics-ip stage 6.5). Run via the OSS CAD Suite:

    source ~/oss-cad-suite/environment
    python3 synth/asic_smoke.py            # all blocks
    python3 synth/asic_smoke.py --check    # exit 1 if any block is not clean
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Representative reusable blocks (the headline is the pipelined foc_core). Each
# resolves its RTL + include files from its generated <block>.core.
BLOCKS = [
    "pwm_generator", "commutation", "svpwm", "current_pi", "circle_limit_seq",
    "divider32", "spi_drv_master", "as5047p_spi_master", "drv_manager",
    "adc_sequencer", "axil_regfile", "wb_regfile", "axis_sampler", "foc_core",
]


def core_sources(block: str) -> tuple[list[str], list[str]]:
    """(rtl .v files, include .vh files) from <block>.core."""
    text = (ROOT / f"{block}.core").read_text()
    rtl = re.findall(r"^\s*-\s+(rtl/\S+\.v)\b", text, re.M)
    inc = re.findall(r"^\s*-\s+(rtl/\S+\.vh):", text, re.M)
    # de-dup, keep order
    seen: set[str] = set()
    rtl = [f for f in rtl if not (f in seen or seen.add(f))]
    return rtl, inc


def smoke_one(block: str) -> dict:
    rtl, inc = core_sources(block)
    reads = "\n".join(f"read_verilog -I rtl -I rtl/gen {f}" for f in rtl)
    script = f"""
{reads}
hierarchy -top {block} -check
proc
flatten
synth -top {block} -flatten
check -assert
select -count t:$dlatch t:$dlatchsr t:$_DLATCH_* %u
stat
"""
    p = subprocess.run(["yosys", "-p", script], cwd=ROOT,
                       capture_output=True, text=True)
    out = p.stdout + p.stderr
    ok = (p.returncode == 0)
    # latch count: `select -count ... %u` prints "<n> objects."
    lm = re.search(r"(\d+)\s+objects\.", out)
    latches = int(lm.group(1)) if lm else None
    # gate-cell count from the flattened `stat` ("<n> cells")
    cm = re.search(r"^\s*(\d+)\s+cells\s*$", out, re.M)
    cells = int(cm.group(1)) if cm else None
    clean = ok and (latches == 0)
    return {"block": block, "ok": ok, "latches": latches,
            "cells": cells, "clean": clean, "log": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any block is not ASIC-clean")
    args = ap.parse_args()

    results = [smoke_one(b) for b in BLOCKS]
    lines = [
        # REUSE-IgnoreStart (this SPDX tag is emitted INTO the report file)
        "<!-- SPDX-License-Identifier: MIT -->",
        # REUSE-IgnoreEnd
        "# ASIC-readiness smoke (robotics-ip stage 10)",
        "",
        "Each reusable block driven to gate-level synthesis with yosys and "
        "checked for ASIC cleanliness: maps to gates, **no inferred latches**, "
        "**no combinational loops / multiple drivers** (`check -assert`).",
        "",
        "> **Honest framing.** This is an ASIC-*synthesizability* smoke, **not** "
        "a tapeout and **not** real-PDK place&route. Post-route timing/area need "
        "OpenROAD + the sky130 PDK (`synth/openlane/` wires that flow). The FPGA "
        "flow reports a real post-route system Fmax of **41.3 MHz** (stage 6.5).",
        "",
        "| Block | Synthesizes | Inferred latches | Multidriver/loop | Gate cells |",
        "| --- | --- | --- | --- | --- |",
    ]
    all_clean = True
    for r in results:
        all_clean &= r["clean"]
        latch = "0 ✓" if r["latches"] == 0 else f"**{r['latches']}** ✗"
        loop = "none ✓" if r["ok"] else "**FAIL** ✗"
        synth = "yes ✓" if r["ok"] else "**no** ✗"
        lines.append(f"| `{r['block']}` | {synth} | {latch} | {loop} | "
                     f"{r['cells'] if r['cells'] is not None else 'n/a'} |")
    lines += ["",
              f"**Result:** {sum(r['clean'] for r in results)}/{len(results)} "
              "blocks ASIC-synthesizable and structurally clean"
              + (" — all clean." if all_clean else " — SEE FAILURES ABOVE."),
              ""]
    (ROOT / "synth" / "asic_smoke_report.md").write_text("\n".join(lines))
    for r in results:
        flag = "OK " if r["clean"] else "FAIL"
        print(f"[asic] {flag} {r['block']:20s} cells={r['cells']} "
              f"latches={r['latches']}")
    print(f"[asic] wrote synth/asic_smoke_report.md "
          f"({sum(r['clean'] for r in results)}/{len(results)} clean)")
    if args.check and not all_clean:
        sys.exit(1)


if __name__ == "__main__":
    main()
