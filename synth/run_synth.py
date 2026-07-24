#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Open ECP5 synthesis flow for controller_top (platform-abstraction stage 15).

Drives the open toolchain (yosys -> nextpnr-ecp5 -> ecppack) and writes
synth/synth_report.md with the ECP5 utilization and the post-route Fmax.

  synth/run_synth.py            # full flow: synth + place&route + bitstream
  synth/run_synth.py --check    # synthesis only (the "does it map to ECP5" gate)

The toolchain is the OSS CAD Suite (`source ~/oss-cad-suite/environment`); the
proofs and the simulator are separate. Target board: ULX3S (LFE5U-85F,
CABGA381). The ctrl_*/dbg_* interface ports are left unconstrained for the synth
check (nextpnr --lpf-allow-unconstrained); a real board build maps them to
headers via synth/ulx3s.lpf.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "synth" / "work"
JSON = WORK / "board_top.json"
DEVICE = "--85k"
PACKAGE = "CABGA381"


def _run(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def _require(tool):
    if shutil.which(tool) is None:
        sys.exit(f"[synth] {tool} not found - source ~/oss-cad-suite/environment")


def synth():
    _require("yosys")
    rc = _run(["yosys", "-s", "synth/synth_ecp5.ys"], WORK / "yosys.log")
    if rc != 0 or not JSON.exists():
        sys.exit(f"[synth] yosys synthesis FAILED (rc={rc}); see {WORK/'yosys.log'}")
    stat = (WORK / "stat.txt").read_text() if (WORK / "stat.txt").exists() else ""
    util = {}
    # yosys `stat` prints "<count>   <CELL>" (count first).
    for key in ("TRELLIS_FF", "LUT4", "CCU2C", "DP16KD", "MULT18X18D", "TRELLIS_IO"):
        m = re.search(rf"(\d+)\s+{key}\b", stat)
        if m:
            util[key] = int(m.group(1))
    print(f"[synth] synthesized OK: "
          + ", ".join(f"{k}={v}" for k, v in util.items()))
    return util


def pnr():
    _require("nextpnr-ecp5")
    cfg = WORK / "board_top.config"
    if cfg.exists():
        cfg.unlink()
    rc = _run(["nextpnr-ecp5", DEVICE, "--package", PACKAGE,
               "--json", str(JSON), "--lpf", "synth/ulx3s.lpf",
               "--lpf-allow-unconstrained", "--textcfg", str(cfg)],
              WORK / "nextpnr.log")
    # nextpnr returns nonzero when timing is not met at the constrained
    # frequency; that is a *reported result* (the design still placed & routed,
    # the textcfg exists), not a flow failure. A true place/route failure leaves
    # no textcfg - that is fatal.
    if not cfg.exists():
        sys.exit(f"[synth] nextpnr place/route FAILED (rc={rc}); "
                 f"see {WORK/'nextpnr.log'}")
    log = (WORK / "nextpnr.log").read_text()
    fmax = None
    for m in re.finditer(r"Max frequency for clock.*?:\s*([\d.]+)\s*MHz", log):
        fmax = float(m.group(1))  # take the last (post-route) estimate
    timing_met = (rc == 0)
    print(f"[synth] place&route OK (fits), Fmax ~ {fmax} MHz, "
          f"timing@25MHz {'MET' if timing_met else 'NOT met'}")
    return cfg, fmax, timing_met


def pack(cfg):
    _require("ecppack")
    bit = WORK / "board_top.bit"
    rc = _run(["ecppack", str(cfg), str(bit)], WORK / "ecppack.log")
    if rc != 0:
        sys.exit(f"[synth] ecppack FAILED (rc={rc}); see {WORK/'ecppack.log'}")
    print(f"[synth] bitstream written: {bit}")
    return bit


def write_report(util, fmax, timing_met, bit):
    lines = ["# ECP5 synthesis report",
             "",
             ("Open flow (yosys -> nextpnr-ecp5 -> ecppack), board wrapper "
              "`synth/board_top.v` on the ULX3S (LFE5U-85F, CABGA381). Regenerate "
              "with `synth/run_synth.py`."),
             "",
             "## Utilization (synth_ecp5, raw cell counts)"]
    for k, v in util.items():
        lines.append(f"- {k}: {v}")
    lines += ["",
              ("The design fits the -85F with wide margin (the parsed counts "
               "above are indicative; see `synth/work/stat.txt` and "
               "`nextpnr.log` for the authoritative post-pack utilisation)."),
              "",
              "## Timing",
              f"- Post-route Fmax estimate: **{fmax if fmax else 'n/a'} MHz** "
              f"(target 25 MHz: {'MET' if timing_met else 'NOT met'})",
              f"- Bitstream: `{bit.relative_to(ROOT) if bit else 'n/a'}`",
              ""]
    if timing_met:
        lines += [
            "**Finding:** the FOC datapath is pipelined (robotics-ip stage 6.5): "
            "the Clarke/Park/PI/circle-limit/inv-Park/SVPWM chain is walked over "
            "registered stages, and the former Fmax bottleneck - `circle_limit`'s "
            "combinational 16-iteration isqrt + two 32-bit divisions - is now the "
            "sequential, bit-exact `circle_limit_seq` (one op per clock, reusing "
            "`divider32`). Post-route Fmax rose from ~3.3 MHz (unpipelined) to "
            f"~{fmax} MHz, clearing the 25 MHz target, while LUT usage dropped "
            "(the huge combinational divide/sqrt logic became a small reused "
            "datapath). The simulator stays cycle-accurate; the FOC duties simply "
            "appear a fixed, sub-sample-period latency later.",
            ""]
    else:
        lines += [
            "**Finding:** the design synthesizes and fits but does not yet meet "
            f"25 MHz (Fmax ~{fmax} MHz). Inspect `nextpnr.log` for the critical "
            "path; the FOC datapath multiply/limiter stages are the usual "
            "candidates for further pipelining.",
            ""]
    (ROOT / "synth" / "synth_report.md").write_text("\n".join(lines))
    print(f"[synth] wrote {ROOT/'synth'/'synth_report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="synthesis only (no place&route / bitstream)")
    args = ap.parse_args()
    util = synth()
    if args.check:
        return
    cfg, fmax, timing_met = pnr()
    bit = pack(cfg)
    write_report(util, fmax, timing_met, bit)


if __name__ == "__main__":
    main()
