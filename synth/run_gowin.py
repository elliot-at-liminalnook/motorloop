#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Open Gowin GW5A place&route for controller_top (release-and-portability §4.2):
a real post-route Fmax on a SECOND FPGA family with no vendor tools. Targets the
Tang Primer 25K (GW5A-25A) via the open flow yosys `synth_gowin` ->
`nextpnr-himbaechel` (the chipdb + apicula DB ship with the OSS CAD Suite).
Complements the open ECP5 flow (synth/run_synth.py) and the yosys resource
estimates (synth/portability.py).

  source ~/oss-cad-suite/environment
  python3 synth/run_gowin.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "synth" / "work"
DEVICE = "GW5A-LV25MG121NC1/I0"          # Tang Primer 25K (GW5A-25A die)
CST = ROOT / "synth" / "tangprimer25k.cst"   # board pin constraints (user-supplied)


def rtl_files() -> list[str]:
    ys = (ROOT / "synth" / "synth_ecp5.ys").read_text()
    files = re.findall(r"(rtl/\S+\.v)\b", ys)        # board_top.v is synth/*, excluded
    seen: set[str] = set()
    return [f for f in files if not (f in seen or seen.add(f))]


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    reads = "\n".join(f"read_verilog -I rtl -I rtl/gen {f}" for f in rtl_files())
    json_out = WORK / "gowin.json"
    ys = f"{reads}\nsynth_gowin -family gw5a -top controller_top -json {json_out}\n"
    ylog = WORK / "gowin_yosys.log"
    with ylog.open("w") as f:
        ry = subprocess.run(["yosys", "-p", ys], cwd=ROOT, stdout=f,
                            stderr=subprocess.STDOUT)
    if ry.returncode != 0 or not json_out.exists():
        sys.exit(f"[gowin] yosys synth_gowin FAILED; see {ylog}")
    print(f"[gowin] synth_gowin OK - controller_top maps to {DEVICE} (json: {json_out})")

    # nextpnr-himbaechel (gowin) requires ALL top I/O pin-constrained - there is
    # no --lpf-allow-unconstrained as on ECP5 - so a real post-route Fmax needs
    # the board's .cst. Without it, the mapping above is the portability result;
    # the open ECP5 flow (run_synth.py) is the authoritative open-P&R Fmax.
    if not CST.exists():
        print(f"[gowin] place&route skipped: supply {CST.name} (Tang Primer 25K "
              f"pin constraints) for the post-route Fmax - himbaechel requires "
              f"constrained I/O. synth_gowin mapping is the portability proof.")
        return 0
    nlog = WORK / "gowin_nextpnr.log"
    with nlog.open("w") as f:
        rp = subprocess.run(
            ["nextpnr-himbaechel", "--device", DEVICE, "--json", str(json_out),
             "--vopt", f"cst={CST}", "--freq", "50",
             "--report", str(WORK / "gowin_report.json")],
            cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    fm = re.findall(r"Max frequency for clock '[^']*':\s*([\d.]+)\s*MHz", nlog.read_text())
    fmax = max(map(float, fm)) if fm else None
    if rp.returncode != 0 and fmax is None:
        sys.exit(f"[gowin] nextpnr-himbaechel FAILED; see {nlog}")
    print(f"[gowin] {DEVICE}: "
          f"{'post-route Fmax ~ %.2f MHz' % fmax if fmax else 'routed'}")
    return 0


if __name__ == "__main__":
    main()
