<!-- SPDX-License-Identifier: MIT -->
# OpenLane (sky130) ASIC flow — robotics-ip stage 10

Two layers, deliberately honest about what each proves:

## 1. ASIC-readiness smoke (runs here, in CI)

`synth/asic_smoke.py` drives **yosys** to gate-level synthesis for every
reusable block and asserts it is ASIC-clean — maps to gates, **no inferred
latches**, **no combinational loops / multiple drivers** (`check -assert`) — and
reports the gate-cell count. See `synth/asic_smoke_report.md`. This needs only
the OSS CAD Suite (no PDK), so it gates every push.

```
source ~/oss-cad-suite/environment
python3 synth/asic_smoke.py --check
```

**What it proves:** the RTL is ASIC-*synthesizable* and structurally clean.
**What it does NOT prove:** real-PDK area, post-route timing, DRC/LVS, or a
tapeout. Those need the full flow below.

## 2. Full OpenLane RTL-to-GDS (runs where OpenLane is installed)

`foc_core.json` is an [OpenLane 2](https://openlane2.readthedocs.io) config for
the pipelined `foc_core` (synth → floorplan → place → CTS → route → STA →
GDS) on sky130. It is **not** run in CI (OpenLane needs OpenROAD + magic +
KLayout + the sky130 PDK, ~GBs, typically via Docker/Nix), but is wired so a
careful org can reproduce it:

```
python3 sim/scripts/gen_rtl_params.py     # produce rtl/gen/sincos_init.vh first
pip install openlane                       # or use the OpenLane Docker image
openlane synth/openlane/foc_core.json      # writes runs/<tag>/ with the GDS + STA
```

The 50 ns (20 MHz) `CLOCK_PERIOD` is a conservative ASIC smoke target. Honest
framing: an OpenLane run here would establish **ASIC-synthesizable + place&route
clean at a stated clock on sky130** — still a *smoke*, not silicon validation.
The credible speed number today is the open ECP5 FPGA flow's **41.3 MHz** real
post-route Fmax (`synth/synth_report.md`, stage 6.5).
