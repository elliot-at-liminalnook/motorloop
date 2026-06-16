<!-- SPDX-License-Identifier: MIT -->
# ECP5 synthesis report

Open flow (yosys -> nextpnr-ecp5 -> ecppack), board wrapper `synth/board_top.v` on the ULX3S (LFE5U-85F, CABGA381). Regenerate with `synth/run_synth.py`.

## Utilization (synth_ecp5, raw cell counts)
- TRELLIS_FF: 1902
- LUT4: 14320
- CCU2C: 2783
- MULT18X18D: 22

The design fits the -85F with wide margin (the parsed counts above are indicative; see `synth/work/stat.txt` and `nextpnr.log` for the authoritative post-pack utilisation).

## Timing
- Post-route Fmax estimate: **3.29 MHz** (target 25 MHz: NOT met)
- Bitstream: `synth/work/board_top.bit`

**Finding:** the design synthesizes and fits, but the unpipelined FOC datapath (the `circle_limit` 16-iteration isqrt and the chained Clarke/Park/PI/SVPWM evaluated in one update) is a long combinational path that caps Fmax well below 25 MHz. A real 25 MHz board build needs that datapath pipelined (multi-cycle / registered stages); the simulator is cycle-accurate regardless. This is the synth flow surfacing exactly the RTL that won't map well at speed (stage 15.4).
