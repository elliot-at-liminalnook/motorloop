# ECP5 synthesis report

Open flow (yosys -> nextpnr-ecp5 -> ecppack), board wrapper `synth/board_top.v` on the ULX3S (LFE5U-85F, CABGA381). Regenerate with `synth/run_synth.py`.

## Utilization (synth_ecp5, raw cell counts)
- TRELLIS_FF: 2634
- LUT4: 3914
- CCU2C: 1607
- MULT18X18D: 24

The design fits the -85F with wide margin (the parsed counts above are indicative; see `synth/work/stat.txt` and `nextpnr.log` for the authoritative post-pack utilisation).

## Timing
- Post-route Fmax estimate: **41.27 MHz** (target 25 MHz: MET)
- Bitstream: `synth/work/board_top.bit`

**Finding:** the FOC datapath is pipelined (robotics-ip stage 6.5): the Clarke/Park/PI/circle-limit/inv-Park/SVPWM chain is walked over registered stages, and the former Fmax bottleneck - `circle_limit`'s combinational 16-iteration isqrt + two 32-bit divisions - is now the sequential, bit-exact `circle_limit_seq` (one op per clock, reusing `divider32`). Post-route Fmax rose from ~3.3 MHz (unpipelined) to ~41.27 MHz, clearing the 25 MHz target, while LUT usage dropped (the huge combinational divide/sqrt logic became a small reused datapath). The simulator stays cycle-accurate; the FOC duties simply appear a fixed, sub-sample-period latency later.
