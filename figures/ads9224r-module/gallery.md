<!-- SPDX-License-Identifier: MIT -->
# Open ADS9224R module — figure gallery

Rendered by [`sim/scripts/gen_ads9224r_figures.py`](../../sim/scripts/gen_ads9224r_figures.py)
(`make ads9224r`). `scaling.png` and `settling.png` are live ngspice runs of the
front-end models (`sim/circuits/ads9224r_*.cir`) — the same models the derivation
tests assert on. **Standing caveat:** designed + simulated against the device
models; values EVM-baseline / datasheet-typical (Q23) — not hardware measurements.
Board docs: [`hw/ads9224r-module/`](../../hw/ads9224r-module/README.md).

### Signal chain
![signal_chain](signal_chain.png)
Low-side shunt → THS4551 FDA → RC charge-bucket → ADS9224R, buffered reference;
one CONVST samples both phases together.

### Simultaneity (why this part)
![simultaneity](simultaneity.png)
Simultaneous sampling vs the single-ADC skew the part-comparison study (T3) measures.

### Scaling (ngspice)
![scaling](scaling.png)
DC transfer: 320 codes/A, ±102 A full-scale at the default shunt/gain.

### Acquisition settling (ngspice)
![settling](settling.png)
The charge-bucket settles below 0.5 LSB within the acquisition window.

## Sim-validation (Tiers 2–4) — see `notes/ads9224r-sim-validation-report.md`

### Front-end noise → ENOB (Tier 2, ngspice `.noise`)
![noise](noise.png)
The antialiasing feedback cap bounds the wideband-noise ENOB cost to <0.5 bit
(the charge bucket alone would cost ~1.9 bits — the SBAA282 finding).

### Validated front-end → FOC current loop (Tier 4)
![loop_budget](loop_budget.png)
The validated front-end contributes ~1.3 mA RMS — balanced against the ADC's own
transition noise — for ~15 effective bits at the loop, with zero inter-channel
skew (Q21).
