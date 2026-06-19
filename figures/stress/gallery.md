<!-- SPDX-License-Identifier: MIT -->
# Stress / extreme-scenario gallery

Pushing the system + code to the limit and showing the protections (and the
math) holding. Rendered by `sim/scripts/gen_stress_figures.py` (`make stress`);
asserted by `sim/tests/test_stress.py`. Write-up:
[`notes/stress-test-report.md`](../../notes/stress-test-report.md). **Caveat:**
placeholder motor gains (Q1) — illustrative edge behaviour, not hardware specs.

## A — System failure boundaries
### A1 Thermal runaway (locked-rotor cook)
![A1](A1_thermal.png)
### A2 Brownout cascade (weak supply → UVLO)
![A2](A2_brownout.png)
### A3 Regen overvoltage (hard decel pumps the bus, guard bounds it)
![A3](A3_regen.png)
### A4 Overcurrent command (iq clamp + zero shoot-through)
![A4](A4_overcurrent.png)
### A5 Fault injection (detected)
![A5](A5_fault.png)

## B — Control & sensor limits
### B1 Reversal cliff to loss-of-lock (AS5600 vs AS5047P)
![B1](B1_reversal_cliff.png)
### B2 Extreme load step (survivable)
![B2](B2_load_step.png)

## C — Analog / ADC extremes
### C1 Settling failure boundary (oversized bucket misses tACQ)
![C1](C1_settle_limit.png)
### C2 Full-scale code clipping (rails, no wrap)
![C2](C2_fullscale_clip.png)

## D — Numerical / code edge cases
### D1 Fixed-point rails (duties clamp; dq wraps only past the rail)
![D1](D1_numeric_rails.png)
### D2 Circle-limiter saturation
![D2](D2_circle_sat.png)
