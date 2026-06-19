<!-- SPDX-License-Identifier: MIT -->
# Stress / extreme-scenario study — pushing the system to the limit

Drives the motorloop system + code to its failure/protection boundaries across
four axes and checks that the **protections — and the fixed-point math — hold.**
Tests: `sim/tests/test_stress.py`; figures: `make stress`
(`sim/scripts/gen_stress_figures.py`) → `figures/stress/`.

**Frame (honest):** simulation with placeholder motor gains (Q1) — the absolute
thresholds are illustrative; the value is the *behaviour at the edge* (does the
protection trip, does the math saturate vs wrap) made visible, each from a live
bench / ngspice / RTL run.

## A — System failure boundaries

- **A1 Thermal runaway** (`A1_thermal.png`): a locked-rotor cook at held current
  drives the winding **26 → 90 °C in 12 s** with no sign of stopping. The honest
  point: a pure current loop *faithfully holds the commanded current*, so nothing
  throttles the heating — **a current loop has no thermal foldback**; temperature
  runs away until a separate thermal limit (or damage) intervenes.
- **A2 Brownout cascade** (`A2_brownout.png`): a weak **0.7 A** supply driven by
  the closed-loop six-step controller — nothing injected. During spin-up the bus
  is dragged below the **5.9 V DRV8301 UVLO four times**, the RTL logs **4 nFAULT
  events**, then it stabilises. The protection (UVLO detect + re-init) works.
- **A3 Regen overvoltage** (`A3_regen.png`): a hard decel (110 → 10 rad/s) dumps
  the rotor's kinetic energy into the bus cap, **pumping it 12 → 15.7 V** — but
  the controller's **duty down-slew guard bounds it under the 20 V alarm**. The
  protection holds.
- **A4 Overcurrent command** (`A4_overcurrent.png`): an absurd torque command
  (iq = 1800) is **clamped to iq_max (~300)** and the dead-time guard holds
  **zero shoot-through at every command** — the protections, not the silicon, set
  the limit.
- **A5 Fault injection** (`A5_fault.png`): a latched DRV fault injected mid-run is
  **detected** (fault_count steps up) — the controller sees it.

## B — Control & sensor limits

- **B1 Reversal cliff** (`B1_reversal_cliff.png`): slamming forward↔reverse faster
  and faster, the **AS5047P survives a strictly higher reversal speed** than the
  AS5600 before the commutation error crosses 90° (loss of lock) — the dynamic
  limit, pushed to failure (ties to the part-comparison study).
- **B2 Extreme load step** (`B2_load_step.png`): an instant 0 → 0.8 N·m load
  spikes the current and dips the speed, but trips **no shoot-through** and the
  bus holds — a survivable extreme.

## C — Analog / ADC extremes

- **C1 Settling failure boundary** (`C1_settle_limit.png`): the design bucket
  (1 nF) settles **< 0.5 LSB within tACQ (140 ns)**; oversizing it (47 nF)
  **misses the window** — the real settling-vs-bucket-size boundary, found in
  ngspice.
- **C2 Full-scale clipping** (`C2_fullscale_clip.png`): beyond ±102 A the ADC code
  **rails at ±32768 (clamps, never wraps)** — a railed reading stays railed, it
  doesn't fold to the opposite sign.

## D — Numerical / code edge cases

- **D1 Fixed-point rails** (`D1_numeric_rails.png`): the **duties (the bridge
  commands) always clamp to [0, 625]** no matter how extreme the inputs — the
  output is never corrupted. Intermediate dq values *do* wrap, but **only past the
  18-bit rail (2× beyond any physical current)**; within the physical range the
  math is monotone, and the circle-limiter + SVPWM saturation downstream contain
  the rest.
- **D2 Circle-limiter saturation** (`D2_circle_sat.png`): the voltage-vector
  magnitude **saturates at the circle radius** however large the command grows —
  it does not run away or overflow.

## The takeaway

Across all four axes the system **degrades safely**: protections trip (UVLO,
overcurrent clamp, dead-time guard, regen down-slew), the arithmetic **saturates
rather than wraps** at the outputs, and the one genuine "no automatic limit" is
called out honestly (thermal — a current loop needs an explicit thermal foldback).
Every claim is a live run; reproduce with `make stress` + `pytest
sim/tests/test_stress.py`.
