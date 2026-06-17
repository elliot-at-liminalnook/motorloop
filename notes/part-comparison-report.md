<!-- SPDX-License-Identifier: MIT -->
# Part-comparison study — results

Implements [`notes/part-comparison-checklist.md`](part-comparison-checklist.md):
a simulation study comparing the BOM part options on precision, position-tracking
speed, rapid full-speed reversals, and absolute limits — with a figure per test.

## The honest frame

- **Simulation, against the device *models*** (datasheet-grounded), not silicon.
- **Relative, not absolute.** Every test holds the FOC controller fixed and
  changes exactly one part; the motor gains are placeholder (Q1), so absolute
  thresholds (max RPM, max reversal speed) are *illustrative* — the orderings
  and ratios are the result.
- **No manufactured differences.** Where the models don't separate two parts, the
  study says so (see "What the models do *not* distinguish", below).

Code: experiments in [`sim/tests/part_compare.py`](../sim/tests/part_compare.py),
metrics in [`sim/tests/part_metrics.py`](../sim/tests/part_metrics.py), assertions
in [`sim/tests/test_part_comparison.py`](../sim/tests/test_part_comparison.py),
figures by [`sim/scripts/gen_comparison_figures.py`](../sim/scripts/gen_comparison_figures.py)
(`make compare`) → [`figures/comparison/`](../figures/comparison/gallery.md).

## What is compared, and how each is isolated

| Comparison | Parts | Isolated by |
|---|---|---|
| Angle sensor | **AS5600** (PWM, ~90 µs) vs **AS5047P** (SPI 14-bit, DAEC) | platforms `zonri_drv8301` vs `zonri_as5047p` — identical except `angle_name` |
| Current sampling | **MCP3208** (sequential skew) vs **ADS9224R** (simultaneous) | the FOC `current_sample_scheme` (1 vs 0) on a fixed stable platform |

The current-sampling comparison is deliberately *not* the whole-BOM `adc_dual_mode`
swap (`zonri_ads9224r` / `ti_reference_hp`): that also shifts the shared
placeholder loop to a different, oscillating operating point (id≈−14, iq≈28 vs
id≈0, iq≈60) — a tuning confound, not a part difference. The `current_sample_scheme`
toggle isolates the Q21 skew with everything else held.

## Findings (angle sensor — the headline)

- **T1 — angle latency.** The AS5600's commutation error grows ~linearly with
  speed (≈6° at 480 rpm → ≈90° around ~6,700 rpm, where torque inverts); the
  AS5047P stays near zero. The ratio is **~260×** at every speed — exactly the
  90 µs / 0.35 µs effective-latency ratio. This is Q22, quantified.
- **T2 — reversal cliff.** Max speed you can slam forward→reverse before the
  commutation error crosses 90°: AS5600 ≈ **3,800 rpm**, AS5047P ≥ **17,000 rpm**
  (sweep-capped; the true ratio tracks the ~260× latency ratio).
- **T5 — snap reversal.** Through an instantaneous ±Ω flip the AS5600's
  commutation error snaps ±40°+ and its phase current peaks higher (≈9.4 A vs
  ≈5.8 A) and phase-shifted; the AS5047P stays tight. Honest: with the
  stiff-supply default *neither latches a fault* — the difference is commutation
  integrity, not a tripped protection.
- **T6 — phase margin.** The AS5600's mean angle lag deepens with speed and it
  loses commutation around ~6,000 rpm; the AS5047P holds across the whole sweep.
- **T7 — delivered resolution.** The AS5047P resolves a finer step (≈0.006° vs
  ≈0.017°, **~3×**) — **not** the full 4× implied by 14-vs-12 bit, because the
  RTL/filter path truncates the AS5047P to 12 bits at the commutation boundary.
  Its real advantage is latency (T1), not raw resolution.
- **T8 — torque penalty.** mean(cos(angle error)) — the fraction of current that
  makes torque — decays for the AS5600 and goes **negative (braking)** at high
  speed; the AS5047P stays ~100%. The "so-what" of latency, in torque terms.
- **T10 — operating envelope.** Over (reversal speed × abruptness) the AS5047P is
  100% locked; the AS5600 ~50%. The boundary is set by reversal **speed**, not
  abruptness — peak angular rate is bounded by Ω, so a faster *slew* to the same
  Ω doesn't break commutation that the speed itself wouldn't.

## Findings (current sampling)

- **T3 — sample skew vs di/dt.** The sequential single-ADC (MCP3208) injects more
  dq current ripple than simultaneous sampling (ADS9224R), and the gap grows with
  di/dt (speed) — clearest at low/mid speed before the fundamental di/dt
  dominates both. This is Q21.
- **T4 — noise floor.** The sequential path sits on a measurably higher
  current-measurement noise floor.

## What the models do *not* distinguish (the honesty list)

- **Mechanical eccentricity (T9) hits both sensors ~equally** (≈2.4° angle-error
  std for each). A magnet/mounting runout is not retired by a better sensor IC —
  it needs a calibration/mounting fix. The disturbance (electrical) realism layer
  leaves the digital angle path essentially unchanged for both sensors.
- **Delivered angular resolution (T7) is closer than the datasheets imply** (~3×,
  not 4×) because the RTL truncates the AS5047P's 14 bits to 12 at the
  commutation interface. The headline AS5047P win is latency, not bit-depth.
- **The whole-BOM current-ADC swap could not be compared head-to-head** for
  "who survives" without confounding it with a loop-tuning shift (above), so the
  ADC comparison is scoped to the sampling-scheme skew (Q21), not a system
  envelope. T5's system snapshot is therefore scoped to the sensor pair.

## Reproduce

```
make test       # runs test_part_comparison.py with the rest of the suite
make compare    # re-renders all 10 figures -> figures/comparison/
```
