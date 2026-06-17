<!-- SPDX-License-Identifier: MIT -->
# Part-comparison study — sensor & feedback-chain stress tests

Goal: a simulation study that **compares the BOM part options** on the axes the
user asked for — position **precision/exactness**, position-tracking **speed**,
**rapid full-speed reversals "without breaking"**, and **absolute limits** — with
a **figure for every test**. Primary pairs: angle sensors **AS5600 vs AS5047P**
and current ADCs **MCP3208 vs ADS9224R**; system-level the default BOM vs
`ti_reference_hp`.

## The honest frame (carried into every test + the report)
- **Simulation, against the device *models*** (datasheet-grounded), not silicon.
- **Relative, not absolute.** Every test holds the FOC controller *fixed* and
  changes exactly **one part**; the part-to-part *difference* is the result. The
  motor gains are still placeholder (Q1), so absolute thresholds (max RPM, max
  reversal rate) are indicative — report them as ratios / orderings, not specs.
- **No manufactured differences.** Where a test finds the models do *not*
  distinguish two parts on an axis, it records "no modeled difference" and the
  report says so — we do not invent a gap the sim doesn't have.
- Each test = (a) a pytest that asserts the **expected ordering** as a regression
  guard *and* dumps its data, (b) a rendered figure.

The method is exactly Q21 (current-sample skew, retired by ADS9224R) and Q22
(angle latency, retired by AS5047P) made visible and quantified.

---

## §0 — Harness foundation (shared by all 10 tests)

### §0.1 Isolated-variable platform profiles (`sim/tests/bench_factory.py`)
- [ ] Add derived profiles so each comparison changes **one part only** (the
      existing 8 profiles vary several fields at once). Build them by copying a
      base `PLATFORMS` entry and overriding a single field:
  - `cmp_angle_as5600` / `cmp_angle_as5047p` — identical except
    `angle_name` (`as5600`→`as5047p`) + `angle_spi_mode`. Same driver
    (`drv8301`), same ADC (`mcp3208`), same gains/shifts.
  - `cmp_adc_mcp3208` / `cmp_adc_ads9224r` — identical except `adc_name`
    (+ `adc_dual_mode`, `foc.current_sample_scheme`). **Verify which field
    actually selects the ADS9224R model** (note: `ti_reference_hp` carries
    `adc_name:"mcp3208"` *with* `adc_dual_mode:True` — confirm dual-mode routes
    to the simultaneous-sampling model regardless of `adc_name`, or fix the
    profile so the ADC pair is clean).
  - System pair reuses existing `zonri_drv8301` (default BOM) vs
    `ti_reference_hp` (DRV8323RS + ADS9224R + AS5047P) for the whole-chain tests.
- [ ] A `COMPARISONS` dict mapping a test's pair-name → `(profile_a, profile_b)`
      so tests/figures share one source of truth for what's being compared.

### §0.2 The comparison harness (`sim/tests/part_compare.py`)
- [ ] `build(platform, **overrides) -> bench` — thin wrapper over the existing
      bench_factory init (DRV config window, `set_mode(3)`, `set_foc_speed_loop`),
      asserting `bench.configured`. **Same FOC mode/gains for both members of a
      pair** — assert it.
- [ ] `run_sampled(bench, drive, dt, n) -> Trace` — the established
      `for _ in range(n): drive(bench, k); bench.run_for(dt); record(...)` loop
      (mirrors `test_foc_scenarios.py`). Each sample captures, as numpy arrays:
  - **truth**: `theta` (plant angle), `omega`, `currents` (3-phase), `bus_v`
  - **measured**: `angle` (RTL-decoded, sensor-agnostic — the angle the loop
    *uses*), `speed`, `encoder_angle_rad`, `foc_id`, `foc_iq`, `foc_cur_a/_b`
  - **break flags** (latched max over the run): `shoot_through_violations`,
    `locked_out`, `stalled`, `drv_dead`, `pvdd_uv_events`, `fault_count`,
    `mismatch_count`, `bus_v_max`/`bus_v_min`, `fet_tj_max_c`
- [ ] Reuse the C++ `bench.trace` dict where a high-rate column already exists
      (`time_s`, `omega_rad_s`, `speed_rad_s`, `theta_rad`, `ia_a/ib_a/ic_a`,
      `vbus_v`); the Python sampling loop covers measured-angle (no trace column
      for it today — do **not** add a C++ column unless a test needs >loop-rate).

### §0.3 Metrics (`sim/tests/part_metrics.py`)
- [ ] `angle_error_elec(measured, truth, pole_pairs)` — `wrap_pi` of the
      electrical-angle difference (reuse `gen_readme_figures.wrap_pi`). This is the
      commutation-relevant error; 30° elec = the "commutation breaks" threshold.
- [ ] `speed_error`, `current_vector_error(foc_id, foc_iq, truth_dq)` (Park the
      truth currents at the truth angle to get the reference dq), `ripple_pct`,
      `noise_floor_fft(signal, fs)` (rfft → dB floor), and
      `is_locked(trace)` → bool (no break flag set **and** speed estimate tracks
      truth within tolerance) — the shared lock/break classifier for T2/T5/T10.
- [ ] `expected_order(a_metric, b_metric, better="lower")` helper so each test
      asserts the predicted winner uniformly, and returns a "no difference"
      sentinel when `|a−b|` < a model-resolution epsilon (feeds §0.1 honesty).

### §0.4 Honesty preamble constant
- [ ] A `FRAME` docstring/constant in `part_compare.py` (the four bullets above),
      imported into the report generator so the caveat ships with the figures.

---

## §1 — Scenario primitives (stress generators; each returns a `drive` fn)

- [ ] `sweep_speed(lo, hi, n)` — `set_speed_clamp(True, ω)` stepped across the
      range (clean way to hold a known truth speed while sampling angle error).
- [ ] `reversal(omega, transition_s)` — `set_speed_clamp(False)` + FOC speed loop;
      command `+omega → −omega` over `transition_s`; the **rate knob** is
      `transition_s` (smaller = steeper slew). Used by T2/T5/T10.
- [ ] `iq_step(di_dt)` — fixed speed, step `set_iq_target` to sweep **di/dt** for
      the current-skew test (T3).
- [ ] `slow_rotate(omega_tiny)` — very low `set_speed_clamp` speed so the measured
      angle climbs in **quantization steps**; characterizes resolution (T7).
- [ ] `realism(level)` — enumerate the *actually-exposed* knobs and ramp them:
      `set_line_glitches`, `inject_magnet_loss` (eccentricity/runout proxy), the
      `chain` ground-shift/CSA-gain fields. **Only test layers the bench really
      exposes** (T9); list them in the report.
- [ ] `grid(speed_axis, rate_axis)` — the 2-D driver for the envelope map (T10);
      coarse default grid, runtime-budgeted.

---

## §2 — The 10 tests (`sim/tests/test_part_comparison.py`)

Group A — **angle sensor / position** (AS5600 vs AS5047P):

- [ ] **T1 Latency curve.** `sweep_speed`, metric = RMS & peak
      `angle_error_elec` vs RPM. Assert: AS5047P error < AS5600 at every speed
      above low-speed, AS5600 grows ~linearly. Dump → `latency_curve.json`.
- [ ] **T2 Reversal cliff.** `reversal` with shrinking `transition_s` until
      `is_locked` goes false; metric = **max reversal rate survived**. Assert:
      AS5047P max-rate ≥ AS5600. Dump the swept rates + the last-locked overlay.
- [ ] **T6 Phase-margin erosion.** Ramp to ceiling; metric = commutation lead/lag
      (electrical) vs RPM and **max RPM holding commutation**. Assert: AS5047P
      sustains ≥ AS5600. Dump margin-vs-RPM.
- [ ] **T7 Resolution staircase.** `slow_rotate`; metric = measured-angle step
      size (LSB) = **smallest resolvable move**. Assert: AS5047P step <
      AS5600 effective step. Dump truth-vs-measured staircase. *(No position-loop
      API — resolution is characterized directly from the slow-rotation angle.)*
- [ ] **T8 So-what penalty.** `sweep_speed`; metric = torque ripple % (from plant
      torque / phase current) **and** efficiency (mech power `τ·ω` vs elec power
      `≈ v_bus·i_bus`) vs RPM. Assert: AS5600 ripple ≥ AS5047P, efficiency ≤.
      Label efficiency a *modeled estimate*. Dump ripple+efficiency curves.

Group B — **current ADC** (MCP3208 vs ADS9224R):

- [ ] **T3 Simultaneity skew vs di/dt.** `iq_step` sweeping di/dt; metric =
      `current_vector_error` during the transient. Assert: ADS9224R ≈ flat,
      MCP3208 grows with di/dt. Dump error-vs-di/dt.
- [ ] **T4 Quantization noise floor.** Steady state; metric =
      `noise_floor_fft` of measured phase current / estimated torque. Assert:
      16-bit floor < 12-bit floor. Dump overlaid spectra.

Group C — **system / whole BOM** (default vs `ti_reference_hp`):

- [ ] **T5 Snap-reversal destruction.** Near-instant full-speed reversal; capture
      **peak phase current, `bus_v_max` (regen pump), `shoot_through_violations`,
      `pvdd_uv_events`, `drv_dead`, `locked_out`**. Report which BOM stays
      controlled+fault-free (relative). Dump currents+bus+fault-markers trace.
- [ ] **T9 Dirty bench.** Ramp `realism(level)`; metric = degradation
      (angle/current error growth, first break) per BOM. Assert: hp BOM degrades
      later. Dump degradation-vs-noise (only the exposed layers).
- [ ] **T10 Operating-envelope map.** `grid(speed × reversal-rate)`, classify
      each point `is_locked`; metric = the **locked region area** per BOM.
      Assert: hp envelope ⊇ default envelope. Dump the 2-D classification grids.
      *(Sweep-heavy → slow tier; coarse default grid, note the runtime.)*

---

## §3 — Figures (`sim/scripts/gen_comparison_figures.py`)
- [ ] Mirror `gen_readme_figures.py` (Agg backend, shared `caveat()` stamp, the
      standing "placeholder gains" note). Render all 10 from the §2 dumps into
      `figures/comparison/`:
  - T1/T6/T8 overlaid two-curve line plots (+ the 30°-elec / max-RPM markers).
  - T2 reversal overlay (truth vs estimate through the reversal) + max-rate bars.
  - T3 error-vs-di/dt, T4 overlaid FFT spectra (log-y, dB floor annotated).
  - T5 phase-currents + bus-voltage timeline with fault-event markers.
  - T7 quantization staircase, T9 degradation curves, **T10 the 2-D envelope
    heatmap (green=locked / red=broke) with both BOMs overlaid** — the finale.
- [ ] A `figures/comparison/gallery.md` (mirrors `figures/gallery.md`) embedding
      all 10 with one-line captions; SPDX-safe (no SPDX strings in generated MD
      body — reuse the existing REUSE handling).

---

## §4 — Report + integration
- [ ] **`notes/part-comparison-report.md`** — the §0.4 frame up top, then one
      section per test: the figure, the headline number/ordering, and explicitly
      **where the model could *not* distinguish the parts** (the honesty list).
      Tie T1/T6/T8→Q22 and T3/T4→Q21.
- [ ] **`make compare`** target → runs the fast tests + `gen_comparison_figures`.
      The fast subset (T1,T3,T4,T6,T7,T8) gates in CI; the sweep-heavy tier
      (T2,T5,T9,T10) runs on demand / nightly (document the runtime, like the
      litex_sim honesty note). Do **not** add the slow tier to the per-push gate.
- [ ] Run the *full* suite with **system `python3`** (not oss-cad — numpy shadow);
      rebuild the pybind module first via `sim/scripts/build_bench.sh` if the
      §0.1 profile changes touch C++ (they shouldn't — profiles are Python).
- [ ] SPDX header on every new `.py`/`.md`; REUSE-Ignore wraps if a generator
      emits an SPDX string. `reuse lint` clean.

## Done-when
All 10 tests pass (each asserts its predicted ordering or records "no modeled
difference"), `figures/comparison/` has 10 rendered figures + the gallery, and
`notes/part-comparison-report.md` summarizes the findings with the honesty list.
`make compare` reproduces it from a clean checkout.

## What NOT to do
- Don't compare a pair where >1 part differs — use the §0.1 isolated profiles.
- Don't report absolute max-RPM / max-reversal-rate as hardware specs — they ride
  on placeholder gains (Q1); report orderings/ratios.
- Don't manufacture a difference the models don't have — record it and say so.
- Don't put the sweep-heavy tier (T2/T5/T9/T10) in the per-push CI gate.

## Implemented (results)

All four sections done; **all 10 tests pass** (`test_part_comparison.py`, ~50 s
in the suite) and **all 10 figures render** (`make compare`, ~1m50s). REUSE
clean (356/356). Full write-up: [`part-comparison-report.md`](part-comparison-report.md);
figures + captions: [`figures/comparison/gallery.md`](../figures/comparison/gallery.md).

- **§0 harness** — `sim/tests/part_metrics.py` (pure metrics) +
  `sim/tests/part_compare.py` (build/sample + 7 runners feeding the 10 tests) +
  `COMPARISONS` in `bench_factory.py`. The angle pair is already isolated
  (`zonri_drv8301` vs `zonri_as5047p` differ only in `angle_name`); no redundant
  profiles added. Measured signal = `bench.encoder_angle_rad` vs truth
  `bench.theta`; break flags = shoot-through/locked/stalled/pvdd_uv/fault_count.
- **§1 primitives** — speed sweep, prescribed (dyno) reversal, di/dt skew sweep,
  steady spectrum, snap, slow-rotate staircase, realism breakdown, 2-D envelope.
- **§2 the 10 tests** — assert the predicted ordering or record a no-difference
  sentinel (T9). Sized coarse for CI; figures re-run finer.
- **§3 figures** — `sim/scripts/gen_comparison_figures.py` → `figures/comparison/`
  (10 PNGs + `gallery.md`). `make compare` target added.

**Three honest findings baked in:** (1) the AS5047P win is **latency** (~260×,
T1/T2), not bit-depth — delivered resolution is only ~3× (T7, RTL truncates
14→12 bit); (2) **eccentricity hits both sensors equally** (T9) — an IC upgrade
doesn't fix a mounting problem; (3) the ADC comparison is the **skew toggle**
(Q21), because the whole-BOM `adc_dual_mode` swap confounds the placeholder loop.

**Key empirical discoveries during build:** the closed-loop speed PI *runs away*
with placeholder gains (so reversals are dyno-prescribed via `set_speed_clamp`,
which isolates feedback-chain breakage cleanly); the `cur_norm_shift` scaling is
consistent across platforms (foc/truth ratio ≈24.7) so the confound was the
operating point, not the units.
