<!-- SPDX-License-Identifier: MIT -->
# Motor selection: implement 3 real motors + compare performance

Retire the placeholder motor (Q1) by implementing three concrete, available
motors and comparing them on the bench:

| Tier | Motor | Key datasheet facts |
| --- | --- | --- |
| Budget | **iPower GM2804** (w/ AS5048A encoder) | 14 poles (**7 pp**), R ≈ 9 Ω, Kv ≈ 165, ≤25 W; L/J not published |
| Mid | **Nanotec DB42S03** | R_ll = **1.5 Ω**, L_ll = **2.1 mH**, 26 W, 24 V; ~4 pp (confirm); full datasheet |
| Premium | **maxon EC 45 flat** (12 V, 50 W) | full datasheet R/L/Kt/J; ~8 pp (confirm) |

**North star:** make each motor a provenance-tracked profile that drops into the
bench, then run a motor-comparison study (mirroring the part-comparison study) so
the three can be compared head-to-head — and tie the result back to the sensor
choice (pole-pairs × speed sets the angle-latency sensitivity).

## The honest boundary (built in)

- **Model-based comparison.** Datasheet numbers where published; gimbal L/J, all
  align offsets, and any `assumed` value need the physical motor + the bench
  motor-ID to become `measured`. This *partially* retires Q1 (datasheet) — full
  validation is still a bench session.
- **Pole pairs are build-time.** `gen_rtl_params.py` bakes `POLE_PAIRS`,
  `speed_num` and `EXTRAP_NUM` from `motor.pole_pairs`. The plant's pole_pairs and
  the RTL's must match, so **each motor with a different pole count needs a regen +
  re-Verilate** — the motor comparison is per-motor *build*, not a runtime swap
  (unlike the BOM/sensor swaps). Three different pole counts (7/4/8) → three builds.
- **12 V / ~1 A bench.** The GM2804 fits as-is; the DB42S03 and EC 45 run at 12 V
  but below rated — flag where the supply current limit must be raised.

---

## §0 — Motor-profile infrastructure (the foundation)

- [ ] **Motor overlays, not a global edit.** Add `sim/config/motors/<name>.toml`
      overlays, each a `[motor_spec]` (the measurables: `r_line_line`,
      `l_line_line`, `ke_line_line_peak`, `pole_count`) + the non-derivable
      `[motor]` fields (`J`, `B`, `trapezoid_blend`, cogging) + the per-motor
      `foc.align_offset`, each with `status`/`source`/`blocked_by`. Load via
      `sim_params.load(overlay)` so **the existing `derive_params` check validates
      `motor.R = r_line_line/2`, etc. per motor** — reuse, don't reinvent
      (`kv_to_ke_line_line`, the /2, /√3, //2 conversions already exist).
- [ ] **`MOTORS` registry** in `bench_factory.py` (mirrors `PLATFORMS`): name →
      overlay path + a one-line provenance summary; a `motor_cfg(params, name)`
      helper returning the `cfg["motor"]` override for runtime selection of the
      *plant* parameters.
- [ ] **Per-motor build workflow** `sim/scripts/build_motor.sh <name>`:
      load the overlay → `python3 sim/scripts/gen_rtl_params.py` (re-bakes
      `POLE_PAIRS`/`speed_num`/`EXTRAP_NUM`) → `bash sim/scripts/build_bench.sh`.
      Document loudly that this is build-time (the BOM swaps are not).
- [ ] **Conversion unit test** (`test_motor_profiles.py`): every overlay
      re-derives consistently (the `derive_params.check` must pass for each),
      and the measurable→per-phase conversions match the hand-checks.
- [ ] *(Optional, future)* a runtime-configurable `POLE_PAIRS` register in
      `controller_top` would make motor swaps runtime like the BOMs — note as a
      possible enhancement; not required here.

## §1 — Capture the three motors (params + provenance)

- [ ] **Archive datasheets** to `docs/datasheets/` (gitignored proprietary-
      reference): iPower GM2804, Nanotec DB42 (DB_Brushless PDF), maxon EC 45 flat
      (p/n 251601). Add them to the datasheets README with source URLs.
- [ ] **GM2804** (`motors/gm2804.toml`): `pole_count = 14` (7 pp);
      `ke_line_line_peak` from Kv ≈ 165 (`kv_to_ke_line_line`); `r_line_line` —
      **interpret the quoted 9 Ω** (terminal-to-terminal vs per-phase — note the
      ambiguity, `status="assumed"`, measure to confirm); `l_line_line` — **not
      published → estimate + flag `assumed`/measure**; `J`,`B` estimate;
      `trapezoid_blend ≈ 0` (sinusoidal gimbal); cogging order = LCM(12,14) = 84.
      Mostly `assumed` — honest (hobby part, sparse datasheet).
- [ ] **DB42S03** (`motors/db42s03.toml`): `r_line_line = 1.5`,
      `l_line_line = 2.1e-3` (`status="datasheet"`), `pole_count` from the
      datasheet (confirm; DB42 ~ 8-pole → 4 pp), `ke_line_line_peak` from rated
      torque/current (Kt → Ke), `J` from the datasheet rotor inertia,
      `trapezoid_blend` per the BEMF shape (confirm). **This is the provenance
      win** — most fields become `datasheet`.
- [ ] **EC 45 flat** (`motors/maxon_ec45.toml`): R, L, Kt→Ke, `J`, `pole_count`
      all from the maxon datasheet (`status="datasheet"`). Note the 12 V variant.
- [ ] Each overlay: set `status`/`source`/`blocked_by` honestly; what isn't on the
      datasheet stays `assumed`/`blocked_by="Q1"` (measure on the bench).

## §2 — Per-motor bring-up + calibration

- [ ] **align_offset per motor:** run the FOC torque-maximizing sweep
      (foc-checklist stage 5 method) for each motor → write `foc.align_offset` into
      its overlay. (Each motor + sensor mounting has its own dq alignment.)
- [ ] **Validate each motor runs** (`test_motors.py`, mirror `test_platforms.py`,
      one build per motor): init reaches `configured`, six-step spins, FOC closed
      loop holds zero shoot-through and tracks a speed target, current loop stable.
- [ ] **Operating-point sanity:** electrical time constant L/R, base speed
      (≈ V_bus/Ke) and rated current vs the 12 V / supply limit — flag any motor
      that needs the supply current limit raised or won't reach rated at 12 V.

## §3 — Motor-comparison study (the performance comparison)

Mirror the part-comparison study: `sim/tests/motor_compare.py` (runners),
`test_motor_comparison.py` (assert orderings), `gen_motor_figures.py` (figures).
The harness orchestrates **build-per-motor** (§0.3), runs the battery, collects
results, then plots cross-motor. Axes (each a metric + figure):

- [ ] **M1 Torque constant Kt** (N·m/A) — the fundamental torque-per-amp; bar.
- [ ] **M2 Torque–speed envelope at 12 V** — max torque vs speed, base speed
      ≈ V_bus/Ke; the classic motor chart, one curve per motor.
- [ ] **M3 Electrical time constant L/R** + current-loop step response — each motor
      needs different current-loop dynamics (and PI gains).
- [ ] **M4 Acceleration / dynamics** — torque-to-inertia (Kt·I / J); a speed-step
      response per motor.
- [ ] **M5 Torque ripple / smoothness** — cogging + commutation ripple (differs by
      slot/pole combo: 12N14P vs the DB/maxon); FFT of torque/current.
- [ ] **M6 Efficiency at a rated operating point** — mech power (τ·ω) / elec power;
      bar (expect maxon highest, GM2804 lowest).
- [ ] **M7 Thermal rise at rated current** — I²R heating vs time (reuse the stress
      A1 thermal scenario per motor).
- [ ] **M8 FOC angle-latency sensitivity (the tie-in):** the commutation error ≈
      pole_pairs × ω × t_latency, so **higher-pole motors (GM2804 7 pp, EC 45 8 pp)
      are far more sensitive to angle latency than the DB42 (4 pp)** — they *need*
      the AS5047P where the DB42 tolerates the AS5600. A figure of latency-induced
      torque loss per motor × sensor, connecting this study to the
      part-comparison/sensor result.

## §4 — Graphics

- [ ] `gen_motor_figures.py` → `figures/motors/`: the M1–M8 figures + a gallery.
      Carry the standing caveat (placeholder/datasheet provenance, model-based).
      Where a value is `assumed`/measured-pending, mark it on the figure.

## §5 — Report, tests, integration

- [ ] **`notes/motor-comparison-report.md`** — per-axis findings + the headline:
      which motor for which use (GM2804 = cheap/sensored/teaching; DB42 = the
      characterized mid; EC 45 = precision), and the **motor↔sensor coupling** (M8).
- [ ] **Assertions** in `test_motor_comparison.py`: each motor runs; predicted
      orderings hold (e.g. Kt(maxon) > Kt(DB42) > Kt(GM2804) at comparable size;
      GM2804/EC 45 more latency-sensitive than DB42) — or record "no modeled
      difference" honestly.
- [ ] **`make motors`** target (build-per-motor + render figures); CI runs the
      fast per-motor validation, the build-heavy sweep on demand (document runtime).
- [ ] **REUSE/SPDX** on all new files; datasheets proprietary-reference.
- [ ] **Promote Q1:** for DB42 / EC 45, the motor params move `placeholder →
      datasheet`; update `notes/open-questions.md` Q1 with what's now datasheet vs
      still bench-only.

## §6 — Honest boundary (what stays open)

- The comparison is **model-based**: datasheet + estimates. The GM2804's L/J, every
  `align_offset`, and any `assumed` value need the **physical motor + the bench
  motor-ID routine** to reach `measured`.
- The **12 V / ~1 A bench** under-drives the DB42/EC 45 at rated — the comparison at
  12 V is valid relative, but rated-power numbers need a stronger supply (note it).
- Pole-pair build-time coupling means the comparison harness rebuilds per motor —
  call out the per-motor `POLE_PAIRS`/`speed_num`/`EXTRAP_NUM` in the report.

## Done-when

Three motor overlays exist (each `derive_params`-consistent, provenance-tagged),
each builds + runs (init/six-step/FOC, zero shoot-through) with its own
`align_offset`, and the motor-comparison study reports M1–M8 with figures + a
report — including the motor↔sensor (pole-pairs × latency) coupling. `make motors`
reproduces it. Q1 is partially retired (DB42/EC 45 datasheet; GM2804 + all J/align
flagged measure-on-bench).

## What NOT to do

- Don't hand-edit the global `[motor_spec]` per motor — use overlays so the
  derive-check + provenance apply per motor.
- Don't compare motors at mismatched `POLE_PAIRS` (plant vs RTL) — always regen +
  rebuild per motor; a mismatch silently breaks commutation.
- Don't report absolute torque/efficiency as hardware truth for the `assumed`
  fields — label datasheet vs estimate vs measure-pending.
- Don't fabricate the gimbal's missing L/J — estimate with a flagged `assumed` and
  leave it for the bench.

## Implemented (results)

Done; full write-up + numbers in
[`motor-comparison-report.md`](motor-comparison-report.md). All new tests green.

- **§0/§1 profiles:** `sim/tests/motors.py` — three provenance-tagged `Motor`
  profiles (GM2804 / DB42S03 / EC 45 flat) holding the datasheet measurables +
  the measurable→per-phase conversions (reusing the bench's R=Rll/2, L=Lll/2,
  Ke=Ke_ll/√3, pp=poles/2 relations; checked vs `derive_params`). Honest per-motor
  provenance: DB42 mostly `datasheet`, GM2804 mostly `assumed`/measure, EC 45
  datasheet-typical (confirm 251601).
- **§0.3 per-motor build:** `sim/scripts/build_motor.sh <name>` +
  `gen_rtl_params.py` env override (`MOTORLOOP_POLE_PAIRS`) — the pole count is
  build-time. **Fixed a latent bug** (the `POLE_PAIRS` define re-read the param,
  desyncing it from `SPEED_NUM`/`EXTRAP_NUM` under an override); default builds
  stay byte-identical.
- **§2 bench validation:** the DB42 (4 pp = current build) runs the FOC loop with
  no rebuild — `test_db42_runs_on_current_build` (iq tracks, zero shoot-through).
  GM2804/EC 45 (7/8 pp) use `build_motor.sh` for a full bench run (on-demand; the
  comparison covers them rebuild-free).
- **§3/§4 comparison:** `test_motor_comparison.py` (7 tests: conversions +
  M2/M3/M6/M8 orderings + DB42 run) and `gen_motor_figures.py` (`make motors`) →
  `figures/motors/` (torque-speed, dynamics, efficiency, **M8 latency-coupling**,
  summary) + gallery. Headline: **more pole pairs → more angle-latency-sensitive**
  (EC 45 8pp loses ~53 % torque on the AS5600 at speed vs DB42 4pp ~14 %) — the
  motor choice and the sensor choice are one coupled decision.
- **§5 Q1:** adopting the DB42 (or EC 45) promotes the motor params
  `placeholder → datasheet` (see `notes/open-questions.md` Q1). Report +
  `make motors` reproduce.
- **Honest boundary held:** model-based; the GM2804's L/J, all `align_offset`s,
  and rated-power numbers (12 V under-drives the DB42/EC 45) still need the
  physical motor + the bench motor-ID to become `measured`.
