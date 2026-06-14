# FOC Conversion Checklist

Ordered tasks and code to convert the controller from sensored six-step
(trapezoidal) to field-oriented control (FOC) of a sinusoidal PMSM, and to
extend the bench so the result is verified to the same standard as the
six-step loop. Companion to the completed
[simulation-checklist](simulation-checklist.md),
[derivation-checklist](derivation-checklist.md), and
[realism-checklist](realism-checklist.md); architecture context in
[architecture](architecture.md). Created 2026-06-14; **completed
2026-06-14** — all stages implemented, FOC runs closed-loop in the bench
(mode 3) alongside six-step, suite green. Findings and recorded regression
numbers at the bottom.

**Why this is tractable in a week:** the FOC math blocks are textbook and
small; the genuine engineering is (a) fixed-point scaling discipline,
(b) the synchronized phase-current sampling architecture, and (c) angle
latency compensation. The bench is what makes it sane — the pipeline comes
up incrementally (Clarke alone → Park → current loop at fixed angle → full
cascade), every fixed-point sign/scale/overflow bug caught in pytest against
real plant physics before any board.

**Definition of done:** `pytest sim/tests` green with the new FOC tiers,
where green includes: (a) Clarke/Park/inv-Park/SVPWM unit parity against a
Python reference at fixed-point tolerance; (b) a closed current loop at a
*fixed* electrical angle regulating id and iq to commanded values with
bounded ripple and zero shoot-through; (c) a full closed-loop FOC speed run
spinning the simulated PMSM to a commanded speed with id≈0 held and iq
tracking load torque; (d) a torque-linearity scenario (commanded iq vs
achieved plant torque); (e) the angle-latency ripple study quantifying
torque ripple vs AS5600 lag across the speed range (feeds Q18/Q3); (f) a
field-weakening scenario above base speed; (g) the dead-time distortion
finding recorded. Every new parameter provenance-flagged. The three-way
electromechanical parity (C++/Python/Modelica) stays green throughout — the
control law change must not perturb the plant.

**Design decisions (pre-resolved):**

- **The plant does not change.** It is phase-variable abc with body diodes,
  so it already represents a sinusoidal PMSM correctly. FOC is enabled by
  *configuration* (`motor.emf_trapezoid_blend` → sinusoidal end) plus new
  RTL, not a plant rewrite. The oracle trio's scope stays frozen
  (realism-checklist design decision); it keeps validating the
  electromechanics with ideal physics regardless of control law.
- **Reuse over rewrite in the RTL.** `pwm_generator.v` is already
  center-aligned, complementary, dead-time-inserting, min-pulse-enforcing —
  it becomes the SVPWM output stage with one change (per-leg duty instead of
  one shared `duty_compare`). `speed_pi.v` stays as the outer loop, its
  output reinterpreted as iq\* (torque-current command). `commutation.v` is
  deleted; the sector-aware half of `adc_sequencer.v` is replaced. All
  protection (`drv_manager`), comms (UART/SPI/ADC masters), init, and the
  shoot-through/stall logic are untouched.
- **Modulation = min/max common-mode injection, not sector-dwell SVPWM.**
  Identical line-to-line result, a fraction of the logic, no sector-detect
  state. (Sector-dwell SVPWM is a drop-in alternative later if wanted.)
- **Fixed-point, documented Q-format.** Currents, voltages, and the dq
  states carry an explicit Q-format with saturation at every transform
  boundary; the convention lives in a short `notes/foc-fixed-point.md` and is
  enforced by the unit parity tests. Starting recommendation: signed Q15
  fractionals scaled to full-scale ADC range for currents/voltages, 16-bit
  electrical angle (sensor 12-bit × pole_pairs, wrapped). One serial
  multiplier reused across the chain (1250 clk/period at 20 kHz / 25 MHz is
  ample).
- **`sincos` = quarter-wave BRAM LUT.** Simpler and cheaper than CORDIC at
  this resolution; CORDIC stays an option if BRAM is tight on the GW5A.
- **Current-sampling architecture is a deliberate, bench-evaluated choice
  (new Q21), not assumed.** The single sequential MCP3208 cannot deliver two
  simultaneous phase currents (already recorded: Q17, architecture.md). The
  bench models the candidate schemes and the decision is made from their
  measured torque ripple, before any hardware commitment.
- **Angle latency is now first-order (new Q22).** For six-step the AS5600
  frame+filter lag is a minor commutation-timing shift; for FOC it rotates
  the dq frame off-true and the error grows with speed. Compensation
  (extrapolation by ω·t_latency) is part of the design, and its viability is
  quantified on the bench (extends Q18).

**Out of scope:** sensorless FOC (observer/flux estimation — separate
effort), MTPA / saliency exploitation (surface-PMSM assumed, Ld≈Lq), true
sector-dwell SVPWM (min-max injection is equivalent for this purpose),
dead-time *compensation* beyond recording the distortion it causes (a
follow-on if the bench shows it matters), hardware bring-up, motor purchase,
the second-ADC PCB work itself (the bench evaluates whether it is needed).

**Dependency notes:** Stage 0 first. Stages 1–3 are mutually independent
after it (plant config, modulator, math primitives) and each is testable
standalone. Stage 4 (sampling architecture) is independent of 1–3 but blocks
5. Stage 5 (current loop) needs 1–4. Stage 6 (full cascade) needs 5. Stages
7–8 need 6. Stage 9 last. Ordering below is by bring-up dependency, matching
the incremental-against-the-plant strategy.

## Stage 0 — Scaffolding and conventions ✅

- [x] 0.1 params.toml additions (all provenance-flagged): `[foc]` section —
      `id_target` (decided, 0 for surface PMSM), current-loop `kp_d/ki_d`,
      `kp_q/ki_q` (assumed/Q1, tuned against placeholder motor params),
      `v_circle_limit` (decided, fraction of bus), `modulation` enum
      (min_max_injection — decided), `angle_extrap_enable` and
      `angle_latency_s` (assumed/Q22, seeded from `[angle_sensor]`),
      `current_sample_scheme` enum (assumed/Q21). Keep `[rtl]` Q-format
      widths if any become build-time constants in `rtl_params.vh`.
- [x] 0.2 Open-questions: add **Q21 — synchronized phase-current sampling
      architecture for FOC** (dual ADC on separate SPI vs single-shunt
      DC-bus reconstruction vs sequential-with-skew; resolved by the
      stage-4 bench study + a board decision) and **Q22 — angle-latency
      compensation viability** (does ω·t_latency extrapolation recover
      enough frame accuracy across the speed range; extends Q18). Cross-link
      from the new params; note Q17/architecture.md already record the
      single-MCP3208 limitation.
- [x] 0.3 `notes/foc-fixed-point.md`: the Q-format convention (scaling of
      currents/voltages/angle, saturation rules, the single-multiplier
      reuse contract). One page, referenced by every math-block test.
- [x] 0.4 Bench config plumbing: `bench_factory.py` grows `[foc]` with
      defaults; a `foc(**overrides)` helper (mirrors `realism(...)`) selects
      sampling scheme and latency-compensation on/off per scenario.

## Stage 1 — Sinusoidal PMSM configuration and oracle ✅

- [x] 1.1 Confirm/standardize `motor.emf_trapezoid_blend` at the sinusoidal
      end for the FOC scenarios (document which end is which in the param
      note); leave the six-step scenarios on their existing value so both
      control laws keep their own honest plant config.
- [x] 1.2 dq-frame sanity oracle (Python): a standalone steady-state PMSM dq
      model (vd=R·id−ωe·Lq·iq, vq=R·iq+ωe·Ld·id+ωe·λm) used only to
      cross-check that the abc plant, driven by ideal sinusoidal voltages at
      a fixed dq operating point, lands on the expected id/iq. Not a new
      plant — a verification probe that the abc physics and the dq math
      agree. (This is the FOC analogue of the existing three-way parity.)
- [x] 1.3 Test: ideal-voltage open-loop at a fixed electrical frequency
      produces sinusoidal phase currents with the expected magnitude and the
      dq-projection matches 1.2 within tolerance.

## Stage 2 — Modulator (testable standalone, unblocks everything) ✅

- [x] 2.1 `pwm_generator.v`: per-leg duty. Replace the single
      `duty_compare` with `duty_compare[3]` (or three ports); `pwm_on`
      becomes per-leg (`counter < duty_eff[g]`). Keep center-alignment,
      dead-time, min-pulse, and the `kill` path exactly as-is. Six-step can
      still drive it (all-legs-PWM is a superset of the old behavior).
- [x] 2.2 `svpwm.v`: vα,vβ → three duties via min/max common-mode injection
      (compute three phase references, subtract (max+min)/2, scale to
      `PWM_HALF_PERIOD`, clamp). Fixed-point per 0.3.
- [x] 2.3 Tests: (a) RTL `svpwm` vs Python reference across the (vα,vβ)
      plane at fixed-point tolerance; (b) generated line-to-line voltages
      match ideal SVPWM (bus utilization ~15% over pure sine); (c)
      per-leg `pwm_generator` still passes the existing dead-time and
      min-pulse assertions; (d) shoot-through checker clean across the duty
      plane.

## Stage 3 — Fixed-point math primitives ✅

- [x] 3.1 `sincos.v`: quarter-wave BRAM LUT, 16-bit electrical angle in →
      signed sin/cos out. Test vs `math.sin/cos` at LUT tolerance.
- [x] 3.2 `clarke.v`: ia,ib → iα,iβ (iα=ia, iβ=(ia+2ib)/√3, √3 a Q-constant).
      Combinational. Parity test vs Python.
- [x] 3.3 `park.v` / `inv_park.v`: rotate by θe using 3.1. Parity tests,
      including the round-trip (Park∘inv-Park ≈ identity within tolerance).
- [x] 3.4 Bring-up note: each block gets a pybind-free Python twin in the
      test (or a numpy reference) so the parity tier mirrors the existing
      "executable spec" pattern.

## Stage 4 — Current-sampling architecture (the hardware crux, Q21) ✅

- [x] 4.1 Modeled the candidate schemes behind `current_sample_scheme`:
      (a) **dual ADC / external S/H** — `FeedbackChain` freezes all three
      low-side-shunt currents at the counter peak (bench latches on the
      `pwm_up` 1→0 edge), so the sequencer's two sequential reads both reflect
      the peak; (b) **sequential single-ADC** — the natural single-MCP3208
      behaviour, no S/H, each read live, carrying the real inter-conversion
      timing. **(c) single-shunt DC-bus reconstruction is NOT implemented** —
      it is a different sensor topology (bus-current sampling + active-vector
      reconstruction with minimum-vector blind zones); deferred and recorded
      as future work in the findings, since (a) vs (b) already resolves Q21.
- [x] 4.2 Added the FOC branch to `adc_sequencer.v` (`foc_mode`): samples
      ch0 (ia) then ch1 (ib) starting near the counter peak, removes the
      DC_CAL offsets, emits signed currents + `foc_valid`. The six-step
      sector-aware schedule is untouched (guarded), DC_CAL offset capture and
      the adc_stuck/offset guards are shared. Wired through `controller_top`
      (`ctrl_foc_sample`, `dbg_foc_*`) and the bench (`set_foc_sample`,
      `foc_cur_a/b`, `foc_valid`).
- [x] 4.3 Study (`test_foc_sampling.py`): scheme (a) recovers both phase
      currents (dq measurement error ~0.13 A — just peak-vs-read ripple);
      scheme (b) loses ~half of phase B (it samples ~22 µs late, after leg
      B's low-side conduction window closes), giving a **~12× larger dq
      measurement error (~1.5 A)** — the torque-command error a single
      sequential ADC would inject. **Q21 recommendation: simultaneous
      sampling (dual ADC or external S/H) is required; a single sequential
      MCP3208 is inadequate for FOC.** Recorded in the findings.

## Stage 5 — Current loop (torque control at fixed angle) ✅

- [x] 5.1 `current_pi.v` ×2 (d and q): parallel-form PI with anti-windup
      (clamp + back-calculation), fixed-point per 0.3.
- [x] 5.2 Voltage-circle limiter: cap √(vd²+vq²) ≤ `v_circle_limit`·vbus,
      prioritizing vd (field-weakening headroom) when saturated. Approximate
      magnitude allowed (e.g. octagon/α-max+β-min) — test the bound holds.
- [x] 5.3 `foc_current_path` wiring: Clarke → Park → {id,iq} PI → inv-Park →
      `svpwm` → per-leg `pwm_generator`, triggered on current-sample-ready,
      duties latched at `period_start`. Serial-multiplier FSM per 0.3.
- [x] 5.4 Scenario: **closed current loop at a fixed electrical angle** (rotor
      mechanically held, or angle forced) regulating id→0 and iq→commanded
      with bounded ripple, zero shoot-through. The first true closed-loop FOC
      milestone, and the cleanest place to shake out fixed-point bugs.
- [x] 5.5 Torque-linearity scenario: sweep iq\*, confirm achieved plant
      torque is linear in iq (Kt·iq) with id held at 0.

## Stage 6 — Full FOC cascade ✅

- [x] 6.1 `controller_top.v`: route the AS5600 angle (×`pole_pairs`, wrapped)
      to the dq transforms; reinterpret `speed_pi` output as iq\*; set id\*
      from `[foc].id_target`. New `ctrl_mode` value for FOC (keep six-step
      modes 0–2 intact so both control laws coexist for comparison).
- [x] 6.2 Closed-loop FOC speed run: spin the simulated PMSM to a commanded
      speed, id≈0 held, iq tracking load torque, bounded current, zero
      shoot-through. Plot against the six-step startup for the README.
- [x] 6.3 Suite hygiene: the three-way electromechanical parity stays green
      (the plant is unchanged) and the existing six-step scenarios stay green
      (modes preserved). Lint clean.

## Stage 7 — Angle latency compensation (Q22) and the ripple study ✅

- [x] 7.1 Angle extrapolation in the RTL: θe_used = θe_meas + ωe·t_latency
      (ωe from the speed estimate; t_latency from `[foc].angle_latency_s`),
      gated by `angle_extrap_enable`.
- [x] 7.2 **Ripple-vs-latency study:** sweep speed with the AS5600 model's
      real frame+filter latency, with and without 7.1, and quantify torque
      ripple and torque loss vs frame lag. This is the standout FOC demo —
      and the honest answer to "is the AS5600 good enough for FOC, and to
      what speed." Feeds Q18/Q3; record the numbers and a figure.

## Stage 8 — FOC scenarios, parity, and findings ✅

- [x] 8.1 `foc_reference.FocController` — the full control law (speed PI +
      id/iq current PIs + circle limit + transforms + SVPWM) in fixed-point,
      closed around the averaged C++ plant (`test_foc_reference_loop.py`). It
      reaches the commanded speed and **agrees with the RTL FOC loop's settled
      speed within 6%** (behavioral, not bit-exact — the RTL uses the switched
      bridge + ADC sampling + sensor latency, the reference the averaged plant
      with ideal sampling). Extends the executable-spec story to the control
      law.
- [~] 8.2 **Field weakening — NOT APPLICABLE to the placeholder motor
      (documented finding, no contrived test).** With Ke = 0.02 (placeholder)
      the voltage-limited base speed is ~570 rad/s, ~5× above the AS5600's
      usable range (~120 rad/s, Q18), so in the real operating envelope FOC
      never field-weakens. Demonstrating it requires artificial high-speed
      clamping where the simple controller (no automatic FW loop; linear
      ω·latency extrapolation that overshoots past ~300 rad/s) behaves
      pathologically. **Revisit when a real motor is identified (Q1)**: if its
      base speed falls within the sensor range, add the automatic FW loop
      (drive id\* on the circle-limiter saturation flag — the `sat` output is
      already exposed). Recorded honestly rather than forced green.
- [x] 8.3 **Dead-time distortion finding** (`test_foc_scenarios.py`): FOC
      chops all three legs continuously (no floating phase), so the bridge
      dead time + body-diode conduction distort the phase current — the bench
      models both. Measured at **~2–3 % residual-to-fundamental at low speed**,
      bounded, with the loop running cleanly through it (zero shoot-through).
      Surfaced, not fixed: dead-time *compensation* is a follow-on if a real
      motor shows it matters (out of scope here).
- [x] 8.4 New figures in `gen_readme_figures.py` (regen with `--only
      foc_startup,foc_sampling,foc_latency`): **FOC startup** (speed loop to
      target, id held at 0, iq carrying torque), **FOC sampling** (the Q21
      dq-error spikes of the sequential scheme vs the simultaneous one), and
      **FOC angle latency** (torque vs speed, raw angle vs extrapolation, the
      Q22 result). Same provenance caveat footer; added to the gallery.

## Stage 9 — Integration and docs ✅

- [x] 9.1 A UART host commanding mode 3 drives the FOC speed loop end-to-end
      (`test_foc_uart.py`): write align (reg 3), speed target (reg 2), take
      the control mux (reg 8), command mode 3 (reg 0) → the PMSM spins;
      telemetry reads back over the same link. Mirrors the six-step S11.
- [x] 9.2 `architecture.md` gains a FOC control-law section (Q21 sampling
      decision + Q22 latency outcome confirming the ADC-domain prediction);
      the README "Not FOC" caveat becomes "two control laws, one bench" with
      links to the checklist and the FOC gallery figures.
- [x] 9.3 Open-questions annotated: Q21 and Q22 bench-resolved (with the
      remaining hardware decisions), Q17 cross-linked to Q21's quantified
      result, Q18 updated with the FOC sensor-ceiling finding.

## Findings

**Completed 2026-06-14.** The controller runs FOC closed-loop in the bench
(mode 3): the speed loop spins the simulated PMSM to target, the current loop
holds id = 0, zero shoot-through. New tests: `test_foc_plant.py` (dq
cross-check), `test_foc_math.py` (RTL↔reference bit-parity, 184 cases),
`test_foc_sampling.py` (Q21), `test_foc_current_loop.py` (stage 5),
`test_foc_speed.py` (stage 6), `test_foc_latency.py` (Q22),
`test_foc_reference_loop.py` (8.1), `test_foc_scenarios.py` (8.3),
`test_foc_uart.py` (9.1). The shoot-through and dead-time guarantees the FOC
scenarios *observe* are now also formally **proven** for all states
(`notes/formal-checklist.md`, `formal/proof_report.md`) — and bringing FOC up
through formal flushed out a non-synthesizable `isqrt` in `circle_limit.v`
(data-dependent loop bounds), now fixed. All green; the pre-existing 361 tests
stay green
(parity + six-step + edge cases unperturbed).

Recorded numbers and gotchas:

1. **Current-PI gain was ~12× too high at first.** With KP = 24 (duty/LSB)
   even a small current error slammed the output to the voltage rail, where
   the anti-windup *freeze* locked it — presenting as a baffling sign
   asymmetry (one polarity of each axis converged, the other railed). The
   plant gain is ~0.95 LSB-of-current per duty-unit, so id = 60 needs vd ≈ 63;
   KP = 2 (KI shift 4) fixed it and all four sign combinations now regulate
   cleanly. Lesson: derive the loop gain from the plant's V→I gain, don't
   guess.

2. **FOC needs its own alignment offset.** The six-step
   `sensored_align_offset` (3755, a sector convention) gives ~75 % torque
   efficiency under FOC (≈41° frame error). A bench torque-maximizing sweep
   found `foc.align_offset = 160` → 100 % efficiency. Different control law,
   different sensor-to-flux alignment; hardware needs a real alignment routine
   (Q1/Q20).

3. **Q21 (sampling) — the headline finding.** Dual/simultaneous sampling:
   ~0.13 A dq measurement error (just peak-vs-read ripple). Sequential
   single-ADC: ~1.5 A (≈12×), because the second conversion samples phase B
   ~22 µs after the peak, by which point B's low-side shunt has stopped
   conducting. Confirms quantitatively the ADC-domain decision's prediction.

4. **Q22 (angle latency).** Raw AS5600 angle → torque falls with speed
   (≈10 %+ by 120 rad/s, placeholder params); ω·t_latency extrapolation
   recovers it and is neutral at low speed. Caveat: the *linear*
   extrapolation overshoots past ~300 rad/s (Δθ wraps) — irrelevant for this
   sensor's ~120 rad/s ceiling, but a real higher-speed sensor would want a
   wrapped/clamped form.

5. **Dead-time distortion** is present in every FOC run (all three legs chop
   continuously, unlike six-step's floating phase): ~2–3 % residual-to-
   fundamental at low speed, bounded, loop tolerant. Compensation is a
   follow-on if a real motor shows it matters.

6. **Field weakening is moot for the placeholder motor** (base speed
   ~570 rad/s ≫ the AS5600's ~120 rad/s range), so no automatic FW loop was
   wired and no contrived test forced — recorded as 8.2. The circle limiter's
   `sat` flag is exposed for a future FW loop once a real motor is identified.

7. **Plant test affordances added** (physics unchanged): `set_averaged_phase`
   (continuous 3-phase averaged drive) and `set_speed_clamp` (dyno hold) —
   both let FOC tests read clean electrical steady states without waiting out
   the J/B ≈ 2 s mechanical time constant.

Verification note: the full suite was run serially (per the
`sim/README.md` rule — never two pytest sessions at once; the conftest
rebuilds the shared module).
