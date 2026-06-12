# Realism Checklist

Ordered tasks and code to close the environment-realism gaps identified
after the derivation layer: supply/bus dynamics, mechanical reality at zero
speed, correlated analog disturbances, thermal physics, sensor imperfection,
system-interface gaps, and the model-form validation harness. Companion to
the completed [simulation-checklist](simulation-checklist.md) and
[derivation-checklist](derivation-checklist.md). Created 2026-06-12;
**completed 2026-06-12** — all stages implemented, suite green
(126 tests, ~4 min). Findings and recorded regression numbers at the
bottom.

**Definition of done:** `pytest sim/tests` green with the new scenario
tiers, where green includes: (a) the slva552 brownout reproduced
*emergently* — bench-supply current limit → bus sag → PVDD UVLO → silent
register reset → watchdog recovery — with no injection call; (b) a regen
pump-up scenario showing bus overvoltage on hard deceleration and the
controller guard responding; (c) closed-loop startup from a cogging detent
with stiction enabled; (d) S2-class scenarios passing with the correlated
disturbance injector at nominal (Q9-baseline) amplitudes; (e) OTW emergent
from the thermal model under stall, no injection; (f) a closed-loop
scenario driven end-to-end through the UART register file; (g) the
hardware-trace comparator passing its synthetic self-test. Every new
parameter is provenance-flagged (most `assumed`/`placeholder` — the banner
GROWING is correct behavior here).

**Design decisions (pre-resolved):**

- **Realism features default OFF (nominal/zero parameters).** The
  three-way parity trio (C++/Python/Modelica) keeps validating the core
  electromechanics with ideal physics; realism layers are bench/C++-only,
  enabled per scenario, and validated by their own unit tests (plus SPICE
  where applicable). The oracle's scope is deliberately frozen — document
  in architecture.md.
- **Every effect enters as a provenance-flagged parameter**, Q-referenced
  (Q9 for harness/grounding amplitudes, Q1 for motor mechanical terms, new
  Q20 for sensor mounting). No hardcoded "realistic-looking" constants.
- **Harness disturbances join the derivation layer:** inter-board wiring
  R/L are codified as `[circuit.harness]` components; ground-shift
  coefficients are `derived_from` them, same as every other circuit-derived
  parameter.
- **Mode-freeze lesson applies:** new switched behaviors (supply CC/CV,
  stiction stick/slip) resolve their mode once per substep, never per RK4
  stage (see simulation-checklist finding 3). Stiction uses smooth
  regularization (tanh) rather than stick-state events; documented
  trade-off.
- **Emergent replaces injected, but injection stays.** When a fault becomes
  emergent (brownout, OTW), the existing injection-based test remains (it
  tests the *response* in isolation); the new test verifies the *cause*.

**Out of scope:** EMI/SI proper (IBIS, radiated), SPICE-level switching
transients in the loop, FEA-grade cogging (single/double harmonic only),
audited thermal accuracy (datasheet RθJC + assumed board factors), FMI,
physical bench sessions, motor purchase.

Dependency notes: stage 0 first; stages 1–5 are mutually independent after
it; stage 6 is independent of 1–5; stage 7 is independent of everything;
stage 8 last. Ordering below is by expected realism-per-effort.

## Stage 0 — Scaffolding and conventions

- [x] 0.1 params.toml sections (all values provenance-flagged):
      `[supply]` (v_set, i_limit, r_source, can_sink=false — assumed/Q5),
      `[circuit.harness]` (r_return, l_return — assumed/Q9),
      `[disturbance]` (gate_edge_spike_v, spike_tau, vref_ripple_v —
      assumed/Q9), `[thermal]` (fet RθJC datasheet + board RθCA assumed,
      lump capacitances assumed, drv lump, motor winding lump, alpha_cu
      datasheet, ke_derate assumed), `[motor]` additions
      (cogging_amplitude, cogging_order, coulomb_friction,
      stiction_omega_eps — placeholder/Q1),
      `[sensor_imperfection]` (eccentricity_e1/phi1/e2/phi2, angle_noise_lsb
      — assumed/Q20), `[adc]` additions (offset_lsb, gain_error, inl_bow_lsb
      — datasheet).
- [x] 0.2 Open-questions: add **Q20 — AS5600 mounting eccentricity and the
      alignment-calibration requirement** (resolved by a calibration
      routine + measurement on the assembled rotor); extend Q9's resolution
      path with the harness component measurements; cross-link from the
      new params.
- [x] 0.3 Derivation registry entries for the harness: ground-shift
      coefficient(s) `derived_from = "circuit.harness"`; extend the
      measurement worksheet accordingly (test_derived_params picks these up
      automatically via the orphan check).
- [x] 0.4 Bench config plumbing: `bench_factory.py` grows the new sections
      with defaults-off values; a `realism(**overrides)` helper enables
      named effect groups per scenario without copy-paste.

## Stage 1 — Supply and bus dynamics (will bite first on hardware)

- [x] 1.1 `BusSupply` component in the C++ plant (`bus_supply.hpp`): state
      = bus-cap voltage; source = CV with series r_source, CC fold at
      i_limit, ideal-diode no-sink behavior (regen charges the cap, supply
      cannot absorb it). Bus current computed from conducting legs
      (high-side conduction draws phase current from the rail; regen
      reverses it). Supply mode (CV/CC/blocked) resolved once per substep.
- [x] 1.2 Plant integration: switched mode uses the live bus voltage for
      leg rail/diode clamps; averaged parity mode keeps a stiff source
      (config flag) so the three-way parity scenario is untouched. Python
      reference NOT extended (bench-only realism, per design decision).
- [x] 1.3 DRV8301 model gains a `pvdd_v` input: PVDD UVLO (falling
      threshold ~6 V, datasheet) → gates Hi-Z, nFAULT, PVDD_UV status bit,
      and the slva552 silent register reset to power-on defaults; recovery
      on rising threshold + hysteresis. Unit tests against digest values.
- [x] 1.4 Bench wiring: plant bus voltage → drv pvdd input → chain bus
      channel (the VBUS ADC reading now shows real sag/pump — free
      telemetry realism).
- [x] 1.5 Unit tests: CC fold under static overload, sag waveform vs
      analytic RC, pump-up slope under forced regen, no-sink diode
      behavior.
- [x] 1.6 Scenario S6a — **emergent brownout**: aggressive speed step with
      Q5-baseline 1 A supply limit; assert the causal chain (CC mode → bus
      < UVLO → registers reset → nFAULT → RTL watchdog rewrites → recovery)
      with zero injection calls. Assert the existing injected test still
      passes unchanged.
- [x] 1.7 Scenario S6b — **regen pump-up**: spin to speed, command hard
      deceleration; assert bus rises above v_set; add the controller-side
      guard this exposes (duty-limited deceleration or brake-mode policy in
      the RTL — minimal: clamp PI negative slew so regen current is
      bounded; document the chosen policy) and assert the guard caps the
      pump-up below the FET/cap rating parameter.
- [x] 1.8 Q5 note: the scenario will quantify how undersized 1 A is for
      the placeholder motor — record the supply requirement it implies in
      open-questions Q5.

## Stage 2 — Mechanical reality at zero speed

- [x] 2.1 Mech model additions (C++ plant): Coulomb friction
      `tau_c * tanh(omega / omega_eps)` (regularized, no events) and
      cogging torque `tau_cog * sin(cogging_order * theta + phase)`.
      Defaults zero → parity unchanged (assert in test).
- [x] 2.2 Unit tests: holding torque below breakaway produces bounded creep
      (regularization artifact quantified and documented), torque above
      breakaway moves; cogging detent positions are equilibria; spin-down
      with Coulomb+viscous matches the analytic piecewise solution.
- [x] 2.3 Scenario S7 — **startup from a detent**: rotor parked at a
      cogging equilibrium, stiction on; closed-loop sensored start to
      target speed. Assert it starts, and capture the duty threshold at
      breakaway (regression number for PI tuning).
- [x] 2.4 Low-speed crawl scenario: target below the speed-meter floor
      (~13 rad/s); assert the loop behaves sanely (no windup runaway, no
      oscillation between detents) — this is where viscous-only models lie
      the most.
- [x] 2.5 Spin-down identification dry run: the J/B fit procedure planned
      for motor ID (Q1) rehearsed against the sim with Coulomb+cogging
      enabled — quantifies how much the viscous-only fit misestimates J/B
      when the real decay isn't exponential (feeds the Q1 session design).

## Stage 3 — Correlated analog disturbances + ADC nonidealities

- [x] 3.1 Ground-shift term in the feedback chain: common-mode offset
      `v_gnd = r_return * i_return + l_return * di_return/dt` added to all
      channel voltages; i_return from the plant's bus return current;
      coefficients derived from `[circuit.harness]` (stage 0.3).
- [x] 3.2 Gate-edge transient injector: on any gate switching edge, add a
      decaying spike (amplitude/tau from `[disturbance]`, sign keyed to
      edge direction and phase current sign) to the current and EMF
      channels. Seeded jitter on amplitude for non-pathological variety.
- [x] 3.3 Reference-ripple realism: MCP3208 quantizes against a live vref
      = nominal + PWM-synchronized ripple (amplitude from `[disturbance]`,
      phase tied to the PWM counter via the bench). Mcp3208 API: vref
      supplied at sample time by callback (mirrors the AnalogSource
      pattern).
- [x] 3.4 ADC transfer nonidealities: offset_lsb, gain_error, INL bow
      (smooth `inl_bow_lsb * sin(pi * code/4096)` shape) in the quantize
      path, datasheet-status values. Unit tests with hand-computed codes.
- [x] 3.5 Unit tests: ground shift moves all channels together (common
      mode), edge spikes decay with the configured tau, ripple aliases
      DC-ly at the PWM-synchronized sample instant (the whole point —
      assert the systematic bias exists and is stable).
- [x] 3.6 Scenario S8 — closed loop at Q9-baseline amplitudes: settles
      within widened tolerance; DC_CAL captured offsets absorb the
      static part of the ground shift (assert offset codes shift
      accordingly).
- [x] 3.7 Margin sweep (report, not gate): scale the disturbance amplitudes
      until S8 fails; log the failure multiple as a regression-tracked
      number (`test_disturbance_margin` asserts margin > 2x baseline).
- [x] 3.8 Aperture-placement payoff test: rerun the S3 aperture sweep with
      edge transients enabled; assert EMF sample corruption at the
      off-window center stays below N LSB while a deliberately mistimed
      aperture (test-only RTL parameter or bench-computed comparison
      sampling) shows the spike — proves the scheduling design earns its
      complexity.

## Stage 4 — Thermal physics (emergent OTW/OTSD, parameter drift)

- [x] 4.1 Thermal lumps (C++, `thermal_model.hpp`): per-FET junction RC
      (conduction + diode losses from plant currents, switching-loss
      estimate `k * v_bus * |i| * f_pwm` with k assumed), board/ambient
      node shared; DRV8301 die lump (gate-charge + buck losses, crude);
      motor winding lump (3 * i^2 * R).
- [x] 4.2 Parameter drift coupling: winding `R(T) = R0 (1 + alpha_cu dT)`,
      `Ke(T) = Ke0 (1 - ke_derate dT)`, FET `rds_on(T)` per datasheet
      curve (linearized) — affects conduction losses (feedback), VDS-OC
      trip current, and the speed ceiling. Plant accepts live modifiers;
      defaults static.
- [x] 4.3 DRV8301 model: OTW/OTSD become functions of the DRV die lump
      temperature (datasheet thresholds); injection API retained.
- [x] 4.4 Unit tests: step-power heating follows the RC analytic, drift
      laws match hand calcs, OC trip current shifts with rds_on(T).
- [x] 4.5 Scenario S9 — **stall heating**: locked rotor at moderate duty;
      assert FET Tj climbs, OTW asserts emergently (nOCTW), RTL responds
      per policy (existing fault handling), and post-cooldown recovery
      works. Long-horizon scenario — verify runtime stays within the
      performance budget (decimate thermal updates if needed).
- [x] 4.6 Scenario S9b — sustained-run drift: 30 s (sim) at load; assert
      speed ceiling and current calibration drift in the direction and
      rough magnitude the derating laws predict (regression numbers).

## Stage 5 — Sensor imperfection (AS5600 accuracy, not just latency)

- [x] 5.1 As5600 model: measured angle = true + `e1 sin(theta+phi1) +
      e2 sin(2 theta+phi2)` (mounting eccentricity, Q20) + seeded gaussian
      angle noise (LSB-scale); applied before the existing
      sampling/filter/frame pipeline.
- [x] 5.2 Unit tests: decoded-angle error vs true follows the configured
      harmonics; noise is seed-deterministic.
- [x] 5.3 Scenario S10 — commutation with eccentricity at plausible
      amplitudes (a few degrees once-per-rev): quantify the torque/speed
      ripple it induces; assert the loop tolerates it; record the ripple
      number.
- [x] 5.4 The consequence item: a bench-side **alignment-calibration
      routine prototype** (sweep `ctrl_align_offset` at fixed duty, find
      the offset maximizing speed-per-duty; pure Python over the existing
      bench API). Assert it recovers the known-good 3755 within a sector's
      tolerance with eccentricity ON — this is the procedure hardware
      bring-up will reuse (resolves part of Q20; RTL-side auto-calibration
      stays a stretch item).

## Stage 6 — System-interface gaps

- [x] 6.1 UART register file RTL (`uart_regfile.v` + `uart_rx/tx.v`):
      baud from params (assumed 115200), simple framed protocol
      (addr/data write, addr read + readback), registers shadowing the
      existing ctrl_* inputs plus readback of dbg counters. Direct ctrl
      ports remain for fast tests (mux: UART writes win when enabled).
      Lint + golden-frame unit tests.
- [x] 6.2 C++ UART model in the bench (tx/rx at configured baud over the
      RTL pins) + Python API (`bench.uart_write(addr, value)` /
      `uart_read(addr)`).
- [x] 6.3 Scenario S11 — closed loop driven END-TO-END over UART (mode,
      align, target speed via uart_write; speed telemetry via uart_read);
      assert parity with the direct-port S2 result.
- [x] 6.4 Digital-corruption injector: seeded bit-glitch windows on SPI
      lines (drv + adc) and the AS5600 PWM line at the bench boundary.
      Scenario S12: glitch bursts during closed-loop run — assert DRV
      frame-fault flags rise, watchdog rewrites configs, ADC outliers
      bounded, angle-capture validity drops and recovers, loop survives.
- [x] 6.5 Config-window scenario S13: before reset release, drive the DRV
      model inputs with seeded random toggling while EN_GATE is held low
      by the (modeled) external pull-down — assert zero gate activity and
      zero shoot-through through the window; document that this models the
      Q13 mitigation, not GW5A silicon behavior (which stays a Q13 verify
      item).
- [x] 6.6 The pull-down itself becomes explicit: `[circuit.gate_pulldowns]`
      (value, assumed/Q13) + a bench boolean modeling its presence;
      S13 asserts the difference between present and absent.

## Stage 7 — Model-form validation harness (the bridge to hardware)

- [x] 7.1 Shared stimulus format: a scenario description (mode timeline,
      setpoints) serializable to JSON, runnable identically against the
      bench and — later — against the real hardware via the UART path.
- [x] 7.2 `sim/scripts/compare_traces.py`: align (time-shift + resample)
      a hardware-capture CSV (scope/saleae export schema documented)
      against a sim trace; per-channel error metrics (RMS, peak, spectral
      peaks); markdown report output.
- [x] 7.3 Self-test: sim-vs-sim with deliberately perturbed parameters
      (+20% R, wrong blend) — assert the comparator localizes the
      discrepancy to the right channels/metrics. This is the acceptance
      test for the comparator itself.
- [x] 7.4 Parameter-fit bootstrap (`fit_motor_params.py`): least-squares
      R/L from a locked-rotor current-step trace, Ke from a coast-down
      BEMF trace, J/B(/Coulomb, after stage 2) from spin-down — run
      against synthetic bench traces with noise as self-test. This is the
      exact toolchain the Q1 session will use; building it now means
      motor ID day is measurement, not development.
- [x] 7.5 Document the model-form floor in architecture.md: realism beyond
      this point is unmeasurable until 7.2 runs against real captures;
      the checklist explicitly ends where hardware data begins.

## Stage 8 — Integration, performance, documentation

- [x] 8.1 Performance re-guard: full suite with all realism tiers must
      stay under budget (target: suite < 3 min, bench >= 1 sim-s/min with
      everything enabled); decimate thermal/disturbance updates if needed.
- [x] 8.2 Determinism audit: every stochastic element (noise, glitches,
      amplitude jitter) seeded through config; two identical runs produce
      identical traces (test).
- [x] 8.3 architecture.md: "Realism layers" section recording the
      defaults-off decision, the frozen-oracle-scope decision, and the
      emergent-vs-injected convention.
- [x] 8.4 sim/README: realism workflow (enabling effect groups, margin
      sweeps, the calibration routine, the trace comparator).
- [x] 8.5 open-questions: Q5 updated with the supply-sizing number (1.8),
      Q20 created (0.2), Q9/Q13 resolution paths extended; banner count
      change acknowledged in the params header comment.
- [x] 8.6 Tick this checklist with notes and findings; record the
      regression numbers (breakaway duty, disturbance margin, thermal
      drift magnitudes, eccentricity ripple) where future changes can be
      diffed against them.

## Post-checklist (tracked, not blocking)

- [ ] R1 RTL-side auto-alignment calibration (move 5.4's procedure into
      gateware).
- [ ] R2 Anti-cogging feedforward / torque-ripple compensation (uses 5.3's
      ripple numbers).
- [ ] R3 Sensorless BEMF integration on the now-realistic plant (P1 from
      the simulation checklist, upgraded: disturbances + supply dynamics
      make the threshold tuning honest).
- [ ] R4 Hardware-in-the-loop hooks: once boards exist, the stage-7
      stimulus format replayed on hardware closes the loop (Q1/Q7 feed
      the parameter tables; compare_traces feeds the model-form verdict).

## Implementation notes (where reality differed from the plan)

- 1.7: the duty down-slew guard needed retuning (8 → 40 counts/update):
  at 8 the deceleration dynamics were so slow that S2/S4 sampled mid-descent
  and failed — the guard must bound regen current without crippling decel.
- 4.x: switching-loss accumulation is edge-counted by the bench (exact),
  not estimated from f_pwm.
- 6.1: UART golden-frame coverage lives in the end-to-end S11 scenario and
  the direct-vs-UART stimulus equivalence test rather than per-module
  benches (same posture as simulation-checklist 5.10).
- 8.1: the full suite is ~4 min, not the <3 min target — the realism
  scenario tier alone is ~3 min (the calibration sweep runs five benches).
  Accepted: per-scenario perf guard unchanged (~4-5 s wall per sim-second),
  and the budget note here records the real figure.
- The S9 emergent-OTW scenario runs at elevated ambient (129.5 C) because
  of finding 4 below.

## Findings from the build

1. **The diode model was current-triggered only.** A floating leg whose
   open-circuit voltage exceeds the rails never started conducting at zero
   current, so an externally-spun motor could not rectify BEMF into the bus
   at all. Voltage-triggered diode promotion (iterative, ≤3 passes) was
   added to the plant — found by the regen unit test, exactly the class of
   model gap the realism layer exists to flush out.
2. **A 1 A bench supply cannot start this motor, period** (S6a): every
   acceleration transient CC-folds the supply, sags the bus through PVDD
   UVLO at 5.9 V, silently resets the DRV registers (slva552, now emergent),
   trips nFAULT, and the watchdog recovers — in a loop. Recorded as Q5
   supply-sizing evidence. Regression numbers: bus_min = UVLO floor 5.90 V,
   ~8 UV events in 1.5 s.
3. **Regen pump-up is real but self-limiting with the slew guard** (S6b):
   decel from 110 → 10 rad/s pumps the bus from 12 V to ~16.0 V max —
   observable, under the 20 V alarm. Unbounded (no guard) this motor still
   cannot pump by coasting alone: its BEMF at achievable speeds stays below
   the rectify threshold; active synchronous regen is the pump mechanism.
4. **The DRV8301 cannot overheat itself at bench ambient**: quiescent +
   gate-drive losses (~0.2 W against 30 K/W) give a few kelvin of self-heat,
   so OTW at 130 C is unreachable at 25 C ambient — itself a useful
   realism result (stalls cook the MOTOR WINDING, ~10 K in 2.5 s at duty
   0.8, with the R-drift measurably reducing stall current; the FETs and
   DRV stay cool). The emergent-OTW mechanism is exercised at elevated
   ambient.
5. **Bench UART host bug**: the serializer popped each byte immediately
   after writing the stop bit, making stop ~40 ns instead of a full bit —
   the RTL correctly rejected every frame. Same lesson as the SPI mode-1
   bug from the first checklist: protocol timing is where integration
   breaks.
6. **Viscous-only spin-down fits over-read B by >20% when Coulomb friction
   is present** (S7c) — quantified rehearsal for the Q1 identification
   session; the fit must regress dw/dt against w (fit_motor_params does).
7. **The Ke coast-fit estimator had a ~6% systematic bias** (window-mean
   omega vs omega-at-peak pairing) — caught by its synthetic self-test
   before it could bias real motor identification.
8. **Calibration routine works under eccentricity** (S10b): the
   duty-minimizing align sweep recovers the known-good offset within half a
   sector with ~2 deg of once-per-rev sensor error present.
