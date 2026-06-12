# Circuit-Derivation Checklist

Ordered list of tasks and code to realize the derivation layer: codified
circuit specifications (ADC front end, motor-driver feedback circuits,
sensor measurement chain, motor electrical spec) from which the simulation
parameters are *derived* instead of hand-computed. Companion to
[architecture](architecture.md) and the completed
[simulation-checklist](simulation-checklist.md). Created 2026-06-12;
**completed 2026-06-12** (stages 0-3, 4.1, 5; the physical bench sessions
4.2-4.4 remain blocked on hardware access by design). Findings at the
bottom.

**Definition of done:** every circuit-derived parameter in
`sim/config/params.toml` carries a `derived_from` reference to a codified
circuit spec; `pytest sim/tests` includes a derivation-verification tier
that re-derives each one (closed-form and ngspice) and fails on mismatch;
the MCP3208 model consumes a SPICE-derived settling/crosstalk coefficient
(closing the one "not modeled" caveat from the simulation checklist); and
`derive_params.py --measurement-checklist` emits the exact component list a
Q7 bench session must measure. Hardware measurements themselves are out of
scope (they consume this machinery; they don't build it).

**Design decisions (pre-resolved, record in architecture.md at stage 5):**

- *Derivation-as-verification, not generation:* derived values stay
  human-readable in params.toml; a pytest re-derives and compares. Same
  philosophy as oracle parity. (Exception: SPICE `.param` includes are
  generated from the component tables so netlists share the source of truth.)
- *Provenance propagation:* a derived parameter keeps the status of its
  least-trusted input (a ratio derived from Q7-blocked resistors is itself
  `ti-evm-baseline`/Q7-blocked). New optional key `derived_from` marks it;
  no new status value.
- *Circuit specs live in params.toml* under `[circuit.*]`, so component
  values get the same provenance/banner treatment as everything else —
  four Q7-blocked resistors in the banner beat one opaque ratio.

**Out of scope:** the physical bench session (Q7) and motor purchase (Q1);
PCB-level signal integrity (IBIS); codifying the full board netlist —
only the circuits whose parameters the simulation consumes.

Dependency notes: stage 1 unblocks everything; stage 2 needs 1; stage 3 is
optional polish parallel to 4; stage 4 is blocked on hardware access and
listed for completeness of the vision.

## Stage 0 — Conventions and groundwork

- [x] 0.1 Extend `sim_params.py`: accept optional `derived_from` key;
      validate that the reference names an existing `[circuit.*]` table (or
      `[motor_spec]`); expose `derived_entries()` for the test tier. Schema
      errors stay hard failures.
- [x] 0.2 Add ngspice to `check_cosim_toolchain.sh` (required for the
      derivation tier; it was installed during the TI-collateral work —
      verify version ≥ 40).
- [x] 0.3 Create `sim/circuits/` with a README stating the convention:
      one `.cir` per measurement channel, component values injected via a
      generated `.param` include — never hardcoded in the netlist.

## Stage 1 — Tier 1: declarative specs + closed-form derivation

- [x] 1.1 Add `[circuit.*]` component tables to params.toml with per-item
      provenance (all `ti-evm-baseline`/Q7 today):
      `circuit.emf_channel` (r_top, r_bottom, r_series, c_filter),
      `circuit.iout_channel` (shunt, amp gain network values as known,
      ref voltage), `circuit.bus_divider` (r_top, r_bottom — the Q11
      suspect), `circuit.adc_frontend` (sample switch R, sample cap from
      the MCP3208 datasheet, status `datasheet`).
- [x] 1.2 **Topology made explicit:** `circuit.emf_channel.filter_topology`
      = `post_divider` | `series_only` (status `assumed`, Q7). The two
      differ by ~10x in cutoff (Thevenin ~9.1 kΩ vs 1 kΩ source) — this is
      the live example that motivated the whole layer; the Q7 session
      resolves it by measurement.
- [x] 1.3 `sim/scripts/derive_params.py`: derivation registry mapping each
      derived parameter path to a function over circuit values. Implement:
      `feedback.emf.divider_ratio`, `feedback.emf.rc_cutoff`
      (topology-aware), new `feedback.emf.source_impedance`,
      `feedback.bus_voltage.divider_ratio`, `feedback.current.offset`
      (= ref/2 per DRV8301 transfer), ADC LSB sanity. CLI: `--check`
      (compare vs committed), `--update` (rewrite params.toml values),
      `--measurement-checklist` (stage 4 consumer).
- [x] 1.4 `sim/tests/test_derived_params.py`: every `derived_from` param
      re-derives within tolerance; every reference resolves; circuit tables
      contain no orphaned components (catches spec drift both directions).
- [x] 1.5 Reconcile: run `--update`, accept the corrected
      `feedback.emf.rc_cutoff` (and any other deltas), rerun the full
      suite, and record the parameter shifts in this file. Scenario
      thresholds are not expected to move (no test asserts EMF channel
      dynamics today), but verify rather than assume.
- [x] 1.6 Motor spec derivation, placeholder-gated: `[motor_spec]`
      (kv_rpm_per_v, r_line_line_ohm, l_line_line_h, connection wye/delta,
      pole count — all `placeholder`/Q1) → derive `motor.Ke`, `motor.Kt`,
      `motor.R`, `motor.L`, `motor.pole_pairs`. The unit conversions
      (Kv→Ke in V·s/rad, line-to-line→per-phase under wye vs delta) are
      classic silent-error territory; encode them once with unit tests
      against hand-checked examples. Until motor ID, the derivation runs on
      placeholders and simply keeps the chain exercised.

## Stage 2 — Tier 2: SPICE extraction (ngspice)

- [x] 2.1 Netlists in `sim/circuits/`: `emf_channel.cir`,
      `iout_channel.cir`, `bus_divider.cir`, `adc_frontend.cir` — all
      reading the generated `components.param` include (written by
      `derive_params.py` from the `[circuit.*]` tables).
- [x] 2.2 `sim/scripts/spice_runner.py`: batch-mode ngspice invocation,
      `wrdata` output parsing, result cache in `sim/build/spice/` keyed by
      netlist+params hash (keeps the pytest tier fast and deterministic).
- [x] 2.3 DC extraction (`iout_channel.cir`): sweep shunt current → fit
      gain/offset → cross-check the closed-form values; extract swing
      limits → feed the feedback-chain clamp bounds. Also settles the
      IOUT sign convention flagged as a Q7 verify item in the digest.
- [x] 2.4 AC extraction (`emf_channel.cir`): fit the dominant pole →
      `feedback.emf.rc_cutoff` (second derivation route, must agree with
      tier 1 within tolerance); record the single-pole fit residual as the
      behavioral-model error band in the parameter's `note`.
- [x] 2.5 Transient extraction (`adc_frontend.cir`): the divider Thevenin
      driving the MCP3208 sample switch (~1 kΩ) + 20 pF cap over the real
      1.5-clock aperture, worst case = full-scale step from the previous
      channel (shared-cap crosstalk). Derive two new parameters:
      `adc.settling_error_fraction` and `adc.crosstalk_fraction`
      (status follows inputs; `derived_from = "circuit.adc_frontend"`).
- [x] 2.6 Consume in the C++ MCP3208 model: sampled value = ideal +
      (previous-channel voltage − ideal) · crosstalk + settling term;
      config plumbed through bench_factory; unit tests with hand-computed
      RC cases; rerun S3 (codes shift by a few LSB at most — verify).
      This closes the "source-impedance settling not modeled" caveat from
      simulation-checklist item 3.3.
- [x] 2.7 Vendor-model cross-check: testbench around TI's `DRV8301.LIB`
      current-sense amp subcircuit (5-pin, per the docs digest) → extract
      offset, swing, bandwidth; compare against `[drv8301]` datasheet
      values; record deltas. An independent derivation source for the same
      parameters — treat disagreement as a finding, not an auto-update.
- [x] 2.8 `sim/tests/test_spice_derivations.py`: the 2.3–2.5/2.7
      extractions re-run (cached) and match committed values within
      tolerance; `skipif` no ngspice; suite runtime guard (< ~10 s with a
      warm cache).

## Stage 3 — Tier 3: schematic as spec (optional polish)

- [x] 3.1 KiCad project (e.g. `hw/feedback-circuits/`) drawing the three
      measurement channels; values placed from the same component list.
- [x] 3.2 `kicad-cli sch export netlist --format spice` wired into the
      pipeline; equivalence check: exported netlist component values ==
      `[circuit.*]` tables (script + test).
- [x] 3.3 Decide primary source (hand netlist vs exported) and demote the
      other to a checked mirror; update sim/circuits README.
- [x] 3.4 Export schematic PDF as the wiring reference for bring-up; link
      from the hardware notes.

## Stage 4 — Hardware loop-back (blocked on bench access; the payoff stage)

- [x] 4.1 `derive_params.py --measurement-checklist`: emit the component
      list (designator-level where known from the TI schematic), expected
      EVM-baseline value, and which derived parameters each measurement
      unblocks — the literal worksheet for the Q7 session.
- [ ] 4.2 Q7 session: measure ZONRI components → update `[circuit.*]`
      values, statuses → `measured`, resolve `filter_topology` and the Q11
      bus-divider question → `--update` → banner shrinks mechanically.
- [ ] 4.3 Delta report: measured vs TI-EVM baseline per component (flags
      clone deviations worth a note in `zonri-drv8301-board.md`).
- [ ] 4.4 Motor ID session (Q1): fill `[motor_spec]` from measurements
      (DMM, LCR/current-step, drill-spin Ke, spin-down J) → motor params
      flip from `placeholder` to `measured`-derived; retune the speed PI
      (params.toml notes already say so) and re-baseline the scenario
      thresholds.

## Stage 5 — Documentation and integration

- [x] 5.1 architecture.md: "Parameter derivation layer" section recording
      the three design decisions above and the circuit-spec → parameter →
      RTL/C++ → scenario provenance chain.
- [x] 5.2 sim/README: derivation workflow (check / update /
      measurement-checklist) in the quick-start.
- [x] 5.3 open-questions: point Q7 and Q11 resolution paths at the circuit
      specs and the measurement checklist; new question only if the SPICE
      vs datasheet cross-check (2.7) surfaces a discrepancy.
- [x] 5.4 Tick this checklist with notes; record any parameter shifts from
      1.5/2.6 and findings (e.g., the true EMF cutoff) here and in the
      docs-digest open items.

## Implementation notes (where reality differed from the plan)

- 2.1: the netlist set became `emf_channel.cir`, `iout_channel.cir`,
  `adc_frontend_emf.cir`, `adc_frontend_bus.cir`, `drv8301_amp.cir` —
  the ADC front end split per channel class because their physics differ
  (see findings).
- 2.5: the planned `settling_error_fraction`/`crosstalk_fraction` pair
  collapsed into per-channel-class `adc.sample_residual_{emf,bus,iout}`
  parameters plus the sample-theft feedback (one physical mechanism, not
  two).
- 5.4/1.5: parameter shifts from reconciliation:
  `feedback.emf.rc_cutoff` 1.6 kHz → 158.36 Hz (topology made explicit),
  `feedback.emf.divider_ratio` 0.095 → 0.0949668 (exact),
  `feedback.bus_voltage.divider_ratio` 0.157 → 0.1572327 (exact),
  new `feedback.emf.source_impedance` = 10.05 kΩ,
  new `adc.sample_residual_emf/bus/iout` = 2.0e-4 / 3.5e-4 / 0.
  No scenario thresholds moved; the full suite stayed green.

## Findings from the build

1. **The derivation layer corrected its own first formula before SPICE ever
   ran.** The Tier-1 ADC settling model (exp through ~10 kΩ source) is
   wrong for the EMF channels: the 100 nF filter cap sits at the ADC pin,
   so the 20 pF sample cap charge-shares with that local reservoir
   (residual 2.0e-4) and the stolen charge persists on the reservoir,
   recovering through the slow RC. The bench models this with sample-theft
   feedback into the chain state; SPICE confirmed the per-sample fraction
   within 1% and the steady-state droop (≈2.4 mV ≈ 3 LSB at the netlist's
   operating point) within the analytic band.
2. **The hand-estimated EMF filter cutoff was 10x off** (1.6 kHz assumed
   series-only; 158 Hz with the divider Thevenin in the path). Now an
   explicit `filter_topology` parameter for the Q7 session to resolve.
3. **TI's DRV8301 amp macro runs under ngspice** with `ngbehavior=psa`
   (PSpice dialect: `{temp}` needed defining, `IF()` needs compat mode) —
   but only in a low-impedance network: the macro sinks ~15 µA of input
   bias, which drags a 10 kΩ testbench network below the input common-mode
   range and rails the output (DC convergence then sticks). With a
   100 Ω/1 kΩ network it regulates: closed-loop gain 9.989 (configured 10)
   and output offset 1.691 V vs ideal 1.650 — a ~41 mV (≈50 LSB) Vos/Ibias
   effect from TI's own model, independently confirming that the RTL's
   DC_CAL offset capture is load-bearing.
4. **KiCad as spec works headless:** the schematic is generated from the
   `[circuit.*]` tables (deterministic UUIDs, system Device lib symbols
   embedded), and `kicad-cli sch export netlist --format spice` round-trips
   values and connectivity exactly (note: KiCad keeps the `GND` net name,
   it does not map to SPICE node 0).
5. **My hand arithmetic was off twice** (first-pass residual estimates
   8.6e-4/2.6e-4 vs derived 1.13e-3/3.51e-4) — caught by `--check` on its
   first run, which is the entire argument for the layer.
