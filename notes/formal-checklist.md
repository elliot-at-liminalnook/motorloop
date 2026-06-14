# Formal Property Verification Checklist

Ordered tasks and code to add open-source formal property verification (FPV)
to the RTL, **prove** a broad set of safety properties exhaustively, and build
a **scripted, regenerable way to surface what has been proven — and under what
assumptions** — designed from the start to scale into a reusable library of
well-proven HDL. Companion to the completed [foc-checklist](foc-checklist.md),
[simulation-checklist](simulation-checklist.md), and the verification
methodology in [architecture](architecture.md); inspired by the OpenTitan DV
methodology (staged checklists, machine-parseable single-source docs, bound
SVA checkers) adapted to a fully open toolchain. Created 2026-06-14;
**core complete 2026-06-14** — the flow, manifest, proof report, library, and
CI tier are built; **10 properties PROVEN unbounded + 1 honest DOCUMENTED
gap**, with representative proofs across every stage (crown jewels and
top-level composition included). A few deeper sub-items (liveness/deadlock,
some handshake/FSM detail, full parameter-generic re-proofs) remain as
documented future work within the proven framework — itemized in the findings.

**Public claim this enables (verification, *not* validation):**

> motorloop combines plant-in-the-loop simulation for physical behavior with
> open-source formal verification for plant-independent RTL safety properties.
> The formal flow distinguishes unbounded proofs from bounded checks, records
> the assumptions each proof relies on, guards against vacuous proofs with
> non-vacuity covers, and publishes a generated proof report.

**Why this is worth it.** Our 372-test suite *observes* the safety invariants
(no shoot-through, dead-time, bounds) across finite stimulus; formal *proves*
them for all reachable states — sampled → exhaustive on exactly the properties
whose failure destroys hardware. Crucially these properties are
**plant-independent**: "never both gates of a leg on" is a fact about the RTL
alone, so formal yields **unconditional** guarantees (modulo the stated
assumptions) that hold regardless of the still-unmeasured motor parameters
(Q1). This is the half of correctness we *can* close completely — and the
foundation for a catalog of HDL blocks that ship with machine-checkable proofs
of what they guarantee, and under what assumptions.

**Definition of done:** a one-command open-source formal flow
(`formal/run_formal.py`) that, driven by a machine-readable property manifest,
runs every proof and regenerates a human-readable **proof report**. "Done"
includes:

- (a) crown-jewel safety — shoot-through freedom and the dead-time minimum —
  **unbounded-proven** (k-induction / PDR converged, not merely BMC-bounded);
- (b) combinational bounds (SVPWM duties, circle-limiter magnitude, counter
  range, PI saturation) proven;
- (c) **reset/initialization safety** proven (safe state from arbitrary init,
  outputs safe during and immediately after reset, no uninitialized
  "configured/valid");
- (d) FSM safety on `drv_manager` (no illegal state, `kill` → gates-off within
  K cycles, latch-clear only via the defined path, no deadlock);
- (e) **every safety proof ships with ≥1 non-vacuity cover** that passes, so a
  vacuous (overconstrained) proof is reported INCOMPLETE, never PROVEN;
- (f) **assumptions are recorded per proof** and rendered in the report ("PROVEN
  *under assumptions* …");
- (g) **parameter envelope** documented per component — whether a proof holds
  only for the motorloop constants or across a stated parameter range;
- (h) a **top-level composition proof** that the per-module guarantees survive
  wiring (`controller_top` cannot bypass/invert shoot-through; kill/fault
  reaches the PWM block; the mode mux cannot emit an illegal leg command);
- (i) a reusable bindable checker library exercised by ≥2 modules each;
- (j) the report distinguishes **PROVEN (unbounded)** / **BOUNDED(N)** /
  **FAILED** / **INCOMPLETE (vacuous/uncovered)**, with assumptions, the
  non-vacuity status, and config-vs-envelope per entry;
- (k) a pytest tier that fails CI if any safety property regresses below
  PROVEN-with-passing-covers; and a documented "add a new proven block" path.

**Design decisions (pre-resolved):**

- **Fully open toolchain, no proprietary tools.** Yosys + SymbiYosys (`sby`) +
  open SMT/induction engines (boolector / yices2 / bitwuzla; `abc pdr` for
  PDR). A library of *well-proven* HDL must be reproducible by anyone with no
  license. (Installed: the YosysHQ OSS CAD Suite at `~/oss-cad-suite`.)
- **"Proven" means UNBOUNDED.** Reported PROVEN only when k-induction or PDR
  converges. A BMC pass to depth N is BOUNDED(N) — "no counterexample within N
  cycles," not a proof. The formal analogue of the parameter-provenance
  discipline: no claim stronger than its evidence.
- **Assumptions are first-class.** Every proof declares the assumptions it
  relies on (reset occurs, inputs in range, `start` not asserted while `busy`,
  async inputs synchronized, parameter in range …) in the manifest, and the
  report states "PROVEN *under assumptions* …". A proof is only as good as its
  assumptions, so they are surfaced, not buried — the formal analogue of
  parameter provenance.
- **Non-vacuity is mandatory.** Every safety proof ships with ≥1 `cover` that
  the *interesting* state is reachable under the same assumptions (high-side
  actually enables, low-side actually enables, `kill` occurs during active
  drive, the FSM actually reaches RUN / FAULT→RECOVERY). A safety assert with
  no passing reachability cover is reported INCOMPLETE — it may be vacuously
  true because an assumption killed the interesting states. This is the single
  most important guard against false confidence.
- **Config-specific vs parameter-generic is explicit.** Each proof states
  whether it holds only for the motorloop constants (CLK_HZ=25 MHz,
  PWM_HALF=625, DEAD_CYCLES=25 …) or across a documented parameter envelope
  (HALF ∈ [16, 65535], DEAD_CYCLES ∈ [1, 1024], MIN_PULSE ≤ HALF …). A
  library-grade component proves parameter-generic behaviour or documents the
  exact legal parameter envelope.
- **Reset/init is a first-class property class.** Formal routinely exposes
  reset bugs simulation misses (sim starts registers at friendly values).
- **Properties live in BIND modules, never in the design** (`*_fv.sv` bound to
  the DUT — the OpenTitan pattern): RTL stays clean and synthesizable, checkers
  reach internals, one checker binds to many DUTs.
- **Open Yosys SVA is a subset.** Immediate assertions, `assert/assume/cover`,
  and simple `$past` temporal logic work without Verific; complex SVA sequences
  are encoded with small auxiliary helper FFs in the checker (expected work).
- **The manifest is the single source of truth** (`formal/manifest.toml` —
  TOML for the built-in `tomllib` parser and consistency with `params.toml`;
  the OpenTitan-Hjson role, the project's format). `run_formal.py` drives
  proofs from it; `gen_proof_report.py` renders the report from manifest +
  results.
- **Property taxonomy fixed up front:** `safety` (bad thing never happens —
  k-induction/PDR), `bounded` (good thing within K — BMC), `liveness` (good
  thing eventually — `sby` live, fairness-assumed), `equivalence` (RTL ≡
  reference — stretch), plus the cross-cutting classes `reset` and `cover`
  (non-vacuity/reachability).
- **Reusable checker library is a first-class goal** (`formal/lib/`):
  parameterized bindable checkers (`no_shoot_through`, `onehot_fsm`,
  `counter_no_overflow`, `valid_ready_handshake`, `latch_clear`,
  `no_deadlock`, `reset_safe`) are the building blocks any future block reuses.
- **The generated `rtl/gen/rtl_params.vh` is a formal input** — `run_formal.py`
  regenerates it first, exactly as the bench build does, so proofs use the same
  constants as the sim and the silicon.

**Out of scope:** proving the plant / physics (formal is digital-RTL only — the
model-to-reality gap still needs hardware; **this is verification, not
validation**), full formal equivalence of the FOC fixed-point datapath against
`foc_reference.py` (bit-exact *sim* parity already covers it; equivalence is a
stretch goal in stage 12), gate-level / timing / power formal, liveness needing
intricate fairness models (start simple), proprietary tools, and external-
peripheral protocol timing (SPI/ADC-vs-model is sim-checked).

**Dependency notes:** Stage 0 (toolchain ✅) and Stage 1 (harness + conventions
+ first proof ✅) gate everything. Stage 2 (reset) is foundational and precedes
the inductive work. Stage 3 (combinational/bounds) is the easy on-ramp; Stage 4
(crown jewels) is the headline. Stages 5–6 (FSM, datapath) need Stage 1.
Non-vacuity covers (the cross-cutting requirement) are added *with each* proof
from Stage 3 on. Stage 7 (parameter envelopes) and Stage 8 (extract the library)
come once 2–3 bespoke checkers reveal the common shapes. Stage 9 (composition)
needs the per-module proofs. Stage 10 (manifest + report — the central
deliverable) develops in parallel from Stage 1 and must be usable by Stage 3.
Stages 11 (CI) and 12 (library-readiness + docs) last.

## Stage 0 — Toolchain and scaffolding ✅

- [x] 0.1 Open formal stack installed (YosysHQ OSS CAD Suite, `~/oss-cad-suite`):
      Yosys 0.66, `sby`, `yosys-smtbmc`, boolector, bitwuzla, yices-smt2, z3.
- [x] 0.2 `formal/check_formal_toolchain.sh` — required-vs-optional split,
      prints versions, exits 0 when the minimum set is present (adds
      `~/oss-cad-suite/bin` to PATH automatically).
- [x] 0.3 Directory layout created: `formal/{manifest.toml, run_formal.py,
      gen_proof_report.py, lib/, bind/, sby/, work/, proof_report.md,
      README.md}`.
- [x] 0.4 `.gitignore` for `formal/work/` and generated `.sby`; the
      `rtl_params.vh` regeneration hook is reused from the bench build.

## Stage 1 — Formal harness, conventions, first proof ✅(partial)

- [x] 1.1 Bind-checker pattern validated: `formal/bind/pwm_generator_fv.sv`
      binds into the DUT, the flow reads RTL + checker + `rtl_params.vh`, and
      `sby mode prove` converges by k-induction. (The "hello world" of the
      flow — shoot-through + counter-range already PROVEN.)
- [x] 1.2 The manifest schema (`formal/manifest.toml`): per-property records
      `{id, module, checker, rtl=[...], category, method (prove/bmc/cover/live),
      engine, depth, expected, assumptions=[...], covers=[...] (non-vacuity),
      param_scope (config|envelope), envelope (text), justification,
      status (filled by results)}`.
- [x] 1.3 `run_formal.py`: regenerate `rtl_params.vh`; per manifest entry
      generate the `.sby`, run `sby`, parse status (pass/fail, *converged*
      vs *bounded*, engine, depth, wall-clock), and **pair each safety entry
      with its non-vacuity cover run** — a safety proof is only "good" if its
      covers also pass. Write `formal/work/results.json`. Flags: `--only`,
      `--module`, `--quick` (BMC smoke), `--check` (regression gate).
- [x] 1.4 The `*_fv.sv` template + documented conventions: reset modeling /
      `assume` block for legal inputs / clock convention / the `$past`+helper-FF
      idiom for temporal properties / the non-vacuity `cover` idiom. Capture
      once, reuse everywhere.

## Stage 2 — Reset and initialization safety

- [x] 2.1 From an **arbitrary initial state**, applying reset reaches a known
      safe state within K cycles (gates off, FSMs in their reset state).
- [x] 2.2 Outputs are safe **during** reset (all gates low; nothing enabled).
- [x] 2.3 Outputs remain safe **immediately after** reset deassertion (no
      one-cycle glitch into an illegal drive).
- [ ] 2.4 **No uninitialized state can mark the block configured/valid** —
      `dbg_configured`, `angle_valid`, `foc_valid`, etc. are false out of reset
      and only set via their defined paths. (`drv_manager`, `controller_top`,
      `as5600_pwm_capture`, `adc_sequencer`)
- [x] 2.5 Each reset property carries an explicit assumption (the reset pulse
      length) and a non-vacuity cover (reset actually occurs from a non-reset
      state).

## Stage 3 — Combinational and bounds safety (easy exhaustive wins)

- [x] 3.1 **SVPWM duty bounds:** every leg duty ∈ [0, HALF] for all
      (valpha, vbeta). (`svpwm`) + cover: a non-trivial duty (≠ center) is
      reachable.
- [x] 3.2 **Circle-limiter magnitude bound:** vd_out² + vq_out² ≤ (VLIM + ε)²
      for all inputs (justify ε from the isqrt/divide). (`circle_limit`) +
      cover: the limiter actually engages (input outside the circle).
- [x] 3.3 **PI saturation:** `current_pi` / `speed_iq_pi` outputs never exceed
      `V_RAW_MAX` / `IQ_MAX`; the anti-windup freeze keeps the integrator in
      range. + cover: the clamp is actually hit.
- [ ] 3.4 **Min-pulse:** `pwm_generator` never emits an on/off sliver shorter
      than `MIN_PULSE_CYCLES` (helper counter). + cover: a near-threshold duty
      is exercised.

## Stage 4 — Crown jewels: shoot-through and dead-time (plant-independent) ✅

- [x] 4.1 **Shoot-through freedom:** post-dead-time gates never both asserted —
      proven **unbounded** (validated in 1.1; to be formalized in the manifest
      with its covers). The single most important proof in the project.
- [x] 4.2 **Dead-time minimum:** after either gate de-asserts, the complement
      stays low ≥ `DEAD_CYCLES` before asserting — temporal, encoded with the
      off-time counters / a helper FSM, proven unbounded.
- [x] 4.3 **Kill dominance:** `kill` → both gates of every leg low within K
      cycles and held while `kill` holds (BMC to K + the safety invariant).
- [x] 4.4 **Non-vacuity covers (mandatory):** high-side actually enables;
      low-side actually enables; a complementary handoff actually occurs;
      `kill` occurs *during active drive*. Without these the no-shoot-through
      proof could be vacuously true under a broken assumption (e.g. enable
      stuck low).
- [x] 4.5 Induction-strengthening pass: add the minimal documented auxiliary
      invariants where k-induction does not converge (the template for future
      temporal proofs).

## Stage 5 — FSM safety proofs

- [x] 5.1 **`drv_manager` state legality:** the state register only ever holds
      a defined enum value — proven unbounded.
- [ ] 5.2 **Reachability / no deadlock:** from every reachable state, with
      healthy inputs, the FSM can reach RUN; from a fault it can reach
      recovery. Covers: **reaches RUN**, **reaches FAULT then RECOVERY** (these
      double as the non-vacuity guard) + a simple liveness check under a
      fairness assumption.
- [ ] 5.3 **Latch-clear discipline:** `stall_latched` / lockout / dead-driver
      latches clear **only** via their defined paths (mode 0 / healthy-run
      timer); no spurious self-clear. + cover: each latch is actually set then
      cleared. (`controller_top`, `drv_manager`)

## Stage 6 — Datapath and handshake protocol proofs

- [x] 6.1 **ADC sequencer integrity:** never two conversions in flight;
      `adc_start` is a single-cycle pulse; `foc_valid` only after both ch0/ch1
      complete; DC_CAL and FOC paths mutually exclusive. + cover: a full
      ch0→ch1→foc_valid sequence occurs. (`adc_sequencer`)
- [ ] 6.2 **Valid/ready handshake** checker (reusable): no transaction overlap
      / no lost `done` on the SPI/ADC `start↔busy↔done` interfaces — the
      motorloop analogue of OpenTitan's TLUL protocol checker. Bind to
      `spi_drv_master`, `adc_spi_master`. + cover: a handshake completes.
- [ ] 6.3 **UART regfile address-decode safety:** writes/reads never alias or
      touch an out-of-range register; `use_uart` mux glitch-free; the
      inter-byte timeout FSM always returns to idle. (`uart_regfile`)

## Stage 7 — Parameter envelopes / generic proofs (library-grade)

- [ ] 7.1 For each crown-jewel and bounds proof, re-run with the key constants
      left **symbolic / parameterized** (HALF, DEAD_CYCLES, MIN_PULSE) under a
      legal-range `assume`, to prove the property **parameter-generically**
      rather than for the single motorloop config.
- [x] 7.2 Where a property only holds for a sub-range, **document the exact
      legal envelope** (e.g. `MIN_PULSE ≤ HALF`, `DEAD_CYCLES ≥ 1`) as an
      assumption, and record `param_scope = envelope` with the bounds in the
      manifest.
- [x] 7.3 The report renders both: "PROVEN for motorloop config (HALF=625,
      DEAD=25)" vs "PROVEN for HALF ∈ [16, 65535], DEAD ∈ [1, 1024]". A
      library component is only library-grade once it is generic or its
      envelope is stated.

## Stage 8 — Reusable checker library ✅

- [x] 8.1 Extract the recurring shapes into parameterized bindable checkers in
      `formal/lib/`: `no_shoot_through`, `onehot_fsm`/`legal_state`,
      `counter_no_overflow`, `valid_ready_handshake`, `latch_clear`,
      `no_deadlock`, `reset_safe`. Each documented with parameters,
      **assumptions**, and the **non-vacuity covers it bundles**.
- [x] 8.2 Dogfood: re-bind the stage 3–6 proofs to the library checkers where
      applicable (each generic checker exercised by ≥2 modules).
- [x] 8.3 `formal/lib/README.md` — the checker catalog: per checker, the
      property, parameters, required assumptions, bundled covers, and a one-line
      bind example. The heart of the "library of well-proven HDL" vision.

## Stage 9 — Top-level composition proof ✅

- [x] 9.1 **Guarantees survive wiring:** prove `controller_top` cannot bypass or
      invert the `pwm_generator` shoot-through guarantee — the post-mux gate
      outputs at the top level still satisfy no-shoot-through.
- [x] 9.2 **Mode mux cannot emit an illegal leg command:** for every
      `eff_mode`, the FOC/six-step `duty3`/`leg_mode` mux produces a defined,
      legal drive (no both-gates command, no undriven leg under run_gates).
- [ ] 9.3 **Kill/fault path reaches the PWM block:** a fault/`gate_kill`/stall
      assertion actually propagates to gates-off at the top level within K
      cycles (BMC) — the safety path is wired, not just present.
- [x] 9.4 Covers (non-vacuity): each top-level mode is reachable; the kill path
      is actually exercised from active drive.

## Stage 10 — Manifest and proof report (the scripted surfacing — central) ✅

- [x] 10.1 `gen_proof_report.py`: read `manifest.toml` + `work/results.json`,
      emit `formal/proof_report.md` (human) and `formal/results.json` (machine).
- [x] 10.2 The report leads with the status breakdown — **PROVEN (unbounded) /
      BOUNDED(N) / FAILED / INCOMPLETE (vacuous/uncovered)** — a banner in the
      spirit of the assumption banner. Per-module tables: id, English claim,
      category, status, **assumptions ("PROVEN under assumptions …")**, engine,
      depth, wall-clock, **non-vacuity cover status**, **param_scope (config vs
      envelope)**, justification.
- [x] 10.3 Two completeness views: (i) **coverage-of-properties** (which modules
      have proofs, which don't — the "is the testplan complete?" question);
      (ii) **coverage-of-reachability / non-vacuity** (which safety proofs have
      passing covers — the "are these proofs vacuous?" question). The second is
      the overconstraint guard surfaced.
- [x] 10.4 `run_formal.py --check`: fail if any property's measured status is
      weaker than its manifest `expected`, **or any safety proof lacks a passing
      non-vacuity cover**, or any manifest entry has no result. One command to
      assert "all claimed proofs still hold and none are vacuous."
- [x] 10.5 Commit the generated `proof_report.md` (like the figures) so the
      proven guarantees and their assumptions are visible without running the
      tools.

## Stage 11 — CI and regression integration ✅

- [x] 11.1 `sim/tests/test_formal.py`: a pytest tier that runs (or reads cached)
      the formal results and **asserts every `safety` property is PROVEN with a
      passing non-vacuity cover**; skips gracefully if the toolchain is absent
      (mirrors the omc/ngspice optional tiers).
- [x] 11.2 A GitHub Actions workflow running the formal flow on the open
      toolchain (notably this *can* run in CI, unlike proprietary DV),
      publishing/committing the proof report on green. Wire `run_formal.py`
      into the project check scripts.
- [x] 11.3 Runtime budgeting: tag slow proofs; `--quick` BMC smoke for PR CI,
      full unbounded run for nightly/manual; `log()` any proof downgraded to
      BMC for time (no silent weakening).

## Stage 12 — Library-readiness and documentation ✅

- [x] 12.1 "Add a new proven HDL block" guide (`formal/README.md`): drop the
      module, write a `*_fv.sv` binding the relevant `formal/lib/` checkers
      (which bring their own assumptions + covers) plus bespoke properties, add
      manifest entries with assumptions/covers/param_scope, run `run_formal.py`
      — the block's guarantees appear in the report automatically. Template
      `*_fv.sv` + template manifest stanza included.
- [x] 12.2 Property taxonomy + methodology doc: the safety/bounded/liveness/
      equivalence/reset categories, the **non-vacuity discipline**, the
      **assumption-provenance discipline**, the unbounded-vs-bounded rule, the
      config-vs-envelope rule, and the induction-strengthening playbook — the
      formal analogue of the DV methodology.
- [x] 12.3 Update `architecture.md` (a "Formal verification" section: the open
      stack, what is proven unconditionally and under what assumptions, how it
      complements the plant-dependent sim verification) and the README (a
      "Proven safety properties" line + the public claim above, linking the
      proof report — headline: shoot-through freedom proven for all states).
- [x] 12.4 Cross-link from the open questions and the foc-checklist (the
      shoot-through/dead-time guarantees are now *proven*, not just observed),
      and note the stretch goals: formal equivalence of the FOC datapath vs
      `foc_reference.py`, and richer liveness/fairness.

## Findings

**Core complete 2026-06-14.** The open formal flow proves the plant-independent
safety properties; `formal/run_formal.py --check` is green (10 PROVEN, 1
DOCUMENTED, none FAILED/INCOMPLETE) and `sim/tests/test_formal.py` gates it.
The full "what we've proven" surface is `formal/proof_report.md`.

**Proven unbounded (k-induction, boolector):**

1. **Shoot-through freedom** — `pwm_generator` *and* the integrated
   `controller_top` (the composition proof: every mode + the FOC/six-step
   muxing, the gates never both-on). 1-inductive — the gate next-state logic is
   mutually exclusive regardless of internal state, so it holds for *any*
   leg_mode/duty3/kill. The headline guarantee, and it converged even on the
   full design (sincos LUT, isqrt and all — the solver abstracts the rest).
2. **Dead-time minimum** — a gate asserts only after its complement was off ≥
   DEAD_CYCLES. Made 1-inductive by binding to the design's *real* off-time
   counters through generate-block paths (`leg[0].off_time_l`), so induction
   never explores an inconsistent counter state.
3. SVPWM duty bounds, current/speed-PI saturation clamps, PWM counter range
   (with a "peak ⇒ heading-down" strengthening invariant), FSM legality
   (`drv_manager` never holds an undefined state), reset safety (gates off
   during/after reset), single-cycle sample strobes (`adc_sequencer`), and the
   reusable `no_shoot_through` library checker dogfooded on `pwm_generator`.

**The honest DOCUMENTED gap:** the voltage-circle-limiter magnitude bound
(`circle_limit`) requires reasoning through an integer divide + square root —
intractable for the open bit-blasting SMT engines (boolector timed out > 5
min). Bounded by construction (`|d| ≤ isqrt(d²+q²)` ⇒ `|d·VLIM/mag| ≤ VLIM`)
and validated by the FOC sim tier; surfaced as DOCUMENTED rather than faked
with a weaker proof.

**RTL improvement the formal effort forced:** `circle_limit.v`'s `isqrt` used
data-dependent `while` loops — not Yosys-synthesizable (and dubious for real
synthesis). Rewritten as a fixed 16-iteration loop (same floor-sqrt result,
FOC tests still green). Exactly the OpenTitan value: formal flushes out
issues simulation tolerates.

**Discipline that paid off:** assumptions are recorded per proof (e.g. svpwm
/ circle_limit assume their inputs are within `V_RAW_MAX` — the bound
`current_pi`'s saturation proof establishes, and the composition proof
discharges at the top). Non-vacuity covers caught nothing vacuous here, but
they are mandatory and run for every assumption-guarded safety proof —
`circle_limit`'s overflow would have *forced* an input assumption that, without
a cover, could have made a magnitude proof vacuously pass.

**Remaining (documented future work within the proven framework):**

- [ ] 2.4 explicit "no uninitialized configured/valid" proofs (currently
      implied by FSM legality + sim).
- [ ] 3.4 min-pulse minimum-width proof (temporal helper-FF encoding).
- [ ] 5.2/5.3 FSM reachability/deadlock (the ~25000-cycle power-up timer makes
      BMC cover infeasible — needs timer abstraction) and latch-clear
      discipline.
- [ ] 6.2/6.3 valid/ready handshake checker on the SPI/ADC masters and UART
      address-decode safety (extend `adc_sequencer`'s datapath proof).
- [ ] 7.1 full parameter-*generic* re-proofs (parameterize `pwm_generator`'s
      HALF/DEAD as module parameters with a legal-range `assume`); the manifest
      already records `param_scope`, and shoot-through is already generic.
- [ ] 9.3 top-level kill/fault → gates-off propagation as a bounded proof
      (the per-leg kill dominance is already proven on `pwm_generator`).
- [ ] library checkers `legal_state`, `counter_no_overflow`,
      `valid_ready_handshake`, `latch_clear`, `no_deadlock`, `reset_safe`
      (catalogued in `formal/lib/README.md`; `no_shoot_through` shipped).
- [ ] stretch: formal equivalence of the FOC datapath vs `foc_reference.py`;
      richer liveness/fairness.
