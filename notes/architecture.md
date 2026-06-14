# Architecture Decision Record: Lockstep Verilator Bench

Date: 2026-06-12. Status: adopted and **implemented** (same date) —
see [simulation-checklist](simulation-checklist.md) for the completed build
and its findings. Supersedes the original FMI/OMSimulator co-simulation plan.

## Decision

Verify the Verilog controller in a single lockstep C++ testbench process:
Verilated RTL + C++ plant ODE library + behavioral C++ peripheral models, with
a Python shell (pybind11) for scenarios, regression tests, and telemetry.
OpenModelica is demoted from simulation backbone to independent cross-check
oracle. FMI/OMSimulator move off the critical path entirely.

## Why the FMI backbone was dropped

- FMI earns its complexity when heterogeneous tools from different vendors must
  interoperate without sharing a process. This project has exactly two sides —
  a Verilated C++ object and a ~15-ODE plant — and we own both.
- PWM cannot cross an FMI co-simulation boundary cleanly: gate signals
  exchanged at a fixed communication step quantize duty resolution (a 1 µs step
  against a 50 µs PWM period is ~50 duty levels), while steps fine enough to
  fix that make multi-second runs impractical. This forced a two-mode fidelity
  split (averaged vs switched) that the lockstep design makes unnecessary.
- Debugging through four processes and three serialization boundaries versus
  one process in one debugger.
- Fault injection into peripherals (the highest-value verification this project
  can do) is nearly impossible to script through an FMU boundary and trivial
  with in-process behavioral models.

## Components

1. **Verilated RTL** — controller top-level, stepped at its real clock.
2. **C++ plant library** — three-phase R/L with parameterized
   trapezoidal-to-sinusoidal back-EMF blend, switched bridge with body-diode
   conduction states (CRSS052N08N: Ron ≈ 4.6 mΩ, Vf ≈ 0.95 V), inertia/
   friction/load mechanics. Hand-rolled RK4 or semi-implicit integration.
   Motor parameters live in one struct (R, L, Ke, J, poles, EMF shape) —
   values pending motor identification.
3. **Behavioral peripheral models** (parameterized from
   [docs-digest](docs-digest.md)):
   - DRV8301: mode-1 SPI with N+1 pipelined responses, full register file,
     dead-time insertion (DTC floor), EN_GATE 5–10 ms ready sequencing and
     <10 µs quick-reset pulse, VDS overcurrent comparator
     (trip = OC_ADJ_SET threshold / Ron), nFAULT/nOCTW latching and 64 µs
     pulse stretching.
   - MCP3208: bit-accurate SPI slave with the real ~20-clock conversion frame,
     12-bit quantization, source-impedance/settling effects optional.
   - AS5600: 150 µs internal sampling, slow-filter settling, PWM-frame or I2C
     output latency. Sensor latency is modeled deliberately — at speed it is
     likely the dominant feedback nonideality.
4. **Python shell** — pybind11 module exposing scenario runs; pytest regression
   suite; plotting/telemetry dashboard fed from the C++ loop.

## Scheduling

Gate edges change only ~6 times per 50 µs PWM period, so the plant integrates
event-to-event between gate changes with substeps capped near 1 µs. Edge timing
is exact (dead-time correctness resolvable at nanosecond scale) at close to
averaged-model cost. Expected throughput: roughly real time for full
multi-second startup sequences (Verilator at tens of MHz-equivalent on a design
this small; plant cost negligible). The averaged/switched fidelity split the
FMI design required collapses into one configuration.

## Test tiers

1. **Protocol unit benches**: SPI master against golden DRV8301 frames (mode 1,
   N+1 response, frame-fault bit), PWM + dead-time generator, ADC sequencer,
   commutation table. cocotb over Verilator is acceptable at this tier if
   Python-first ergonomics are wanted.
2. **Closed-loop system scenarios**: mapped from the spraby9 incremental build
   levels — open-loop spin, sensored six-step, current loop, fault injection,
   sensorless BEMF handoff later. Fault scenarios include the slva552 silent
   register reset, AS5600 magnet-loss flags, nOCTW storms, ADC noise.
3. **Oracle parity**: identical scenarios run against the OpenModelica plant
   (standalone `omc`, CSV output — no FMU export) and the dependency-free
   Python reference runner. Three independent implementations of the same
   equations catching each other's sign/scale errors is the model-validation
   strategy until hardware traces exist.

## What survives from the previous plan

- The Modelica package and Python reference runner continue as oracle
  implementations; the study trail in `openmodelica-example-tour.md` still
  applies to oracle modeling.
- All docs-digest electrical facts parameterize the C++ peripherals instead of
  Modelica blocks.
- FMI remains available as a decoupled learning track: the C++ plant can be
  wrapped as an FMU later (fmu4cpp in the Modelica-Inspiration collection)
  once it is trusted, rather than debugging FMI plumbing and motor physics
  simultaneously. OMSimulator is no longer a required tool.

## Parameter configuration with provenance

All simulation parameters live in one commented config,
`sim/config/params.toml`. Every parameter is a table carrying `value`, `unit`,
and a `status` declaring how trustworthy it is:

`measured` > `datasheet` > `decided` > `ti-evm-baseline` > `assumed` >
`placeholder` — the last three are **unconfirmed**, and any unconfirmed
parameter must name the open question that resolves it
(`blocked_by = "Qn"` → [open-questions](open-questions.md)).

The point is to make moving forward under uncertainty safe: uncertainty is
machine-readable, not buried in comments. Enforced by the loader
(`sim/scripts/sim_params.py`, stdlib-only):

- Every run prints a loud banner listing all unconfirmed parameters before
  producing results.
- Every output artifact gets a `.assumptions.txt` sidecar with the same list,
  so a trace can never be mistaken for a hardware prediction after the fact.
- Schema violations (missing status, unconfirmed without a Q reference, bare
  values) are hard errors.

All plant implementations consume the same file: the Python reference runner
reads it directly, the C++ bench will read it via toml++ (or receive values
through the pybind11 layer), and the Modelica oracle's literals are checked
against it by the parity tests. Resolving an open question means flipping
statuses to `measured`/`decided` in one place; the banner shrinking to zero is
the definition of a fully-grounded model.

## Parameter derivation layer (implemented 2026-06-12)

Circuit-derived parameters are not hand-computed: the component level is
codified ([circuit.*] and [motor_spec] tables in params.toml, with the same
per-item provenance), and every derived parameter carries
`derived_from = "circuit.<name>"`. Build plan and findings:
[derivation-checklist](derivation-checklist.md).

Three design decisions:

- **Derivation-as-verification, not generation.** Derived values stay
  human-readable in params.toml; `derive_params.py --check` (run by
  `test_derived_params.py`) re-derives each one and fails on mismatch;
  `--update` rewrites them mechanically when component values change. Same
  philosophy as oracle parity. (Exception: `components.param` for the
  ngspice netlists IS generated, so the netlists share the source of truth.)
- **Provenance propagation.** A derived parameter keeps the status of its
  least-trusted input — a ratio derived from Q7-blocked resistors stays
  `ti-evm-baseline`/Q7. No new status value.
- **Three derivation tiers.** Closed-form formulas in the registry
  (dividers, Thevenin impedances, motor unit conversions); ngspice
  extraction over `sim/circuits/*.cir` for everything the formulas might
  get wrong (AC poles, transient sampling physics, TI's own DRV8301 amp
  macro as an independent source); and a generated KiCad schematic mirror
  (`hw/feedback-circuits/`, verified by netlist round trip) as the
  human-reviewable wiring/measurement reference.

The provenance chain is now machine-checked end to end: circuit spec →
derived parameter → RTL header / C++ model → scenario assertions. The
Q7 bench session consumes `derive_params.py --measurement-checklist` and
feeds measured component values back into the same tables.

## Realism layers (implemented 2026-06-12)

Environment realism on top of the circuit-faithful core — build record:
[realism-checklist](realism-checklist.md). Three conventions:

- **Defaults off.** Realism features ship with zero/nominal parameters; the
  C++/Python/Modelica parity trio keeps validating ideal electromechanics
  untouched, and the oracle's scope is deliberately frozen. Scenarios enable
  named effect groups via `bench_factory.realism(params, "supply",
  "mechanical", "disturbance", "thermal", "sensor")`.
- **Every effect is a provenance-flagged parameter** (Q1 mechanical, Q5
  supply, Q9 harness/disturbance, Q13 pull-downs, Q20 sensor mounting); the
  banner growing is correct behavior.
- **Emergent replaces injected, but injection stays.** The slva552 brownout
  now reproduces causally (supply CC fold → bus sag → PVDD UVLO → silent
  register reset → watchdog rewrite) with zero injection calls; OTW/OTSD
  emerge from the DRV die-temperature lump. The injection APIs remain for
  isolated response testing.

What exists: bus-supply dynamics in the plant (CV/CC/no-sink, bus-cap
state, voltage-triggered diode rectification), Coulomb friction + cogging,
correlated disturbances (ground shift derived from `[circuit.harness]`,
gate-edge spikes, PWM-synchronized vref ripple), ADC transfer nonidealities,
FET/DRV/winding thermal lumps with R/Ke/rds drift feedback, AS5600
eccentricity + angle noise, the UART command/telemetry register file (RTL +
bench host model), digital-line corruption injection, the FPGA
configuration-window model, and the model-form validation harness
(portable stimulus format, hardware-trace comparator, motor parameter-fit
bootstrap). The RTL gained a duty down-slew guard bounding regenerative
pump-up.

## Edge-case policies (implemented 2026-06-12)

The spec gaps from [edge-case-scenarios](edge-case-scenarios.md) were
resolved as follows (constants in params.toml `[rtl]`, tests in
`test_edge_cases.py`):

- **E1 sector hysteresis:** the sensored sector only advances to an
  adjacent sector after the position penetrates `sector_hysteresis` counts
  past the shared boundary; non-adjacent jumps are taken immediately.
- **E4 EMF aperture:** above `HALF - emf_skip_duty_margin` duty the EMF
  conversion is skipped for that period (code holds last valid) — never
  mis-sampled inside the on-window.
- **E5 reverse rotation:** the speed meter votes direction per 6-edge
  window; the PI consumes a *signed* measurement, so a backdriven rotor
  produces maximum forward torque instead of a falsely satisfied loop.
- **E10 calibration plausibility:** DC_CAL offsets are accepted only within
  2048 ± `dc_cal_offset_tol`; rejects keep the previous value and raise
  `offset_fault`.
- **E13 dead peripheral:** `drv_dead_threshold` consecutive verify-rewrite
  failures *while otherwise healthy* latch a DEAD state (safe-off, flag)
  instead of livelocking. A mismatch with nFAULT low routes to the fault
  path instead (it is a brownout, not dead hardware — E18).
- **E14 stuck ADC:** `adc_stuck_threshold` consecutive rail-pinned
  conversions raise a telemetry flag (control is speed-based; the flag is
  for the host).
- **E15 capture hysteresis:** angle validity tolerates ±25% carrier drift
  once valid but requires ±15% to (re)validate — a marginal sensor refuses
  to run rather than flapping.
- **E16 UART timeout:** `uart_byte_timeout` of mid-frame silence resets the
  command FSM; a torn frame cannot poison the link.
- **E20 fault lockout:** `fault_lockout_threshold` recoveries without a
  100 ms healthy interval latch LOCKOUT. Lockout/dead/stall all clear the
  same way: the host holds mode 0 (idle) for >100 ms — an explicit
  acknowledgement, not an automatic retry.
- **E21 stall:** near-max duty with zero speed for `stall_timeout` latches
  a stall fault (gates killed); cleared by idling.
- **E25 counters:** fault/mismatch counters saturate at 0xFF (and the
  dead/lockout states bound how far they can run in practice).
- **E27 integrator envelope:** the plant constructor rejects parameter sets
  whose electrical time constant the substep cannot resolve
  (max_substep > 0.5·L/R throws), so motor-ID day cannot silently hand the
  sim an unstable configuration.

## Resolved design decisions

### ADC domain: 3.3 V (decided 2026-06-12)

The MCP3208 runs at 3.3 V on the common logic rail, not in a 5 V domain behind
the TXB0108.

Rationale:

- Throughput at 3.3 V (~1 MHz SCLK, ~20–24 µs per conversion, 2 conversions
  per 50 µs PWM period) is sufficient for both roadmap control strategies.
  Sensored six-step needs one sector-selected current sample per period;
  BEMF-integration sensorless needs exactly two (floating-phase EMF + active
  current). Overcurrent protection is not on the ADC path at all — the
  DRV8301's VDS comparator and nFAULT/nOCTW handle it at hardware speed.
- The 5 V argument's strongest case (FOC-grade sampling) fails anyway: a
  single muxed SAR cannot sample two phase currents simultaneously at either
  voltage (~22 µs skew at 3.3 V, ~11 µs at 5 V — both poison the Clarke
  transform). If FOC ever happens, the fix is a second MCP3208 in parallel
  (shared SCLK/DIN, separate CS/DOUT → simultaneous apertures, doubled
  throughput, still 3.3 V) or a simultaneous-sampling ADC — not a 5 V domain.
- A 3.3 V VREF spans the board's analog design exactly (1.65 V-centered,
  0–3.3 V signals; LSB ≈ 0.81 mV), using all 4096 codes.
- The TXB0108/HW-221 leaves the critical path entirely (AS5600 runs natively
  at 3.3 V): one supply domain, no VA≤VB sequencing, no weak-keeper artifacts
  on jumper-wired SPI.

Obligations this creates:

- RTL: the ADC sequencer must be sector-aware and priority-scheduled (active
  current + floating EMF per period, slow round-robin for the rest), and must
  launch conversions at a computed offset from the PWM counter so the ~1.5–2 µs
  sample aperture lands in the PWM off-window (only ~5 µs wide at 90 % duty).
- Plant model: simulate the real per-period schedule and aperture placement,
  not an idealized synchronous ADC.
- PWM frequency is effectively pinned near 20 kHz (40 kHz would leave one
  conversion per period) unless a second ADC is added.
- Optional margin: bench-validate SCLK at 1.3–1.5 MHz (the 1 MHz figure is the
  2.7 V spec) for a free ~30–50 % throughput bump.

### FOC control law (implemented 2026-06-14)

The controller runs both sensored six-step (modes 1–2) and field-oriented
control of a sinusoidal PMSM (mode 3), selectable per scenario. The FOC
datapath (`rtl/foc_core.v`) is Clarke → Park → id/iq current PIs → voltage-
circle limiter → inverse Park → SVPWM (min/max common-mode injection), with
an outer speed PI (`speed_iq_pi.v`) commanding iq\*. Fixed-point throughout,
with a bit-exact Python twin (`sim/scripts/foc_reference.py`) used as the
executable spec. The plant is unchanged — FOC is a configuration
(`emf_trapezoid_blend = 0`, sinusoidal) plus new RTL — so the three-way
electromechanical parity keeps validating it. Build record and findings:
`foc-checklist.md`.

Two hardware questions the bench resolved before any board (full write-ups in
`open-questions.md`):

- **Q21 — current sampling.** Confirmed the ADC-decision prediction
  quantitatively: with the single sequential MCP3208 the second phase-current
  conversion lands ~22 µs late, after that leg's low-side shunt conduction
  window has closed, giving a ~12× larger dq measurement error than
  simultaneous sampling (~1.5 A vs ~0.13 A). **Decision: a second
  MCP3208 / external S&H (simultaneous sampling) is required for FOC** — the
  six-step ADC schedule does not generalize.
- **Q22 — angle latency.** The AS5600 frame+filter lag (~1.5 ms) rotates the
  dq frame off-true, and the torque loss grows with speed (≈10 %+ by
  120 rad/s on placeholder params). Advancing the angle by ω·t_latency in the
  RTL recovers it; the bench quantifies the trade-off per speed.

FOC also needs its own sensor-to-flux alignment offset (`foc.align_offset`),
distinct from the six-step sector-convention offset — both are bench-derived
placeholders pending a hardware alignment routine (Q1/Q20).

## First slice

Port the one-phase averaged plant to C++ with a pybind11 binding and a pytest
parity check against `sim/scripts/run_one_phase_reference.py`. One-evening
slice that proves compiler, bindings, and test harness before any RTL exists.

The complete ordered path from here to a fully running, verified simulation on
this machine is tracked in [simulation-checklist](simulation-checklist.md)
(stages 0–7 with a definition of done).
