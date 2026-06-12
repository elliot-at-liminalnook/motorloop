# Simulation Build Checklist

Ordered list of everything that must exist before the simulation runs and
verifies fully on this machine. Architecture context: [architecture](architecture.md).
Created 2026-06-12; **completed 2026-06-12** — all stages done, suite green:
`python3 -m pytest sim/tests` → 63 passed (~50 s). Findings from the build
are at the bottom.

**Definition of done (met):** `pytest sim/tests` is green on this machine,
where the suite includes (a) three-way plant parity (C++ / Python / Modelica
oracle), (b) peripheral protocol unit tests against datasheet golden values,
(c) a closed-loop sensored six-step scenario in which the Verilated RTL spins
the simulated motor to a commanded speed with bounded current and zero
shoot-through violations, and (d) the fault-injection scenarios. Every run
prints the assumption banner and writes `.assumptions.txt` sidecars.

**Out of scope for this milestone:** hardware bring-up, sensorless BEMF
commutation (stretch, after milestone), FOC, the telemetry dashboard, FMI
export. Motor parameters remain `placeholder` (Q1) — the milestone proves
the machinery, not the hardware prediction.

## Stage 0 — Toolchain ✅

- [x] 0.1 Installed: Verilator 5.032, cmake 4.2.3, ninja 1.13.2, gtkwave
      3.3.126, python3-dev, pybind11 3.0.1, pytest 9.0.2, matplotlib 3.10.
- [x] 0.2 omc 1.27.0-dev from the OpenModelica apt repo (`resolute` is
      directly supported). OMSimulator not installed (intentionally).
- [x] 0.3 `check_cosim_toolchain.sh` rewritten: required vs optional split,
      OMSimulator demoted to optional; exits 0 on this machine.

## Stage 1 — C++ plant core ✅

- [x] 1.1 CMake + pybind11 under `sim/cpp/`, module `bldcsim` into
      `sim/build/cpp`. Config ingestion decision: parameters flow
      Python→C++ through pybind11 (sim_params.py is the only TOML parser).
- [x] 1.2 C++ one-phase plant mirrors the Python runner stage-for-stage.
- [x] 1.3 `sim/tests/conftest.py`: session fixture builds the bench, prints
      the assumption banner once, exposes typed params.
- [x] 1.4 Parity #1: C++ vs Python at abs/rel 1e-9 (identical trajectories);
      reusable comparator in `sim/tests/trajectory_compare.py`.
- [x] 1.5 Analytic tests: locked-rotor L/R step, no-load steady speed.

## Stage 2 — Three-phase plant ✅

- [x] 2.1 Per-phase R/L, EMF blend (`clamp(2 sin)` trapezoid, C0), isolated
      neutral via connected-leg mean.
- [x] 2.2 Switched bridge with body diodes; floating-phase terminal voltage
      verified in a hand-checked dead-time case (va = −Vf, vb = vbus + Vf
      during freewheel, decay to float).
- [x] 2.3 Averaged six-step mode (ideal freewheel clamps) for parity runs.
- [x] 2.4 Mechanics: inertia, viscous damping, constant load torque.
- [x] 2.5 Event-to-event integration (bench syncs the plant on every gate
      edge, lag capped at sim.max_substep); step-halving convergence test.
      **Finding:** leg modes must be frozen across RK4 stages — stage-level
      re-resolution flip-flops diode modes around zero crossings and creates
      a phantom equilibrium (currents never decay).
- [x] 2.6 Unit tests: locked rotor, open-loop sync speed, currents-sum-zero,
      diode freewheel voltages, energy-consistent torque coupling
      (sum(e·i) == torque·omega), shoot-through counting.
- [x] 2.7 Python three-phase averaged reference
      (`run_three_phase_reference.py`), C++ parity at 1e-8.
- [x] 2.8 Modelica oracle `ThreePhaseAveragedOpenLoop` (self-contained, no
      MSL dependency); compiles and simulates under omc 1.27.
- [x] 2.9 params.toml extended: `[sim]`, `[scenario.three_phase_open_loop]`,
      `[rtl]`; every new parameter carries status/blocked_by.

## Stage 3 — Behavioral peripheral models ✅

- [x] 3.1 DRV8301: 6-PWM truth table, DTC floor, EN_GATE sequencing +
      quick-reset semantics, mode-1 SPI slave (N+1 pipelined, frame-fault
      bit), verified register file, VDS OC with all four OCP modes,
      nFAULT/nOCTW with 64 µs stretch; injections: register reset, OTW,
      latched fault.
- [x] 3.2 DRV8301 unit tests against digest golden values (12 tests
      including write-before-ready rejection and reset semantics).
- [x] 3.3 MCP3208: mode-0,0 frame, hold aperture on the falling edge of the
      5th clock after start (recorded for aperture assertions), 12-bit
      quantization, overclock/CS-gap/differential guards. Source-impedance
      settling not modeled (noted optional).
- [x] 3.4 MCP3208 unit tests: exact codes, aperture instant, channel
      addressing, guard counters.
- [x] 3.5 AS5600: 150 µs sampling, slow filter (tau = settling/4), 4351-unit
      PWM frame (128 init + 4096 data + 127 error), magnet-loss injection;
      round-trip and latency tests.
- [x] 3.6 Feedback chain: low-side-conduction-gated shunt/amp transfer with
      1.65 V offset, EMF dividers + RC, bus divider, DC_CAL shorting, rail
      clamps, optional seeded gaussian noise; scale factors unit-tested.

## Stage 4 — Lockstep bench core ✅

- [x] 4.1 controller_top verilated through CMake (`verilate()` +
      POSITION_INDEPENDENT_CODE); clock/reset driving and VCD dump proven
      from pytest (test_vcd_dump). VCD ≈ 6 GB per simulated second — short
      windows only.
- [x] 4.2 Scheduler in `bench.cpp::tick()`: posedge eval → DRV/ADC/encoder
      models → plant event-to-event sync → peripheral outputs → negedge.
      Performance: ~4–5 s wall per simulated second (budget was 60 s);
      guarded by test_performance_budget.
- [x] 4.3 Signal binding lives in one place (`Bench::tick`), mirroring the
      future physical wiring map.
- [x] 4.4 Trace recorder with the shared CSV schema + assumption sidecars
      (`run_bench_scenario.py` writes both); adc-sample log carries the
      PWM-counter at each hold instant.
- [x] 4.5 Scenario API via pybind11 (mode/duty/speed/align controls, run,
      injections, probes); deterministic seeded noise source in the
      feedback chain.

## Stage 5 — Controller RTL ✅ (rtl/ now has 12 modules)

- [x] 5.1 `verilator --lint-only` gate in pytest; `[rtl]` section +
      `gen_rtl_params.py` renders `rtl/gen/rtl_params.vh` (single source of
      truth preserved).
- [x] 5.2 `pwm_generator.v`: center-aligned, per-leg complementary drive
      with dead-time insertion and min-pulse snap; off-window centered on
      the counter peak by construction.
- [x] 5.3 `commutation.v`: six-step table + low/float phase bookkeeping.
- [x] 5.4 `spi_drv_master.v` (true mode-1 timing: launch on rising, sample
      on trailing) + `drv_manager.v`: power-up, EN_GATE ready wait, DC_CAL
      window, CR1/CR2 write + readback-verify, periodic refresh watchdog
      (slva552), quick-reset fault recovery.
- [x] 5.5 `adc_spi_master.v` + `adc_sequencer.v`: two conversions per PWM
      period — current of the solid low-side phase (down-slope launch), EMF
      of the floating phase with the hold aperture at the off-window center;
      every 8th period samples bus voltage; DC_CAL offset capture.
- [x] 5.6 `as5600_pwm_capture.v`: period/high-time measurement, 4351-unit
      decode via sequential divider, carrier validation, loss-of-signal
      timeout. The milestone closed loop runs entirely off this path.
- [x] 5.7 `open_loop_ramp.v`: 32-bit phase accumulator, linear ramp
      (1 increment / 4096 clk minimum ≈ 56 rad/s² floor).
- [x] 5.8 `speed_meter.v` (6-edge full-electrical-revolution measurement —
      see findings) + `speed_pi.v` (conditional-integration anti-windup;
      kp=12, ki_shift=4 tuned against placeholder motor).
- [x] 5.9 Fault response: nFAULT/nOCTW synchronizers, gate kill via
      drv_manager, quick-reset pulse, reconfigure, resume; nOCTW edge
      counter.
- [x] 5.10 Module verification note: protocol golden tests run against the
      C++ peripheral models (stage 3) and the RTL is verified end-to-end
      through the bench (S0 proves SPI frames land; S3 proves ADC timing;
      the dead-time checker runs in every scenario). No separate per-module
      verilated benches were built; cocotb unused.

## Stage 6 — Scenario & verification suite ✅

- [x] 6.1 pytest scenarios via `bench_factory.py` over params.toml.
- [x] 6.2 S0 init: no gate activity before ready, SPI config verified in
      the model's registers, DC_CAL offsets = 2048, motor untouched.
- [x] 6.3 S1 open-loop spin: ≥70% of sync speed, all six sectors, bounded
      currents.
- [x] 6.4 S2 closed-loop sensored six-step: settles at 79/80 rad/s
      (±15% asserted); global shoot-through checker + minimum observed
      dead-time ≥ rtl_dead_time in every scenario (`finished()` helper).
- [x] 6.5 S3 aperture sweep duty 0.10–0.95: every EMF hold instant lands in
      the off-window (asserted against the RTL PWM counter at the hold
      tick); sector-aware channel selection ≥ 90%.
- [x] 6.6 S4 transients: speed step 60→90 tracks; 0.02 N·m load step
      droops but keeps spinning, no faults.
- [x] 6.7 S5 fault injection: slva552 silent register reset detected and
      rewritten by the refresh watchdog within ~2 periods; latched-fault →
      quick-reset → reconfigure → resume; OTW reported (nOCTW counted), not
      fatal; magnet loss → safe gate-off, auto-recovery; seeded ADC noise
      (≈6 LSB rms) — loop still settles.
- [x] 6.8 Performance guard: 0.2 s scenario must finish in <12 s wall
      (currently ~1 s); suite total ~50 s.

## Stage 7 — Oracle parity & reporting ✅

- [x] 7.1 `run_three_phase_oracle.py`: omc batch run with EVERY parameter
      overridden from params.toml (stronger than literal-checking — drift
      is impossible); output converted to the shared schema.
- [x] 7.2 Three-way parity: C++↔Python aligned at 1e-8; C++↔Modelica
      interpolated — currents <2% RMS (measured ~0.2%), omega/theta
      pointwise 1%, final speed <0.5% (measured ~0.03%).
- [x] 7.3 `plot_trace.py`: multi-panel plots from any shared-schema CSV.
- [x] 7.4 Documentation pass: sim/README quick start + layout; this
      checklist ticked; open-questions updated (Q15/Q17 resolved, Q18
      partial evidence).

## Post-milestone (tracked, not blocking)

- [ ] P1 Sensorless BEMF-integration controller + switched-plant scenarios
      (floating-phase sensing, threshold tuning sweeps — Q18 evidence).
- [ ] P2 AS5600 viability-vs-speed study (Q18 → resolves Q3).
- [ ] P3 Telemetry dashboard (PyGame or similar) fed from the trace stream.
- [ ] P4 fmu4cpp wrap of the trusted C++ plant (optional FMI learning track).

## Findings from the build (things the bench caught or taught)

1. **DRV8301 SPI mode-1 is easy to get wrong:** the first RTL master
   launched the next MOSI bit on the falling edge (where the slave samples),
   silently corrupting every frame by one bit position. Caught by the
   register-readback mismatch counter on the very first integration run —
   exactly the class of bug that would have cost a bench debugging session
   with a logic analyzer.
2. **ADC framing off-by-one:** the MCP3208's null/data bit positions plus
   the bench's one-cycle feedback delay shift the master's collect window to
   clock cycles 8–19. Symptom: every code exactly halved.
3. **Diode-mode RK4 freeze:** integrator stages must hold leg modes fixed
   within a substep or freewheel currents stop decaying at a phantom
   equilibrium (~mA). Applies identically to the C++ and Python plants.
4. **AS5600 PWM-frame latency quantizes commutation timing:** single-sector
   period measurement reads only SPEED_NUM/(n·frames); the speed meter
   measures across a full electrical revolution (6 edges) instead.
5. **Speed-meter timeout vs slow windows:** a standstill timeout shorter
   than the 6-edge window at low speed corrupts measurements and biases the
   loop ~25% high; quick timeout now requires zero edges.
6. **ADC slot scheduling:** a current conversion launched early on the up
   slope is still busy at the EMF launch point; the current slot moved to
   the down slope. (S3 caught the EMF slot silently never firing.)
7. **Open-loop ramp floor:** 1 freq-word per 256 clk ≈ 890 rad/s² exceeded
   the placeholder motor's pull-in torque; ramp tick widened to 4096 clk.
8. **Q18 evidence (preliminary):** with placeholder motor parameters,
   AS5600-PWM sensored six-step tops out near 120 rad/s mech — the
   filter+frame latency (~1.5–2 ms) costs ~40–50 elec degrees of
   commutation lag at speed. Recorded against Q18/Q3; the full study is P2.
