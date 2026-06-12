# Simulation Workspace

The lockstep verification bench is **built and running**: the Verilated
controller RTL, behavioral peripheral models (DRV8301, MCP3208, AS5600), and
the C++ plant advance together in one process, verified by the pytest suite.
Architecture decision record: `../notes/architecture.md`; build checklist
(complete): `../notes/simulation-checklist.md`.

## Quick Start

```bash
# 0. toolchain sanity (verilator, cmake, ninja, pybind11, pytest, omc)
sim/scripts/check_cosim_toolchain.sh

# 1. build the bench module (generates rtl/gen/rtl_params.vh, cmake, ninja)
sim/scripts/build_bench.sh

# 2. run the full verification suite (150 tests, ~12 min; the parity,
#    peripheral, and derivation tiers alone finish in seconds)
python3 -m pytest sim/tests
```

Do not run two pytest sessions concurrently: the conftest rebuilds the
shared `bldcsim` module, and rebuilding it underneath a running suite
produces spurious failures.

The suite covers: params-loader convention checks, three-way plant parity
(C++/Python/Modelica), plant analytics and integrator convergence,
peripheral protocol golden tests, RTL lint, and the closed-loop scenarios
S0–S5 (init sequencing, open-loop spin, sensored six-step speed control,
ADC aperture schedule, transients, fault injection) with an always-on
shoot-through checker.

## Manual Scenario Runs

```bash
# closed-loop sensored six-step to 80 rad/s, trace CSV + assumptions sidecar
python3 sim/scripts/run_bench_scenario.py closed_loop --seconds 1.2

# open-loop forced commutation; add --vcd for waveforms (short runs only,
# ~6 GB per simulated second)
python3 sim/scripts/run_bench_scenario.py open_loop --seconds 1.0

# plot any shared-schema trace CSV
python3 sim/scripts/plot_trace.py sim/build/bench_closed_loop.csv
```

## Layout

- `config/`: `params.toml`, single source of truth for all parameters, each
  carrying a provenance `status`; unconfirmed ones reference open questions
  (`blocked_by = "Qn"` → `../notes/open-questions.md`). The RTL consumes the
  same values via the generated header `rtl/gen/rtl_params.vh`
  (`scripts/gen_rtl_params.py`).
- `cpp/`: the bench — plant ODE library (averaged + switched bridge with
  body diodes), behavioral peripherals, lockstep scheduler (`bench.cpp`),
  pybind11 bindings. Built into `build/cpp/bldcsim*.so`.
- `modelica/`: OpenModelica oracle models, run standalone through `omc`
  with every parameter overridden from params.toml (no FMU on the
  verification path).
- `scripts/`: params loader + RTL header generator, reference runners,
  oracle runner, scenario runner, plotter, toolchain check.
- `tests/`: the pytest suite (`conftest.py` builds the bench and prints the
  assumption banner once per session).
- `build/`: generated outputs (CSV traces + `.assumptions.txt` sidecars,
  compiled bench, oracle workdir).

## The Three Plant Implementations

The same physics is implemented three times on purpose; parity between them
is the model-validation strategy until hardware traces exist:

1. `scripts/run_one_phase_reference.py` / `run_three_phase_reference.py` —
   dependency-free Python, the executable spec.
2. `modelica/BldcCosimTestbench/package.mo` — the oracle (dassl,
   event-located switching). Run via `scripts/run_three_phase_oracle.py`.
3. `cpp/src/*_plant.*` — the primary implementation inside the bench.

Measured agreement (open-loop ramp scenario): C++ vs Python ~1e-8 (shared
integrator), C++ vs Modelica ~0.2% RMS on currents, <0.5% on speed.

## Parameters And Assumption Flags

Every run loads `config/params.toml`, prints a banner listing all
unconfirmed parameters (with their Q-numbers), and writes the same list as a
sidecar next to each output artifact. Inspect the current assumption state:

```bash
python3 sim/scripts/sim_params.py
```

Do not treat simulation output as a hardware prediction while the banner is
non-empty: motor parameters in particular are placeholders (Q1).

## Circuit-Derived Parameters

Circuit-derived parameters (feedback dividers, filter poles, ADC sampling
residuals, motor unit conversions) are codified at the component level —
`[circuit.*]` and `[motor_spec]` tables in params.toml — and re-derived
mechanically (see `../notes/derivation-checklist.md`, complete):

```bash
# verify every derived_from parameter against its circuit spec
python3 sim/scripts/derive_params.py --check

# after changing component values (e.g. Q7 measurements): recompute
python3 sim/scripts/derive_params.py --update

# the bench-session worksheet: what to measure, what each value unblocks
python3 sim/scripts/derive_params.py --measurement-checklist
```

The pytest suite enforces consistency three ways: closed-form re-derivation
(`test_derived_params.py`), ngspice extraction over the netlists in
`sim/circuits/` including TI's own DRV8301 amp macro
(`test_spice_derivations.py`, cached in `build/spice/`), and a KiCad
round-trip of the generated schematic mirror in `hw/feedback-circuits/`
(`test_kicad_spec.py`; regenerate with `sim/scripts/gen_kicad_sch.py`).

## Realism Layers

Environment realism is implemented and tested (build record:
`../notes/realism-checklist.md`), default-off and enabled per scenario:

```python
from bench_factory import realism
cfg = realism(params, "supply", "mechanical", "disturbance",
              "thermal", "sensor")   # any subset of effect groups
```

Highlights: the slva552 brownout is *emergent* (1 A bench supply → bus sag
→ UVLO → silent register reset → watchdog recovery — scenario S6a, zero
injection calls); regen deceleration observably pumps the bus and is
bounded by the RTL's duty down-slew guard (S6b); startup happens from a
real cogging detent with stiction (S7); the loop holds at 2x the assumed
disturbance amplitudes (S8); thermal lumps make OTW/OTSD and parameter
drift emergent (S9); sensor eccentricity is tolerated and the
alignment-calibration routine recovers the correct offset (S10); the whole
loop runs end-to-end over the UART register file (S11) and survives
digital-line corruption (S12); the FPGA configuration window demonstrates
why the EN_GATE pull-down is mandatory (S13).

The model-form validation harness is the bridge to hardware:

```bash
# compare a hardware capture against a sim trace (shared schema)
python3 sim/scripts/compare_traces.py reference.csv candidate.csv --report r.md
```

plus `stimulus.py` (portable scenario timelines, identical via direct ports
or UART) and `fit_motor_params.py` (the Q1 identification fits, self-tested
against synthetic traces).
