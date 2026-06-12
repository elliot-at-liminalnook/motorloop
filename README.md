# motorloop

**Motor-in-the-loop verification for Verilog.** Run your controller RTL
closed-loop against a simulated motor, gate driver, and ADCs — before
hardware exists.

The RTL runs against behavioral models of the actual chips around it —
DRV8301 gate driver, MCP3208 ADC, AS5600 angle sensor — and an ODE model of
the inverter, motor, and bench power supply, all compiled into a single
process that advances in lockstep. The controller spins a simulated motor
through its real feedback circuits; tests assert on physics (speed, current,
bus voltage, die temperature), not just on waveforms.

## The problem

Open-source FPGA motor controllers are almost universally verified
open-loop: drive the RTL with canned stimulus, inspect the PWM in a waveform
viewer, then go to the bench. The bugs that survive this are exactly the
ones that only exist closed-loop — a wrong SPI mode against the gate driver,
an off-by-one in ADC framing that silently halves every reading, a
commutation policy that limit-cycles only when back-EMF feeds back into the
next sector decision. On hardware these are slow to localize and sometimes
destructive; shoot-through does not offer a second attempt.

Commercial workflows solve this with Simulink/HDL-coder co-simulation or HIL
rigs. There was no open equivalent for hand-written Verilog: a way to
develop RTL against a plant instead of against a waveform viewer. This repo
is one.

## What it caught

The point of the bench is the class of bug it finds before hardware does.
All of these were found by failing tests here; none would have been found by
stimulus-replay testbenches:

- SPI master launched MOSI on the wrong edge for the DRV8301's mode-1
  timing — every register write silently corrupted.
- ADC framing collected one bit cycle early — all current/voltage codes
  halved. Closed-loop, the speed PI compensated and *almost* hid it.
- The stall detector false-tripped on a fast rotor whose angle aliased the
  sensor's PWM frame rate.
- Regenerative braking during a fast decel pumped the bus voltage into the
  supply's no-sink region; the RTL needed a duty down-slew guard.
- A 1 A bench-supply current limit reproduced TI's slva552 brownout
  app-note failure (bus sag → UVLO → silent register reset) with no fault
  injected — and verified the watchdog recovery path.

## How it works

```
            one process, one clock authority
┌──────────────┐  SPI   ┌──────────────────────────┐
│ Verilated    │◄──────►│ DRV8301 model (regs,     │
│ controller   │        │ faults, UVLO, OC, OT)    │
│ RTL          │  SPI   ├──────────────────────────┤
│ (your        │◄──────►│ MCP3208 model (aperture, │
│  Verilog)    │        │ residual, INL)           │
│              │  PWM   ├──────────────────────────┤
│              │◄───────│ AS5600 model             │
│              │ gates  ├──────────────────────────┤
│              │───────►│ inverter + motor + supply│
│              │        │ ODE plant (RK4, diodes)  │
└──────────────┘        └──────────────────────────┘
```

- **Lockstep, not FMI.** Verilator compiles the RTL to C++; peripherals and
  plant are C++ libraries; a single scheduler ticks everything. No
  co-simulation middleware, no time-sync drift, deterministic runs.
- **Protocol fidelity at the boundaries.** The DRV8301 model implements the
  N+1 pipelined SPI response, register semantics, and fault behavior from
  the datasheet; the ADC model samples during the real aperture window with
  charge-sharing residual from the previous channel.
- **Physics where it matters.** Switched bridge with body-diode conduction,
  event-to-event RK4 integration, and optional realism layers (all
  default-off, enabled per scenario): supply CV/CC/no-sink dynamics, cogging
  and stiction, ground-shift and gate-edge coupling into the feedback
  dividers, thermal RC lumps that feed resistance and Ke drift back into the
  plant, sensor eccentricity.
- **Driven from Python.** pybind11 bindings; tests are pytest. 155 tests
  cover protocol golden vectors, plant parity, closed-loop scenarios, fault
  injection, and 29 edge cases (stall, flooded UART, dead driver, corrupted
  calibration, inertia extremes...). A shoot-through checker runs in every
  scenario.

## Every parameter states what it's worth

Simulation results are only as good as the numbers underneath them, so every
parameter in `sim/config/params.toml` carries a provenance status
(`measured` > `datasheet` > `decided` > `assumed` > `placeholder`). Every
run prints the unconfirmed ones and writes them as a sidecar next to each
output artifact:

```
==========================================================
  !! UNCONFIRMED ASSUMPTIONS: 75 parameter(s) !!
----------------------------------------------------------
  [placeholder] motor.R  = 0.5 Ohm   (Q1)
  [assumed    ] bus.vbus = 12.0 V    (Q5)
  ...
  Results below are NOT hardware predictions.
==========================================================
```

Parameters that come from circuit topology aren't hand-entered at all:
component values live in `[circuit.*]` tables and the derived values
(divider ratios, filter poles, ADC sampling residuals) are recomputed
mechanically — closed-form, by ngspice over the same netlists (including
TI's own amplifier macro model), and round-tripped through a generated KiCad
schematic. `derive_params.py --check` fails if any derived number drifts
from its spec.

## What this is not

- **Not validated against hardware.** The bench is internally consistent —
  three independent plant implementations (C++, Python, OpenModelica) agree
  to <0.5%, and SPICE agrees with the closed forms — but internal
  consistency is verification, not validation. Motor parameters are
  placeholders until measured. The repo includes a model-form harness
  (`compare_traces.py`, `fit_motor_params.py`, portable stimulus timelines)
  so that the day hardware traces exist, the comparison is a command, not a
  project.
- **Not FOC.** The controller is sensored six-step with a speed PI loop.
  The bench doesn't care — any Verilog that talks SPI/PWM to these
  peripherals can be dropped in — but the reference RTL is trapezoidal.
- **Not analog simulation.** Feedback circuits are behavioral (validated
  against SPICE, but ODE/algebraic, not transistor-level). Sub-ns gate
  timing and EMI are out of scope.

## Quick start

```bash
git clone git@github.com:elliot-at-liminalnook/motorloop.git
cd motorloop
sim/scripts/check_cosim_toolchain.sh   # verilator, cmake, ninja, pybind11, pytest
sim/scripts/build_bench.sh
python3 -m pytest sim/tests            # full suite ~12 min; parity tiers in seconds

# watch it spin a motor
python3 sim/scripts/run_bench_scenario.py closed_loop --seconds 1.2
python3 sim/scripts/plot_trace.py sim/build/bench_closed_loop.csv
```

Optional tiers need `ngspice` (SPICE derivation checks), `kicad-cli`
(schematic round-trip), and OpenModelica `omc` (oracle parity). Tests for
missing tools skip rather than fail.

## Target hardware

The reference design models a specific bench so that every parameter has a
physical referent:

- FPGA: Sipeed Tang Primer 25K Dock (Gowin GW5A family)
- Power stage: ZONRI DRV8301-based 3-phase board (CRSS052N08N MOSFETs),
  a derivative of TI's DRV830x High Current EVM reference design
- ADC: Microchip MCP3208 (12-bit SPI); angle: ams OSRAM AS5600 (PWM out)
- Level shift: TXB0108 module (see `notes/hardware-bringup-notes.md` for
  why I2C through it is avoided)

## Vendor collateral

`docs/` holds datasheets, TI app notes, EVM design files, and TI's DRV8301
SPICE model — copyrighted material that is not redistributed here. Each
`docs/*/README.md` (committed) is an index of exactly what to download and
from where. Everything works without them; the one affected test
(`test_spice_derivations.py`'s DRV8301 macro cross-check) skips if
`DRV8301.LIB` is absent.

## Layout

- `rtl/` — the controller (six-step commutation, speed PI, fault manager,
  UART register file; 12 modules, lint-gated). `rtl/gen/rtl_params.vh` is
  generated from params.toml so RTL constants share the config's provenance.
- `sim/cpp/` — plant, peripheral models, lockstep scheduler, bindings
- `sim/config/params.toml` — single source of truth, provenance-flagged
- `sim/circuits/`, `hw/` — SPICE netlists and generated KiCad mirror of the
  feedback circuits
- `sim/modelica/` — independent plant oracle (run standalone via `omc`)
- `sim/scripts/` — build, derivation, scenario, plotting, and
  trace-comparison tools
- `sim/tests/` — the pytest suite (see `sim/README.md` for the tier map)
- `notes/` — architecture decision record, open questions (every
  assumption's Q-number resolves here), build checklists with findings,
  edge-case catalog, bring-up notes
