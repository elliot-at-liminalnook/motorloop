# OpenModelica Example Tour

> **Architecture note (2026-06-12):** since this tour was written, the project
> moved from an FMI/OMSimulator co-simulation backbone to a lockstep Verilator
> bench — see [architecture](architecture.md). The continuous/hybrid modeling
> examples below still apply directly to the Modelica **oracle** models. The
> FMU export/import and OMSimulator composition sections are now an optional
> FMI learning track, not the verification critical path.

This note captures the first source tour through a local clone of the
OpenModelica sources (https://github.com/OpenModelica/OpenModelica, not
included in this repo). Paths below are relative to that clone's root.

The examples folder (`OMCompiler/Examples/README.md`) explains that it stores
Modelica examples and scripts used to build the OpenModelica User's Guide.

## Tiny Continuous Models

`OMCompiler/Examples/HelloWorld.mo`

- Defines state `x`, parameter `a`, and the differential equation `der(x) = -a * x`.
- Useful as the smallest mental model for a plant state.

`OMCompiler/Examples/SimpleIntegrator.mo`

- Defines constant input-like variable `u` and state `x`.
- Core equation is `der(x) = u`.
- Useful for remembering that Modelica states are implied by derivative equations and initialization.

## Hybrid Events

`OMCompiler/Examples/BouncingBall.mo`

- Shows event handling with `when`.
- Uses `pre(v)` to refer to the previous event value.
- Uses `reinit(v, v_new)` to reset a continuous state at an event.

This is relevant to our hardware model for:

- fault assertions,
- overcurrent shutdown,
- ADC sample events,
- PWM edge behavior,
- sensor capture events,
- mode changes in the plant or controller boundary.

`OMCompiler/Examples/sim_BouncingBall.mos`

- Shows the script flow: `loadFile`, `simulate`, and `plot`.

## Physical Component Connections

`OMCompiler/Examples/dcmotor.mo`

- Imports `Modelica.Electrical.Analog.Basic`.
- Instantiates resistor, inductor, ground, rotational inertia, EMF, step source, and signal voltage.
- Wires components with `connect(...)`.

This is the most relevant early example because it composes electrical and mechanical domains using physical ports. Our BLDC plant should grow from this idea:

```text
phase voltage -> winding R/L -> back-EMF coupling -> shaft inertia/load -> sensor outputs
```

`OMCompiler/Examples/sim_dcmotor.mos`

- Loads the Modelica library.
- Loads `dcmotor.mo`.
- Simulates from 0 to 10 seconds.
- Plots angular speed and position.

## Simple Mode Switching

`OMCompiler/Examples/Switch.mo`

- Shows conditional equations using `if open then ... else ...`.
- Useful as a small example of topology/mode changes, though our inverter switching should be modeled carefully to avoid numerical trouble.

## External C Hooks

`OMCompiler/Examples/ExternalLibraries.mo`

- Demonstrates external C functions with `external` declarations and library/include annotations.
- Relevant if we later need a Modelica oracle component to call out to a small C implementation. (Under the lockstep-bench architecture the Verilated controller is never coupled to the Modelica side — it pairs with the C++ plant in-process; see `architecture.md`.)

## FMU Export / Import

`testsuite/openmodelica/fmi/ModelExchange/2.0/BouncingBall.mos`

- Embeds a small Modelica model with `loadString`.
- Exports an FMU with `buildModelFMU(BouncingBallFMI20, version = "2.0", fmuType="me_cs")`.
- Imports the FMU with `importFMU`.
- Loads the generated wrapper model.
- Simulates the generated FMU wrapper.

This is the first concise example of the FMU flow we need for the plant model.

`testsuite/openmodelica/fmi/CoSimulation/2.0/simpleStiffFMU.mos`

- Builds a co-simulation FMU.
- Sets FMI flags for the internal solver with `--fmiFlags=s:cvode`.
- Runs the FMU using OMSimulator.

This is relevant because solver choice and step size can dominate whether a plant FMU behaves well under co-simulation.

`testsuite/openmodelica/fmi/CoSimulation/2.0/fmi_interpolation_01.mos`

- Builds a co-simulation FMU.
- Extracts `modelDescription.xml`.
- Prints `ModelVariables` and `ModelStructure`.

This is useful because `modelDescription.xml` is the contract the master sees: variable names, causality, derivatives, outputs, and initialization metadata.

## OMSimulator Composition

`testsuite/omsimulator/DualMassOscillator.mo`

- Defines `System1`, `System2`, `CoupledSystem`, and `ReferenceSystem`.
- `System1` exposes signal ports such as force input and position/speed/acceleration outputs.
- `System2` consumes matching position/speed/acceleration inputs and outputs force.

`testsuite/omsimulator/DualMassOscillator.mos`

- Loads the model.
- Builds `System1` and `System2` as FMUs.
- Creates an OMSimulator model and system.
- Adds both FMUs as submodels.
- Connects ports by name with `oms_addConnection`.
- Sets result file, stop time, and step-size settings.
- Initializes, simulates, reads values, terminates, and deletes the model.

This is the best local blueprint for our eventual architecture:

```text
controller FMU <-> plant FMU
```

## Clocked / Sampled Behavior

`testsuite/omsimulator/testSynchronousFMU_02.mos`

- Uses `Modelica.Clocked.Examples.CascadeControlledDrive.AbsoluteClocks`.
- Simulates the model directly.
- Exports it as an FMU.
- Runs the FMU through OMSimulator.
- Compares direct simulation results against FMU simulation results.

This is worth bookmarking for sampled/clocked behavior such as ADC sampling, PWM period modeling, sensor updates, and controller-facing discrete timing.

## Practical Next Step

Start from `dcmotor.mo` and create a tiny project-local model for one motor phase:

```text
voltage command -> winding R/L -> back-EMF term -> current output
```

This model was built (`sim/modelica/BldcCosimTestbench/package.mo`) alongside a dependency-free Python reference runner. Under the lockstep-bench architecture, the next step is the C++ port of the same plant with a pytest parity check (see `../notes/architecture.md`), not FMU export — the FMU/OMSimulator flow remains available as an optional learning track once the plant is trusted.
