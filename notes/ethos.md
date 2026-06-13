# Ethos: Knowing What The Bench Knows

Hardware has a strange kind of truth.

In software, a test can usually run the thing itself. If a parser accepts an
input or a database migration preserves rows, the test is close to the real
event. In hardware, especially before the board exists or before it is safe
to power, most of the work happens at a distance. We read datasheets, copy
schematics, pick placeholder motors, write timing diagrams, and tell
ourselves a story about what will happen.

That story can be useful. It can also become dangerous if it silently turns
into certainty.

This project is built around a simple habit: keep facts, assumptions, models,
and measurements separate. Let each one be useful, but make it visible what
kind of claim it is.

## What Counts As Knowing?

The bench treats knowledge in layers.

Some things are measured facts. A resistor value on the actual board, a motor
back-EMF trace, a captured SPI frame, or a hardware current waveform belongs
in this category.

Some things are documented facts. A datasheet timing limit or a reference
board schematic is not a measurement of this exact bench, but it is still
better than a guess.

Some things are design decisions. A PWM frequency, a UART register map, or a
chosen fault policy may not be "true" in nature, but it can be intentionally
chosen and then tested.

Some things are assumptions. Placeholder motor parameters, guessed harness
inductance, board values copied from a related TI EVM, or expected sensor
mounting errors are allowed in the model, but they must not pretend to be
measurements.

The goal is not to avoid assumptions. Early hardware work cannot move without
them. The goal is to keep them labeled.

## Why The Architecture Looks This Way

The architecture follows from that habit.

The RTL does not run against canned waveforms. It runs against models of the
things it will actually have to talk to: a DRV8301 gate driver, an MCP3208
ADC, an AS5600 angle sensor, feedback circuits, an inverter, a motor, and a
bench supply. That changes the question from "did the PWM waveform look
reasonable?" to "did the controller survive the same kind of closed-loop
feedback it will see on the bench?"

The simulation is lockstep and in-process because determinism matters. One
scheduler owns time. The Verilated RTL, peripheral models, and plant advance
together. If a test fails, it should fail the same way again. That makes the
bench feel more like a software regression suite and less like a demo.

The tests assert on outcomes, not just pictures. A run can fail because the
speed did not settle, an ADC aperture landed inside the PWM on-window, a DRV
register failed to verify, the bus fell through UVLO, the controller allowed
shoot-through, or a stall failed to latch safe-off. Plots exist so humans can
understand a run after the fact. They are not the pass/fail mechanism.

The plant exists in more than one form on purpose. The primary C++ plant is
checked against a Python reference and an OpenModelica oracle. Agreement
between three implementations does not prove the model is the real motor, but
it does catch sign errors, unit mistakes, and accidental drift. That is
verification of the model form, not validation against hardware.

The circuit-derived values are also checked rather than trusted. Divider
ratios, filter poles, ADC sampling residuals, and motor unit conversions are
re-derived from component-level tables. Some are cross-checked with ngspice
or a generated KiCad schematic. The important part is not that any one tool
is sacred. The important part is that independent paths must agree before a
number gets treated as stable.

## Why Parameters Carry Provenance

Every parameter in `sim/config/params.toml` carries a status such as
`measured`, `datasheet`, `decided`, `assumed`, or `placeholder`.

That is not bookkeeping for its own sake. It changes how simulation output
should be read.

If the motor resistance is a placeholder, then a speed plot is not a hardware
prediction. It may still test whether the RTL initializes the DRV8301,
samples the ADC at the right time, avoids shoot-through, and recovers from
faults. But it should not be mistaken for "this exact motor will spin at this
exact speed."

That is why every run prints the unconfirmed assumptions and writes them next
to generated artifacts. A trace should carry its caveats with it. Six months
later, the file should still say what was known and what was guessed.

## What The Bench Can And Cannot Say

The bench can say:

- this RTL followed the modeled DRV8301 SPI protocol;
- this ADC schedule placed its sampling aperture where intended;
- this control policy survived the modeled brownout, regen, stall, noise, and
  sensor-latency scenarios;
- this plant implementation agrees with the independent references under the
  tested assumptions;
- this parameter was derived consistently from the current circuit spec.

The bench cannot say, by itself:

- the real motor has these exact parameters;
- the clone board matches the TI EVM values;
- the wiring harness has this exact noise behavior;
- the thermal model is accurate enough for safety limits;
- the simulation has been validated against hardware.

Those claims need measurement. The architecture is meant to make those
measurements easy to plug in later, not to pretend they have already
happened.

## The Practical Payoff

This is not about making a perfect model before touching hardware. That would
be its own trap.

The practical payoff is narrower and more useful: catch the bugs that do not
need real hardware to catch, make the remaining unknowns explicit, and turn
future hardware measurements into updates to a checked system instead of a
pile of notes.

The bench is a way to ask better questions before power-up:

- What do we believe?
- Where did that belief come from?
- What would falsify it?
- Is this failure in the RTL, the model, the parameter set, or the hardware?

That is the connection between the ethos and the architecture. The code is
not just simulating a motor. It is trying to make the boundary between
knowledge and belief visible enough that a novice can work safely, learn
quickly, and not confuse a convincing plot with the truth.
