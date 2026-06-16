<!-- SPDX-License-Identifier: MIT -->
# Why This Exists

This project grew out of a very ordinary feeling: coming from software into
hardware and realizing that the ground is less solid than it first looks.

In software, the feedback loop is usually direct. You write code, run it,
inspect the failure, change the code, and run it again. A test is often
exercising the thing itself. It may be incomplete, but the relationship
between the test and the object under test is usually easy to understand.

Hardware feels different. Long before a board is safe to power, you are
already making claims about it. You read datasheets. You trace reference
schematics. You copy values from a board that is only probably similar to
yours. You choose a motor you have not measured yet. You imagine what the
current path is doing. You look at timing diagrams and decide what the RTL
should emit.

None of that is bad. It is how hardware projects move before the hardware can
answer back. But it creates a problem that is easy to underestimate: after a
while, it becomes hard to tell the difference between what you know, what you
believe, what you copied, what you guessed, and what your model merely made
plausible.

That is the problem this project is really about.

## Simulation As A Claim

A simulation is not truth. It is a claim written in executable form.

That claim can be extremely useful. It can catch mistakes before anything
smokes. It can make invisible timing visible. It can let you ask "what if?"
without risking a board. It can turn a confusing closed-loop behavior into
something repeatable enough to inspect.

But a simulation can also become a very convincing lie. A smooth plot can
feel like evidence even when the motor parameters are placeholders. A green
test can feel like validation even when it only proves that three pieces of
software agree with each other. A number copied from a reference design can
look official long after everyone has forgotten that it was never measured on
this board.

So the first rule here is not "trust the simulation." It is almost the
opposite:

Trust the simulation only for the kind of claim it is actually able to make.

That rule shaped the architecture.

## Why The Controller Runs Inside A World

The RTL is not tested by feeding it a few canned waveforms and checking that
PWM toggles appear. That would answer too small a question.

A motor controller is a loop. It talks to a gate driver. The gate driver has
registers, timing rules, fault behavior, and power-rail limits. The ADC does
not return "the current"; it returns a sampled, delayed, quantized version of
some voltage that came through a real analog path. The angle sensor has
latency. The motor pushes back. The supply sags. Braking can pump the bus.
Faults can reset registers underneath you.

The interesting bugs live in that loop.

That is why the controller runs against a modeled world: gate driver, ADC,
angle sensor, feedback circuits, inverter, motor, supply, and thermal effects.
The point is not to pretend that this world is perfect. The point is to make
the controller deal with consequences instead of a script.

If the SPI mode is wrong, the DRV registers do not configure. If the ADC
framing is off, the controller believes the wrong current. If the supply is
too weak, the bus falls through undervoltage and the gate driver silently
resets. Those are not pretty waveform problems. They are system problems.

## Why It Runs As A Regression Suite

Plots are useful, but they are not the judge.

The project generates figures because humans need to see. A plot can explain
a brownout, a startup transient, a bad sampling aperture, or a speed ripple in
a way that a table cannot. But visual inspection is not a testing strategy.

The actual workflow is automated. The bench runs headless. Tests assert on
behavior: did the controller reach speed, did the ADC sample at the intended
time, did the driver registers verify, did the gates avoid shoot-through, did
the fault recovery path work, did the bus voltage stay inside the expected
range, did the stall detector latch safe-off?

That matters because it changes the posture of the project. It is not a demo
where success means "the waveform looked reasonable this time." It is closer
to a software regression suite for hardware behavior. If a change breaks a
claim the project already made, the suite should say so.

This is one of the useful things software culture brings to hardware work:
the habit of making regressions cheap, boring, and repeatable.

## Why Every Parameter Has A Status

The uncomfortable part of modeling is not that some values are unknown. The
uncomfortable part is forgetting which ones are unknown.

So every parameter carries a status. Some are measured. Some come from
datasheets. Some are design decisions. Some are assumptions. Some are plain
placeholders.

That means the project can keep moving without pretending to be more certain
than it is. A placeholder motor can still be good enough to test the DRV
startup sequence, ADC timing, fault handling, and basic closed-loop behavior.
It is not good enough to predict the exact speed or current of a real motor.

Both statements can be true at the same time. The status labels are what keep
them from collapsing into each other.

The assumption banner and sidecar files are part of the same idea. A trace
should carry its own caveats. Months later, it should still be obvious
whether the run was based on measurements or on scaffolding.

## Why The Same Physics Appears More Than Once

The motor plant exists in multiple forms because disagreement is valuable.

The C++ model is the one used in the fast lockstep bench. The Python model is
small and readable. The OpenModelica model is an independent oracle. When
they agree, that does not prove the model matches hardware. It does prove
something narrower and still important: the equations are probably being
implemented consistently, with fewer sign errors and unit mistakes hiding in
the code.

The same reasoning applies to circuit-derived parameters. A divider ratio or
filter pole should not survive just because someone typed it once. It should
be re-derived from the component values. Where practical, it should be
checked through another path, such as SPICE or a generated schematic.

This is not bureaucracy. It is a way of making beliefs pay rent.

## What This Is Preparing For

The project is not trying to avoid hardware. The whole point is to arrive at
hardware with better questions.

When the real board finally produces a trace, the useful question is not
"does it match the simulation?" in some vague sense. The useful questions are
more specific:

- Which parameter was wrong?
- Which model effect was missing?
- Which assumption can now become a measurement?
- Which behavior was a controller bug rather than a model bug?
- Which safety claim survived contact with the bench?

That is why the architecture includes shared trace formats, comparison tools,
stimulus timelines, and motor-identification helpers. The goal is for
measurement day to shrink uncertainty instead of creating a new pile of
unstructured notes.

## The Larger Motivation

This project is a motor-control testbench, but the deeper pattern is broader:
build systems that remember the difference between belief and knowledge.

That matters especially for novices. A beginner does not just need answers.
A beginner needs a way to tell what kind of answer they are holding. Is this
from a datasheet? A measurement? A copied reference design? A simulation? A
guess that has become load-bearing?

The architecture is an attempt to make that distinction visible. Not perfect,
not final, and not a substitute for the physical bench. But useful.

It lets the project say:

This is what the controller did.

This is the world we tested it in.

This is where the numbers came from.

This is what remains unproven.

That is the spirit of motorloop. It is not just about spinning a simulated
motor. It is about learning how to make claims carefully, so that when the
hardware finally gets a vote, the project is ready to listen.
