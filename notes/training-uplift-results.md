<!-- SPDX-License-Identifier: MIT -->
# Training-uplift results

> **Document status:** Historical · **Audience:** Robot-learning contributors · **Last reviewed:** 2026-07-12 · **Canonical for:** Durable verdict extracted from the July 2026 uplift execution record

## Verdict

The July 2026 audit found that generated MJCF omitted the actuator `gear`
attribute. The intended joint envelope was about thirteen times larger than the
effective effort. Contract tests were added around compiled actuator outcomes,
the generator was corrected, and a parametric-body PPO run then passed the
project’s rendered locomotion gate.

The recorded deployment evaluation reached a 0.83 m/s mean under a 1.2 m/s
command, survived 600 of 600 steps, and showed a visible alternating stride in
the preserved rendered video. The exact dated commands, intermediate runs, and
artifact paths remain in [`uplift-execution-plan.md`](uplift-execution-plan.md).

## What this demonstrated

- The missing compiled actuator ratio was capable of explaining the earlier
  underpowered behavior.
- Outcome-based model contracts can catch a configuration that intent-only
  validation missed.
- With corrected effort and revised PPO/PD settings, the tested parametric body
  could learn visible commanded locomotion in simulation.

## What this did not demonstrate

- Hardware validation of the motor, actuator, or robot.
- Transfer to the later 6 lb, twelve-ST3215-HS physical design contract.
- A final combat or open-ended self-play policy.
- General parity between every historical MJX experiment and the active Warp
  backend.

The current robot status is [`locomotion-status.md`](locomotion-status.md), and
the project-wide maturity boundary is [`current-status.md`](current-status.md).
