# SPDX-License-Identifier: MIT
"""`arena` — an elegant framework wrapping the co-design training harness.

A kernel/scheduler/runner split over a universal trace spine:
  * kernel    = train_adversarial.py (trains ONE stage; owns reward/benchmark/keep-best/resume)
  * trace     = arena.trace          (one Event model; context flows across process/machine bounds)
  * Stage     = arena.stage          (declarative unit of work -> a kernel CLI invocation)
  * Schedule  = arena.schedule       (Curriculum / League / Pipeline)
  * engine    = arena.engine         (the ONE drive() loop: warm-from-best -> gate -> keep-best -> resume)
  * Runner    = arena.runner         (LocalRunner / PodRunner)
  * Run       = arena.run            (Run(name, schedule, runner).go() + errors/timeline/metrics/figure)

Built incrementally per notes/framework-build-checklist.md (tracked in BUILD_STATE.json).
"""

__version__ = "0.1.0"
