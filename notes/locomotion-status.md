<!-- SPDX-License-Identifier: MIT -->
# Locomotion and combat status

> **Document status:** Active · **Audience:** Robot-learning contributors · **Last reviewed:** 2026-07-14 · **Canonical for:** Short current summary of locomotion promotion and combat prerequisites

The active work is no longer “can any corrected model walk?” It is whether the
current robot design and its learned/teacher stack can satisfy strict, composable
navigation gates before combat training is promoted.

## Current verified state

- The active backend is MuJoCo-Warp with Torch; retired MJX/JAX launch procedures
  are not supported.
- The 6 lb, twelve-servo physical envelope is defined and contract-tested in
  [`robot-hardware-contract.md`](robot-hardware-contract.md).
- A sequential thirty-one-rung curriculum, one 256-observation/14-action
  task-conditioned universal controller, immutable candidate archives, rotating
  promotion seeds, real old-task replay, and a persistent regression matrix are
  implemented.
- Rungs 1–6 are accepted. Rung 6 passed five fresh full-episode promotion seeds
  plus retention replays for rungs 2–5; its worst held-out foot was airborne for
  8.99% of the episode against a 5% requirement, with no physical constraint
  terminations or falls.
- The rung-6 objective is phase-free: every foot must physically leave the
  ground while the base remains in place, upright, and within physical safety
  limits. Clock agreement, diagonal pairing, cadence, symmetry, and duty shape
  are diagnostics rather than rewards or promotion gates.
- The live handoff, frozen checkpoint hash, and exact resume command are recorded
  in [`training-ladder-runbook.md`](training-ladder-runbook.md); the governing
  reward/gate rules are in
  [`training-objective-contract.md`](training-objective-contract.md).
- Rung 7 is actively training forward travel directly from the accepted stepping
  policy. It has no reference gait, implicit CPG action prior, or fixed
  gait/airtime/pose/clearance style reward. Its new task channel is initialized
  to reproduce rung 6 exactly before specializing; a legacy teacher is available
  only as an optional post-failure exploration fallback.
- The complete verification entry point covers walker, mesh, combat, ladder,
  diagnostics, and grouped co-design mechanics on CUDA.

## Current gate

A ladder artifact is promotable only when it:

1. passes every task-specific outcome and physical-constraint threshold;
2. passes each threshold on fresh rotating promotion seeds over complete
   800-step episodes;
3. replays every prior skill for a complete episode within its stored tolerance;
4. uses real prior-task state/action replay with adaptive pressure while learning;
5. has compatible model, observation, action, reward, and dependency metadata;
   and
6. keeps style-only measurements out of promotion, while reporting contact
   diversity, per-leg behavior, and reward-role shares for diagnosis; and
7. produces a plausible rendered rollout before a behavioral claim is promoted.

Until that gate passes for the active design, downstream open-ended combat is a
frontier rather than a result.

## Detailed active records

- [`locomotion-bootstrap-teacher-checklist.md`](locomotion-bootstrap-teacher-checklist.md)
  is the append-only implementation and experiment log.
- [`training-ladder-runbook.md`](training-ladder-runbook.md) owns the current
  sequential run state, resume procedure, and adaptive-contract lessons.
- [`sparc-learning-log.md`](sparc-learning-log.md) records the active combat
  scoring/router experiments.
- [`codesign-win-exchanges-checklist.md`](codesign-win-exchanges-checklist.md)
  records the fight-curriculum milestones.

Those files preserve exact runs and intermediate failures. This page owns the
short status; do not reconstruct the current verdict from the last paragraph of
each log.

## Historical result boundary

The earlier parametric body’s rendered 0.83 m/s result remains valid for its
tested model and is summarized in
[`training-uplift-results.md`](training-uplift-results.md). It predates the active
physical contract and is not the promotion evidence for this route.
