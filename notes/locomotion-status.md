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
- A sequential thirty-one-rung curriculum, task-conditioned FiLM policy,
  fixed-seed promotion, prior-skill replay, and a persistent regression matrix
  are implemented.
- Rungs 1–5 are accepted. The resumable rung-6 stepping policy is at step
  10,002,432 and passes every physical/safety gate; step-clock agreement is
  0.699512 on the worst of three deterministic seeds against a 0.70 gate, so it
  remains correctly unpromoted. The other two seeds score 0.700459 and 0.700644;
  promotion uses the adverse seed tail rather than the passing mean.
- The stopped RunPod artifacts and exact resume command are recorded in
  [`training-ladder-runbook.md`](training-ladder-runbook.md).
- The complete verification entry point covers walker, mesh, combat, ladder,
  diagnostics, and grouped co-design mechanics on CUDA.

## Current gate

A ladder artifact is promotable only when it:

1. passes every task-specific physical and competence threshold;
2. passes each threshold at the conservative tail of its deterministic
   multi-seed evaluation bank;
3. reproduces the pass at the separate fixed retention seed;
4. replays every prior same-family skill within its stored tolerance;
5. has compatible model, observation, action, reward, and dependency metadata;
   and
6. produces a plausible rendered rollout before a behavioral claim is promoted.

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
