<!-- SPDX-License-Identifier: MIT -->
# Locomotion and combat status

> **Document status:** Active · **Audience:** Robot-learning contributors · **Last reviewed:** 2026-07-12 · **Canonical for:** Short current summary of locomotion promotion and combat prerequisites

The active work is no longer “can any corrected model walk?” It is whether the
current robot design and its learned/teacher stack can satisfy strict, composable
navigation gates before combat training is promoted.

## Current verified state

- The active backend is MuJoCo-Warp with Torch; retired MJX/JAX launch procedures
  are not supported.
- The 6 lb, twelve-servo physical envelope is defined and contract-tested in
  [`robot-hardware-contract.md`](robot-hardware-contract.md).
- Cardinal CPG priors, route-aware search, transition datasets, behavior cloning,
  residual-policy paths, rendering, and strict checkpoint evaluation exist.
- The best documented route scaffold reaches two of four strict waypoints without
  falling. It does not satisfy the three-of-four teacher/policy promotion gate.
- The complete verification entry point covers walker, mesh, combat, and grouped
  co-design mechanics on CUDA.

## Current gate

A locomotion artifact is promotable only when it:

1. reaches at least three of four waypoints at the strict radius;
2. passes fixed-direction survival and regression checks;
3. has compatible model, observation, action, and checkpoint metadata; and
4. produces a plausible rendered rollout.

Until that gate passes for the active design, downstream open-ended combat is a
frontier rather than a result.

## Detailed active records

- [`locomotion-bootstrap-teacher-checklist.md`](locomotion-bootstrap-teacher-checklist.md)
  is the append-only implementation and experiment log.
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
