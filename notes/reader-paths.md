<!-- SPDX-License-Identifier: MIT -->
# Reader paths

> **Document status:** Current · **Audience:** Newcomers and contributors · **Last reviewed:** 2026-07-12 · **Canonical for:** Goal-oriented routes through the repository

Each route ends with a concrete outcome. Read only the documents needed for
that outcome; the full historical catalog is optional.

## Understand the project in 15 minutes

**Outcome:** explain the two systems, current maturity, and largest open boundary.

1. [`getting-started.md`](getting-started.md)
2. [`system-architecture.md`](system-architecture.md)
3. [`current-status.md`](current-status.md)

Stop when you can explain why “simulation-verified” is not “hardware-validated.”

## Run or modify the motor-control bench

**Outcome:** build the Verilator/C++ bench, run a focused regression, and locate
the parameter assumptions.

1. Read [`sim/README.md`](../sim/README.md).
2. Inspect `sim/config/params.toml` and [`open-questions.md`](open-questions.md).
3. Follow the local prerequisites in [`reproduce.md`](reproduce.md).
4. Run `make test`; use the manual scenarios in `sim/README.md` for debugging.
5. Read [`architecture.md`](architecture.md) before changing process boundaries
   or plant scheduling.

Expected time after dependencies are installed: one focused work session.

## Review formal and implementation evidence

**Outcome:** determine what is proven, simulated, or merely documented for an
RTL block.

1. Read [`formal/README.md`](../formal/README.md).
2. Find the block in [`formal/proof_report.md`](../formal/proof_report.md).
3. Open its file under `rtl/contracts/`.
4. Check the generated [`status-matrix-generated.md`](status-matrix-generated.md).
5. Use [`synth/synth_report.md`](../synth/synth_report.md) only for the current
   ECP5 implementation result, not for real-silicon behavior.

## Train or evaluate a robot

**Outcome:** enter the active Warp/Torch workflow without following retired MJX
commands or launching an unverified long run.

1. Read [`current-status.md`](current-status.md), especially the separated
   walking claims.
2. Read [`robot-hardware-contract.md`](robot-hardware-contract.md).
3. Follow [`pre-gpu-test-entrypoint.md`](pre-gpu-test-entrypoint.md) exactly.
4. Use the active `make gpu-warp-*` targets only after the full gate exits zero.
5. Use [`rl-verification-playbook.md`](rl-verification-playbook.md) when changing
   rewards, observations, generated physics, or checkpoint semantics.

The local no-argument precheck is useful but cannot authorize a long CUDA run.

## Assess hardware readiness

**Outcome:** identify which values are measured, datasheet-backed, assumed, or
still blocked before physical bring-up.

1. Read [`current-status.md`](current-status.md).
2. For the motor bench, inspect [`open-questions.md`](open-questions.md),
   [`hardware-bringup-notes.md`](hardware-bringup-notes.md), and
   `sim/config/params.toml`.
3. For the robot, inspect
   [`robot-hardware-contract.md`](robot-hardware-contract.md) and the current
   CAD/physics metadata.
4. Treat simulation plots as predictions with named assumptions, not measurements.

## Reuse an RTL block

**Outcome:** consume one block without pulling in the whole controller.

1. Choose a contract under `rtl/contracts/`.
2. Check proof and simulation status in
   [`status-matrix-generated.md`](status-matrix-generated.md).
3. Follow the single-block acceptance example in [`reproduce.md`](reproduce.md).
4. Use the corresponding FuseSoC core or `Bender.yml` source map.

## Contribute documentation

**Outcome:** add a page that remains findable and cannot silently become stale.

1. Follow [`documentation-guide.md`](documentation-guide.md).
2. Add or refine its lifecycle classification in `docs-lifecycle.toml`.
3. Link it from [`README.md`](README.md) if it is a curated entry, otherwise let
   the generated catalog expose it.
4. Run `python3 scripts/check_docs.py` and `make docs`.
