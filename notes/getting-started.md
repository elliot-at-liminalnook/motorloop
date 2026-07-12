<!-- SPDX-License-Identifier: MIT -->
# Getting started

> **Document status:** Current · **Audience:** First-time visitors · **Last reviewed:** 2026-07-12 · **Canonical for:** The first 15 minutes in the repository

## What this project is

Motorloop is a verification and simulation workspace with two main systems:

- a Verilog motor controller running in lockstep with modeled power electronics,
  sensors, supply, and motor physics; and
- a robot-design and learning stack using generated MuJoCo models,
  MuJoCo-Warp, and Torch.

The shared question is: **does a design still work when the rest of its modeled
system pushes back?** The complete boundary is shown in
[`system-architecture.md`](system-architecture.md).

## What it is not

- It is not yet a hardware-validated motor controller or robot.
- The current servo robot does not run the Verilog controller inside every
  training step.
- Historical MJX/JAX runbooks are not the active robot workflow.
- A green local robot precheck is not a complete GPU-host verification verdict.

See [`current-status.md`](current-status.md) before treating any result as a
hardware or behavioral claim.

## Choose a first outcome

### Understand the design without installing anything

Read, in order:

1. [`current-status.md`](current-status.md)
2. [`system-architecture.md`](system-architecture.md)
3. The matching route in [`reader-paths.md`](reader-paths.md)

Time: about 15 minutes.

### Inspect available commands

```bash
make help
```

The exact prerequisites and full verification sequence live in
[`reproduce.md`](reproduce.md).

### Run a focused component regression

After installing the pinned HDL and Python dependencies:

```bash
make test
```

This builds the Verilator/C++ bench and runs the component regression. It is a
focused development command, not every repository gate.

### Run the fast robot precheck

With the pinned Warp environment installed:

```bash
bash scripts/run_pre_gpu_tests.sh
```

This is intentionally a local edit/check loop. Complete launch authorization
requires the CUDA-host form documented in
[`pre-gpu-test-entrypoint.md`](pre-gpu-test-entrypoint.md).

## How to read results

The project distinguishes:

- finite automated tests;
- unbounded formal properties under assumptions;
- simulation demonstrations and rendered behavior; and
- hardware validation against measurements.

Those levels are not interchangeable. The exact vocabulary is in
[`glossary.md`](glossary.md), and the current claim/evidence table is in
[`current-status.md`](current-status.md).

## Where to go next

| Interest | Next page |
| --- | --- |
| Motor-control simulation | [`sim/README.md`](../sim/README.md) |
| Formal RTL guarantees | [`formal/README.md`](../formal/README.md) |
| Robot training and launch | [`pre-gpu-test-entrypoint.md`](pre-gpu-test-entrypoint.md) |
| Physical robot constraints | [`robot-hardware-contract.md`](robot-hardware-contract.md) |
| Reusing an RTL block | [`reader-paths.md`](reader-paths.md#reuse-an-rtl-block) |
| Documentation and contribution rules | [`documentation-guide.md`](documentation-guide.md) |
