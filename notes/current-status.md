<!-- SPDX-License-Identifier: MIT -->
# Current project status

> **Document status:** Current · **Audience:** All readers · **Last reviewed:** 2026-07-14 · **Canonical for:** High-level maturity, current frontiers, and evidence precedence

This page answers “what is true now?” It deliberately avoids copying volatile
test counts and timing numbers. Follow the evidence links for the latest generated
values.

## Status at a glance

| Area | Current maturity | Best evidence | Important boundary |
| --- | --- | --- | --- |
| Motor-control RTL | Implemented, simulation-tested, and partly formally proven | [`formal/proof_report.md`](../formal/proof_report.md), [`status-matrix-generated.md`](status-matrix-generated.md) | No correlation against a physical motor bench |
| Component co-simulation | Implemented with Verilated RTL, C++ plant, behavioral peripherals, Python scenarios, and independent plant checks | [`sim/README.md`](../sim/README.md), [`architecture.md`](architecture.md) | Model parameters still include explicitly flagged assumptions |
| FPGA synthesis and packaging | Open ECP5 place-and-route meets the target clock; reusable blocks have packaging and contracts | [`synth/synth_report.md`](../synth/synth_report.md), [`rtl/contracts/`](../rtl/contracts/) | Other vendor flows are portability evidence unless run in their authoritative tools |
| Active robot backend | Plain MuJoCo oracle plus batched MuJoCo-Warp physics and Torch PPO | [`pre-gpu-test-entrypoint.md`](pre-gpu-test-entrypoint.md), [`runpod-warp-validation-2026-07-10.md`](runpod-warp-validation-2026-07-10.md) | CUDA reductions are bounded-repeatable, not promised bitwise identical |
| Active robot hardware envelope | 6 lb maximum, twelve ST3215-HS bus servos | [`robot-hardware-contract.md`](robot-hardware-contract.md) | Continuous torque, output inertia, power delivery, and the physical mass budget need measurement |
| Locomotion | Active development through a gated 31-rung universal-controller curriculum with rotating full-episode promotion, adaptive real replay, and outcome-based objectives | [`locomotion-status.md`](locomotion-status.md), [`training-ladder-runbook.md`](training-ladder-runbook.md), [`training-objective-contract.md`](training-objective-contract.md) | Rungs 1–6 are accepted; outcome-only rung-7 forward walking is training from the promoted stepping policy, and no later locomotion/combat rung is yet promoted |
| Predictive control | Experimental morphology-token GRU with selectable recurrent or causal temporal-Transformer future decoder, masked interaction commands, calibrated gradient planning, and a matched L40S first-result proof | [`predictive-universal-controller.md`](predictive-universal-controller.md), [`predictive-transformer-proof-2026-07-15.md`](predictive-transformer-proof-2026-07-15.md) | Transformer predictor compute is faster, but its short-run final held-out accuracy regressed; neither predictor nor diffusion is promoted into the accepted ladder |
| Combat and self-play | Environments, scoring, curricula, opponent machinery, and verification fixtures exist | [`locomotion-status.md`](locomotion-status.md), [`rl-verification-playbook.md`](rl-verification-playbook.md) | Open-ended self-play combat is an active frontier, not a completed result |
| Real2Sim2Real | Identification, residual, adaptation, and robust-ranking hooks exist | [`codesign-realization-report.md`](codesign-realization-report.md) | No complete hardware fit or sim-to-real deployment has been validated |

## The walking claims, separated

Two claims that used to be blended together are different:

1. A **historical parametric-body policy** reached a rendered 0.83 m/s mean
   deployment evaluation after the missing actuator-gear bug was fixed. The
   dated evidence is preserved in [`training-uplift-results.md`](training-uplift-results.md).
2. The **current physical design contract**, adopted later, is the 6 lb,
   twelve-servo mesh-derived robot. Its locomotion work has separate morphology,
   actuator, route, and promotion gates. Its current state is summarized in
   [`locomotion-status.md`](locomotion-status.md).

The first result proves that the corrected training stack can produce a walking
policy in its tested model. It is not a hardware result and does not automatically
transfer to the later physical design.

## Evidence precedence

When numbers or status disagree, use this order:

1. A freshly produced machine artifact from the canonical command.
2. The generated report committed from that artifact.
3. This high-level status page.
4. A subsystem explanation or current checklist.
5. Historical plans, reports, surveys, and learning logs.

In particular:

- `synth/synth_report.md` owns current ECP5 timing and utilization.
- `formal/proof_report.md` owns the last rendered proof results.
- `notes/status-matrix-generated.md` combines the last rendered proof and
  synthesis evidence.
- `notes/reproduce.md` owns full repository reproduction.
- `notes/pre-gpu-test-entrypoint.md` owns the robot/RL launch gate.
- `notes/robot-hardware-contract.md` owns the active robot physical envelope.

## What is not yet claimed

- The simulated motor controller has not been correlated with the intended
  inverter, sensors, motor, supply, and harness on a physical bench.
- The robot’s datasheet stall-torque envelope is not a continuous-duty torque
  measurement.
- A rendered simulation behavior is not hardware validation.
- The current combat stack has not produced a final, open-ended self-play result.
- Real2Sim2Real infrastructure is not evidence of successful sim-to-real transfer.

The project’s shortest honest description is therefore: **a broad verification
and simulation system with strong executable checks, active robot-learning
experiments, and intentionally unfinished hardware validation.**
