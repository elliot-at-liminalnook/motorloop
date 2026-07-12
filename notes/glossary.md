<!-- SPDX-License-Identifier: MIT -->
# Glossary

> **Document status:** Current · **Audience:** Newcomers and contributors · **Last reviewed:** 2026-07-12 · **Canonical for:** Project terminology and evidence vocabulary

Use this page to decode project-specific acronyms and to distinguish levels of
evidence that must not be treated as interchangeable.

## System and control terms

| Term | Meaning in this repository |
| --- | --- |
| **BLDC** | Brushless DC motor. The component bench models a three-phase motor, inverter, sensors, and controller. |
| **FOC** | Field-oriented control: current control in a rotor-aligned `d/q` coordinate frame. |
| **RTL** | Register-transfer-level Verilog intended for FPGA or ASIC implementation. |
| **co-simulation** | The lockstep process in which Verilated RTL and the C++ plant/peripheral models advance together. It does not mean the robot trainer runs the RTL in its inner loop. |
| **plant** | The modeled physical system being controlled: inverter, electrical motor dynamics, mechanics, supply, and optional realism effects. |
| **peripheral model** | A behavioral model of a chip boundary such as a gate driver, ADC, or angle sensor. |
| **contract** | An explicit interface, timing, assumption, or outcome that a test or proof can check. |
| **oracle** | An independently implemented reference used for comparison, such as plain MuJoCo or the Modelica plant. |

## Robot and learning terms

| Term | Meaning in this repository |
| --- | --- |
| **MJCF** | MuJoCo XML model format generated from the robot design sources. |
| **MuJoCo** | The single-world CPU physics implementation used as the semantic oracle and for rendering. |
| **MuJoCo-Warp / Warp** | The active batched physics backend used for robot training on CPU or NVIDIA CUDA. |
| **MJX** | MuJoCo’s older JAX-oriented backend used by historical experiments. It is not the active training backend. |
| **PPO** | Proximal Policy Optimization, the policy-learning algorithm used by the active Torch trainers. |
| **PD control** | A proportional-derivative controller that turns policy position targets into joint effort. |
| **CPG** | Central pattern generator: a compact oscillatory gait prior used by locomotion experiments and teacher search. |
| **RND** | Random Network Distillation, an exploration bonus based on prediction novelty. |
| **HER** | Hindsight Experience Replay, relabeling achieved outcomes as goals. |
| **PBT** | Population-Based Training, periodically copying and mutating trainers. |
| **PFSP** | Prioritized Fictitious Self-Play, sampling opponents according to matchup difficulty. |
| **SPARC** | The project’s combat scoring and evaluation surface; use the defining code and current active-work summary for exact metrics. |
| **HoF** | Hall of fame: frozen opponent checkpoints retained for evaluation or self-play. |
| **Real2Sim2Real** | Hooks for fitting simulation from measurements, training with uncertainty, and returning policies to hardware. The hooks exist; hardware fitting is not complete. |

## Evidence and maturity terms

| Term | Required evidence |
| --- | --- |
| **Implemented** | The code path exists and is reachable. This alone says nothing about correctness. |
| **Tested** | An automated finite test exercises the stated behavior. |
| **Formally proven** | A plant-independent RTL property has an unbounded proof under recorded assumptions. |
| **Simulation-verified** | Tests or cross-engine checks support a claim inside the stated model and configuration. |
| **Demonstrated** | A bounded experiment or rendered rollout showed the behavior once or over a stated evaluation set. |
| **Hardware-validated** | Simulation or control behavior was correlated against measurements from the actual physical system. This project has not reached that maturity for the complete motor or robot systems. |
| **Historical** | Correctly describes an earlier repository state but must not be read as current guidance. |
| **Canonical** | The designated source to update and follow for a particular topic. |

See [`current-status.md`](current-status.md) for how these terms apply to the
current repository.
