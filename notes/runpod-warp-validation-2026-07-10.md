<!-- SPDX-License-Identifier: MIT -->
# MuJoCo-Warp launch-gate validation, 2026-07-10

This is evidence for the procedure in
[`pre-gpu-test-entrypoint.md`](pre-gpu-test-entrypoint.md), not a second runbook.

## Local

The canonical local stages were run from the working tree represented by the
files in this report:

- component/Verilator suite: **439 passed, 3 optional-tool skips**;
- full robot suite after final fixes: **153 passed**;
- body trainability, generated-variant, and contact/damage proofs: **passed**;
- source audit: no retired robot-backend imports or environment hooks.

The local run found and fixed a MuJoCo 3.10 `mj_fullM` call-signature mismatch.
The complete body proof is now called directly from pytest.

## RunPod

Target: secure-cloud NVIDIA L40S, 46,068 MiB, driver 580.126.09. Stack:
Python 3.12.3, Torch 2.8.0+cu128, Warp 1.14.0, MuJoCo 3.10.0, and
MuJoCo-Warp 3.10.0.1.

`bash scripts/run_pre_gpu_tests.sh --require-gpu --gpu-only` passed:

| geometry | worlds | control steps | environment-steps/s |
| --- | ---: | ---: | ---: |
| walker | 256 | 100 | 38,690.5 |
| mesh | 256 | 100 | 45,100.3 |
| fused combat | 256 | 100 | 30,140.4 |

The fresh-process same-seed canary passed walker, mesh, combat, and grouped
co-design. Final-pair worst reported drift was:

| geometry | update relative | evaluation relative | checkpoint relative |
| --- | ---: | ---: | ---: |
| walker | 1.87e-7 | 6.46e-8 | 1.25e-7 |
| mesh | 8.13e-7 | 2.35e-7 | 3.28e-7 |
| combat | 2.28e-2 | 8.29e-4 | 4.18e-3 |
| grouped co-design | 6.97e-8 | 1.54e-7 | 2.62e-7 |

Parallel ground-contact reductions are numerically repeatable, not bitwise
repeatable. The acceptance bounds are explicit and unit-tested in
`gpu_determinism_canary.py`; semantic contracts, tensor shapes/keys, steps, and
seeds remain exact. The combat held-out outcome bound is tighter than its
optimizer and running-normalizer bounds.

The run also exposed and fixed an undersized combat constraint pool before the
final pass. The final gate emitted no `nefc overflow` warnings.

The preserved final log is `/tmp/bldc_gpu_gate_final.log` with SHA-256
`5778718d6171394e15dbf4b1c81cc0520a44ee84c17408f9da7f8574d8779621` on the
validation host. The pod was terminated after the log was copied; the RunPod
API reported zero remaining pods.
