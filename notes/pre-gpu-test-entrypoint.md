<!-- SPDX-License-Identifier: MIT -->
# Simulation and RL verification entry point

> **Document status:** Current · **Audience:** Robot-learning developers · **Last reviewed:** 2026-07-12 · **Canonical for:** Robot/RL verification and long-run launch authorization

## The only entry point

Run commands from the repository root. For a fast local edit/check cycle:

```bash
bash scripts/run_pre_gpu_tests.sh
```

This is a **precheck, not full verification**. The only command that produces a
complete verification verdict must run in a CUDA environment:

```bash
bash scripts/run_pre_gpu_tests.sh --require-gpu
```

`make pre-gpu` and `make pre-gpu-gpu` are aliases for those two modes. Do not
assemble a launch decision from individually green pytest or benchmark commands.
The old `--gpu-only` spelling is accepted temporarily, but is deprecated and no
longer skips the CPU-only stages.

## Why full verification needs a GPU host

The active robot stack is MuJoCo plus MuJoCo-Warp with Torch. Plain MuJoCo is the
single-world CPU oracle. MuJoCo-Warp executes batched physics on CPU or CUDA, and
Torch owns policies, PPO, checkpoints, and offline helpers.

The most expensive robot rollouts and PPO integration tests are marked `gpu`.
They run 64 to 256 worlds on CUDA and cover walker, mesh, combat, gait tracking,
autoreset, constraint-pool capacity, CaT termination, checkpoint/resume, and PPO.
The same-seed canary then runs fresh training processes for walker, mesh, combat,
and grouped co-design.

Verilator and the component/co-simulation tests cannot execute as GPU kernels.
They remain CPU tests, but full verification runs them on the GPU host with
independent tests distributed across pytest-xdist workers. The shared `bldcsim`
extension is built once before workers start. This is intentionally part of the
same full command: a CUDA result without the RTL/component regression is not a
complete result.

## What each mode runs

| Stage | Local precheck | Full GPU-host verification |
| --- | --- | --- |
| Dependency and patch hygiene | yes | yes |
| Fast parameter/model-form/compiled-FOC checks | yes | covered by full suite |
| Complete component and Verilator regression | no | yes, parallel CPU workers |
| Exact CPU MuJoCo oracles and deterministic robot contracts | yes | yes |
| Tests marked `gpu` | no | yes, CUDA required |
| Body, generated-variant, and contact/damage proofs | yes | yes |
| Walker, mesh, and combat CUDA execution benches | no | yes |
| Fresh-process same-seed CUDA training canary | no | yes |

The robot source guard is in the collected tree and fails if a retired backend
import is reintroduced. Missing legacy dependencies therefore cannot turn an
unported code path into a false-green skip.

## Setup

For local work, install the platform-appropriate PyTorch build and the pinned
stack in one environment:

```bash
python3 -m venv .venv-warp
.venv-warp/bin/python -m pip install torch
.venv-warp/bin/python -m pip install -r requirements-warp.txt
bash scripts/run_pre_gpu_tests.sh
```

For a CUDA/RunPod host, place the same working tree at `/root/proj` and run:

```bash
cd /root/proj
bash sim/robot/setup_warp_pod.sh
source /root/proj/out/warp_env.sh
bash scripts/run_pre_gpu_tests.sh --require-gpu
```

`setup_warp_pod.sh` installs the pinned Python stack and the native tools needed
by the CPU-only component regression. Set `PRE_GPU_CPU_WORKERS` to tune the
parallel component tests; the default is 8. Set `PRE_GPU_STAGE_TIMEOUT` to change
the default 7200-second timeout per stage, or `PRE_GPU_TMPDIR` to retain outputs.

Only after the full command exits zero should a long job start:

```bash
make gpu-warp-train
make gpu-warp-combat
make gpu-warp-selfplay
make gpu-warp-codesign
```

## Determinism contract

Exact model, schema, reset, trajectory, reward, and CPU same-seed checks remain
on CPU. CUDA ground-contact kernels use parallel reductions, so bitwise equality
is not a portable GPU contract. `gpu_determinism_canary.py` instead requires
exact shapes, keys, seeds, steps, and checkpoint semantics plus explicit bounded
numeric drift for each geometry. Its `--report-only` mode is diagnostic and is
never used by the entry point.

## Exit contract

- Exit 0 without `--require-gpu`: fast local precheck passed; full verification
  remains outstanding.
- Exit 0 with `--require-gpu`: complete CPU and CUDA verification passed on the
  current host and working tree.
- Any other exit: do not launch. Fix the first failure and rerun the entry point.

## Scope and new tests

This establishes deterministic CPU behavior, bounded target-GPU repeatability,
and simulated-physics consistency. It does not validate unmeasured hardware such
as continuous servo torque, supply droop, bus timing, thermal behavior,
transmission loss, or physical mass distribution.

Put deterministic robot tests under `sim/robot/test_*.py`. Mark expensive batched
rollout or trainer tests with `@pytest.mark.gpu` and use the shared `gpu_device`
fixture; they must not silently fall back to CPU. Add new long-run geometries to
`gpu_determinism_canary.py`. Put component tests under `sim/tests`; the full
parallel regression collects the entire directory automatically.
