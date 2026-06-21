<!-- SPDX-License-Identifier: MIT -->
# GPU runbook — MJX co-design (provision → install → run → cost)

The co-design realization (`notes/codesign-realization-checklist.md`) needs a CUDA GPU;
the local box has none. This is the end-to-end recipe to reproduce every GPU result on a
rented box (RunPod 4090 used here). The CPU-only Phase-R/RS sim-to-sim verifications run
locally with no GPU — see `make codesign-rs`.

> **Exact from-scratch pod recipe (provision → ssh → ship → install → env → terminate) +
> the hard-won gotchas is in `notes/gpu-pod-setup.md`.** The pod disk is ephemeral; the
> trained checkpoints/metrics this session produced are saved locally in `sim/build/gpu/out/`
> (re-ship them to a fresh pod to resume warm-starts). The section below is the summary.

## 0. Provision (RunPod 4090; see notes/gpu-pod-setup.md + the motorloop-runpod-gpu memory)
- Deploy on-demand via the GraphQL API (curl, not urllib — Cloudflare blocks urllib's UA);
  image `runpod/pytorch:2.x-py3.10-cuda12-*`, port 22, `PUBLIC_KEY` env = your SSH pubkey.
- SSH in with the dedicated key (`~/.ssh/runpod_ed25519`). Cost discipline: spin up for a
  burst, **terminate to stop billing** when idle (4090 ≈ $0.34–0.69/hr).

## 1. Install
```
tar czf - sim/robot sim/tests/motors.py requirements-gpu.txt | ssh POD "tar xzf - -C /root/proj"  # ship code+reqs
ssh POD 'bash /root/proj/sim/robot/setup_pod.sh'   # ONE command: pinned install + out/env.sh + smoke test
```
`setup_pod.sh` installs the **pinned** `requirements-gpu.txt` and writes `out/env.sh` (the
mandatory `MUJOCO_GL=""`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `CODESIGN_OUT`,
`JAX_COMPILATION_CACHE_DIR` exports — `source` it before every run). Full gotchas:
`notes/gpu-pod-setup.md`. Gotcha: pod is Python 3.10 → `tomllib` is 3.11+; `gen_robot_mjcf`
falls back to `tomli`.

## 2. Run order (each `make` target = one script; `CODESIGN_OUT` = where checkpoints land)
**Always leak-test tiny first** (the E2E-first rule — one GPU ⇒ stages are sequential):
```
make gpu-validate            # tiny run of EVERY stage end-to-end (~25-40 min); MUST be green
```
Then the real-scale runs, in dependency order:
| step | target | what | rough wall-clock (4090) |
|---|---|---|---|
| 0/1 | `make gpu-baseline`   | throughput baseline + baseline locomotion policy | ~5–15 min |
| 1/9 | `make gpu-parity`     | MJX↔MuJoCo parity gate                            | ~2 min |
| 2/3/5 | `make gpu-codesign` | universal policy + trained-return CEM + Pareto    | ~20–60 min |
| R1/R7 | `make gpu-rederive` | re-derive rankings under the calibrated sim       | ~5 min |
| B    | `make gpu-adversarial`| warm-start the SPARC fighter from the walker      | ~30–60 min |
| 4    | `make gpu-selfplay`   | self-play Hall-of-Fame league                     | ~30–90 min |
| 4    | `make gpu-coevolve`   | two-body co-evolution (CPU physics, no GPU)       | ~2 min |
| 6/7  | `make gpu-extra`      | topology GA (CPU) + differentiable co-design (GPU)| ~3 min |

Throughput note: this body is **contact-bound** in MJX (~35k env-steps/s saturated at
batch 16384 on a 4090; scales 6140× over a single env). brax PPO's cost floor is
`batch·unroll·minibatches` env-steps/step — shrink all three for `--tiny`.

## 3. Real2Sim2Real (Phase R/RS) — framework-now, hardware-gated-fit (NO GPU needed)
```
make codesign-rs            # all sim-to-sim self-tests (posterior recovery, residuals,
                            # RMA adaptation gap, robust QD, info-gain, multifidelity, NSGA-II)
make gpu-residual           # RS2/RS3 learned actuator + contact residuals (sim-to-sim)
make gpu-rma                # RS4 teacher->student adaptation gap (sim-to-sim)
make gpu-robust-codesign    # R6/RS6 CVaR + MAP-Elites archive
make gpu-active-id          # RS5/RS8 info-gain test selection + ranking correlation
```
Each prints `PROVEN: ...`. The real fits (motor bench logs, drop/ram tests) swap the
synthetic "real" stand-in for measured logs once hardware exists — the code paths are wired.

## 4. Reproduce + tear down
```
make gpu-validate           # everything green tiny
# ... real-scale targets as needed ...
# TERMINATE the pod (podTerminate) to stop billing — disk is ephemeral, re-ship next time.
```
Results + the proxy↔real / generalization / speedup numbers land in
`notes/codesign-realization-report.md`.
