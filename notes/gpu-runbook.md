<!-- SPDX-License-Identifier: MIT -->
# Retired GPU runbook

> **Document status:** Retired · **Audience:** Historical readers · **Last reviewed:** 2026-07-12 · **Replacement:** [`pre-gpu-test-entrypoint.md`](pre-gpu-test-entrypoint.md)

The workflow previously documented here is historical and must not be used for
new runs. Robot simulation, training, evaluation, combat, and co-design now use
MuJoCo-Warp with Torch through one pinned environment.

Use [the single pre-GPU entry point](pre-gpu-test-entrypoint.md):

```bash
bash scripts/run_pre_gpu_tests.sh
```

That form is only a fast local precheck. Complete verification must run on the
actual CUDA host:

```bash
bash sim/robot/setup_warp_pod.sh
source /root/proj/out/warp_env.sh
bash scripts/run_pre_gpu_tests.sh --require-gpu
```

This command also runs the CPU-only RTL/component regression on the GPU host;
`--gpu-only` is deprecated and no longer omits that work. Only then use the
active `make gpu-warp-*` targets. The old requirements, environment exports,
checkpoint formats, and target ordering are unsupported.
