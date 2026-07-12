<!-- SPDX-License-Identifier: MIT -->
# Retired GPU pod setup

> **Document status:** Retired · **Audience:** Historical readers · **Last reviewed:** 2026-07-12 · **Replacement:** [`pre-gpu-test-entrypoint.md`](pre-gpu-test-entrypoint.md)

This file is retained only so historical links do not break. It is **not a
runbook**. The former commands installed a retired robot backend and used a
different environment, checkpoint format, and launch path.

The only supported local and RunPod procedure is:

- [Pre-GPU deterministic test entry point](pre-gpu-test-entrypoint.md)
- `scripts/run_pre_gpu_tests.sh`
- `requirements-warp.txt`
- `sim/robot/setup_warp_pod.sh`

For an ephemeral RunPod, copy the same working tree used for the local precheck:

```bash
cd /root/proj
bash sim/robot/setup_warp_pod.sh
source /root/proj/out/warp_env.sh
bash scripts/run_pre_gpu_tests.sh --require-gpu
```

Do not launch a long job unless that command exits `0`. Terminate the pod after
validation or after copying any required artifacts off its ephemeral disk.
