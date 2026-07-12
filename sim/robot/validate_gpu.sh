#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Compatibility entry point for the canonical target-GPU gate.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$ROOT/scripts/run_pre_gpu_tests.sh" --require-gpu --gpu-only "$@"
