#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Compatibility entry point for the canonical MuJoCo-Warp pod setup.
set -Eeuo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup_warp_pod.sh" "$@"
