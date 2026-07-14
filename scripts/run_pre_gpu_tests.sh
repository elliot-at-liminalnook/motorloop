#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Single entry point for local prechecks and complete GPU-host verification.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WARP_PY="${WARP_PY:-$ROOT/.venv-warp/bin/python}"
STAGE_TIMEOUT="${PRE_GPU_STAGE_TIMEOUT:-7200}"
REQUIRE_GPU=0
GPU_ONLY=0
CPU_WORKERS="${PRE_GPU_CPU_WORKERS:-8}"

usage() {
  cat <<'EOF'
Usage: scripts/run_pre_gpu_tests.sh [--require-gpu] [--gpu-only]

With no options, runs the fast deterministic CPU precheck. This is useful while
editing, but it is not full verification. On a CUDA machine, --require-gpu runs
the complete component and robot regression, CUDA-marked MuJoCo-Warp tests,
batched execution checks, and the same-seed training canary.

--gpu-only is a deprecated compatibility option. It no longer skips CPU-only
RTL/component tests because a complete GPU-host verification must include them.

Environment overrides: WARP_PY, PRE_GPU_TMPDIR, PRE_GPU_STAGE_TIMEOUT (seconds,
default 7200 per stage), PRE_GPU_CPU_WORKERS (default 8).
EOF
}

while (($#)); do
  case "$1" in
    --require-gpu) REQUIRE_GPU=1 ;;
    --gpu-only) GPU_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if ((GPU_ONLY && !REQUIRE_GPU)); then
  echo "--gpu-only requires --require-gpu" >&2
  exit 2
fi
if ((GPU_ONLY)); then
  echo "warning: --gpu-only is deprecated; running the complete GPU-host verification" >&2
fi
[[ -x "$WARP_PY" ]] || { echo "required interpreter is missing: $WARP_PY" >&2; exit 2; }
export WARP_PY

TMP_ROOT="${PRE_GPU_TMPDIR:-$(mktemp -d -t bldc-pre-gpu-XXXXXX)}"
mkdir -p "$TMP_ROOT/out"
if [[ -z "${PRE_GPU_TMPDIR:-}" ]]; then
  trap 'rm -rf "$TMP_ROOT"' EXIT
fi

export CODESIGN_OUT="$TMP_ROOT/out"
export MUJOCO_GL=""
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1

STAGE=0
run_stage() {
  STAGE=$((STAGE + 1))
  printf '\n========== VERIFY %02d: %s ==========\n' "$STAGE" "$1"
  shift
  timeout --foreground "$STAGE_TIMEOUT" "$@"
}

run_stage "MuJoCo-Warp dependency contract" "$WARP_PY" -c \
  "import torch,warp,mujoco,mujoco_warp,pydantic,xdist; print('stack:', torch.__version__, warp.__version__, mujoco.__version__)"
run_stage "patch hygiene" git diff --check

if ((REQUIRE_GPU)); then
  run_stage "target CUDA availability" "$WARP_PY" -c \
    "import torch; assert torch.cuda.is_available(), 'CUDA is required'; print(torch.cuda.get_device_name(0))"

  # Verilator is CPU-only. Build once, then let independent test files run in
  # isolated processes on the GPU host's CPU without racing the shared module.
  run_stage "complete component/co-simulation regression (parallel CPU workers)" \
    make test-parallel "TEST_WORKERS=$CPU_WORKERS" "WARP_PY=$WARP_PY" \
      "COMPONENT_PY=$WARP_PY"

  run_stage "complete deterministic CPU oracle and robot contracts" \
    env ROBOT_WARP_DEVICE=cpu "$WARP_PY" -m pytest sim/robot -q -m "not gpu"
  run_stage "training-scale MuJoCo-Warp CUDA pytest" \
    env ROBOT_REQUIRE_GPU=1 "$WARP_PY" -m pytest sim/robot -q -m gpu
else
  run_stage "fast component contracts and compiled FOC smoke" \
    python3 -m pytest -q \
      sim/tests/test_params_loader.py \
      sim/tests/test_derived_params.py \
      sim/tests/test_model_form_harness.py \
      sim/tests/test_foc_math.py::test_svpwm_bus_utilization
  run_stage "deterministic CPU oracle and robot contracts (CUDA tier excluded)" \
    env ROBOT_WARP_DEVICE=cpu "$WARP_PY" -m pytest sim/robot -q -m "not gpu"
fi

run_stage "body trainability proof" "$WARP_PY" sim/robot/validate_body.py
run_stage "generated-variant proof" "$WARP_PY" sim/robot/prove_robot.py
run_stage "contact/damage ordering" "$WARP_PY" sim/robot/test_contact.py

if ((REQUIRE_GPU)); then
  run_stage "walker CUDA execution" "$WARP_PY" sim/robot/walker_warp_env.py \
    --device cuda --nworld 256 --warmup 10 --steps 100
  run_stage "mesh CUDA execution" "$WARP_PY" sim/robot/mesh_warp_env.py \
    --device cuda --nworld 256 --warmup 10 --steps 100
  run_stage "combat CUDA execution" "$WARP_PY" sim/robot/combat_warp_env.py \
    --device cuda --nworld 256 --warmup 10 --steps 100
  run_stage "ladder locomotion CUDA execution" "$WARP_PY" sim/robot/ladder_warp_env.py \
    --rung 23 --device cuda --nworld 256 --warmup 10 --steps 100
  run_stage "commanded-leg combat CUDA execution" "$WARP_PY" sim/robot/ladder_warp_env.py \
    --rung 26 --device cuda --nworld 256 --warmup 10 --steps 100
  run_stage "target-GPU same-seed training repeatability canary" \
    "$WARP_PY" sim/robot/gpu_determinism_canary.py
  printf '\nVERIFICATION RESULT: PASS (complete GPU-host verification)\n'
else
  printf '\nPRECHECK RESULT: PASS (fast local tier only; not full verification)\n'
  printf 'Before a long simulation or RL run, execute this entry point with --require-gpu on a CUDA host.\n'
fi
