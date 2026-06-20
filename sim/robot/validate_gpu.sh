#!/bin/bash
# SPDX-License-Identifier: MIT
# Sequential tiny GPU validation of the whole co-design pipeline (one GPU => no overlap;
# overlapping JAX procs throw cuSolver errors). Each stage is leak-tested at micro-scale
# BEFORE any long run (the E2E-first rule). Logs -> $OUT/validate_<stage>.log
set -u
cd "$(dirname "$0")"
OUT="${CODESIGN_OUT:-/root/proj/out}"; mkdir -p "$OUT"
PY="${PY:-python3}"
export MUJOCO_GL=""        # headless pod: osmesa/egl GL libs are broken; no stage renders,
                          # and an empty backend imports mujoco cleanly (mj_step needs no GL)
run () { # name  timeout  cmd...
  local name="$1" to="$2"; shift 2
  echo "========== STAGE $name =========="
  timeout "$to" "$PY" -u "$@" > "$OUT/validate_$name.log" 2>&1
  local rc=$?
  echo "[$name] rc=$rc"; grep -aE "PROVEN|METRIC|VERDICT|rho=|Error|Traceback|SKIP|adaptation|Pareto" "$OUT/validate_$name.log" | grep -avE "WARNING|warn" | tail -6
  echo
}
run parity     900 test_parity.py
run walker     900 codesign_gpu.py --tiny
run rederive   900 rederive_r7.py --tiny
run buildpack  600 codesign_validate.py --build-pack --tiny
run fighter    900 train_adversarial.py --tiny --resume "$OUT/universal_ckpt.pkl"
run score      600 codesign_validate.py --score-fighter --tiny
run selfplay   900 selfplay_mjx.py --tiny
run diff       600 codesign_diff.py
run phase2pol  900 optimize_design.py --fitness policy --pop 4 --gens 2 --policy-budget 12000
echo "ALL GPU VALIDATION STAGES ATTEMPTED"
