#!/bin/bash
# SPDX-License-Identifier: MIT
# Run the validated GPU sequence:
#   1. command-conditioned walking
#   2. deploy + waypoint validation + visible walking render
#   3. self-play fighting seeded from the current curriculum/self-play best

set -euo pipefail

cd "${PROJECT_ROOT:-/root/proj}"
PY="${PY:-venv/bin/python}"
OUT="${CODESIGN_OUT:-/root/proj/out}"
CMD_TAG="${CMD_TAG:-walk7}"
CMD_STEPS="${CMD_STEPS:-8000000}"
CMD_ENVS="${CMD_ENVS:-4096}"
CMD_BATCH="${CMD_BATCH:-512}"
CMD_EVALS="${CMD_EVALS:-8}"
CMD_HOLD="${CMD_HOLD:-80}"
CMD_VMAX="${CMD_VMAX:-0.35}"
CMD_TRACK_SIGMA="${CMD_TRACK_SIGMA:-0.05}"
CMD_HOLD_STEPS="${CMD_HOLD_STEPS:-80}"
NAV_STEPS="${NAV_STEPS:-160}"
NAV_RADIUS="${NAV_RADIUS:-0.07}"
SP_ROUNDS="${SP_ROUNDS:-8}"
SP_ROUND_STEPS="${SP_ROUND_STEPS:-6000000}"
SP_ENVS="${SP_ENVS:-2048}"
SP_BATCH="${SP_BATCH:-1024}"
SP_TOL="${SP_TOL:-0.0}"
SP_ACCEPT_METRIC="${SP_ACCEPT_METRIC:-min_margin}"
SP_BENCH_SEEDS="${SP_BENCH_SEEDS:-20240601,20240602,20240603}"
SP_MIN_DEALT="${SP_MIN_DEALT:-0.02}"
SP_MAX_EARLY="${SP_MAX_EARLY:-0.8}"
SP_MARGIN_TOL="${SP_MARGIN_TOL:-0.0}"
SP_EARLY_HIT_PENALTY="${SP_EARLY_HIT_PENALTY:-30}"
SP_MIN_HIT_STEP="${SP_MIN_HIT_STEP:-20}"
SP_TAKEN_WEIGHT="${SP_TAKEN_WEIGHT:-24}"
SP_MAX_STALE_ROUNDS="${SP_MAX_STALE_ROUNDS:-0}"
SP_TRAIN_SEP_LO="${SP_TRAIN_SEP_LO:-0.25}"
SP_TRAIN_SEP_HI="${SP_TRAIN_SEP_HI:-0.45}"
SP_TRAIN_AZ="${SP_TRAIN_AZ:-0.5}"
SP_BENCH_SEP_LO="${SP_BENCH_SEP_LO:-0.25}"
SP_BENCH_SEP_HI="${SP_BENCH_SEP_HI:-0.45}"
SP_BENCH_AZ="${SP_BENCH_AZ:-0.5}"
SP_ROBUST_SLICE_1="${SP_ROBUST_SLICE_1:-clean100:0.18:0.45:0.8:100}"
SP_ROBUST_SLICE_2="${SP_ROBUST_SLICE_2:-bridge100:0.25:0.45:0.5:100}"
SP_ROBUST_MARGIN_TOL="${SP_ROBUST_MARGIN_TOL:-0.0}"
SP_ROBUST_JUDGE_TOL="${SP_ROBUST_JUDGE_TOL:-0.0}"
MILESTONE_GAP="${MILESTONE_GAP:-2000000}"
RENDER_GL="${RENDER_GL:-egl}"

# Do not leak a render-only GL backend into training/eval imports. MuJoCo can try
# to initialize EGL at import time even when no rendering is requested.
unset MUJOCO_GL
export CMD_VMAX CMD_TRACK_SIGMA CMD_HOLD_STEPS

mkdir -p "$OUT/figures" "$OUT/videos"

echo "===== validated pipeline start $(date -Is) ====="
echo "OUT=$OUT CMD_TAG=$CMD_TAG CMD_STEPS=$CMD_STEPS SP_ROUNDS=$SP_ROUNDS SP_ROUND_STEPS=$SP_ROUND_STEPS"

echo "===== train commanded walker ====="
CODESIGN_OUT="$OUT" "$PY" -u sim/robot/train_commanded.py \
  --tag "$CMD_TAG" --steps "$CMD_STEPS" --envs "$CMD_ENVS" --evals "$CMD_EVALS"

echo "===== validate commanded deployment ====="
CODESIGN_OUT="$OUT" CMD_CONTROL_MODE="${CMD_CONTROL_MODE:-cpg_pd}" "$PY" -u sim/robot/check_cpg_teacher_equivalence.py
for mode in forward backward left right square random; do
  CODESIGN_OUT="$OUT" "$PY" -u sim/robot/eval_commanded.py \
    --tag "${CMD_TAG}_${mode}" --ckpt "$OUT/${CMD_TAG}.pkl" \
    --hold "$CMD_HOLD" --mode "$mode" --speed "$CMD_VMAX"
done
CODESIGN_OUT="$OUT" "$PY" -u sim/robot/eval_checkpoint_navigation.py \
  --tag "$CMD_TAG" --ckpt "$OUT/${CMD_TAG}.pkl" \
  --radius "$NAV_RADIUS" --steps-per-waypoint "$NAV_STEPS"
CODESIGN_OUT="$OUT" "$PY" -u sim/robot/make_command_figure.py --tag "$CMD_TAG" --out "$OUT/figures"

echo "===== render commanded rollout ====="
if ! CODESIGN_OUT="$OUT" MUJOCO_GL="$RENDER_GL" "$PY" -u sim/robot/render_commanded_video.py \
    --tag "$CMD_TAG" --mode forward --hold 240 --out "$OUT/videos/${CMD_TAG}_forward.mp4"; then
  echo "WARN: commanded render failed; continuing with scalar/plot validation"
fi

echo "===== command gate ====="
CODESIGN_OUT="$OUT" "$PY" -u sim/robot/validate_commanded.py \
  --tag "$CMD_TAG" --out "$OUT" \
  --require-modes forward,backward,left,right,square,random \
  --min-nav-frac 0.75 \
  --render "$OUT/videos/${CMD_TAG}_forward.mp4" \
  --check

echo "===== start milestone renderer ====="
if ! pgrep -f "render_daemon.sh" >/dev/null 2>&1; then
  CODESIGN_OUT="$OUT" MUJOCO_GL="$RENDER_GL" bash sim/robot/render_daemon.sh spr \
    >> "$OUT/render_daemon.log" 2>&1 &
  echo "render_daemon pid=$!"
fi

echo "===== self-play fighting ====="
SP_SEED_CKPT="${SP_SEED_CKPT:-}"
if [ -z "$SP_SEED_CKPT" ]; then
  if [ -f "$OUT/clean_seed.pkl" ]; then
    SP_SEED_CKPT="$OUT/clean_seed.pkl"
  else
    SP_SEED_CKPT="$OUT/curriculum_best.pkl"
  fi
fi
if [ ! -f "$SP_SEED_CKPT" ]; then
  echo "missing self-play seed checkpoint: $SP_SEED_CKPT" >&2
  exit 2
fi
CODESIGN_OUT="$OUT" MILESTONE_GAP="$MILESTONE_GAP" "$PY" -u sim/robot/selfplay_drive.py \
  --seed-ckpt "$SP_SEED_CKPT" --rounds "$SP_ROUNDS" --round-steps "$SP_ROUND_STEPS" \
  --envs "$SP_ENVS" --batch "$SP_BATCH" --tol "$SP_TOL" --lean-contacts \
  --accept-metric "$SP_ACCEPT_METRIC" --bench-seeds "$SP_BENCH_SEEDS" \
  --min-accepted-dealt "$SP_MIN_DEALT" --max-accepted-early-dmg "$SP_MAX_EARLY" \
  --margin-tol "$SP_MARGIN_TOL" --early-hit-penalty "$SP_EARLY_HIT_PENALTY" \
  --min-hit-step "$SP_MIN_HIT_STEP" --taken-weight "$SP_TAKEN_WEIGHT" \
  --max-stale-rounds "$SP_MAX_STALE_ROUNDS" \
  --train-sep-lo "$SP_TRAIN_SEP_LO" --train-sep-hi "$SP_TRAIN_SEP_HI" \
  --train-azimuth "$SP_TRAIN_AZ" --bench-sep-lo "$SP_BENCH_SEP_LO" \
  --bench-sep-hi "$SP_BENCH_SEP_HI" --bench-az "$SP_BENCH_AZ" \
  --robust-slice "$SP_ROBUST_SLICE_1" --robust-slice "$SP_ROBUST_SLICE_2" \
  --robust-margin-tol "$SP_ROBUST_MARGIN_TOL" --robust-judge-tol "$SP_ROBUST_JUDGE_TOL"

echo "===== validated pipeline complete $(date -Is) ====="
