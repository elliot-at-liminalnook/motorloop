#!/bin/bash
# SPDX-License-Identifier: MIT
# One-command scaffold-prior combat curriculum:
#   scaffold checkpoint -> combat-compatible seed -> baseline benchmark/render
#   -> contact-forcing curriculum -> trained benchmark/render -> summary.
#
# Typical pod use:
#   PROJECT_ROOT=/root/proj PY=/root/proj/venv/bin/python CODESIGN_OUT=/root/proj/out \
#     bash scripts/run_scaffold_combat_curriculum.sh
#
# Fast plumbing check:
#   COMBAT_TINY=1 COMBAT_RENDER=0 bash scripts/run_scaffold_combat_curriculum.sh

set -euo pipefail

if [ -n "${PROJECT_ROOT:-}" ]; then
  cd "$PROJECT_ROOT"
else
  cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
fi

PY="${PY:-venv/bin/python}"
if [ ! -x "$PY" ]; then
  PY="${PY:-python3}"
fi

ROOT_OUT="${CODESIGN_OUT:-$PWD/out}"
TAG="${COMBAT_TAG:-scaffold_combat}"
RUN_OUT="${COMBAT_RUN_OUT:-$ROOT_OUT/$TAG}"
TINY="${COMBAT_TINY:-0}"
LEAN="${COMBAT_LEAN_CONTACTS:-1}"
RENDER="${COMBAT_RENDER:-1}"
RENDER_GL="${RENDER_GL:-egl}"
RESUME="${COMBAT_RESUME:-0}"

if [ "$TINY" = "1" ]; then
  STEPS_PER_PHASE="${COMBAT_STEPS_PER_PHASE:-8000}"
  PHASES="${COMBAT_PHASES:-2}"
  BENCH_EPIS="${COMBAT_BENCH_EPIS:-4}"
  BENCH_STEPS="${COMBAT_BENCH_STEPS:-40}"
  RENDER_STEPS="${COMBAT_RENDER_STEPS:-80}"
else
  STEPS_PER_PHASE="${COMBAT_STEPS_PER_PHASE:-4000000}"
  PHASES="${COMBAT_PHASES:-5}"
  BENCH_EPIS="${COMBAT_BENCH_EPIS:-16}"
  BENCH_STEPS="${COMBAT_BENCH_STEPS:-200}"
  RENDER_STEPS="${COMBAT_RENDER_STEPS:-220}"
fi

ENVS="${COMBAT_ENVS:-0}"
BATCH="${COMBAT_BATCH:-0}"
RETRIES="${COMBAT_RETRIES:-1}"
TOL="${COMBAT_TOL:-2.0}"
KEEP_METRIC="${COMBAT_KEEP_METRIC:-sparc}"
MIN_KEEP_DEALT="${COMBAT_MIN_KEEP_DEALT:-0.0005}"
MAX_KEEP_EARLY_DMG="${COMBAT_MAX_KEEP_EARLY_DMG:-1.0}"
FLEE_PENALTY="${COMBAT_FLEE_PENALTY:-0}"
CLOSE_BONUS="${COMBAT_CLOSE_BONUS:-0}"
CLOSE_RADIUS="${COMBAT_CLOSE_RADIUS:-0.45}"
DAMAGE_BONUS="${COMBAT_DAMAGE_BONUS:-0}"
FRAME_SKIP="${COMBAT_FRAME_SKIP:-5}"
BENCH_SEP_LO="${COMBAT_BENCH_SEP_LO:-0.25}"
BENCH_SEP_HI="${COMBAT_BENCH_SEP_HI:-0.7}"
BENCH_AZ="${COMBAT_BENCH_AZ:-3.14159}"
RENDER_SEP="${COMBAT_RENDER_SEP:-0.5}"

mkdir -p "$RUN_OUT/videos" "$RUN_OUT/figures"
export CODESIGN_OUT="$RUN_OUT"
unset MUJOCO_GL

choose_warm() {
  if [ -n "${WARM_CKPT:-}" ] && [ -f "$WARM_CKPT" ]; then
    echo "$WARM_CKPT"
    return 0
  fi
  for c in \
    "$ROOT_OUT/walk22_forward_resid.pkl" \
    "$ROOT_OUT/universal_ckpt.pkl" \
    "$ROOT_OUT/curriculum_best.pkl" \
    "$ROOT_OUT/cval_ckpt.pkl" \
    "$RUN_OUT/curriculum_best.pkl"; do
    if [ -f "$c" ]; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

WARM="$(choose_warm || true)"
if [ -z "$WARM" ]; then
  echo "missing scaffold checkpoint. Set WARM_CKPT=/path/to/walker_or_fighter.pkl" >&2
  exit 2
fi

echo "===== scaffold combat curriculum $(date -Is) ====="
echo "root=$PWD"
echo "out=$RUN_OUT"
echo "warm=$WARM"
echo "steps_per_phase=$STEPS_PER_PHASE phases=$PHASES tiny=$TINY envs=$ENVS batch=$BATCH resume=$RESUME"

LEAN_FLAG=()
if [ "$LEAN" = "1" ]; then
  LEAN_FLAG=(--lean-contacts)
fi

BASE_CKPT="$RUN_OUT/${TAG}_scaffold_seed.pkl"
"$PY" -u sim/robot/prepare_fighter_seed.py \
  --src "$WARM" --out "$BASE_CKPT" --frame-skip "$FRAME_SKIP" \
  --bench-sep-lo "$BENCH_SEP_LO" --bench-sep-hi "$BENCH_SEP_HI" --bench-az "$BENCH_AZ" \
  "${LEAN_FLAG[@]}"

echo "===== baseline scaffold benchmark ====="
"$PY" -u sim/robot/eval_fighter_benchmark.py \
  --tag "${TAG}_scaffold" --ckpt "$BASE_CKPT" \
  --bench-sep-lo "$BENCH_SEP_LO" --bench-sep-hi "$BENCH_SEP_HI" --bench-az "$BENCH_AZ" \
  --bench-epis "$BENCH_EPIS" --bench-steps "$BENCH_STEPS" --frame-skip "$FRAME_SKIP" \
  --out-json "$RUN_OUT/${TAG}_scaffold_benchmark_eval.json" \
  "${LEAN_FLAG[@]}"

render_one() {
  local a="$1" b="$2" out="$3" label="$4"
  if [ "$RENDER" != "1" ]; then
    return 0
  fi
  if ! CODESIGN_OUT="$RUN_OUT" MUJOCO_GL="$RENDER_GL" "$PY" -u sim/robot/render_fight_video.py \
      --a "$a" --b "$b" --out "$out" --steps "$RENDER_STEPS" --sep "$RENDER_SEP" --label "$label"; then
    echo "WARN: GL render failed for $label; writing no-GL top-down trace"
    if ! CODESIGN_OUT="$RUN_OUT" env -u MUJOCO_GL "$PY" -u sim/robot/render_fight_trace.py \
        --a "$a" --b "$b" --out "$out" --steps "$RENDER_STEPS" --sep "$RENDER_SEP" --label "$label"; then
      echo "WARN: trace render failed for $label; continuing with metrics"
    fi
  fi
}

render_one "$BASE_CKPT" "$BASE_CKPT" "$RUN_OUT/videos/${TAG}_scaffold_self.mp4" "${TAG}_scaffold"

echo "===== contact-forcing combat curriculum ====="
CURR_ARGS=(--warm "$BASE_CKPT" --steps-per-phase "$STEPS_PER_PHASE" --tol "$TOL" --retries "$RETRIES")
CURR_ARGS+=(--bench-sep-lo "$BENCH_SEP_LO" --bench-sep-hi "$BENCH_SEP_HI" --bench-az "$BENCH_AZ")
CURR_ARGS+=(--bench-epis "$BENCH_EPIS" --bench-steps "$BENCH_STEPS")
CURR_ARGS+=(--keep-metric "$KEEP_METRIC" --min-keep-dealt "$MIN_KEEP_DEALT")
CURR_ARGS+=(--max-keep-early-dmg "$MAX_KEEP_EARLY_DMG")
CURR_ARGS+=(--flee-penalty "$FLEE_PENALTY" --close-bonus "$CLOSE_BONUS")
CURR_ARGS+=(--close-radius "$CLOSE_RADIUS" --damage-bonus "$DAMAGE_BONUS")
if [ "$RESUME" = "1" ]; then
  CURR_ARGS+=(--resume)
fi
if [ "$PHASES" != "0" ]; then
  CURR_ARGS+=(--phases "$PHASES")
fi
if [ "$ENVS" != "0" ]; then
  CURR_ARGS+=(--envs "$ENVS")
fi
if [ "$BATCH" != "0" ]; then
  CURR_ARGS+=(--batch "$BATCH")
fi
if [ "$LEAN" = "1" ]; then
  CURR_ARGS+=(--lean-contacts)
fi
if [ "$TINY" = "1" ]; then
  CURR_ARGS+=(--tiny)
fi
"$PY" -u sim/robot/curriculum_drive.py "${CURR_ARGS[@]}" 2>&1 | tee "$RUN_OUT/${TAG}_curriculum_drive.log"

TRAINED_CKPT="$RUN_OUT/curriculum_best.pkl"
if [ ! -f "$TRAINED_CKPT" ]; then
  echo "training did not produce $TRAINED_CKPT" >&2
  exit 3
fi

echo "===== trained fighter benchmark ====="
"$PY" -u sim/robot/eval_fighter_benchmark.py \
  --tag "${TAG}_trained" --ckpt "$TRAINED_CKPT" \
  --bench-sep-lo "$BENCH_SEP_LO" --bench-sep-hi "$BENCH_SEP_HI" --bench-az "$BENCH_AZ" \
  --bench-epis "$BENCH_EPIS" --bench-steps "$BENCH_STEPS" --frame-skip "$FRAME_SKIP" \
  --out-json "$RUN_OUT/${TAG}_trained_benchmark_eval.json" \
  "${LEAN_FLAG[@]}"

render_one "$TRAINED_CKPT" "$TRAINED_CKPT" "$RUN_OUT/videos/${TAG}_trained_self.mp4" "${TAG}_trained_self"
render_one "$TRAINED_CKPT" "$BASE_CKPT" "$RUN_OUT/videos/${TAG}_trained_vs_scaffold.mp4" "${TAG}_trained_vs_scaffold"

if ! "$PY" -u sim/robot/make_benchmark_figure.py --src "$RUN_OUT" --out "$RUN_OUT/figures"; then
  echo "WARN: benchmark figure failed; continuing with JSON/JSONL metrics"
fi

"$PY" - "$RUN_OUT" "$TAG" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
tag = sys.argv[2]
base = json.loads((out / f"{tag}_scaffold_benchmark_eval.json").read_text())
trained = json.loads((out / f"{tag}_trained_benchmark_eval.json").read_text())
state = json.loads((out / "curriculum_state.json").read_text()) if (out / "curriculum_state.json").exists() else {}
bench_rows = []
for p in sorted(out.glob("*_benchmark.jsonl")):
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                row = json.loads(line)
            except Exception:
                continue
            row["source"] = p.name
            bench_rows.append(row)
bench_rows.sort(key=lambda r: r.get("cum_step", r.get("step", 0)))

keys = ["sparc", "dealt", "taken", "closing", "fleeing", "dist", "win_rate", "survival_rate", "safe_rate"]
delta = {k: float(trained.get(k, 0.0)) - float(base.get(k, 0.0)) for k in keys}
bench_keys = ["bench_sparc", "bench_dealt", "bench_taken", "closing", "fleeing", "dist",
              "win_rate", "survival_rate", "safe_rate", "bench_margin", "selected_score"]
first = bench_rows[0] if bench_rows else {}
last = bench_rows[-1] if bench_rows else {}
best = max(bench_rows, key=lambda r: float(r.get("selected_score", r.get("bench_sparc", -1e30)))) if bench_rows else {}
safe_rows = [r for r in bench_rows if float(r.get("safe_rate", 0.0)) >= 1.0]
safe_first = safe_rows[0] if safe_rows else {}
safe_last = safe_rows[-1] if safe_rows else {}
trend_delta = {
    k: float(last.get(k, 0.0)) - float(first.get(k, 0.0))
    for k in bench_keys
    if k in first or k in last
}
safe_trend_delta = {
    k: float(safe_last.get(k, 0.0)) - float(safe_first.get(k, 0.0))
    for k in bench_keys
    if k in safe_first or k in safe_last
}
summary = {
    "tag": tag,
    "out": str(out),
    "scaffold_ckpt": str(out / f"{tag}_scaffold_seed.pkl"),
    "trained_ckpt": str(out / "curriculum_best.pkl"),
    "completed_phases": state.get("completed", []),
    "global_best_bench": state.get("global_best_bench"),
    "baseline": {k: base.get(k) for k in keys if k in base},
    "trained": {k: trained.get(k) for k in keys if k in trained},
    "delta": delta,
    "benchmark_first": {k: first.get(k) for k in bench_keys if k in first},
    "benchmark_last": {k: last.get(k) for k in bench_keys if k in last},
    "benchmark_best": {k: best.get(k) for k in bench_keys if k in best},
    "benchmark_trend_delta": trend_delta,
    "benchmark_safe_first": {k: safe_first.get(k) for k in bench_keys if k in safe_first},
    "benchmark_safe_last": {k: safe_last.get(k) for k in bench_keys if k in safe_last},
    "benchmark_safe_trend_delta": safe_trend_delta,
    "videos": sorted(str(p) for p in (out / "videos").glob("*.mp4")),
    "figure": str(out / "figures" / "benchmark_curve.png"),
}
(out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
print("===== summary =====")
print(json.dumps(summary, indent=2))
PY

echo "===== done: $RUN_OUT ====="
