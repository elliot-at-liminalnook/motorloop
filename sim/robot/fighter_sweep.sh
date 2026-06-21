#!/bin/bash
# SPDX-License-Identifier: MIT
# Param sweep for the fighter on the OPTIMIZED fight scene (lean reduced-collision +
# frame_skip + saturated envs + many iterations). Each config is wall-clock-bounded so the
# whole sweep is cost-capped; the host monitors $ and terminates at the budget. Goal: dealt>0.
set -u
cd "$(dirname "$0")"
OUT="${CODESIGN_OUT:-/root/proj/out}"; mkdir -p "$OUT"
export MUJOCO_GL="" XLA_PYTHON_CLIENT_PREALLOCATE=false
PER="${PER:-1500}"                                  # seconds per config (host caps total $)
WARM="$OUT/universal_ckpt.pkl"                       # locomotor warm-start (first run of a config)
COMMON="--envs 8192 --batch 256 --minibatches 32 --updates 8 --unroll 5 --frame-skip 8 --lean-contacts --evals 24"

# RESUME-FROM-LATEST: a config resumes from its OWN checkpoint if present (restart-safe),
# else warm-starts from the locomotor. Every eval re-saves {tag}_ckpt.pkl (incremental).
run(){ tag="$1"; shift
  res="$WARM"; [ -f "$OUT/${tag}_ckpt.pkl" ] && res="$OUT/${tag}_ckpt.pkl" && echo "[$tag] RESUMING from its last checkpoint"
  echo "=== SWEEP $tag : $* (resume=$(basename "$res")) ==="
  timeout "$PER" python3 -u train_adversarial.py $COMMON --tag "$tag" ${res:+--resume "$res"} "$@" > "$OUT/sweep_$tag.log" 2>&1
  echo "$tag rc=$? :: $(grep -aE 'FIGHTER' "$OUT/sweep_$tag.log" 2>/dev/null | tail -1)"; }

run s1 --steps 30000000 --shaping 1.0 --sep 0.9 --lr 3e-4 --entropy 1e-2
run s2 --steps 30000000 --shaping 1.0 --sep 0.9 --lr 5e-4 --entropy 3e-2   # more exploration
run s3 --steps 30000000 --shaping 2.0 --sep 0.6 --lr 3e-4 --entropy 1e-2   # aggressive curriculum (very close)
run s4 --steps 30000000 --shaping 1.5 --sep 0.7 --lr 3e-4 --entropy 2e-2 --unroll 10
echo SWEEP_DONE
