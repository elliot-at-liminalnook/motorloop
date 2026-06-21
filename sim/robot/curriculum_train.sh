#!/bin/bash
# SPDX-License-Identifier: MIT
# LONG contact-forcing curriculum: widen the start-separation range in phases, warm-starting
# each phase from the previous. The close low end (0.4) is kept every phase so the dealt
# reward signal never disappears; the high end grows so the policy learns to CLOSE then strike.
# Resume-safe (each phase resumes its own {tag}_ckpt.pkl if present) + per-eval checkpoints
# that the host pull-loop copies to local => a multi-hour run is never lost.
set -u
cd "$(dirname "$0")"
OUT="${CODESIGN_OUT:-/root/proj/out}"; mkdir -p "$OUT"
export MUJOCO_GL="" XLA_PYTHON_CLIENT_PREALLOCATE=false
PER="${PER:-1800}"
COMMON="--envs 8192 --batch 256 --minibatches 32 --updates 8 --unroll 5 --frame-skip 8 --lean-contacts --shaping 1.0 --evals 24 --steps 40000000"

# phase: tag  prev-tag(warm-start source)  sep_lo  sep_hi
phase(){ tag="$1"; prev="$2"; lo="$3"; hi="$4"
  src="$OUT/${prev}_ckpt.pkl"; [ -f "$OUT/${tag}_ckpt.pkl" ] && src="$OUT/${tag}_ckpt.pkl" && echo "[$tag] RESUMING own ckpt"
  echo "=== CURRICULUM $tag : sep [$lo,$hi] warm=$(basename "$src") ==="
  timeout "$PER" python3 -u train_adversarial.py $COMMON --tag "$tag" --sep-lo "$lo" --sep-hi "$hi" \
    ${src:+--resume "$src"} > "$OUT/curr_$tag.log" 2>&1
  echo "$tag rc=$? :: $(grep -aE 'FIGHTER' "$OUT/curr_$tag.log" 2>/dev/null | tail -1)"; }

phase c1 cval 0.4 0.7
phase c2 c1   0.4 1.0
phase c3 c2   0.4 1.4
phase c4 c3   0.4 1.8
echo CURR_DONE
