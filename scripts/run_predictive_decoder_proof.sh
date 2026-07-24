#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Matched first-result ablation for the recurrent and temporal-Transformer
# future-physics decoders. Run only after the GPU preflight has passed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${WARP_PY:-$ROOT/.venv-warp/bin/python}"
OUT="${1:-$ROOT/out/predictive_decoder_proof}"
STEPS="${PROOF_STEPS:-98304}"
mkdir -p "$OUT"

for decoder in recurrent transformer; do
  tag="$OUT/$decoder"
  echo "=== predictive decoder proof: $decoder ==="
  "$PYTHON" -u "$ROOT/sim/robot/train_mesh_warp.py" \
    --geometry universal_control --rung 2 \
    --steps "$STEPS" --envs 128 --horizon 64 --episode-length 400 \
    --tag "$tag" --evals 3 --eval-envs 64 --eval-steps 200 \
    --diagnostic-eval-seeds 1 --checkpoint-replay-steps 32 \
    --epochs 2 --minibatches 8 --target-kl 0.02 --kl-stop-multiplier 1.5 \
    --hidden 512,256,128 --architecture predictive_token_gru \
    --prediction-decoder "$decoder" --prediction-horizon 32 \
    --prediction-anchors 4 --prediction-loss-weight 0.25 \
    --guidance-horizon 16 --guidance-steps 2 --guidance-interval 4 \
    --device cuda --preflight off --seed 20260715 \
    > "$tag.log" 2>&1
done
