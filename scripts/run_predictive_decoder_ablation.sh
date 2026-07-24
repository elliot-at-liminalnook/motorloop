#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Fair recurrent-versus-Transformer decoder ablation, as called for by
# notes/predictive-transformer-proof-2026-07-15.md: the decoder trains under
# its own constant-rate optimizer (never the adaptive PPO schedule), freezes
# automatically when held-out calibration degrades, and is judged on held-out
# calibration across diverse tasks AND on morphologies compiled only into the
# evaluation environment. Run only after the GPU preflight has passed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${WARP_PY:-$ROOT/.venv-warp/bin/python}"
OUT="${1:-$ROOT/out/predictive_decoder_ablation}"
STEPS="${ABLATION_STEPS:-98304}"
SEED="${ABLATION_SEED:-20260723}"
PRED_LR="${ABLATION_PRED_LR:-3e-4}"
mkdir -p "$OUT"

# Training designs are the co-design corners; the evaluation bank holds
# combinations no training environment ever compiles, so eval-side predictor
# calibration measures generalization to unseen morphology tokens.
cat > "$OUT/train_designs.json" <<'JSON'
[[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]
JSON
cat > "$OUT/holdout_designs.json" <<'JSON'
[[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.5], [0.25, 0.75, 0.5]]
JSON

# rung 2: stand-and-settle baseline matched to the 2026-07-15 proof;
# rung 7: outcome-only forward walking (diverse locomotion commands);
# rung 24: first commanded-leg interaction rung;
# rung 30: design ensemble, judged on the held-out morphology bank.
run() {
  local decoder="$1" rung="$2"; shift 2
  local tag="$OUT/${decoder}_rung${rung}"
  echo "=== predictive decoder ablation: decoder=$decoder rung=$rung ==="
  "$PYTHON" -u "$ROOT/sim/robot/train_mesh_warp.py" \
    --geometry universal_control --rung "$rung" \
    --steps "$STEPS" --envs 128 --horizon 64 --episode-length 400 \
    --tag "$tag" --evals 3 --eval-envs 64 --eval-steps 200 \
    --diagnostic-eval-seeds 1 --checkpoint-replay-steps 32 \
    --epochs 2 --minibatches 8 --target-kl 0.02 --kl-stop-multiplier 1.5 \
    --width 512 --blocks 3 --architecture predictive_token_gru \
    --prediction-decoder "$decoder" --prediction-horizon 32 \
    --prediction-anchors 4 --prediction-loss-weight 0.25 \
    --prediction-lr "$PRED_LR" \
    --guidance-horizon 16 --guidance-steps 2 --guidance-interval 4 \
    --device cuda --preflight off --seed "$SEED" \
    "$@" > "$tag.log" 2>&1
}

for decoder in recurrent transformer; do
  run "$decoder" 2
  run "$decoder" 7
  run "$decoder" 24
  run "$decoder" 30 \
    --design-bank-json "$OUT/train_designs.json" \
    --eval-design-bank-json "$OUT/holdout_designs.json"
done

echo "Judgment: compare eval_predictor_calibration (unseen-design rung 30 runs"
echo "especially) and trajectory_prediction_frozen streaks in the *.stats.json"
echo "files; lower held-out calibration on unseen morphologies wins."
