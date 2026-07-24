<!-- SPDX-License-Identifier: MIT -->
# Predictive universal controller

> **Document status:** Active · **Audience:** Robot-learning contributors · **Last reviewed:** 2026-07-23 · **Canonical for:** Morphology-token conditioning, self-supervised trajectory prediction, and prediction-gradient control

The `predictive_token_gru` actor is the experimental successor to
`task_film_gru`. It keeps the same fast recurrent action loop, then adds a
separate model of what the robot is likely to do over the next 0.5–1 second.
It is a new policy family and must start a new ladder run; it must not overwrite
or resume the accepted GRU ladder.

## Architecture

1. **Reactive GRU:** consumes normalized physical observations and the global
   task command and emits the immediate 14-channel action distribution.
2. **Morphology-token encoder:** reads the actual compiled MuJoCo body, joint,
   and actuator fields. Tokens include mass/inertia, link offsets, joint axes and
   limits, damping/armature, gear, control/force ranges, no-load speed, and PD
   gain. Co-design coordinates can remain in legacy observations for checkpoint
   compatibility, but the new controller conditions on physical tokens.
3. **Selectable trajectory decoder:** the recurrent baseline rolls forward a
   proposed action sequence with a GRU. The experimental temporal Transformer
   uses causal attention to predict every future frame in one parallel GPU pass;
   both start from policy memory plus the morphology embedding and predict body
   position/orientation/velocity, four foot positions, contact, actuator effort,
   and fall risk.
4. **Masked future-interaction command:** each environment describes only the
   physical future that matters. Locomotion supplies desired root displacement
   and velocity while leaving foot timing free. Commanded-leg combat supplies a
   broad opponent region, selected-leg contact at any time in the horizon,
   outward strike direction, wrong-leg avoidance, and an aggregate minimum of
   two supporting non-striking legs. It supplies no joint pose or strike phase.
5. **Prediction-gradient planner:** differentiates task-space tracking error,
   predicted fall, effort, insufficient support, and action discontinuity with
   respect to a short candidate action sequence, then executes a
   confidence-weighted first action.

The pose target is expressed relative to the current body position and yaw. This
is a locally stabilized world frame: translating or turning the whole scene does
not change the meaning of “move the foot forward,” while drift and future body
motion remain visible. A body-frame-only target would hide precisely the shaking
and world-tracking errors the predictor is meant to catch.

## Learning contract

The predictor needs no hand-labeled trajectory dataset. Every physics rollout
already contains its supervision:

- snapshot the root, feet, contacts, and actual actuator force before and after
  actions;
- choose several anchors within the PPO rollout;
- transform future snapshots into the anchor's stabilized local world frame;
- exclude every target whose path crosses an episode reset; and
- minimize balanced position, rotation, velocity, foot, contact, effort, and
  fall losses alongside PPO.

Planning authority is self-tuning. The final valid anchor in every rollout is
excluded from decoder training and used as a held-out forecast. Continuous
errors are divided by physical scales and contact/fall heads use Brier error, so
authority does not depend on arbitrary auxiliary-loss weights. It is exactly
zero for the first ten held-out observations and then follows
`exp(-4 * calibration_error_ema)`. A poorly calibrated predictor can therefore
only suggest a tiny correction while an accurate predictor earns more authority.

The decoder trains under its own constant-rate Adam optimizer
(`--prediction-lr`, defaulting to the trainer's `--lr` ceiling) and its own
gradient-clipping subspace. It never follows the adaptive PPO learning-rate
schedule, so a larger decoder — the temporal Transformer is ~2.5× the recurrent
one — is not forced through the policy's KL-driven step sizes, and its gradient
scale cannot shrink the policy gradient through a shared clip. The same
held-out calibration signal also gates continued decoder training: once the
calibration EMA rises `--prediction-freeze-tolerance` (default 15%) above its
own best for `--prediction-freeze-patience` (default 3) consecutive
observations, decoder updates freeze. Calibration keeps being observed while
frozen, and training resumes automatically when the forecast recovers to half
the tolerance. A predictor can therefore not keep optimizing its training loss
while its out-of-sample forecast degrades, which is exactly the failure the
2026-07-15 Transformer run exhibited.
The planner also stays near the reactive action sequence through an explicit
prior cost. This is a safety mechanism, not evidence that simulated prediction
is correct on hardware.

## Launch contract

A full run uses a separate output directory:

```bash
.venv-warp/bin/python -u sim/robot/training_ladder.py run \
  --out out/training_ladder_predictive \
  --from 1 --to 31 --device cuda \
  --python .venv-warp/bin/python \
  --architecture predictive_token_gru \
  --command-observations --power-model shared_bus \
  --hidden 512,512,512 --envs 128 --horizon 64 \
  --prediction-horizon 32 --prediction-anchors 4 \
  --prediction-decoder recurrent \
  --prediction-loss-weight 0.25 \
  --guidance-horizon 16 --guidance-steps 2 --guidance-interval 4
```

`--command-observations` trains this family on the commands-only v2 contract
([`universal-command-contract.md`](universal-command-contract.md)): the rung
ID is invisible to the policy, all task semantics arrive as explicit command
channels, and zero-shot certification evaluates the parent checkpoint directly
on the next rung's command distribution. The new family must not inherit the
rung identity it exists to retire.

At 50 Hz, 32 prediction frames cover 0.64 seconds and 16 guidance frames cover
0.32 seconds. Keep the accepted `task_film_gru` ladder running or archived as a
baseline until the new family independently passes the same promotion and
retention gates. Use `--prediction-decoder transformer` only for a separate
experimental run until it wins the held-out tests below.

The fair decoder rematch the first proof called for is scripted in
`scripts/run_predictive_decoder_ablation.sh`: both decoders run with the
separate constant-rate optimizer and the calibration freeze across
stand-and-settle, outcome-only forward walking, the first commanded-leg
interaction rung, and the rung-30 design ensemble. The rung-30 runs compile a
held-out design bank only into the evaluation environment
(`--eval-design-bank-json`), so the `eval_predictor_calibration` record in each
run's stats measures forecast quality on morphology tokens the decoder never
trained on. The winner is the decoder with lower held-out calibration on
unseen morphologies, not the one with lower final training loss.

## Diagnostics and acceptance

Training records the aggregate decoder loss, each physical head's loss, the
dimensionless held-out calibration error by head, guidance confidence, gradient
magnitude, action RMS/maximum change, task and feasibility costs before/after
planning, and the realized metrics on guided frames. Every deterministic
evaluation also replays the identical seed with guidance disabled and records
guided-minus-unguided outcome deltas. A useful predictor should:

- improve held-out physical prediction without degrading PPO KL or throughput;
- retain calibration across locomotion, combat, and unseen design tokens;
- improve or preserve the ordinary outcome gates with guidance enabled; and
- lose authority automatically under a morphology or task distribution shift.

The CPU integration proof exercises a full update, evaluation, save/reload, and
deterministic replay. A first matched L40S proof found that the temporal
Transformer materially accelerates predictor optimization but has mixed
held-out accuracy over twelve updates, so `recurrent` remains the default. See
[`predictive-transformer-proof-2026-07-15.md`](predictive-transformer-proof-2026-07-15.md).
