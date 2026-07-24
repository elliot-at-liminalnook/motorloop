<!-- SPDX-License-Identifier: MIT -->
# Sequential robot-training ladder

> **Document status:** Active · **Audience:** Robot-learning contributors · **Last reviewed:** 2026-07-14 · **Canonical for:** Current 31-rung training state, resume procedure, objective roles, adaptive enforcement, and shutdown handoff

This is the operational record for the active MuJoCo-Warp/Torch curriculum. The
ladder contains thirty-one ordered tasks, trains one universal controller,
promotes only checkpoints that pass rotating held-out gates and prior-skill
replay, and stops at the first unproved rung.

## Current handoff

Rungs 1–6 are accepted. Rung 6, **Step in place**, was promoted from the
`6_504_448`-step immutable candidate after passing five fresh full-episode seeds
and retention replays for rungs 2–5. Rung 7, **Walk forward**, is now training
directly from that accepted policy. Its first attempt uses forward progress and
physical safety with no reference gait; a legacy teacher is an optional fallback
after failure, not a prerequisite or a permanent walking objective.

The restart deliberately does **not** continue the obsolete 20M objective. That
run rewarded and gated agreement with a hand-authored gait clock. It produced
real foot motion, but promotion was decided by choreography rather than by the
physical result. The current rung-6 contract instead asks every foot to leave the
ground, keeps the base in place and upright, and enforces only physical safety.

| Accepted rung-6 promotion evidence | Worst held-out value | Gate | Verdict |
| --- | ---: | ---: | --- |
| Worst-foot airborne fraction | 0.0899 | ≥ 0.05 | pass |
| Base speed | 0.0791 m/s | ≤ 0.20 m/s | pass |
| Uprightness | 0.9996 | ≥ 0.85 | pass |
| Physical-constraint rate | 0 | ≤ 0.001 | pass |
| Fall rate | 0 | ≤ 0.0001 | pass |
| Prior-skill retention | rungs 2–5 | every stored boundary | pass |

The accepted checkpoint is
`out/training_ladder/rung_06_step_in_place.candidates/step_000006504448.pt`; a
frozen pre-migration recovery copy is preserved at
`out/training_ladder/baselines/rung_06_step_in_place_6m.pt`. The `out/` tree is
intentionally Git-ignored; pushing the repository does not upload learned
weights.

- Accepted checkpoint step: `6_504_448`
- Accepted checkpoint SHA-256:
  `1eaa4cb7b2a12ee829bfdfbacd46cb2a357c28128a56feecfbafbce7d1b3ec52`
- Actor migration target: universal task-conditioned FiLM network — three
  constant-width 512 residual blocks (the historical `--hidden 512,256,128`
  spelling built exactly this; checkpoints now record the resolved shapes) —
  256 observations, 14 actions, and 31 global task channels
- Accepted prefix in `ladder_state.json`: `[1, 2, 3, 4, 5, 6]`
- Reward migration preserves the actor, observation normalization, physical
  runtime, safety dual, RNG, and monotonic schedules; it resets the critic,
  optimizer moments, and old reward-semantic competence state.
- Run status is read-only through `bash scripts/training_ladder_status.sh`.

The earlier 20M artifact remains useful historical evidence for candidate
archiving and timing-credit failures, but it is not the parent of this run and
must not replace the accepted 6.5M checkpoint.

## Resume on a new RunPod

Use the exact source commit containing this document. Copy the local artifact
tree before launching; otherwise `--resume` has no checkpoint to load.

```bash
# Local workstation: replace host and port with the new pod's SSH endpoint.
rsync -az --partial --append-verify \
  -e 'ssh -i ~/.ssh/runpod_ed25519 -p PORT' \
  out/training_ladder/ root@HOST:/root/proj/out/training_ladder/
```

On the pod:

```bash
cd /root/proj
bash sim/robot/setup_warp_pod.sh
source /root/proj/out/warp_env.sh
bash scripts/run_pre_gpu_tests.sh --require-gpu

nohup .venv-warp/bin/python -u sim/robot/training_ladder.py run \
  --out /root/proj/out/training_ladder \
  --from 6 --to 31 --resume --allow-reward-migration \
  --device cuda --python /root/proj/.venv-warp/bin/python \
  --envs 128 --horizon 64 --episode-length 800 \
  --steps-per-rung 2000000 --scratch-steps 8000000 \
  --attempts 3 --evals 4 --eval-envs 64 --eval-steps 400 \
  --diagnostic-eval-seeds 3 --checkpoint-replay-steps 32 \
  --epochs 4 --minibatches 8 --target-kl 0.02 \
  --kl-stop-multiplier 1.5 --hidden 512,512,512 \
  --architecture task_film \
  >> out/training_ladder/ladder.log 2>&1 &
```

Monitor without mutating the run:

```bash
bash scripts/training_ladder_status.sh
```

Rung 7's old policy candidates and gait prior were deliberately archived because
they depended on the invalidated old rung-6 policy. After rung 6 is accepted, the
ladder must search a new gait prior. New search artifacts carry SHA-256 provenance
for both the accepted rung-6 checkpoint and the legacy walking teacher; mismatched
sources are rejected on resume.

## Promotion and forgetting contract

Each rung follows this sequence:

1. warm-start from the latest accepted checkpoint in the same policy family;
2. make the new task channel exactly inherit its predecessor and attempt a
   zero-shot certification before spending PPO experience;
3. accept that inherited candidate only if every gate passes five fresh,
   rotating, full-episode seeds with at least 10% dimensionless margin and every
   earlier skill passes full retention replay;
4. otherwise learn the new task with PPO and whole-rollout KL control;
5. rank diagnostic seeds by their adverse gate margin;
6. preserve every improving intermediate in an immutable candidate archive;
7. pass every gate on five fresh, rotating, full-episode promotion seeds;
8. replay every earlier accepted skill for the full 800-step episode;
9. train on real old-task states/actions with adaptive per-skill replay pressure;
10. record every gate margin in the persistent regression matrix; and
11. promote only if the new skill and every replay pass.

The zero-shot exam is competency-based curriculum allocation, not a gate bypass.
It is disabled when there is no immediate same-family predecessor, when a rung
already has a durable training attempt, for smoke/no-gate runs, or for an actor
without task conditioning. Failed exams are versioned by parent-checkpoint hash
and gate signature, then fall through to ordinary PPO without being repeated on
resume. `--no-test-out` requires PPO on every rung; `--test-out-margin` changes
the extra certification headroom.

A failing attempt no longer always runs to its full step budget. At every
trainer evaluation the worst early-gate margin (worst deterministic seed) joins
a least-squares projection; when the margin is failing and its projected zero
crossing lies beyond `--plateau-slack` (default 2×) times the remaining budget
for `--plateau-patience` (default 3) consecutive evaluations, the attempt
aborts with a durable `plateau_abort` record and its checkpoint. The ladder's
retry then deliberately changes the dynamics instead of buying identical
steps: it reinjects exploration (`--plateau-entropy-boost`, capped at the
from-scratch entropy start) and warm-restarts the KL-adapted learning rate to
its ceiling. Gates are never waived — a plateau abort is a cheaper route to
the next attempt, not a promotion. `--no-plateau-intervention` restores plain
retries; the trainer's `--plateau-slack 0` disables the abort entirely.

The ordinary trainer evaluation is a diagnostic and candidate-ranking bank.
For every `>=` gate the ladder uses the minimum value across its deterministic
seeds; for every `<=` gate it uses the maximum. Final promotion uses a separate
seed bank that rotates by rung and attempt and evaluates complete episodes. This
prevents both the first/mean seed and repeated tuning against one public fixed
seed from hiding a brittle policy.

All learning rungs now share the `universal256x14:v1` controller contract: 211
padded physical channels, a 14-channel actuator-availability mask, and one
31-channel global task command. Locomotion and morphology expose twelve servo
actions; combat exposes those plus two pneumatic actions. Compatible older
task-FiLM locomotion checkpoints widen into this namespace without discarding
learned weights.

The optional `predictive_token_gru` family adds physical morphology tokens, a
masked future-interaction command, and a self-supervised locally stabilized
trajectory decoder while retaining the GRU action loop. Its planner optimizes
task-space intent and physical feasibility, earns authority from a held-out
dimensionless calibration anchor, and is evaluated against an identical-seed
guidance-disabled control. It is intentionally a new checkpoint family: launch
it from rung 1 in a separate output tree and compare it against the accepted GRU ladder. See
[`predictive-universal-controller.md`](predictive-universal-controller.md) for
the model, confidence-controlled prediction-gradient planner, diagnostics, and
launch command.

Rung 26 locks and evaluates FL, FR, RL, and RR separately, gates on the worst leg,
then changes requested legs repeatedly without resetting physics. PFSP mixes
weighted hall opponents across parallel worlds and requires a final pass against
every member. Co-design uses common random numbers for ranking, an unseen held-out
bank for its gate, and alternates search with universal-policy adaptation.

If an accepted gate contract changes, the ladder re-audits the accepted prefix.
A failing rung and all dependent successors are quarantined rather than silently
grandfathered. Unaccepted downstream candidates and derived gait priors are also
archived when their prerequisite changes.

## Objective roles and adaptive enforcement

The success requirement stays fixed; only training pressure may self-tune.
Reward-related signals have four distinct roles, defined fully in
[`training-objective-contract.md`](training-objective-contract.md):

| Role | Example | Rule |
| --- | --- | --- |
| Outcome | commanded progress, stopping, reaching, or selected-leg contact | define externally visible success without dictating a motion style |
| Physical constraint | fall, planted-foot slip, unsafe orientation, joint speed, or violent body motion | enforce the real boundary; adapt a separate normalized dual when useful |
| Efficiency cost | energy, impact, heat, or wear | include only when the real robot pays it and report the trade-off |
| Temporary scaffold | teacher action, reference gait, or missing-foot exploration pressure | bootstrap discovery, report separately, and retire with measured competence |

Do not self-tune the acceptance requirement until the policy passes. That would
let training redefine success. Normalize each violation by its physical limit,
give unrelated constraints separate multipliers, and bound the multipliers. A
new diagnostic reports each adaptive metric, target, observed value, multiplier,
ceiling, and saturation state. A saturated multiplier with an unmet contract is
a warning to investigate feasibility, credit assignment, or controller authority;
it is not permission to hand-increase an arbitrary reward weight. Style signals
such as clock agreement, diagonal pairing, cadence, symmetry, duty shape, or a
reference joint trajectory belong in diagnostics or a retiring scaffold—not in
permanent rewards, terminations, or promotion gates.

## Lessons carried forward

- Measure cyclic contracts over the complete cycle. An endpoint EMA remembers
  late swing events better than early ones and can treat identical feet unfairly.
- Demonstrations should teach intent, not freeze actuator timing. The physical
  foot lagged its commanded lift by about a quarter-cycle; action imitation was
  therefore converted to a scaffold that yields to contact and clearance.
- Describe the result, not the choreography. Rung 6 now asks every foot to become
  airborne while the body stays still and safe; it does not specify foot order,
  diagonal pairing, cadence, symmetry, phase, or joint trajectories.
- Preserve behavior across new task identities. Running normalization sent the
  unseen rung-7 one-hot channel to `10.0` while rung 6's learned channel was
  `1.36`, changing deterministic joint targets before learning began and
  collapsing the robot to all-feet-planted. New task columns now reproduce the
  preceding accepted task exactly at initialization, while categorical task
  normalization is frozen and physical sensor normalization continues adapting.
- An absent CLI teacher must mean no teacher. The optimizer previously rebuilt
  an implicit action prior from the retention anchor even when no prior artifact
  was supplied. Action-prior weight is now exactly zero until a versioned prior
  is explicitly configured, and ladder rungs 6+ remove the base walker's fixed
  gait-clock, airtime, nominal-pose, and clearance style rewards.
- Audit reward credit, not just reward names. At rung 7's 0.10 m/s discovery
  command, the generic Gaussian tracker paid 4.56 of its 5.0 maximum to a nearly
  stationary robot and consumed 78.4% of absolute reward; the two direct
  progress terms together supplied only 5.4%. Rung 7 now removes that saturated
  proxy and learns from target-normalized signed progress. A failing gate lowers
  the automatic single-component dominance warning from 85% to 60%.
- Separate outcome, physical constraint, efficiency, and scaffold reward
  magnitudes. If scaffold credit dominates outcome credit, the robot is probably
  learning the hint instead of the task.
- Keep behavioral diversity in telemetry. Contact-pattern entropy and the full
  foot-correlation matrix reveal collapse without turning a preferred gait into
  a requirement.
- An adaptive controller must measure the same physical event over a compatible
  window as evaluation. A short contact EMA reported rung-6 competence as 0.186
  while the full evaluation measured 0.414, so it falsely raised scaffold
  pressure. Exact episode contact coverage now drives the controller; the EMA is
  diagnostic only.
- Keep adaptive reward scale bounded. Increasing scaffold pressure must change
  acquisition emphasis without continuously inflating return magnitude and
  forcing the critic to chase a moving target.
- Normalize value loss by the rollout return scale. The optimum remains the
  same, while raw reward units no longer make every critic minibatch hit the
  gradient clip and render the clip unable to distinguish ordinary updates from
  true spikes.
- Average duty can hide one permanently planted foot. Keep per-leg values and a
  strict worst-foot gate.
- Dense shaping and termination must share the same physical threshold. Rewarding
  clearance below the contact threshold creates a region called success by one
  subsystem and failure by another.
- PPO clipping is not a hard trust region. Measure whole-rollout KL after every
  epoch and stop reusing stale rollout data when policy movement is too large.
- Actor and critic gradients need separate clipping and diagnostics; the critic
  must not consume the actor's clipping budget.
- A saved tensor shape is not dependency provenance. Checkpoint and derived-prior
  source hashes must match the accepted prerequisite.
- Promotion is evidence, not a filename: fixed-seed reproduction, checkpoint
  replay, multi-seed tails, and regression replay are mandatory.
- A mean can hide a boundary failure. Gate on the adverse deterministic-seed
  tail, then use the separate fixed seed to test reproduction rather than
  lowering a threshold because the aggregate is close.
- Never overwrite the only good policy. Archive improving intermediates and
  select among them using fresh evidence after training.
- Retention detection without repair is incomplete. Replay real old-task states
  and actions, and raise pressure only for skills that measurably regress.
- One universal policy needs an explicit global task namespace and actuator
  mask; separate families cannot prove that one controller composes all skills.

## Safe shutdown

1. Let an imminent scheduled evaluation finish so the checkpoint is durable.
2. Send `SIGTERM` to the active `train_mesh_warp.py` child.
3. Wait for `training_ladder.py` to record the intentional interruption and exit.
4. Copy the ignored `out/training_ladder/` artifacts locally with resumable rsync.
5. Compare local and remote checkpoint SHA-256 values and load the local file.
6. Only then terminate the RunPod through its API.

Never terminate an ephemeral pod merely because the source tree was pushed: the
learned weights and runtime state are not Git artifacts.
