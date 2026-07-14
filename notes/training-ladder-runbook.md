<!-- SPDX-License-Identifier: MIT -->
# Sequential robot-training ladder

> **Document status:** Active · **Audience:** Robot-learning contributors · **Last reviewed:** 2026-07-14 · **Canonical for:** Current 31-rung training state, resume procedure, adaptive reward contracts, and shutdown handoff

This is the operational record for the active MuJoCo-Warp/Torch curriculum. The
ladder contains thirty-one ordered tasks, promotes only checkpoints that pass a
fixed-seed gate and prior-skill replay, and stops at the first unproved rung.

## Current handoff

Training was intentionally stopped at 2026-07-14 04:16 UTC before the RunPod was
terminated. Rungs 1–5 are accepted. Rung 6, **Step in place**, has a resumable
candidate but is not accepted: its final deterministic evaluation was only
`0.0004883` below the step-clock threshold.

| Evidence | Saved value | Gate | Verdict |
| --- | ---: | ---: | --- |
| Environment step | 10,002,432 | — | saved |
| Aggregate duty | 0.799434 | ≤ 0.95 | pass |
| Worst-foot cycle duty | 0.861567 | ≤ 0.95 | pass |
| Stationary speed | 0.059957 m/s | ≤ 0.20 m/s | pass |
| Step-clock agreement | 0.699512 | ≥ 0.70 | **not yet pass** |
| Mean swing clearance score | 0.631111 | ≥ 0.30 | pass |
| Upright score | 0.999925 | ≥ 0.85 | pass |
| Catastrophic-termination rate | 0.0003125 | ≤ 0.001 | pass |
| Fall rate | 0 | ≤ 0.0001 | pass |

The candidate checkpoint is locally preserved at
`out/training_ladder/rung_06_step_in_place.pt`. The `out/` tree is intentionally
Git-ignored; pushing the repository does not upload learned weights.

- Checkpoint SHA-256:
  `e7cb3282fb0d570b4fc02f1461472eb911b34c1b4d2be8fb51132f8200b6d1b1`
- Checkpoint step: `10_002_432`
- Reward semantics: `ladder:06:step_in_place:v9:dual2`
- Actor: task-conditioned FiLM network, hidden sizes `512,256,128`
- Accepted prefix in `ladder_state.json`: `[1, 2, 3, 4, 5]`
- The recorded rung-6 `returncode: -15` is the intentional shutdown, not a
  physics or learning failure.

The last accepted checkpoints and the rung-6 candidate, optimizer/runtime state,
normalizers, metrics JSONL, resolved configs, ladder state, and logs were copied
from the pod to the local ignored output directory. The copy is about 240 MB.
The ladder state records the regression-matrix path, but no matrix file exists
yet because no candidate has reached fixed-seed promotion since that diagnostic
was introduced; the runner initializes it on resume and writes its first row at
the next promotion check.

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
  --kl-stop-multiplier 1.5 --hidden 512,256,128 \
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
2. learn the new task with PPO and whole-rollout KL control;
3. pass every rung gate on the training evaluation;
4. reproduce the pass on the fixed retention seed;
5. replay every earlier accepted same-family skill;
6. record every gate margin in the persistent regression matrix; and
7. promote only if the new skill and every replay pass.

If an accepted gate contract changes, the ladder re-audits the accepted prefix.
A failing rung and all dependent successors are quarantined rather than silently
grandfathered. Unaccepted downstream candidates and derived gait priors are also
archived when their prerequisite changes.

## Fixed contracts, adaptive enforcement

The physical requirement stays fixed; only the pressure used to satisfy it is
self-tuning. Reward-related signals have four distinct roles:

| Role | Example | Rule |
| --- | --- | --- |
| Objective | forward progress | maximize the task outcome |
| Physical constraint | slip or per-foot duty | increase its own normalized dual while violated; release after success |
| Competence floor | step-clock ≥ 0.70 | increase a lower-bound dual while short of target; release after success |
| Temporary scaffold | demonstrated lift action | bootstrap exploration, then yield to physical contact and clearance |

Do not self-tune the acceptance requirement until the policy passes. That would
let training redefine success. Normalize each violation by its physical limit,
give unrelated constraints separate multipliers, and bound the multipliers. A
new diagnostic reports each adaptive metric, target, observed value, multiplier,
ceiling, and saturation state. A saturated multiplier with an unmet contract is
a warning to investigate feasibility, credit assignment, or controller authority;
it is not permission to hand-increase an arbitrary reward weight.

## Lessons carried forward

- Measure cyclic contracts over the complete cycle. An endpoint EMA remembers
  late swing events better than early ones and can treat identical feet unfairly.
- Demonstrations should teach intent, not freeze actuator timing. The physical
  foot lagged its commanded lift by about a quarter-cycle; action imitation was
  therefore converted to a scaffold that yields to contact and clearance.
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

## Safe shutdown

1. Let an imminent scheduled evaluation finish so the checkpoint is durable.
2. Send `SIGTERM` to the active `train_mesh_warp.py` child.
3. Wait for `training_ladder.py` to record the intentional interruption and exit.
4. Copy the ignored `out/training_ladder/` artifacts locally with resumable rsync.
5. Compare local and remote checkpoint SHA-256 values and load the local file.
6. Only then terminate the RunPod through its API.

Never terminate an ephemeral pod merely because the source tree was pushed: the
learned weights and runtime state are not Git artifacts.
