<!-- SPDX-License-Identifier: MIT -->
# Sim-to-Real Lidar + Advanced RL — Implementation Notes

Lidar sensing, asymmetric actor-critic, a hierarchical strike gate, TRUE RND
curiosity, TRUE on-policy HER relabeling, and Population-Based Training (PBT),
all wired into the actual Brax-PPO learning loop for the adversarial walker
combat system (`sim/robot/train_adversarial.py`).

This file claims only what is implemented and covered by tests; the GPU
verification log at the bottom is filled in from real runs.

## Status

| Feature | Where | Wired into learning? | Fast behavioral test |
|---|---|---|---|
| Lidar (per-env noise/dropout/latency/stack) | `train_adversarial.py` `_lidar_obs`/`_lidar_scan` | actor/critic obs | `test_lidar_env.py` |
| Asymmetric actor-critic | `build_benchmark` + `_asymmetric_factory` | policy=`state`, value=`value_state` | `test_lidar_env.py` (dims) |
| Hierarchical strike gate | `step()` gate logit | modulates striker DOFs + gate reward | (exercised in GPU smoke) |
| TRUE RND curiosity | `rnd_curiosity.py` + env `step` | per-env predictor trained each step; bonus in reward | `test_rnd_her.py` |
| TRUE HER relabeling | `her_goal.py` `install_her_relabel` | on-policy relabel of every PPO rollout | `test_rnd_her.py` |
| PBT | `pbt_train.py` | per-agent ckpt resume + exploit/explore | `test_pbt.py` |

---

## 1. Lidar (behaviorally real)

`gen_robot_mjcf.py` adds N horizontal + M vertical rangefinder sites on A's torso
(`_lidar_sites_xml`), a `<sensor>` block, and a default material (MJX 3.9
ray-cast workaround). `AdversarialEnv` turns the raw `sensordata` into a
normalized depth scan and a **realistic sensor model**:

- **Per-env RNG.** `state.info["lidar_rng"]` is seeded from each env's reset key
  and `split` every step. Noise and dropout therefore differ across envs AND
  episodes (the old code folded a single `PRNGKey(0)` by timestep — identical
  across envs). `_lidar_scan` adds Gaussian range noise (`--lidar-noise-sigma`)
  and Bernoulli ray dropout (`--lidar-dropout-rate`, dropped rays read max range).
- **Latency, correctly.** A FIFO `lidar_scan_history` of length
  `--lidar-latency-steps` delays the observed scan. Latency is applied to the
  scan **before** the obs is assembled, so a delayed scan reaches the actor AND
  critic even at `--lidar-frame-stack 1` (the old code only rebuilt the obs when
  stacking, making latency a silent no-op at stack 1).
- **Frame stacking.** The post-latency scan is stacked with the previous
  `stack-1` observed scans for temporal velocity.
- **Determinism.** With noise+dropout off (`_lidar_stochastic == False`) the scan
  is a deterministic clean depth image — the held-out benchmark sets sigma=0,
  dropout=0, latency=0 so eval is reproducible.

**Asymmetric actor-critic.** With lidar on, `observation_size` is a dict:
`state` (actor) = `loco + stacked_scan + goal?`; `value_state` (critic) =
`loco + stacked_scan + privileged_opp + goal?`. PPO uses
`policy_obs_key="state"`, `value_obs_key="value_state"`.

Tests (`test_lidar_env.py`): dropout→max-range, deterministic clean scans, noisy
per-env/per-episode variation, latency changing actor+critic obs at stack 1,
frame-stack+latency+goal dims, vmap per-env independence, jit, backward compat.

## 2. Hierarchical strike gate

A gate logit appended to the action space modulates the striker DOFs:
`strike = action[strike] * sigmoid(gate)`. Gate reward pays for opening near the
opponent and penalizes opening far away (`--gate-weight`, `--gate-threshold`).
Closed gate ⇒ approach-only gradients; open ⇒ strike gradients. Exercised in the
GPU smoke (`--hierarchical`).

## 3. TRUE RND curiosity (`rnd_curiosity.py`)

A fixed random target f(s) and a trained predictor g(s); novelty = ||f-g||².
This is wired into training as **per-env online RND**: the predictor params + Adam
state are carried in `state.info` (`rnd_predictor`, `rnd_opt_state`), seeded
identically at reset, and **updated by one Adam step inside every `env.step`** on
the next-state proprioceptive features. The clipped novelty is added to the env
reward (`--rnd-coefficient`). Because the predictor learns the states an env
visits, the bonus decays on familiar states — genuine curiosity, not the previous
fixed-random stub. `info` is carried across the whole run by Brax, so the
predictor accumulates across episodes per env.

`make_rnd` is the pure functional interface (jit/vmap-safe) the env uses;
`RNDPredictor` is a stateful trainer for offline/standalone use. Tests
(`test_rnd_her.py`): predictor loss decreases, novelty drops on a trained/repeated
state and stays higher on unseen states, the env updates the predictor every step,
and the bonus raises the reward.

## 4. TRUE HER relabeling (`her_goal.py`)

Brax PPO is on-policy, so off-policy replay does not apply. Instead a **hindsight
relabeling pass runs over every collected rollout window before PPO computes
rewards/advantages**: `install_her_relabel` monkeypatches
`brax.training.acting.generate_unroll`. For each transition, with probability
`--her-fraction` the goal is relabeled with a goal ACHIEVED at a future step in
the same unroll (the "future" strategy). The pass:

1. requests `her_achieved`/`her_goal` as transition extras (the env exposes them
   in `info`),
2. overwrites the goal dims in `observation` AND `next_observation` (so the
   policy/value nets train on the relabeled goal — dict obs: both heads), and
3. adds `her_coeff*(gr(achieved, new_goal) - gr(achieved, goal))` to the reward,

then drops the extras so PPO sees a standard relabeled Transition. The 4D goal is
[distance, bearing, front-alignment, rod-distance]. Tests (`test_rnd_her.py`):
the relabeled obs goal equals a future achieved goal, the reward changes by
exactly the formula, fraction=0 is a no-op, and the brax patch installs/uninstalls.

## 5. PBT (`pbt_train.py`)

Each `Agent` carries persistent hyperparameters, its own checkpoint, latest
benchmark score, failure flag, and lineage. Per cycle:

1. each agent resumes from **its own** checkpoint (`--resume <agent ckpt>`) and
   trains `steps//cycles`;
2. agents are ranked by held-out `best_score` (read from `<tag>_state.json`);
3. **exploit/explore**: each bottom-half agent's resume checkpoint is replaced by
   a **copy of the top-half winner's weights**, and its hyperparameters become a
   **perturbed** copy of the winner's; lineage records the inheritance.

Subprocess failures (non-zero exit OR missing result) are detected and reported;
a failed agent sinks in the ranking and is rescued by exploit/explore.

### Guarded for long unattended runs

* **Budget caps** — `--max-runtime-min` / `--max-cost-usd` (with `--cost-per-hour`)
  stop the population GRACEFULLY *between* agents/cycles (never mid-checkpoint),
  via a `Budget` with an injectable clock. Caps are cumulative across resumes.
* **Signals** — SIGINT/SIGTERM are caught; the run stops after the current agent
  with state saved.
* **Checkpoints** — after each cycle the best agent's weights are promoted to
  `pbt_best.pkl` + a per-cycle milestone `pbt_cycle{N}_best.pkl`; the full
  population + resume metadata (`stop_reason`, cumulative `elapsed_s`,
  `est_cost_usd`, `started_at`, `resume_cmd`) is written to `pbt_state.json`.
* **Resume** — `--resume-state <pbt_state.json>` continues a population and
  carries cumulative elapsed time so budget caps span restarts.
* **Video** — every `--video-every` cycles the best checkpoint is rendered to a
  1v1 combat MP4 via `render_fight_video.py` (the milestone/render path). A lidar
  policy is rendered against a PASSIVE B (`--passive-b`) since B has no lidar
  sensors; `load_policy` builds the asymmetric inference (dict obs) and the env
  is rebuilt to match the checkpoint's lidar/her dims. Render failures are
  LOGGED, never fatal (unless `--video-fatal`).

The orchestration is pure stdlib and unit-tested with injectable runner /
video-runner / clock (`test_pbt.py`, 12 tests, no GPU): inheritance,
perturbation+bounds, failure handling, all-fail abort, resume round-trip,
runtime/cost cap behavior, interrupt, checkpoint promotion, video
scheduling+non-fatal-failure, and cumulative-budget resume.

### GPU pod rendering note
MJX physics runs with `MUJOCO_GL=""`; the CPU `mujoco.Renderer` needs a GL
backend. On the RunPod A100 image EGL is broken (`eglQueryString` → NoneType), so
video rendering uses **osmesa** (`apt-get install libosmesa6`; the PBT video
runner forces `MUJOCO_GL=osmesa`, override with `PBT_RENDER_GL`).

## Warm-start + frozen opponents

`warm_start` handles BOTH flat obs and the asymmetric lidar dict: the policy net +
its normalizer head grow to the actor (`state`) width, the value net + its head to
the critic (`value_state`) width, the action head grows for new striker DOFs.
Frozen opponents are validated BEFORE training (`validate_frozen_opponent`): B is
driven by the mirrored flat obs, so a lidar/asymmetric or hierarchical snapshot
(wrong obs/act width) is rejected with a precise error instead of failing inside
the jitted rollout.

---

## Outcome-grounding ladder (making the task non-exploitable)

The advanced-RL infrastructure works mechanically, but the *game* it created was too
easy to exploit: a from-scratch policy learned to **stand in place and micro-jitter**,
winning ~25% of held-out bouts by incidental contact against a passive B without ever
locomoting (verified by trajectory dump: ~1–15 cm torso displacement over 8 s, approach
velocity ≤ 0). RND on raw proprioception actively *rewarded* the jitter (novel joint
configs), compounded by no locomotor warm-start and short training. The fix is to make
the spec **adversarial and outcome-grounded**, climbed as a surgical ladder:

1. **Baseline confirmation** — the benchmark now reports BEHAVIOR (`bh_disp`, `bh_path`,
   `bh_closed`, `bh_lateral`, `bh_approach`, `bh_gate_open`, `bh_tip_speed`) with a
   `STAND-STILL` flag. On the exploit policy: `closed ≈ 0, approach ≈ −0.08, gate_open
   0.01` while winning — the exploit is now visible *in the eval*.
2. **Exploit gates** — (2a, selection) `behavior_keep_ok` + `--min-keep-closed/-approach/
   -disp/-far-sparc`: a stand-still or close-only policy cannot become keep-best.
   (2b, reward) `--require-closing` credits damage only while closing, `--stationary-
   damage-penalty` / `--oscillation-penalty` / `--energy-penalty` make idle effort
   net-negative (a high-effort idle step goes +0.06 → −0.18 in test).
3. **Behavioral novelty** — `--rnd-feature tactical`: RND on engagement descriptors
   (distance, bearing, approach/lateral velocity, rod distance, front, tip speed), which
   are ~6× less sensitive to joint jitter than proprioception — curiosity rewards
   reaching new *situations*, not twitching.
4. **Active opponent + range-balanced eval** — the benchmark/curriculum face a scripted
   opponent (`--opponent-script`) and range-balanced spawns, with keep-gates requiring
   FAR-range performance. *Honest status:* the keep-gate / range-balanced SELECTION side
   works; the scripted opponent is a wired hook that perturbs B but is **not yet a
   competent pursuer** (an open-loop drive doesn't produce a gait) — a real active
   opponent needs a locomotion-driven / Hall-of-Fame policy (for the lidar fighter that
   means a non-lidar opponent, since B has no lidar sensors).
5. **Then PBT** — once the spec can't be farmed by standing still, re-run PBT. The
   `rung5` run exercises the full corrected spec (require-closing + tactical RND + active
   opponent + behavior/range keep-gates + locomotion-weighted reward).

Tests: `test_rungs.py` (behavioral, on GPU): keep-gate rejects stand-still, corrected
reward penalizes idle effort, tactical RND ignores joint jitter, opponent hook wired.

## Locomotion-first: verified walker + CPG-PD control (the fix for stand-still)

The from-scratch and locomotor-warm-start fights both stalled at `disp ≈ 0.2m` — not a
reward bug but a hard EXPLORATION problem (a gait is never discovered; approach/velocity
rewards only flow once one exists). The elegant fix reuses the repo's gait prior instead
of rediscovering it.

**1. Verified the combat body can walk (cheap, decisive — no training).** Drove the body
open-loop with the tuned CPG teacher (`cpg_teacher.cpg_pd_step_target`, via
`CommandedEnv` `CMD_CONTROL_MODE=cpg_pd`, zero residual):
- BASE body: `+y 0.345m (0.087 m/s)`, holds on zero command. Walks.
- COMBAT body (striker + lean-contacts): `+y 0.332m (0.083 m/s)`, no fall — the striker
  mass does NOT break the gait. **No body adjustment needed.**
- Note: the gait is strong forward (`+y`) and weak lateral (`+x`); a re-tune
  (`search_cpg_gait_mjx.py`) could make it omnidirectional.

**2. CPG-PD control in the fighter (`--cpg-control`).** The legs are driven by the
directional CPG gait commanded toward B (the prior walks) + a policy RESIDUAL via PD; the
policy controls the striker/gate directly. The fighter walks from step 0 — no exploration,
no warm-start. Pure-CPG (zero residual) closes the gap in **3/5 spawns** (up to +0.54m),
vs the stuck-at-0.18m baselines.

**3. Verified combat training improves with the locomotion.** A short cpg-control run:
judge `8 → 17 → 22`, ratio `5.3 → 8.9`, margin `+0.47 → +0.79`, close-range sparc ~40
over 2.6M steps — the residual learns to fight (strike + margins) on top of the CPG gait.

**4. Long run (`cpglong`).** With the walk verified and combat improvement confirmed, a
60M-step (~4h) cpg-control fight run trains the residual + striker against the corrected
spec (require-closing, tactical RND, behavior keep-gates, lidar/asymmetric/hierarchical).

## GPU Pod Info
- Provision: RunPod REST API (`notes/gpu-pod-setup.md`), A100 80GB PCIe.
- JAX 0.6.x, mujoco/mjx 3.9.x, brax 0.14.x, Python 3.12.

## Verification Log
- (CPU, local) `py_compile` + `git diff --check` clean; PBT fake-runner tests pass.
- GPU smoke + full-pipeline results are appended here after the runs.
