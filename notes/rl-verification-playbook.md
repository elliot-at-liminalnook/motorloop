# RL Verification Playbook — Catching Insane Results as Early as Possible

Date: 2026-07-01. Companion to `training-uplift-audit.md`. That audit found the
project's failures were dominated by verifiable-but-unverified assumptions (missing
actuator gear, frozen resets, never-passed CLI flags, a 0.66 s credit horizon, an
untested control mode). This file is about *why our existing verification missed
them* and the layered checks that would have caught each one within minutes instead
of weeks of GPU time.

The executable policy for this playbook is centralized. Use
`bash scripts/run_pre_gpu_tests.sh` for a fast local precheck, and run
`bash scripts/run_pre_gpu_tests.sh --require-gpu` in a CUDA environment for the
only complete verification verdict. The full form includes the CPU-only RTL and
component suite; it then runs the expensive batched physics and trainer tests on
CUDA. See [pre-gpu-test-entrypoint.md](pre-gpu-test-entrypoint.md).

## Part 1 — Why the checks we already had didn't fire

We were not short on verification tools: `validate_body.py`, `prove_robot.py`,
`anti_cheat.py` (24 metrics!), the behavior benchmark, keep-gates. Every one of them
missed the torque bug. The failure modes are general and worth naming:

**1. We verified attributes, not outcomes.** `validate_body.py` reads the torque
limit from `m.actuator_forcerange` — the attribute that *documents* intent — instead
of commanding `ctrl=1` and measuring what force actually comes out. The gear bug made
forcerange decorative, so the validator's "torque_margin" was computed against a
torque the actuator could never produce. *Rule: a check must exercise the same causal
path the training env uses (write ctrl → step → measure force/acceleration), never a
metadata field along the way.*

**2. Adaptive thresholds hide regressions.** `validate_body.py` searches for the
"tallest STABLE stand" — it lowers the bar until something passes. A robot that can
only hold a spring-propped crouch still gets a PASS. *Rule: bars are absolute and
task-derived (must hold THE stance pose used in training envs; must reach the 0.3 m/s
stride the combat design assumes), never "best achievable."*

**3. We verified the spec, not the compiled artifact.** `robot.toml` said gear=12,
comments said "stands nearly for free," and everyone (humans and agents) reasoned
from that. The compiled `model.xml` disagreed for the project's entire lifetime.
*Rule: every claim in a spec/comment that matters gets an assertion against the
built MjModel. If it's worth writing in a comment, it's worth asserting.*

**4. Anti-cheat was reactive.** `anti_cheat.py` was written after the catapult
exploit; the rung gates after the stand-still exploit; tactical RND after jitter
farming. Each exploit got one post-hoc patch. *Rule: red-team the reward before
training — scripted exploit policies are cheap fixtures, and a reward that pays
them is a bug found in seconds.*

**5. Scalar metrics were allowed to stand in for behavior.** "Walking" was claimed
twice from net-displacement numbers (0.08 m/s drift). Both times a 30-second video
falsified it. *Rule: no behavioral claim without a rendered rollout; milestone
promotion requires video artifacts, not just JSONL rows.*

**6. Mechanisms were assumed, not tested.** Per-episode spawn randomization was
built, reviewed, and believed — but brax's AutoResetWrapper replays one cached reset
state per env forever, so it mostly never ran. Likewise `--max-grad-norm` and
`--lr-schedule` were plumbed and never passed by any script. *Rule: test that the
mechanism observably fires (two episodes differ; the flag changes a logged value),
not that the code for it exists.*

## Part 2 — The layered check stack (cheapest first, run in this order)

### T0 — Model-build assertions (< 1 s, run on every MJCF generation)
Extend `gen_robot_mjcf.py` (or a `test_model_contract.py`) to build the model and
assert, for every hinge actuator:

```python
m = mujoco.MjModel.from_xml_string(xml)
tau = joint_torque_limit(spec)
assert np.allclose(m.actuator_gear[hinges, 0], tau, rtol=0.01)   # gear actually emitted
d = mujoco.MjData(m); d.ctrl[hinge] = 1.0; mujoco.mj_forward(m, d)
dof = m.jnt_dofadr[m.actuator_trnid[hinge, 0]]
assert abs(d.qfrc_actuator[dof] - tau) < 0.05 * tau              # OUTCOME, not attribute
# NB: qfrc_actuator (joint space, post-gear), NOT d.actuator_force — that field is
# actuator-space (pre-gear) and reads 1.0 on a CORRECT gear=tau model too.
```

Also assert: total mass within spec ±5%, joint ranges match `robot.toml`, timestep ×
frame_skip = the control dt the reward math assumes, foot geoms exist and collide
with the floor under the contact mask. **Would have caught: the gear bug, on day one.**

### T1 — Static physical capability (seconds, run after every body/spec change)
Fix `validate_body.py`:
- measure torque via commanded-ctrl outcomes (T0 style), not forcerange;
- hold the *canonical training stance* (the `stand` pose both envs PD around), not
  the tallest stance that happens to work;
- add a torque-margin bar: max gravity-holding torque per joint ≤ 60% of measured
  deliverable torque;
- add a drop test (release from spawn height, assert it settles standing, no limb
  penetration, no explosion) and a max-effort test (full ctrl on one leg → joint
  acceleration within an expected band; catches both weakness AND catapult-strength).

**Would have caught: gear bug, spring-propped false stands, and (inverted) the
post-fix catapult risk.**

### T2 — Training-config preflight (milliseconds, print + assert at trainer startup)
Add a preflight block to `train_adversarial.py` / `train_commanded.py` that computes
and PRINTS derived quantities, with hard failures on red lines:

| Derived quantity | Formula | Red line |
|---|---|---|
| env-steps per PPO iteration | batch × minibatches × unroll | — |
| total PPO iterations | steps / above | **fail if < 200 for from-scratch** |
| credit horizon | dt / (1 − γ) | **fail if < 2× stride period / task timescale** |
| episode duration | episode_length × dt | warn if < 2× time-to-contact |
| first-layer fan-in ratio | obs_dim / hidden[0] | warn if > 4:1 |
| flags actually received | echo argv + resolved config dict | fail on unknown/ignored flags |

Log the resolved config (post-defaults, post-overrides) to the run directory as
JSON — "what did this run *actually* train with" must never require code archaeology.
**Would have caught: the 37-iteration budget, γ=0.97 myopia, the 4×32 network, and
the never-passed grad-clip/KL flags.**

### T3 — Mechanism liveness tests (CPU, seconds, in `test_*.py`)
For every stochastic or conditional mechanism, a test that observes it firing:
- **Reset diversity**: step a wrapped env through 3 auto-resets; assert the
  post-reset qpos/spawn separations are pairwise distinct. (Catches the
  AutoResetWrapper replay defect — the single highest-value missing test.)
- **Flag reachability**: for each CLI flag, one test asserting it changes the
  resolved config or a logged value.
- **Mode matrix**: every control/obs mode combination that any production script
  uses gets a 10-step smoke rollout (pure-`pd` mode ran in production without ever
  having one — the only tested mode was `cpg_pd`).
- **RNG advance**: per-env noise streams differ across envs AND across steps
  (already done well in `test_lidar_env.py` — apply the same standard everywhere).

### T4 — Reward red-teaming (CPU/GPU minutes, before any long run)
Pre-registered exploit fixtures as scripted policies, run through the *actual*
reward path (`env.step`, not a reimplementation):
- stand-still, in-place oscillation (max joint speed, zero displacement), torso
  jitter, catapult-style single impulse, spin-in-place;
- an honest reference: a scripted approach/drag toward the opponent, however crude.

Assert every exploit fixture's episode return < the honest reference's. New reward
terms don't merge without passing the fixture suite; every discovered exploit gets
added as a permanent fixture (stand-still and oscillation would have been caught
pre-GPU). This generalizes `test_rungs.py` from "the gates work" to "the whole
reward is exploit-ordered."

### T5 — Cheap decisive experiments before expensive ones (the ladder rule)
Order runs so each one can invalidate everything downstream of it:
1. upper-bound probes first — scripted/MPC/assisted-harness rollouts answer "CAN
   this body do X at all" for minutes of compute (if trajectory optimization at
   full torque can't find a 0.3 m/s cycle, no RL run should be launched);
2. the shortest run that can falsify the hypothesis (1 h pure-pd validation before
   a 4 h combat run);
3. one variable per run — the audit found four simultaneous confounders (torque,
   budget, γ, network) precisely because runs changed many things at once.

### T6 — In-run monitoring with tripwires (during GPU runs)
- Behavioral metrics (bh_disp, bh_approach, foot air-time, contact-break count) on
  every eval, not just reward/judge score — reward up + air-time flat = exploit alarm.
- **Stagnation tripwire**: if bh_disp hasn't crossed a task-derived floor by N
  iterations, stop the run automatically. The 12M-step 0.18 m run should have
  self-terminated at ~2M.
- **Regression tripwire**: judge score dropping > X% from peak for K consecutive
  evals triggers checkpoint freeze + alert (the cpglong peak-then-collapse burned
  ~40M steps after the peak).
- Videos ARE monitoring: render 1 episode per milestone (the machinery exists in
  `pbt_train.maybe_render_video`) and treat "no video artifact" as a failed
  milestone, not a cosmetic gap.

### T7 — Checkpoint and transfer semantics (whenever checkpoints cross a boundary)
- Embed metadata in every checkpoint: model hash of the MJCF it trained on, action
  semantics (torque | pd_target + scale), obs-layout version, trainer config hash.
- `validate_frozen_opponent` must reject on semantics mismatch, not just shape
  mismatch (shapes matched while torque-trained opponents were silently mis-driven
  under new semantics).
- Warm-start acceptance is behavioral: after loading, roll the policy in the target
  env and assert it reproduces its source behavior (e.g., ≥ 50% of source
  displacement). "WARM-START ok: policy 38->470" verified plumbing, not transfer —
  the transferred walker was behaviorally dead and nothing noticed.
- Obs-layout as an explicit schema (named slices, single source of truth) with a
  test that HER's goal-tail and RND's feature-slice assumptions match it — today
  those are implicit positional contracts that any obs change silently breaks.

### T8 — Adaptive reward contracts and dependency provenance

- Keep physical thresholds and competence floors fixed; adapt only their
  normalized enforcement multipliers.
- Give slip, aggregate duty, per-foot duty, and competence shortfalls separate
  bounded controllers. Never collapse unrelated failures into one reward weight.
- Log every controller's metric, target, observed value, multiplier, ceiling, and
  saturation state. Saturated pressure plus a failed contract is an alert, not a
  successful self-tuning claim.
- Compute gait-cycle contracts over the entire cycle. Endpoint EMAs are useful
  trends but are order-biased pass/fail measurements.
- Treat demonstrations as temporary scaffolds that automatically yield to
  physical contact, clearance, safety, and task outcomes.
- Hash the prerequisite checkpoints used to derive later policies and searched
  priors. If a prerequisite is relearned, quarantine every unaccepted descendant
  and derived artifact before resuming.

The current implementation and the evidence that motivated these rules are in
[`training-ladder-runbook.md`](training-ladder-runbook.md).

## Part 3 — Pitfall → earliest catching check

| Pitfall (as it happened) | Cost as incurred | Check that catches it | Earliest firing point |
|---|---|---|---|
| gear never emitted (~8% torque) | entire project history | T0 ctrl→force assertion; T1 stance-hold at real torque | first model build |
| validate_body used forcerange as tmax | false PASSes | T0/T1 outcome-based measurement | first validator run |
| 37 PPO iterations "12M-step run" | weeks of misread evidence | T2 iteration-count red line | trainer startup |
| γ=0.97 (0.66 s horizon) | all runs | T2 credit-horizon vs task timescale | trainer startup |
| grad-clip/KL flags plumbed, never passed | all runs | T2 resolved-config echo + T3 flag tests | first CI run |
| 4×32 net on 470-dim obs | all lidar runs | T2 fan-in warning | trainer startup |
| AutoReset replays one cached state | all runs' curricula | T3 reset-diversity test | first CI run |
| pure-pd mode never validated | locomotion strategy built on cpg_pd only | T3 mode-matrix smoke | first CI run |
| stand-still / oscillation exploits | 2 GPU cycles + rework | T4 exploit fixtures | pre-GPU, seconds |
| "walking" claimed from displacement metrics (×2) | 2 review cycles | T6 mandatory render gates | first milestone |
| cpglong collapse after 20M peak | ~40M wasted steps | T6 regression tripwire | during run |
| warm-start plumbing-OK, behaviorally dead | 1 GPU cycle | T7 behavioral acceptance | at load time |
| frozen opponents valid-by-shape after semantics change | silent (future) | T7 semantics metadata | at load time |

## Part 4 — Process rules (the human/agent loop)

1. **Every comment-level belief gets an assertion.** "Stands nearly for free" lived
   in a comment for months while being false. If a claim justifies a design
   decision, it must exist as executable code somewhere in T0–T3.
2. **Falsify before scaling.** No run > 1 GPU-hour without a written one-line
   hypothesis and the cheapest experiment that could kill it having already passed.
3. **Render before claiming.** Any sentence of the form "the robot now Xs" requires
   a video artifact. Metrics select candidates; videos confirm them.
4. **New exploit ⇒ new permanent fixture** in the T4 suite, same day. Gates patch
   instances; fixtures prevent the class from returning silently.
5. **Preflight is not optional.** T0–T2 run as a single `preflight` entry point at
   the top of every trainer and in CI; a training process that starts without a
   green preflight is itself a bug.
6. **When results are mysterious, audit the physics before the algorithm.** Months
   went into RND/HER/PBT/curricula while the robot could not physically stand. The
   cheaper layer is always upstream: body → config → mechanism → reward → algorithm.
