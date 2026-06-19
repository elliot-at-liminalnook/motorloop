<!-- SPDX-License-Identifier: MIT -->
# RL environment: NN learns multi-joint locomotion over the FOC (MuJoCo + Gymnasium + SB3)

A fast, standards-based reinforcement-learning environment where a neural network
learns a **multi-joint / locomotion** task by outputting **high-level setpoints**
that a per-joint **FOC + motor inner-loop model** executes — so the policy learns
under *realistic motor limits* (torque-speed envelope, current limit), not ideal
torque. Train **faster-than-real-time** in MuJoCo with **Stable-Baselines3**,
watch it in the MuJoCo viewer, and keep a path to validate the inner loop against
the cycle-accurate RTL bench.

**North star:** the NN policy commands joints; the FOC/motor model turns those
commands into the torque the chosen real motor could actually deliver; MuJoCo
integrates the multibody dynamics; the agent learns. The motor choice (the
GM2804/DB42/EC 45 profiles) is *part of the environment* — so "which motor can the
robot walk with?" becomes a learnable, measurable question.

## The honest boundary (built in)

- **Fast surrogate, NOT cycle-accurate.** This env is for *training at scale* — it
  is explicitly *not* the RTL co-sim. The inner-loop FOC is a **reduced model**
  (torque-speed envelope + current limit + a first-order L/R lag), parity-checked
  against the cycle-accurate bench separately (§6). The two tiers stay distinct.
- **Train-in-sim-first.** "Real-time" (HIL, hardware) is a *later* tier — the same
  motor/FOC contract is the bridge, but this checklist builds the fast trainer.
- **Sim-to-real gap is real.** Domain randomization (§2) mitigates it; the
  verification tier (§6) + eventual HIL are how it's closed honestly.
- **Compute.** RL locomotion is compute-heavy; CPU works for a modest task/step
  budget, GPU (or Isaac Lab) for scale — note the chosen budget, don't pretend a
  CPU run is a fully-converged policy.

## §0 — Stack + scope decisions

- [ ] **Deps:** `pip install mujoco gymnasium "stable-baselines3[extra]" tensorboard`
      (SB3 pulls PyTorch). Optionally clone **MuJoCo Menagerie** (the standard
      high-quality MJCF robot library) for a ready robot model.
- [ ] **Robot (MJCF):** recommend starting from a **Menagerie quadruped** (e.g. a
      Unitree/ANYmal-class model) for locomotion — *or* de-risk with a **2–3 DOF
      planar arm reach** as milestone-0, then move to locomotion. Decide; the env
      is robot-agnostic.
- [ ] **Control hierarchy + action space:** NN policy (high-level, per actuated
      joint) → per-joint **FOC+motor inner loop** (§1) → joint torque into MuJoCo.
      Recommend the action = **per-joint target velocity** (or position/PD target)
      bounded by the motor envelope; the inner loop realizes it. Control rate
      (e.g. 50–100 Hz) < physics rate (e.g. 500–1000 Hz).

## §1 — The motor + FOC inner-loop model (the realism layer — reuses the project)

- [ ] **`sim/rl/actuator.py`** — a fast per-joint actuator: inputs the NN's
      high-level command + the joint's current ω (from MuJoCo); applies a reduced
      **FOC current loop** (saturating to the **torque-speed envelope** + current
      limit, with an optional first-order L/R electrical lag); outputs the
      realizable **joint torque** through the gear ratio.
- [ ] **Plug in the motor profiles** from `sim/tests/motors.py` (GM2804 / DB42S03 /
      maxon EC 45): the actuator's Kt, current/torque limit, torque-speed envelope,
      pole pairs and gear ratio come from the chosen `Motor`. The policy is thus
      *constrained by the real motor* — the RL↔motor coupling.
- [ ] Note the parity obligation: this fast FOC is the reduced form of the RTL FOC;
      §6 cross-checks it against the cycle-accurate bench (single joint).

## §2 — The Gymnasium environment

- [ ] **`sim/rl/foc_locomotion_env.py`** — a `gymnasium.Env` (register an id, e.g.
      `MotorloopLocomotion-v0`) wrapping the MJCF robot + the §1 actuators.
  - **Observation:** joint pos/vel, base orientation/height/velocity (locomotion),
    the command/target, last action.
  - **Action:** per-joint high-level setpoint (bounded by the motor envelope).
  - **`step()`:** NN action → §1 inner loop → joint torques → N MuJoCo substeps at
    the control rate → reward + obs + termination.
  - **Reward:** locomotion = forward velocity + alive bonus − energy/torque penalty
    − orientation/fall penalty (or, for the arm: −tracking error − smoothness).
  - **Reset / termination** (fall, joint limits) + **domain-randomization** hooks
    (motor params, friction, mass, latency) for sim-to-real robustness.

## §3 — Training (Stable-Baselines3)

- [ ] **`sim/rl/train.py`** — **PPO** (locomotion default; SAC optional) over a
      **vectorized** env (`SubprocVecEnv`, many parallel) for throughput;
      TensorBoard logging, checkpoints, an eval callback, fixed seeds, a config
      file. State the step budget honestly (demo vs converged).

## §4 — Visualization (watch it learn / watch it walk)

- [ ] **MuJoCo viewer** (`mujoco.viewer`) — `eval.py --render` to watch a trained
      policy live (the native, standard viewer for MuJoCo).
- [ ] **Headless rollout video** → `figures/rl/<task>.mp4`/`.gif` (the "see it"
      artifact for the README, like `motorloop.gif`).
- [ ] **TensorBoard** learning curves (reward, episode length, energy).

## §5 — The motor-coupling study (ties RL to the motor/sensor work)

- [ ] Train/evaluate the **same task with each motor profile** (GM2804 / DB42 /
      EC 45) in the §1 inner loop → "which motor can the robot walk/reach with, and
      how well?" Report achieved speed/reward + energy per motor — the motor
      envelope shapes the learnable policy. Extend with the **sensor coupling**
      (M8): does the policy degrade with the AS5600's latency at high pole-pairs/
      gear vs the AS5047P? Connects RL to the part-comparison + motor studies.

## §6 — Verification tier (the sim-to-real-gap anchor)

- [ ] **Inner-loop parity:** validate `actuator.py`'s reduced FOC against the
      **cycle-accurate RTL bench** for a single joint (same command + ω → same
      torque/dynamics within tolerance). The trust anchor that the fast env's
      actuator ≈ the real FOC.
- [ ] Flag the full **RTL-in-the-loop policy eval** and **HIL** as the deployment
      phase (out of scope here; the actuator contract is the bridge).

## §7 — Integration

- [ ] **`make rl-train` / `make rl-eval`** targets; `requirements-rl.txt`;
      REUSE/SPDX on all new files (MJCF/configs covered in `REUSE.toml`); the env
      registered + importable.
- [ ] **`notes/rl-environment-report.md`** — the task, the reward, the learning
      curve, the rollout video, and the motor-coupling result (§5).

## Done-when

A Gymnasium env (`MotorloopLocomotion-v0`) trains an SB3 policy on the multi-joint
task with the **FOC+motor inner loop in the loop** (real motor envelope), a
trained policy is watchable in the MuJoCo viewer + a rollout video, the
inner-loop FOC is parity-checked against the cycle-accurate bench (§6), and the
motor-coupling study (§5) reports how the motor choice shapes what the agent can
learn. `make rl-train` + `make rl-eval` reproduce it.

## What NOT to do

- Don't train against the cycle-accurate RTL bench — it's the verifier, not the
  trainer (orders of magnitude too slow). Keep the tiers separate.
- Don't give the NN ideal torque — route every command through the §1 motor
  envelope, or the learned policy won't transfer.
- Don't trust the fast actuator until §6 parity passes.
- Don't claim convergence/real-time you didn't reach — state the step budget and
  that this is the train-in-sim tier.
- Don't hand-fork robot geometry — use a standard MJCF (Menagerie) so the same
  model feeds the dynamics and the viewer.

## Implemented (results)

Done end-to-end; full write-up + numbers in
[`rl-environment-report.md`](rl-environment-report.md).

- **§0 stack:** `~/rl-venv` (uv-managed **Python 3.12**, since 3.14 has no
  torch/mujoco wheels) + **CPU torch** + mujoco/gymnasium/SB3; `requirements-rl.txt`.
- **§1 actuator:** `sim/rl/actuator.py` — pure-numpy motor torque-speed envelope
  (reuses `motors.py`), importable in both venvs (so the parity test runs in the
  system suite).
- **§2 env:** `sim/rl/foc_env.py` — standard Gymnasium MuJoCo robot wrapped with
  the motor-envelope action layer; registered as `MotorloopLocomotion-v0`.
- **§3 training:** `sim/rl/train.py` (PPO + **VecNormalize**) — the policy
  **learned to run** (raw return **3131** vs **−233** random) in ~5 min / 1.5 M
  steps on CPU.
- **§4 viz:** `eval.py` (record) + `render_rollout.py` (torch-free render) →
  `figures/rl/halfcheetah_db42.mp4`. Decoupled to dodge the torch+OSMesa heap
  crash (the record→render philosophy).
- **§5 coupling:** the same policy under each motor → monotonic 3096/3131/3166
  (gimbal<DB42<maxon); `gen_rl_figures.py` → `motor_envelope.png` +
  `coupling_returns.png`.
- **§6 parity:** `test_rl_actuator_parity.py` — the reduced FOC actuator matches
  the cycle-accurate bench in the operating regime (green in the system suite).
- **§7 integration:** `make rl-train/rl-eval/rl-figures`; REUSE clean.
- **Honest boundary held:** fast surrogate (not cycle-accurate); reduced FOC
  (parity-anchored); CPU demo budget (runnable gait, not SOTA); per-motor full
  training + domain-randomized sim-to-real + HIL are the later tiers.
