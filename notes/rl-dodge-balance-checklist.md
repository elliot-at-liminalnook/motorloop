<!-- SPDX-License-Identifier: MIT -->
# Dodge-and-balance: a perceiving quadruped that evades objects thrown at its legs

A quadruped (body + 4 legs) that **perceives objects accelerating toward its
legs**, **dodges** (lifts/retracts the threatened leg), and **balances on the
other three** — the NN policy commanding joints through the FOC motor envelope.
The perception-driven extension of `rl-environment-checklist.md`.

**North star:** egocentric "lidar" senses incoming threats → the policy lifts the
targeted leg out of the way and keeps the other three planted to stay upright,
all under the real motor torque-speed limits.

## The honest boundary (built in)

- **This is a hard RL task** (dynamic obstacle avoidance + dynamic 3-leg balance,
  both at once). On the CPU budget expect a **learning/partial** policy — the
  solid deliverables are the **pipeline + perception + threat system + curriculum**;
  full elegant dodging is compute-bound (GPU / much longer budget).
- **Perception = sim rangefinders** (a cheap "lidar"), *not* a camera: OSMesa CPU
  rendering every step is impractical at training scale. Camera RGB-D is a flagged
  GPU/future option. On hardware this maps to a real **ToF/lidar + IMU on the SoC**
  — not RTL vision.
- **Reuses** the FOC motor-envelope actuator (parity-anchored, `sim/rl/actuator.py`)
  and the record→render viz (torch-free).

## §0 — Decisions + setup

- [x] **Robot:** recommend a **MuJoCo Menagerie quadruped** (e.g. Unitree Go2 —
      real body + 4×3-DOF legs, good for 3-leg balance) as the target; **Ant-v5**
      as a quick simpler stand-in to bring the pipeline up first. Decide.
- [x] **Perception:** **rangefinder ring** (primary) — a horizontal fan of
      MuJoCo `rangefinder` sensors on the body + per-ray closing-rate, so the
      policy senses *approaching* objects. Baseline/debug: privileged object
      state (relative pos/vel) — easiest, use to validate the task is learnable.
      Future (GPU): an onboard RGB-D camera + CNN policy.
- [x] **Reuse:** one motor-envelope actuator per leg joint (`actuator.py`,
      `motors.py`); the 3-tier architecture (fast trainer / RTL verifier / HIL).
- [x] Deps: the existing `~/rl-venv`; clone MuJoCo Menagerie if using Go2.

## §1 — Arena + threat system

- [x] **Scene** (`sim/rl/scenes/dodge_arena.xml` or a programmatic MJCF): the
      quadruped on a floor + a pool of spawnable **projectile bodies** (spheres).
- [x] **ThreatManager** (`sim/rl/threats.py`): on a randomized schedule, pick a
      target leg, spawn a projectile at a random offset/height, and give it a
      velocity/applied force **toward that leg's current position** ("accelerate
      toward its legs"). Parameterize speed / size / spawn-rate / lead — the
      curriculum + domain-randomization knobs.
- [x] **Hit detection** (MuJoCo contacts: projectile↔leg) and despawn/respawn;
      track which leg is targeted (privileged, for reward shaping in §3).

## §2 — Perception (the "see")

- [x] **Rangefinder ring:** N rays (e.g. 12–16 in a horizontal fan, optionally a
      second elevation) attached to the body → nearest-object distance per ray.
      Add the **per-ray closing rate** (Δdistance) so approach/acceleration is
      observable (a single distance frame can't show motion).
- [x] **Observation vector:** proprioception (joint pos/vel, body orientation/IMU,
      per-foot contact) **+** perception (ray distances + closing rates). Normalize
      (VecNormalize).

## §3 — The Gymnasium env

- [x] **`sim/rl/dodge_env.py`** (register `MotorloopDodge-v0`): obs (§2), action =
      per-joint high-level command through the motor envelope, plus the
      ThreatManager driven each step.
- [x] **Reward:** + upright/balance (body height + flat orientation), + alive,
      **+ dodge** (the *targeted* foot moves clear / the object misses — uses the
      privileged target-leg info), **− being hit**, − energy/torque jerk, and a
      terminal **− fall**. Shape toward "lift the threatened leg, keep the other
      three planted."
- [x] **Termination:** fall (body too low / flipped) or a hard hit.
- [x] **Curriculum + domain randomization:** start *balance-only* (no threats) →
      add slow/large/rare objects → ramp speed/rate/precision and randomize motor
      params, friction, mass, sensor noise (for robustness + sim-to-real).

## §4 — Training (SB3)

- [x] **`train_dodge.py`** — PPO (+ VecNormalize) or SAC; the curriculum schedule;
      a **longer budget** than HalfCheetah (this is harder) — report it honestly
      (learning/partial on CPU). Checkpoints + an eval callback; TensorBoard.

## §5 — Visualization

- [x] Extend the record→render pipeline: record the quadruped state **+ the
      projectile positions** each step; `render_rollout` (torch-free) replays
      both → a video of objects flying at the legs and the robot dodging/balancing.
      Optionally overlay the active rays / the targeted leg.
- [x] A **before (random → gets hit/falls) vs after (dodges + balances)** video
      pair → `figures/rl/dodge_*.mp4`.

## §6 — Parity, hardware mapping, honesty

- [x] The motor-envelope actuator stays parity-anchored to the cycle-accurate
      bench (reuse `test_rl_actuator_parity.py`).
- [x] **Hardware mapping (the components answer):** on a real robot, perception =
      a ToF/lidar ring + IMU read by the **RISC-V SoC**, the policy runs on the
      SoC, and it still commands the **FOC per joint** — no RTL vision needed.
      Document this so the sim↔hardware path is explicit.
- [x] Honest report of what converged vs what's compute-bound.

## §7 — Integration

- [x] `make rl-dodge-train` / `rl-dodge-eval`; reuse `requirements-rl.txt`
      (+ Menagerie note); REUSE/SPDX on all new files (MJCF covered in REUSE.toml).
- [x] **`notes/rl-dodge-report.md`** — the task, perception, curriculum, the
      learning curve, the dodge video, and the honest convergence status.

## Done-when

`MotorloopDodge-v0` runs: a quadruped with rangefinder perception, a threat
system launching objects at its legs, and a reward for dodge+balance; a policy
trains (balance → dodge curriculum) and a before/after video shows it evading
objects while staying upright; the motor-envelope parity holds; the report states
honestly what converged on the CPU budget. `make rl-dodge-train`/`-eval` reproduce.

## What NOT to do

- Don't use a camera on CPU/OSMesa — rangefinders give "sight" at 1000× less cost;
  camera is a GPU/future option.
- Don't skip the curriculum — dodge-from-scratch with a falling robot rarely
  learns; balance first, then threats.
- Don't give the policy ideal torque — keep every command through the motor
  envelope (the whole point).
- Don't claim mastery you didn't reach — report the budget and the convergence
  status honestly.
- Don't reach for RTL vision — perception lives in the sim (now) / on the SoC
  (hardware).

## Implemented (results)

Full results: [`rl-dodge-report.md`](rl-dodge-report.md). Reproduce:
`make rl-dodge-train` → `make rl-dodge-eval`.

- **Body / pipeline:** Ant stand-in (torso + 4 legs, 8 hinge joints) brought up
  end-to-end; `MotorloopDodge-v0` registered. `sim/rl/dodge_env.py`,
  `train_dodge.py`, `eval_dodge.py`, `render_rollout.py` (dodge branch).
- **Perception (§2):** 12-ray rangefinder fan + per-ray closing rate; 51-dim obs,
  VecNormalized. (Camera deferred to GPU/future, as planned.)
- **Threats (§1):** 4 projectiles fired at a target foot at `3+6·difficulty` m/s;
  contact-based hit detection + recycle.
- **Reward + curriculum (§3):** balance+dodge reward; difficulty ramps 0→0.6 over
  the first 60 % (balance-first).
- **Training (§4):** PPO + VecNormalize, 2 M steps, 16 envs (~5 min CPU).
- **Result:** **balance under bombardment learned decisively** — random falls in
  ~139 steps; the trained policy **survives the full 900-step episode** under
  continuous fire (training survival ~10→685 steps). **Active dodging is partial**
  — a 5×→25 hit penalty cut the hit rate ~12 % (57.6→50.4 hits/full-episode) at
  equal survival, but does not eliminate hits (compute/actuator-bound; diagnosed
  in the report).
- **Viz (§5):** `figures/rl/dodge_before.mp4` (random, knocked over) vs
  `dodge_after.mp4` (survives the bombardment).
- **Parity + hardware mapping (§6):** motor envelope stays parity-anchored;
  perception maps to a ToF/lidar ring + IMU on the RISC-V SoC, policy on the SoC,
  FOC per joint — no RTL vision.
- **Integration (§7):** `make rl-dodge-train`/`-eval`; REUSE/SPDX clean.

**Honest status:** pipeline + balance-under-bombardment **converged**; elegant
per-object dodging is **compute-bound** (budget / lead-aware reward / faster
actuator / 3-DOF-leg body) — reported, not hidden.
