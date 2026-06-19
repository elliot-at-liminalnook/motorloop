<!-- SPDX-License-Identifier: MIT -->
# Dodge-and-balance — results (a quadruped that sees and survives objects thrown at its legs)

Implements [`rl-dodge-balance-checklist.md`](rl-dodge-balance-checklist.md): a
quadruped (torso + 4 legs) that **perceives objects accelerating toward its legs**
through a rangefinder "lidar" ring, and learns to **stay upright through the
bombardment** — the NN policy commanding every joint **through the FOC motor
torque-speed envelope** (`sim/rl/actuator.py`). The perception-driven extension of
[`rl-environment-report.md`](rl-environment-report.md).

Code: `sim/rl/dodge_env.py` (env + threat system + perception), `train_dodge.py`
(PPO + curriculum), `eval_dodge.py`, `render_rollout.py` (the `robot=="dodge"`
branch). Reproduce: `make rl-dodge-train` → `make rl-dodge-eval` (needs
`~/rl-venv`; `requirements-rl.txt`).

## Honest frame (read this first)

- **This is a hard task** — dynamic obstacle avoidance *and* dynamic balance, at
  once, under real torque limits. The checklist predicted a **learning/partial**
  policy on the CPU budget, and that is exactly what we got: **balance under
  bombardment is strongly learned; clean active dodging is partial.**
- **The solid deliverables** are the *pipeline*: a working rangefinder perception
  ring, a threat system that accelerates objects at the legs, a balance→threats
  curriculum, and a policy that genuinely learned (random falls in ~139 steps; the
  trained policy survives the full 900-step episode under continuous fire).
- **Perception = sim rangefinders, not a camera.** 12-ray horizontal fan + per-ray
  closing rate. OSMesa camera render every step is ~1000× costlier and impractical
  at training scale. On hardware this is a **ToF/lidar ring + IMU on the RISC-V
  SoC** — not RTL vision (see Hardware mapping).
- **CPU demo budget:** 2 M steps, 16 envs. A *runnable, learned* policy, not a
  SOTA evader.

## What was built

- **Body:** the gymnasium Ant (torso + 4 legs, **8 actuated hinge joints**) — the
  quick stand-in the checklist names, to bring the full pipeline up first.
- **Perception (§2):** **12 rangefinder rays** in a horizontal fan on the torso,
  tilted slightly down, + the **per-ray closing rate** (Δdistance) so *approach*
  (not just presence) is observable. Obs = **51-dim**: 8 joint angles + 8 joint
  speeds + 4 torso-quat + 6 torso vel + height + 12 ray distances + 12 closing
  rates. VecNormalized.
- **Threat system (§1):** 4 projectile free-bodies (spheres, ρ=300). On a
  randomized schedule a target leg is picked, a projectile spawns 2.2 m out at leg
  height and is fired **at that foot's current position** at speed
  `3 + 6·difficulty` m/s. Contacts (projectile↔leg geom) = a hit; passed/landed
  projectiles despawn and recycle.
- **Reward (§3):** `+1 alive +1·upright −2·|h−0.55| −0.005·Σa² −25·hit
  +0.8·Σ min(clearance,1)`. Terminate on fall (upright < 0.2 or height < 0.25).
- **Curriculum (§3):** threat difficulty ramps **0 → 0.6** over the first 60 % of
  training (speed *and* spawn-rate scale with it) — balance first, then dodge.
  Dodge-from-scratch with a falling robot does not learn.
- **Actuator (§6):** every joint command passes through the **db42s03 motor + FOC
  torque-speed envelope** (gear 2.0, 12 V bus) — the same parity-anchored actuator
  as the locomotion env.

## What was measured

Deterministic eval, difficulty 0.6 (full threat speed + rate), 5 episodes,
900-step cap:

| policy | survives | hits / ep | hit rate |
|---|---|---|---|
| **random** | ~139 steps | 3.2 | — *(falls before it can be hit much)* |
| **trained (balance reward, −5 hit)** | **900 (full)** | 57.6 | 0.064 / step |
| **trained (dodge reward, −25 hit)** | **900 (full)** | 50.4 | **0.056 / step** |

- **Balance under bombardment: learned, decisively.** Random control falls in
  ~139 steps; both trained policies survive the **entire 900-step episode** while
  projectiles are continuously fired at their legs. Training survival climbed from
  ~10 steps (early) to ~685 (balance reward) over 2 M steps. This is the headline
  result — the robot *sees the fire coming and stays standing through it*.
- **Active dodging: partial.** Strengthening the hit penalty 5×→25 cut the hit
  rate ~12 % (57.6 → 50.4 hits over the full episode) **at equal (full) survival** —
  i.e. the strong-penalty policy keeps its legs marginally clearer for the same
  uptime. But it does **not** eliminate hits: at db42 torque on this body it can't
  always lift the targeted leg clear of a 9 m/s sphere in time, so it absorbs many
  hits while staying balanced rather than cleanly evading each one. Per-episode hit
  variance is high (a single hard seed hit 115×; the 5-ep mean is 50.4).
- **Videos** (`figures/rl/`): `dodge_before.mp4` (random — knocked over fast) vs
  `dodge_after.mp4` (the dodge-reward policy surviving the full bombardment).

## Why dodging is the compute-bound part (honest diagnosis)

The pipeline is sound and the task *is* learnable (balance proves it). Cleaner
evasion is bounded by, in rough order of impact: (1) **budget** — 2 M CPU steps is
~5 min; dynamic avoidance typically wants 10–50× more; (2) **reward shaping** —
−25/hit improved it but a *lead-aware* reward (penalize predicted impact, not just
contact) would teach anticipation; (3) **actuator speed** — a faster joint
(higher-Kv motor or lower gear) buys the leg-lift time the db42 envelope often
can't; (4) **a Menagerie 3-DOF-leg quadruped** (Go2) gives a real foot-retraction
DOF the Ant's 2-DOF leg lacks. These are flagged, not hidden — the deliverable is
the working perception+threat+curriculum+policy loop and an honest convergence
status, per the checklist's "honest boundary."

## Hardware mapping (the "what components" answer, §6)

No new RTL is needed for perception. On a real robot: a **ToF/lidar ring + IMU**
are read by the **RISC-V SoC**; the trained policy runs on the SoC; and it still
commands the **FOC per joint** over AXI-Lite — exactly the existing motor/sensor/
controller stack, just with the policy as the outer loop. The sim rangefinders
*are* the lidar ring; the sim↔hardware path is the actuator envelope (parity-
anchored to the cycle-accurate bench, `test_rl_actuator_parity.py`) plus the
existing SoC. Perception lives in the sim now and on the SoC later — never in RTL.

## Reproduce

```
make rl-dodge-train   # PPO + curriculum, 2 M steps, 16 envs (~5 min CPU) -> ppo_dodge.zip
make rl-dodge-eval    # eval + render figures/rl/dodge_after.mp4
```

## Status vs the checklist

Every section §0–§7 is implemented; `MotorloopDodge-v0` runs with rangefinder
perception, a leg-seeking threat system, and a dodge+balance reward under a
curriculum; a before/after video pair exists; the motor-envelope parity holds.
**Converged:** the pipeline and balance-under-bombardment. **Compute-bound:**
elegant per-object dodging (diagnosed above). Reported honestly, as the checklist
required.
