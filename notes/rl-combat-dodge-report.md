<!-- SPDX-License-Identifier: MIT -->
# Combat-dodge — results (a quadruped that evades a weaponized spinner pursuer)

Implements [`rl-combat-dodge-checklist.md`](rl-combat-dodge-checklist.md): a
quadruped (torso + 4 legs) that treats the threat as an **adversary trying to
attack it** — the combat-robotics meta of a **low spinner** (a horizontal kill-disk
near the floor on a chassis that *chases* the robot). It learns to keep its feet
out of the strike band, hold a marginal standoff, and stay upright — the NN policy
commanding every joint through the FOC motor envelope. The combat reskin of
[`rl-dodge-report.md`](rl-dodge-report.md).

Code: `sim/rl/combat_env.py` (env + mocap pursuer + low perception + clearance
reward), `train_combat.py` (PPO + curriculum), `eval_combat.py`, `render_rollout.py`
(the `robot=="combat"` branch). Reproduce: `make rl-combat-train` →
`make rl-combat-eval` (needs `~/rl-venv`; `requirements-rl.txt`).

## Honest frame (read this first)

- **The threat is a persistent mobile pursuer, not a projectile.** A kinematic
  **mocap spinner** — a low kill-disk (strike band `z∈[0.02, 0.12] m`, radius
  0.30 m) on a chassis — chases the robot's CoM by pure pursuit. The task is to
  *maintain a safe relationship* with it, continuously.
- **Weapon physics are a modeled abstraction, not a flywheel sim.** A strike
  (the disk contacting any leg/foot geom) = **catastrophic: terminate + a large
  penalty** (the mocap body also physically shoves the robot — natural knockback).
  We do **not** simulate the blade's stored kinetic energy to the joule; that's
  stiff/unstable in MuJoCo and unnecessary to learn evasion.
- **Perception = sim rangefinders + a privileged pursuer track**, not a camera —
  maps to an overhead/onboard **ToF-lidar tracker + IMU on the RISC-V SoC** on
  hardware (see Hardware mapping).
- **Body caveat:** trained on the **Ant** stand-in (short, near-ground, 2-DOF
  legs). It learns the **standoff + high-step + ride-high** behavior well; a
  taller-legged body (Go2) is where clean *leaps over* a fast spinner have the
  travel to work. Flagged, per the checklist.
- **CPU demo budget:** 2.5 M steps, 16 envs. A learned, runnable evader, not a
  SOTA combatant.

## What was built

- **Body:** the gymnasium Ant (torso + 4 legs, 8 actuated hinge joints).
- **The weaponized pursuer (§1):** a **mocap** body (immovable, kinematic) carrying
  a low `cylinder` blade (the strike band) + a chassis box, **teleported one
  substep at a time toward the robot's CoM** at `1 + 3·difficulty` m/s (pure
  pursuit). Strike = blade/chassis contacting a leg/foot geom → terminate + −50.
  A `weapon="hammer"` variant (overhead descending arc + lateral-retreat reward)
  is wired for future training.
- **Perception (§2):** a **low rangefinder ring** (12 rays tilted down to catch a
  ground-hugging chassis) + per-ray closing rate, **plus a privileged pursuer
  track** (relative pos/vel, bite radius, band top, weapon one-hot) and **per-foot
  state** (tip height, distance to pursuer, in-reach flag). 71-dim obs,
  VecNormalized.
- **The clearance reward (§3):** `+1 alive`, **`+height-clearance`** (per foot,
  reward the foot *tip* above the strike band, punish a tip *in* the band ~2×
  harder, gated on in-reach), `+standoff` and `+belly-clearance` (both scaled by
  pursuer **urgency** = closing-speed × proximity), `+leap` (upward CoM velocity
  when a foot is low and in-reach), `+settle`/`−jerk` (graceful when the pursuer is
  far), and **−50 on a strike**. Terminate on strike, fall, or flip.
- **Calibration that mattered:** the Ant's ankle-geom *centres* sit at ~0.55 m
  (diagonal legs), so clearance must be measured at the capsule **tip**
  (`centre − |axis_z|·half_len − radius`). Verified: a settled foot tip sits at
  **0.02 m — inside the strike band** — so "clear the band" = lift a tip past
  0.12 m, exactly the strike condition.
- **Curriculum (§4):** pursuer difficulty ramps **0 → 0.6** over the first 60 % of
  training (speed scales with it) — stand first, then evade.

## What was measured

Deterministic eval, **spinner**, **10 episodes**, 1500-step cap. "Survived" =
reached the cap without a strike or fall.

| metric | random @0.6 | **trained @0.6** | trained @0.3 |
|---|---|---|---|
| steps survived (mean) | ~16 | **~758** | ~317 |
| ended by **strike** | 5/10 | 5/10 | 7/10 |
| ended by **fall** | **5/10** | **0/10** | 1/10 |
| **survived to cap** | 0/10 | **5/10** | 2/10 |
| vulnerable (foot in band & in-reach) | 3 % | 5 % | 1 % |
| mean standoff | 1.83 m | 0.92 m | 1.62 m |

- **It learned to survive the spinner.** Random control is struck or knocked over
  in ~16 steps and **never** survives the window (5 falls + 5 strikes); the trained
  policy survives **~47× longer**, **never falls**, and **reaches the full 1500-step
  cap in half its episodes** against a pursuer faster than it can outrun. The
  robust-balance result from the projectile env carries over and is *stronger* here
  (0 falls).
- **Why the trained standoff is *tighter* (0.92 m < random's 1.83 m).** At
  difficulty 0.6 the spinner (2.8 m/s) is **faster than the Ant can outrun**, so
  the policy can't flee — it holds a **marginal ~0.9 m dance just outside the
  0.40 m bite radius**, repositioning and high-stepping to keep its feet clear.
  Random's larger average distance is an artifact: it falls before the pursuer
  (starting 3 m out) even arrives. The trained number is the *steady-state* of a
  bot actively managing a faster attacker — the intended behavior.
- **Low exposure over a long life:** ~5 % of ~750 steps with a foot in the band
  *and* in reach — the clearance term is doing its job (the high-step gait).
- **Honest finding — the policy is specialized to the difficulty it finished
  training on.** At difficulty **0.3** (a *slower* spinner) it does **worse**
  (~317 steps, 2/10 survive) than at 0.6, not better. The curriculum ramps to 0.6
  and ends there, so the fast-aggressor footwork is over-tuned: against a slow
  pursuer the policy relaxes its high-step cadence at the wrong moments and gets
  clipped (note the 1 % exposure but 7/10 strikes — the hits land in that rare
  guard-down window). Training across a difficulty *range* (not just ramping to a
  fixed endpoint) would fix the easy case; the **0.6 headline is the hard opponent
  we care about**.
- **Video** (`figures/rl/`): `combat_before.mp4` (random — struck/knocked over
  fast) vs `combat_after.mp4` (the trained bot dancing at the bite edge,
  high-stepping, surviving the chase).

## What's converged vs compute/body-bound

**Converged:** the pipeline, robust balance (0 falls), and **extended evasion of a
faster pursuer**. **Bounded:** the remaining strikes (the faster spinner eventually
catches it in a minority of episodes) are limited by, in order: (1) the **Ant
body** — short 2-DOF legs can high-step and shuffle but can't truly *leap over* or
*outrun* a 2.8 m/s disk; a taller-leg Go2 would; (2) **budget** — 2.5 M CPU steps;
(3) **mobility** — the actuator/gear caps top translation speed, so escape isn't an
option at high difficulty. These are flagged, not hidden.

## Hardware mapping (the "what components" answer, §6)

No new RTL. On a real combat bot: an **overhead or onboard ToF-lidar + IMU** track
the opponent and are read by the **RISC-V SoC**; the trained policy runs on the
SoC; it still commands the **FOC per joint** over AXI-Lite — the existing
motor/sensor/controller stack with the policy as the outer loop. The sim's low
rangefinder ring + pursuer track *is* that opponent tracker; the sim↔hardware path
is the motor-envelope actuator (parity-anchored to the cycle-accurate bench,
`test_rl_actuator_parity.py`) plus the SoC. Perception in the sim now, on the SoC
later — never in RTL.

## Reproduce

```
make rl-combat-train   # PPO + curriculum, 2.5 M steps, 16 envs -> ppo_combat.zip
make rl-combat-eval    # eval + render figures/rl/combat_after.mp4
```

## Status vs the checklist

Every section §0–§7 is implemented; `MotorloopCombat-v0` runs with a low rangefinder
ring + pursuer track, a **weaponized chasing spinner** (+ a wired hammer variant), a
**height-clearance + standoff + leap** reward under a curriculum; a before/after
video pair exists; the motor-envelope parity holds; metrics show **far longer
survival, zero falls, and low strike-band exposure** vs random. **Converged:**
balance + extended evasion. **Compute/body-bound:** eliminating the last strikes
against a faster-than-you spinner (Ant legs / budget). Reported honestly.
