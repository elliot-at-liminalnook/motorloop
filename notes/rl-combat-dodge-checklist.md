<!-- SPDX-License-Identifier: MIT -->
# Combat-dodge: a quadruped that evades weaponized ground robots with height clearance

A quadruped (body + 4 legs) that treats the threat as an **adversary trying to
attack it** — the combat-robotics meta: **low spinners** (a fast horizontal
kill-disk near the floor) and **low hammer-bots** (an overhead descending arc).
It must **keep its legs/feet out of the strike band** (high-stepping / leg-tuck),
**stand off and run with ground clearance**, and **gracefully hop the plane** when
cornered — all while staying upright and commanding every joint through the FOC
motor envelope. The combat-reskin of [`rl-dodge-balance-checklist.md`](rl-dodge-balance-checklist.md).

**North star:** sense a pursuing weaponized bot → keep every vulnerable limb above
its low strike band whenever inside the bite radius, hold lateral standoff scaled
to how fast it closes, and ride/leap high to let the blade pass under — instead of
tanking hits while balancing.

## The honest boundary (built in)

- **Harder than projectile-dodge:** a *persistent, mobile, pursuing* hazard, not N
  discrete shots — the task is "maintain a safe geometric *relationship*," a
  continuous pursuit-evasion + clearance-posture problem. Expect a
  **learning/partial** policy on the CPU budget; the solid deliverables are the
  **weaponized-adversary system + clearance reward + low perception + curriculum**.
- **Weapon physics are modeled, not simulated to the joule.** A real spinner dumps
  a flywheel's KE into you; faithfully colliding that is stiff/unstable in MuJoCo.
  We model a **strike = catastrophic (terminate + large penalty) + an optional
  scripted knockback impulse** for the eval/video — honest abstraction, not a
  rigid-body weapon sim.
- **Perception = sim rangefinders + a privileged pursuer track**, *not* a camera.
  On hardware this maps to an **overhead/onboard ToF-lidar tracker + IMU on the
  RISC-V SoC** — not RTL vision.
- **Body caveat (decide in §0):** the Ant stand-in has short, weak, near-ground
  legs — it will learn standoff + high-stepping + ride-high posture, but **clean
  leaps over a fast spinner are marginal on that body**. A taller-legged quadruped
  (Go2 / custom long-leg MJCF) is the body where leap-clearance actually has the
  travel to work. Report which we used and what that bounds.
- **Reuses** the FOC motor-envelope actuator (parity-anchored, `sim/rl/actuator.py`),
  the record→render viz (torch-free), and the PPO+VecNormalize+curriculum scaffold.

## §0 — Decisions + setup

- [x] **Body:** **Ant stand-in** to bring the pipeline + reward up first (fast,
      already wired); **taller-leg quadruped** (MuJoCo Menagerie **Go2**, or a
      custom long-leg MJCF) as the target for real leap-clearance. Decide which to
      train; default Ant-now, Go2-stretch.
- [x] **Weapon archetypes:** **spinner** (primary — a low horizontal disk in a
      strike band `z∈[z_lo,z_hi]`, e.g. `[0.0, 0.12] m`, bite radius `R_w`) and
      **hammer** (variant — an overhead arc descending at a forward offset).
      Randomize the type for generalization.
- [x] **Adversary control:** **scripted-aggressive first** — pure-pursuit toward
      our CoM, weapon always live, curriculum on speed→reach→"RPM"/impulse.
      **Self-play** (a second PPO rewarded for *hits*) as the "it's really trying
      to attack" stretch.
- [x] **Code layout:** new **`sim/rl/combat_env.py`** registering
      **`MotorloopCombat-v0`**, reusing `actuator.py` + a shared perception helper;
      **keep `dodge_env.py` (projectile mode) as the A/B baseline**. Don't cram two
      threat models into one file.
- [x] **Reuse:** one motor-envelope actuator per joint; the 3-tier architecture
      (fast trainer / RTL verifier / HIL); the record→render pipeline; deps in the
      existing `~/rl-venv` / `requirements-rl.txt` (+ Menagerie note if Go2).

## §1 — The weaponized adversary (the "battlebot")

- [x] **Adversary body (MJCF, injected like the projectiles):** a driven
      **chassis** + a **spinner blade** geom (low cylinder, radius `R_w`, centered
      at `(z_lo+z_hi)/2`, half-height `(z_hi−z_lo)/2`) and a **hammer** variant
      (arm + head). Free/planar drive joints so it can chase. Params surfaced:
      `R_w`, band `[z_lo,z_hi]`, chassis speed, weapon type, impulse.
- [x] **`AdversaryController`** (scripted): **pure-pursuit** — steer chassis
      velocity toward our CoM's (optionally lead) position at the curriculum speed;
      spin the blade (visual + for knockback direction) / oscillate the hammer
      arc on a phase clock. Park/respawn at arena edge between bouts if desired.
- [x] **Strike detection:** MuJoCo contacts between the **weapon geom** and our
      **vulnerable geoms** (feet + lower-leg links + belly) → **catastrophic**:
      terminate + large penalty + (optional) a scripted **knockback impulse**
      along the blade tangent / hammer normal. Track which limb (for §3 shaping).
- [x] **Curriculum + domain randomization knobs:** chassis speed, bite radius
      `R_w`, band height, weapon type mix (spinner↔hammer), knockback impulse,
      plus friction / mass / motor-param / sensor-noise / arena-size randomization
      (robustness + sim-to-real).

## §2 — Perception (sensing a ground robot)

- [x] **Low rangefinder ring** at ~ankle height (a torso-mounted fan misses a
      ground-hugging chassis) — keep/repurpose the torso fan as a second elevation;
      per-ray + per-ray **closing rate**.
- [x] **Pursuer track (privileged → realistic tracker):** relative position
      `(x,y)`, relative velocity, heading, **weapon band `[z_lo,z_hi]`**, **bite
      radius `R_w`**, and a **weapon-type one-hot** (spinner/hammer).
- [x] **Per-foot state (makes the clearance reward observable):** each foot's
      **height**, its **horizontal distance to the pursuer**, and an **in-reach
      flag**. Without this the §3 gradient has no observable cause.
- [x] **Assemble + normalize** the obs vector (VecNormalize); document the
      dimensionality and layout in the env docstring.

## §3 — The combat-dodge reward (the core)

Helpers: `urg = closing_speed · smoothstep(REACH+0.6, REACH, d_com)` (urgency —
high when close *and* closing); `in_reach(g) = smoothstep(REACH+0.3, REACH−0.1, d_g)`.

- [x] **Term 1 — height clearance (the answer to the brief):** per vulnerable limb
      `g`, with `margin = z_g − z_hi`:
      `+k1·in_reach(g)·clip(margin,0,M)/M  −k2·in_reach(g)·max(0,−margin)`,
      with `k2 ≳ 2·k1` (punish a limb *in/below* the band harder than it rewards
      clearing). This trains the high-stepping / leg-tuck gait.
- [x] **Term 2 — standoff:** `+w2·urg·clip(d_com − REACH, 0, S)` — run away with
      margin, scaled by danger (faster pursuer → flee earlier/farther).
- [x] **Term 3 — belly clearance:** `+w3·urg·clip(h_body − h0, 0, Δ)/Δ` — ride
      high so a low disk passes under the chassis.
- [x] **Term 4 — leap the plane:** `+w4·urg·max(0, com_vz)` when a foot is **low
      and in-reach** — the literal hop over the blade.
- [x] **Term 5 — graceful + alive:** `+1 alive`, `−w_j·‖Δaction‖²` (smooth),
      `+w_s·calm·(up − 0.15·‖ω_torso‖)` with `calm = exp(−Σurg)` (settle &
      re-balance when the pursuer is far).
- [x] **Catastrophic strike:** `−K·strike` (large) **and terminate**. Same for a
      hammer strike.
- [x] **Hammer-specific shaping:** the hammer threatens from *above*, so reward
      **lateral retreat / not being under the descending arc** (timing), not
      height — a separate small term gated on `weapon==hammer`.
- [x] **Termination:** strike, fall (upright/height), or out-of-arena.
- [x] **Weights + smoothstep/urgency helpers** in one place; note that the
      standoff/clearance "potential" terms are potential-based-shaping-flavored
      (dense guidance, policy-invariant-ish) to avoid reward hacking.

## §4 — Training (SB3)

- [x] **`train_combat.py`** (or `train_dodge.py --arena combat`): PPO + VecNormalize;
      **curriculum schedule** — balance-only → slow pursuer → faster/bigger reach →
      spinner↔hammer mix → (stretch) self-play. **Longer budget** than balance
      (this is harder) — report it honestly (learning/partial on CPU). Checkpoints
      + eval callback + TensorBoard. Keep the spawn / lazy-mujoco-import heap-order
      discipline.
- [x] **Self-play (stretch):** co-train an **adversary PPO** rewarded for hits
      (alternating or league); flag as compute-bound — the real "it's trying to
      attack you."

## §5 — Visualization

- [x] **Record** the quadruped state **+ pursuer + weapon (blade angle / hammer
      phase) state** each step; extend `render_rollout.py` with a **combat branch**
      that rebuilds the scene and replays both (torch-free).
- [x] **Overlays (debug):** the **bite circle**, the **strike band**, **in-reach
      feet**, and **min foot-clearance over the blade** — so the dodge is legible.
- [x] **Before (random → clipped/knocked over) vs after (high-steps / stands off /
      leaps clear)** video pair → `figures/rl/combat_*.mp4`.

## §6 — Metrics, parity, hardware mapping, honesty

- [x] **Metrics (A/B vs random + the projectile-dodge baseline):**
      **strike rate** (strikes/ep + steps survived), **min foot-clearance over the
      band when in-reach**, **standoff-distance distribution**, **% time
      vulnerable** (a limb in-band-and-in-reach).
- [x] **Parity:** the motor-envelope actuator stays parity-anchored to the
      cycle-accurate bench (reuse `test_rl_actuator_parity.py`).
- [x] **Hardware mapping (the components answer):** pursuer tracking = an
      **overhead/onboard ToF-lidar + IMU read by the RISC-V SoC**; the policy runs
      on the SoC; it still commands the **FOC per joint** over AXI-Lite — no RTL
      vision. Document the sim↔hardware path.
- [x] **Honest report** of converged vs compute/body-bound (clean leaps over a fast
      spinner on the Ant = the flagged stretch).

## §7 — Integration

- [x] `make rl-combat-train` / `rl-combat-eval`; reuse `requirements-rl.txt`
      (+ Menagerie note); REUSE/SPDX on all new files (MJCF covered in REUSE.toml).
- [x] **`notes/rl-combat-dodge-report.md`** — the threat model, the weapon
      abstraction (honest), perception, the clearance reward, the curriculum, the
      learning curve, the before/after video, the metrics table, and the honest
      convergence status.

## Done-when

`MotorloopCombat-v0` runs: a quadruped with low + torso rangefinder perception and
a pursuer track, a **weaponized adversary** (spinner + hammer, scripted-aggressive
with a curriculum) chasing it, and a **height-clearance + standoff + leap** reward;
a policy trains (balance → pursuit → fast/mixed weapons) and a **before/after video
shows it keeping its legs clear of the strike band / standing off / leaping** while
staying upright; the motor-envelope parity holds; metrics show a **lower strike
rate and higher min foot-clearance than the projectile-dodge baseline**; and the
report states honestly what converged on the CPU budget (and on which body).
`make rl-combat-train`/`-eval` reproduce.

## What NOT to do

- Don't treat the spinner as a projectile — it's a **persistent mobile pursuer**;
  the reward is about *maintaining a safe relationship*, not reacting to shots.
- Don't keep the rigid height-hold — it forbids the hop; punish only **collapse**
  and reward **clearance over the band**.
- Don't reward standoff/leap ungated — gate on `urg` / `calm` or it runs/hops
  forever (reward hacking); cap the leap and require a stable landing.
- Don't put the clearance reward on quantities the policy can't observe — expose
  per-foot height + in-reach (§2) or it won't learn.
- Don't simulate the flywheel KE to the joule — model strike = terminate + penalty
  (+ optional impulse); be honest it's an abstraction.
- Don't use a camera on CPU/OSMesa, and don't reach for RTL vision — perception is
  sim rangefinders now / a ToF-lidar tracker on the SoC later.
- Don't claim leap-mastery on a short-legged body you didn't reach — report the
  body, the budget, and the convergence status honestly.

## Implemented (results)

Full results: [`rl-combat-dodge-report.md`](rl-combat-dodge-report.md). Reproduce:
`make rl-combat-train` → `make rl-combat-eval`.

- **Pipeline:** `MotorloopCombat-v0` (`sim/rl/combat_env.py`), `train_combat.py`,
  `eval_combat.py`, `render_rollout.py` (combat branch); `make rl-combat-train/-eval`.
- **Weaponized pursuer (§1):** a kinematic **mocap spinner** (low kill-disk band
  `[0.02,0.12] m` + chassis) chasing the CoM by pure pursuit at `1+3·difficulty`
  m/s; strike = blade↔leg contact → terminate + −50 (+ natural knockback). Hammer
  variant wired.
- **Perception (§2):** low rangefinder ring + closing rate + privileged pursuer
  track + per-foot tip-height/in-reach; 71-dim obs, VecNormalized.
- **Clearance reward (§3):** height-clearance (foot *tip* vs band — calibration
  fix), standoff + belly (urgency-scaled), leap, settle/jerk, −50 strike.
- **Training (§4):** PPO + VecNormalize, 2.5 M steps, 16 envs, curriculum 0→0.6.
- **Result (spinner, diff 0.6, n=10):** survival **~16 → ~758 steps**, **0/10
  falls** (random 5/10), **5/10 survive the full 1500-step window** (random 0/10),
  exposure 5 %, standoff 0.92 m (a tight dance just outside the 0.40 m bite — it
  *can't* outrun a 2.8 m/s spinner). **Honest finding:** the policy is specialized
  to the curriculum endpoint — at diff 0.3 it does *worse* (over-tuned footwork);
  training over a difficulty *range* would fix the easy case.
- **Viz (§5):** `figures/rl/combat_before.mp4` (random, struck fast) vs
  `combat_after.mp4` (dances at the bite edge, high-steps, survives).
- **Parity + hardware mapping (§6):** motor-envelope parity holds; pursuer track →
  ToF-lidar + IMU on the RISC-V SoC, policy on SoC, FOC per joint — no RTL vision.
- **Integration (§7):** Makefile targets; REUSE/SPDX clean.

**Honest status:** pipeline + robust balance (0 falls) + **extended evasion of a
faster-than-you spinner** converged; eliminating the last strikes is **body/budget-
bound** (Ant's short 2-DOF legs can't truly leap/outrun a 2.8 m/s disk — Go2 is the
next body) — reported, not hidden.
