<!-- SPDX-License-Identifier: MIT -->
# Combat-dodge — results (a quadruped vs a weaponized spinner: an honest skill-ladder study)

Implements [`rl-combat-dodge-checklist.md`](rl-combat-dodge-checklist.md): a
quadruped that must survive a **low spinner** (a horizontal kill-disk on a chassis
that *chases and attacks* it) — keeping legs *and body* out of the strike band,
under the FOC motor envelope. The combat reskin of
[`rl-dodge-report.md`](rl-dodge-report.md).

**Honest bottom line up front.** This is a *partial / negative* result, reported in
full because the journey is the value. Via a **skill-ladder curriculum** the robot
**learned to stand rock-solid (10/10) and to hop / high-step its feet clear of the
strike band (10/10)** — the two motor primitives the dodge needs. But **reactive
dodging of the moving spinner was not achieved on the Ant body** within the CPU
budget: the moment the spinner can reach a foot, the robot is struck (≈12/12). The
study pins down *exactly why*, which is the deliverable. The clear next step is a
**different body (Go2)**, not more Ant training.

Code: `sim/rl/combat_env.py` (oscillating mocap attacker, body-strike, the staged
reward with `hop_reward` / `lethal` / anti-flee modes), `train_combat.py` (PPO +
warm-start `--init-model` + curriculum), `eval_combat.py`, `render_rollout.py`
(combat branch). Reproduce: `make rl-combat-train` (runs the ladder) →
`make rl-combat-eval`.

## The skill-ladder (the method)

Reactive dodging asks the robot to learn two hard things at once — the *motor
skill* (lift a load-bearing foot clear while balancing on the others) and the
*timing* (when, which foot). Trained jointly under a lethal attacker, PPO collapsed
every time. So we **decompose into independently-trainable skills, each
warm-starting the next** (each is a `--tag`ged checkpoint):

| stage | scene | reward added | result |
|---|---|---|---|
| **A — Stand** | attacker parked far | strong always-on balance (`+up`, collapse floor) | **10/10 survive 1500 steps, 0 falls** |
| **B — Hop / high-step** | attacker parked | `hop_reward`: lift feet above the band while upright | **10/10, lifts tips 0.12 m+** — primitive learned |
| **H — Dodge timing** | spinner engages, **non-lethal** + anti-flee | survive strikes (−3, no terminate) so it can *learn* | balance kept; **dodging not acquired** |
| **(target) Dodge** | lethal oscillating attacker | — | **unsolved on the Ant** |

Stages A and B are clean wins. The collapse lives entirely in the transition to a
*moving lethal* threat.

## What was built

- **Oscillating attacker (the "battlebot"):** a kinematic **mocap** spinner (low
  blade disk, band `z∈[0.02,0.12] m`, + chassis) that **darts in (attack) → backs
  off (retreat) → orbits to a new bearing (reposition) → re-attacks** — a dynamic
  adversary, not a glued chaser. Commit distance and speed scale with a difficulty
  curriculum.
- **Exploit fix (the "hops on top" bug):** the first working-ish policy learned to
  **perch its torso on the chassis** for free clearance reward. Fixed: **any robot
  geom touching the weapon is a strike** (not just legs) + an **anti-mount** penalty.
- **Perception:** a low rangefinder ring + a privileged pursuer track + per-foot
  tip-height/in-reach (71-dim obs, VecNormalized). On hardware = a ToF-lidar + IMU
  on the RISC-V SoC.
- **Calibration:** clearance is measured at the capsule **tip** (`centre −
  |axis_z|·half_len − radius`); a settled foot tip sits at 0.02 m — *inside* the
  band — so "clear" means lifting a tip past 0.12 m.

## What was measured

Deterministic lethal eval, spinner, n≥10, 1200-step cap. Difficulty sets attacker
speed (`1.0+2.5·d` m/s) and commit distance.

| policy | @0.10 (parked) | @0.13 (just engages) | @0.20 | @0.30 |
|---|---|---|---|---|
| **random** | 5/10 fall · 5/10 strike | — | — | — |
| **stand (A)** | **10/10 survive** | — | — | — |
| **hop (B)** | **10/10 survive**, high-steps | — | — | — |
| **dodge (H, final)** | 10/12 survive | **0/12 survive · 12/12 strike** | 12/12 strike | 12/12 strike |

- **Stand and hop are solved.** The hopper survives indefinitely and visibly
  **high-steps its feet above the band** while balanced (`figures/rl/combat_hop.mp4`).
- **Dodging is not.** At *every* difficulty where the spinner can actually reach a
  foot (≥0.12), the robot is struck ~12/12. `figures/rl/combat_after.mp4` shows an
  honest engagement: the attacker oscillates in, the robot reacts, and is clipped.
- The trained dodge policy keeps a **standoff of ~1.9 m** — it **flees** rather than
  stand-and-dodge, and the spinner is faster, so it loses.

## Findings (why it's hard — the real deliverable)

1. **Joint learning collapses; staging is mandatory.** From-scratch dodge stalled at
   ~50-step survival. Only warm-starting stand→hop→dodge produced anything.
2. **The training metric lies here.** Episodes have no time-limit truncation, so a
   *successful* (never-terminating) policy is invisible in `ep_len_mean` — only
   failures get logged. Several "stalls" were measurement artifacts; **only truncated
   evals are trustworthy.** (This cost real iterations; documented so it doesn't
   again.)
3. **Catastrophic forgetting under a lethal attacker.** Training the lethal dodge
   *destroyed* the standing skill (a lethal-trained policy fell **8/10 with the
   attacker parked**). Dying on every mistake gives a pure-death gradient and no
   chance to reinforce good behavior.
4. **Non-lethal training (stage H) fixes the collapse, not the dodge.** Letting
   strikes be survivable (−3, no terminate) kept balance perfectly (10/10 parked)
   and let reward *rise* during engagement — but the policy still settled on
   **fleeing**, and anti-flee penalties didn't break it.
5. **The fleeing trap is partly morphology.** The Ant's four legs **splay outward**,
   each foot ~0.5 m from centre in a different direction, all load-bearing. A spinner
   from *any* angle immediately threatens a foot the robot can't lift without losing
   its support polygon — so "run" looks better than "dodge," even though the Ant is
   too slow to run.

## Why a different body, not more training

The Ant is a **poor platform for this specific task**: splayed load-bearing legs,
no leg-tuck-under-body, and a top speed below the spinner's. The likely fixes, in
order: (1) a **Go2-class body** (legs tuck *under* the torso — it can retract a leg
and keep a stable tripod, and is fast enough to reposition); (2) a **longer
harmless-marker (stage H) phase + a lethal fine-tune**, on GPU at 10-50× the steps;
(3) a faster actuator / different action space. The *methodology* (skill-ladder,
non-lethal timing stage, exploit-proof strike model, oscillating adversary) carries
over directly.

## Hardware mapping (unchanged)

No new RTL. On a real combat bot the opponent is tracked by an **overhead/onboard
ToF-lidar + IMU on the RISC-V SoC**; the policy runs on the SoC and commands the
**FOC per joint** over AXI-Lite. Perception lives in the sim now / on the SoC later.

## Reproduce

```
make rl-combat-train   # stand (1.5M) -> hop (1.2M) -> dodge/H (2.5M), warm-started
make rl-combat-eval    # figures/rl/combat_hop.mp4 (high-steps) + combat_after.mp4 (engagement)
```

## Status vs the checklist

`MotorloopCombat-v0` runs with the oscillating weaponized attacker, exploit-proof
strikes (body included), low perception, and the staged clearance reward;
before/after videos and a metrics sweep exist; the motor-envelope parity holds.
**Converged:** the pipeline, stand, hop, and the diagnosis. **Unsolved (body/budget-
bound):** reactive dodging on the Ant — the study shows precisely why and points to
Go2. Reported honestly, in full.
