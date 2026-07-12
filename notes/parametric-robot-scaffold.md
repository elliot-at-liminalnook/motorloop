<!-- SPDX-License-Identifier: MIT -->
# Parametric robot scaffold — `robot.toml → MJCF`, modify-and-retrain proven

A provenance-tracked, single-source body generator so "swap a part and retrain" is a
config edit + a fast (GPU) run — the flexible base for the Go2-scale dodge work and
the custom leg-weapon battlebot ([`motorloop-battlebot-design`](../) memory). Built
to answer "how easy is it to modify the body and retrain?": **easy for parameter
swaps (warm-start), cheap-from-scratch for topology swaps — and free within a
domain-randomized range.**

Code (`sim/robot/`): `robot.toml` (the single source), `gen_robot_mjcf.py`
(toml→MJCF, MJX-native), `prove_robot.py` (the proof), `train_mjx.py` (GPU-ready
training skeleton + local smoke), `optimize_design.py` (co-design over the parameter
space), `sparc_score.py` (the SPARC judging objective), `coevolve.py` (adversarial
co-evolution). Run: `make robot`.

## Why this over a Go2-specific Isaac port

Past small tweaks, your design (5 legs, weapon-legs, retract DOF, spring/pneumatic
actuators) is a big departure from a stock Go2 — so we generate a **parametric**
body with **Go2-scale defaults** (physically-valid legs-under-body start) instead of
marrying Go2's meshes/12-DOF config. It targets **MJX**, which runs the generated
**MJCF directly** — the same format we already hand-edit — so it stays maximally
moddable. (Isaac Lab is the heavier alternative; USD is harder to hand-mod.)

## What was proven (`make robot`)

Five part-modifications generated, loaded, and physics-checked (settle without
exploding); action dim decides warm-start vs retrain:

| variant (a part mod) | bodies | act dim | physically valid | retrain? |
|---|---|---|---|---|
| default (Go2-scale quad) | 14 | 12 | ✅ | baseline |
| **spring-assist** (parallel-elastic) | 14 | 12 | ✅ | **warm-start** |
| **long-legs** | 14 | 12 | ✅ | **warm-start** |
| **heavy + stronger motor** | 14 | 12 | ✅ | **warm-start** |
| **5-leg weapon** (topology) | 17 | 15 | ✅ | from-scratch (12→15) |

- **Parameter swaps keep the action dim → warm-start** the existing policy (the
  skill-ladder move), minutes on GPU.
- **Topology swaps change it → retrain** from scratch — still minutes on GPU sim.
- Spot-check: the spring-assist override moved max joint stiffness 0 → 12 N·m/rad
  (the passive dodge-assist spring actually appears in the model, and that variant
  settles higher).
- **Provenance tie-in:** the actuator force limit is derived from the profiled motor
  envelope (`sim/tests/motors.py`) × gear — the current ST3215-HS model uses
  1.961 N·m at the servo output and a 3:1 abstract reduction — so
  the body is linked to the *measured* motors, same as `derive_params.py` links the
  RTL to the plant.

## Co-design: optimizing the parameter space itself (`optimize_design.py`)

Beyond training control on a fixed body, you can **optimize the body**. A
dependency-free **CEM** (cross-entropy method) outer loop proposes designs over the
continuous params (thigh/calf length, gear, spring stiffness, torso mass); each is
generated and scored by a **physics-based proxy fitness** in MuJoCo — stand within
the motor torque + retract a foot high (dodge clearance) − mass. Proven locally
(no GPU): **fitness 1.38 → 2.80 in ~3 s**, and the optimizer found a *sensible*
dodge-body — raised the gear (more holding torque), **added a parallel-elastic
spring (0 → 5.7 N·m/rad, the passive dodge-assist)**, shortened the calf, and
minimized torso mass.

This is the **co-optimization** structure: CEM/CMA-ES outer (design) + control
inner. The proxy stands in for the real objective; on GPU you **swap
`proxy_fitness` for the trained policy's dodge return** → true morphology+control
co-optimization. Pair it with a DR-conditioned *universal* policy (below) and each
design evaluation is a cheap rollout, not a retrain — the outer search runs fast.
For the weight/cost trade-off, swap CEM for NSGA-II to get a Pareto front
(survival vs mass vs cost, under the motor + SPARC constraints).

## The objective: win the SPARC decision (`sparc_score.py`)

Every reward + fitness references one objective — **maximize OUR points, minimize the
OPPONENT's** across the SPARC Damage/Control/Aggression criteria (6/6/5 = 17 pts,
`tournament_docs/` Judging Guidelines v1.3) — not "just survive." Two rules from the
guidelines are baked in: **aggression credits only translational movement *toward*
the opponent — fleeing and sitting-still score zero** (so the fleeing policy our RL
fell into would *also* lose on the scorecard, not just fail to dodge), and **damage
is graded dealt-vs-taken** (avoiding and dealing damage are one currency).
`step_reward(dealt, taken, closing, fleeing, control)` is the per-step
SPARC-differential reward; `train_mjx.py:reward()` uses it (the weapon-leg body makes
the `dealt` term real). This is the principled fix to the fleeing trap.

## Adversarial co-evolution: an evolving attacker (`coevolve.py`)

A SEPARATE harness co-designs TWO populations against each other — our robot
(maximize SPARC net) and an attacker (strike height / reach / speed; minimize our
net). Each round evolves one side against a **Hall of Fame** sample of the other (the
archive is what stops co-evolution cycling/forgetting — the classic Red Queen
failure). Proven locally: the SPARC net oscillates as each side adapts, both the
robot's clearance and the attacker's strike height climb (co-adaptation), and the
robot evolves *sensibly* (light + strong + high-clearance when it can compete).
**Honest finding:** the current design space ends at a **negative** mean SPARC net vs
the evolved attacker pool — i.e. the harness reveals the robot needs the weapon-leg +
faster actuators to win an arms race, exactly the design direction.

## The self-play match (`match_env.py`)

Where the co-evolution's geometric proxy becomes a real fight: **both robots are full
controllable bodies in one scene** (`build_match`), each with a weapon-leg. Damage is
**force-weighted** — `mj_contactForce` on each weapon→opponent-body contact, so a hard
spear thrust scores far more than a glancing touch (matching SPARC's severity tiers,
trivial→massive); robot-to-robot **body** contact force feeds a **ram → control**
signal (shoving the opponent around). The per-step reward and the final match score
are the **SPARC differential** (our points − theirs), so each policy is trained to
*win the decision*. Proven locally: the two-robot scene builds (A/B act=15 each),
weapon→body damage classification is correct ((1,0)/(0,1)/(0,0)), overlapping torsos
register ram impact force, closing/aggression reads right, and a random-vs-random
match scores and declares a winner (a symmetric 6-6 tie, as expected). `selfplay()` is the GPU-target two-policy PPO loop (league vs
the coevolve Hall of Fame); `train_mjx.py --selfplay` dispatches to it. On GPU: vmap
`MatchEnv` in MJX, train `pol_A`/`pol_B` on `sparc_score.step_reward`.

## The feature that makes part-swaps free: domain randomization

`train_mjx.py:sample_dr()` samples the `[domain_randomization]` ranges (torso mass,
link lengths, gear, spring stiffness) into generator overrides. Train across the
range on GPU (vectorized, free) → one robust policy that already covers the part
space → **swap a part within range with NO retrain.** Topology DR (leg count) is a
regen, not a field-randomization.

## Status / next

Generator + proof + env are wired and validated locally (MuJoCo 3.9). The MJX PPO
loop is a **GPU-target skeleton** (JAX/MJX absent here); `train_mjx.py --smoke`
proves the env (obs 35-dim, actuators move the body, DR body loads) so it's ready to
run the moment a CUDA box is up (`pip install "jax[cuda12]" mujoco-mjx brax`). Next:
plug the `combat_env.py` dodge perception + oscillating attacker + the staged reward
into this body and train the dodge on a morphology that can actually retract a leg.
