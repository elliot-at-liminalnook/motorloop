<!-- SPDX-License-Identifier: MIT -->
# System understanding bundle — morphology/policy co-design

A compact map of how the codebase implements **co-design** (evolve the *body* and the
*controller* together) so it can be reasoned about outside the code. All co-design
lives in **`sim/robot/`**; it stands on the motor-provenance bench (`sim/tests/motors.py`)
and the legacy single-body RL env (`sim/rl/`). Scope here = the co-design system.

## 1. The big picture: three optimization layers + one objective

```
                         sparc_score.py  (THE objective: win the SPARC decision)
                                  ▲ used by every reward & fitness
   robot.toml ──load_spec──► spec(dict) ──overrides──► build_mjcf / build_match ──► MjModel
   (single source)              ▲                          (gen_robot_mjcf.py)        │
                                │                                                     ▼
        ┌───────────────────────┼──────────────────────────┬──────────────────────────┐
        │ INNER: control         │ MIDDLE: design            │ OUTER: populations       │
        │ train_mjx.py (RL/DR)   │ optimize_design.py (CEM)  │ coevolve.py (co-evo+HoF) │
        │ reward = SPARC step    │ fitness = proxy(stand/    │ match_env.py (self-play  │
        │ (per body)             │  clearance/mass)          │  match, SPARC score)     │
        └────────────────────────┴──────────────────────────┴──────────────────────────┘
   constraint feed: motors.py envelope → actuator force ; physical-validity settle ;
   [domain_randomization] ranges ; tournament_docs/ SPARC rules ; mass/cost penalties
```

- **Inner (control):** given a fixed body, learn a policy. `train_mjx.py` (GPU/MJX
  skeleton) with the SPARC per-step reward; `sim/rl/combat_env.py` is the CPU
  precursor (single-body dodge, SB3 PPO).
- **Middle (design):** given a task, optimize one body's continuous params with CEM
  against a fitness. `optimize_design.py`.
- **Outer (populations):** evolve our body *and* an adversary against each other with
  Hall-of-Fame archives (`coevolve.py`), and stage a two-robot scored fight
  (`match_env.py`).
- **Objective:** every reward/fitness reduces to `sparc_score.py` — maximize OUR
  Damage/Control/Aggression points, minimize the opponent's.

## 2. Lifecycle of a candidate design

| stage | where | what happens |
|---|---|---|
| **Generate** | `gen_robot_mjcf.build_mjcf(spec, overrides)` / `build_match` | `spec` (robot.toml) + an override dict → MJCF string → `MjModel`. Topology = the `[[leg]]` list (count/positions, `is_weapon`); a CEM/DR vector becomes overrides via `to_overrides`/`sample_dr`. |
| **Constrain** | `joint_torque_limit` ; `prove_robot.settle` | Actuator `forcerange` = motor envelope (`motors.py` `kt·peak_factor·rated_I`) × gear — the body is bound to a *real measured motor*. Physical validity = drop-and-settle without NaN/height-blowup. DR ranges + SPARC rules + mass/cost are soft constraints. |
| **Train policy** | `train_mjx.reward/obs` (GPU) ; `combat_env` (CPU) | Policy maps obs→joint torques through the motor envelope; reward = `sparc_score.step_reward`. *Currently a skeleton/precursor* — the trained-policy return is the intended fitness. |
| **Evaluate** | `optimize_design.proxy_fitness` ; `match_env.score` ; `coevolve.match` | Three evaluators: (a) static **proxy** (stand-within-torque + retract clearance − mass); (b) a **match** rollout → SPARC differential; (c) a **design-matchup proxy** (offense/defense → SPARC features). |
| **Select** | `cem` elites ; `coevolve` Hall of Fame ; `match_env.run_match` winner | CEM keeps the top-quartile elites and refits a Gaussian; co-evolution appends round winners to `rob_hof`/`att_hof` and samples them as opponents; a match declares a winner by SPARC points. |

## 3. Key abstractions

| concept | representation | key code |
|---|---|---|
| **Morphology** | `spec` dict (TOML); continuous design **vector** (CEM, normalized→real); **overrides** dict deep-merged onto spec; topology = `[[leg]]` list | `robot.toml`, `_deep_merge`, `to_overrides`, `build_mjcf`/`build_match` |
| **Policy** | `obs()`→action through the motor-envelope actuator; per-body in `train_mjx`; per-fighter `pol_a`/`pol_b` callables in matches | `train_mjx.obs/reward`, `MatchEnv.obs/step`, `sim/rl/actuator.MotorActuator.torque_scale` |
| **Constraints** | motor torque envelope → `forcerange`; physical-validity settle; DR ranges; SPARC weight/cost | `joint_torque_limit`, `prove_robot.settle`, `[domain_randomization]`, `sparc_score` |
| **Opponent** | two forms: an **abstract design vector** `[strike_h, reach, speed]` (coevolve) **or** a **full body B** (match_env). Also: scripted oscillating mocap (legacy `combat_env`) | `coevolve.ATT_PARAMS`/`match`, `build_match`, `combat_env._drive_adv` |
| **Scoring** | SPARC categories Damage 6 / Control 6 / Aggression 5 = 17; differential = ours − theirs; rollout→features bridges; damage is **force-weighted** (impact N) | `sparc_score.{points,differential,step_reward,*_fraction}`, `MatchEnv._damage` (`mj_contactForce`) |
| **Archive / population** | CEM Gaussian (mean+std) + elite set; Hall-of-Fame lists per side; a per-design MuJoCo-build cache | `cem` (both files), `coevolve.rob_hof/att_hof`, `coevolve._RCACHE` |

## 4. Main data flows

1. **Body build:** `robot.toml → load_spec → spec` → (`overrides` from CEM `to_overrides`
   / DR `sample_dr`) → `build_mjcf`/`build_match` → `MjModel`. Single robot keeps
   names (`FL_foot`); matches prefix `A_`/`B_`.
2. **Constraint injection:** `motors.MOTORS[motor].kt … → joint_torque_limit → actuator
   forcerange`. (The only place the electrical/motor world enters the body.)
3. **Score:** a rollout/static eval → measurables (clearance, mass, dealt/taken force,
   closing/fleeing, ram) → `sparc_score.*_fraction` → `points`/`differential` →
   scalar fitness.
4. **Optimize:** fitness → `cem` elites → new Gaussian → new vectors → loop. Co-evo
   adds: round winner → Hall of Fame → sampled as next opponent (the Red-Queen loop).
5. **Reproduce:** `make robot` runs generate→prove→smoke→optimize→coevolve→match, each
   printing a `PROVEN:` line.

## 5. Important files / functions

- `sim/robot/robot.toml` — the single provenance-tracked source (torso, leg_defaults,
  actuator motor-ref+gear, `[[leg]]` topology, `[domain_randomization]`).
- `gen_robot_mjcf.py` — `build_mjcf`, `build_match`, `_robot_xml`, `_leg_xml`,
  `joint_torque_limit`, `_deep_merge`, `load_spec`.
- `sparc_score.py` — `points`, `differential`, `step_reward`, `damage/aggression/control_fraction`.
- `optimize_design.py` — `PARAMS`, `to_overrides`, `proxy_fitness`, `_retract_clearance`, `cem`.
- `coevolve.py` — `ATT_PARAMS`, `robot_summary` (+`_RCACHE`), `match`, `robot_fitness`,
  `attacker_fitness`, `cem`, Hall-of-Fame loop in `main`.
- `match_env.py` — `MatchEnv` (`_classify`, `_damage`, `_contact_force`, `_closing`,
  `obs`, `step`, `score`), `run_match`, `selfplay`.
- `train_mjx.py` — `make_model`, `sample_dr`, `obs`, `reward`, `smoke`, `train`, `--selfplay`.
- `prove_robot.py` — `VARIANTS`, `settle`, warm-start-vs-retrain mapping.
- deps: `sim/tests/motors.py` (`Motor`, `MOTORS`, `kt`, `stall_torque_nm`),
  `sim/rl/actuator.py` (`MotorActuator.torque_scale`), `sim/rl/combat_env.py`
  (`CombatDodgeEnv`), `tournament_docs/` (SPARC rules).

## 6. Assumptions, bottlenecks, extension points

**Assumptions (load-bearing):**
- **Proxy-for-real:** every local fitness is a *stand-in* — static `proxy_fitness`
  (stand/clearance/mass) and the geometric co-evo `match`/`_hit_on_us` replace the
  *trained-policy SPARC return*. The whole system is validated on proxies, not on a
  policy that actually fights. This is the single biggest gap.
- **No GPU here:** `train_mjx`/`selfplay` are skeletons; MJX/JAX absent. Real
  throughput and the universal/self-play policies are future.
- **Asymmetric opponent:** the co-evo attacker is 3 abstract params (`strike_h/reach/
  speed`); the match opponent is a full body. They are not the same representation.
- **Coarse Control/Aggression:** at the design layer control is held at 0.5 and
  aggression is an agility proxy; only the match measures them from motion/contact.
- **Hand-tuned constants:** `DAMAGE_REF`/`RAM_REF`, the co-evo erosion/cost weights
  (which needed rebalancing to avoid the disengagement regime), `STAND_TARGET`.
- **DR over params, not topology:** part-swaps within a range are free; leg count is a
  regen + retrain. Single weapon-leg, downward spear.

**Bottlenecks:**
- Each design eval rebuilds an `MjModel` from XML (mitigated only in coevolve via
  `_RCACHE`); fine locally, but the per-candidate MuJoCo build is the inner cost.
- Co-evolution is **regime-sensitive** — it collapses to disengagement/cycling
  without careful cost balancing + the Hall of Fame.
- The proxy→policy gap means none of the "PROVEN" results are about *fighting skill*
  yet — they prove the *machinery*, not the outcome.

**Highest-value extension points:**
1. **Swap proxies → trained-policy return** (the documented `proxy_fitness` /
   `engage` swap) once on GPU — turns the machinery into real co-design.
2. **Design-conditioned universal policy:** put design params in the obs + DR (the
   `sample_dr` hook exists) → one policy covers the design range → design eval becomes
   a cheap rollout, not a retrain. The key efficiency unlock.
3. **Multi-objective:** replace CEM with NSGA-II → Pareto front (survival vs mass vs
   cost under motor + SPARC constraints).
4. **Unify the opponent representation** (attacker as a generated body in coevolve, so
   co-evolution and the match share one morphology space → end-to-end self-play co-evo).
5. **Topology evolution:** GA over the `[[leg]]` list (leg count/placement, weapon
   position), not just continuous params.
6. **Differentiable co-design:** MJX is JAX → gradients of SPARC return w.r.t. design
   (contact-noisy, so ES-assisted).
7. **Richer actuator models in the body:** joint `stiffness` (parallel-elastic spring)
   exists; add the pneumatic/latch energy-store as a custom force for the fast-dodge.
