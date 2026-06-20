<!-- SPDX-License-Identifier: MIT -->
# Co-design realization checklist — proxies → real trained-policy co-design

Turn the co-design **machinery** (proven on proxies, `notes/codesign-architecture.md`)
into **real** morphology+control co-design: every fitness becomes a *trained policy's
SPARC return*, design evaluation is cheap via a universal policy, the attacker is a
real body in a self-play co-evolution, and the search is multi-objective + topology +
gradient-aware — with the three bottlenecks (rebuilds, disengagement, constants) fixed.

**North star:** `optimize_design`/`coevolve` rank bodies by *how well a trained policy
actually fights in them* (SPARC points won), not by static stand/clearance/mass.

## The honest boundary (built in)

- **This needs a GPU.** The local box has none (JAX/MJX absent). Phases 0–1 stand up
  the GPU/MJX foundation; nothing past Phase 0 runs on this machine. Budget real
  compute + wall-clock.
- **The crux is the proxy→real gap.** The whole point is to *measure* how wrong the
  proxies were and replace them — so every phase has a correlation/parity check, not
  just a "it runs" check.
- **Order note (#1 vs #2):** #1 (trained-return swap) is implemented first in its
  *direct, costly* form (fine-tune per candidate) to validate the concept; #2 (the
  universal policy) then makes that eval cheap. Build the trainer once (Phase 1),
  prove the swap (Phase 2), then optimize it (Phase 3). This honors the stated order
  and the real dependency.
- **Reuse, don't fork:** keep `sparc_score.py` as the one objective, `gen_robot_mjcf`
  as the one body factory, `robot.toml` as the one source. New code is the MJX env +
  trainers + search layers, not a parallel universe.

## Phase 0 — GPU / MJX foundation (prerequisite)

- [ ] Provision a GPU box (RunPod/Lambda/Vast — see `notes/`); standard CUDA image.
- [ ] `requirements-gpu.txt`: `jax[cuda12]`, `mujoco-mjx`, `brax` (PPO), `flax`,
      `optax`; pin versions. Install into a fresh venv.
- [ ] Smoke: `jax.devices()` shows the GPU; `mjx.put_model(make_model())` succeeds; a
      single `mjx.step` runs on a generated body.
- **Verify:** GPU visible to JAX; one generated `robot.toml` body loads + steps in MJX;
      record a per-step throughput baseline (steps/s for 1 vs N envs).

## Phase 1 — MJX env + RL trainer + a real baseline policy

- [ ] **`sim/robot/mjx_env.py`** — port the env to JAX-functional form: pure
      `reset/step/obs/reward` over `mjx.Data` (no Python loops/side effects), so it
      `vmap`s over thousands of envs. Start with the single-body dodge/locomotion task.
- [ ] **JAX SPARC reward:** mirror `sparc_score.step_reward` in `jnp` (the arithmetic
      is JAX-trivial; keep `sparc_score.py` as the numpy source of truth + a `_jax`
      twin, or make it backend-agnostic). Damage = force-weighted via MJX contact
      forces (`efc`/`contact` arrays in JAX).
- [ ] **Trainer:** wire **brax PPO** (or a flax/optax PPO) over the vmapped `mjx_env`.
- [ ] **Parity gate (critical):** cross-check MJX obs/reward against the CPU
      `match_env`/`combat_env` on a *fixed* model + control sequence — must agree within
      tol. Add as a test.
- [ ] Train a baseline policy on a *fixed* body; checkpoint it.
- [ ] **Constant calibration (bottleneck):** histogram the real weapon→body impact
      forces from MJX rollouts → set `DAMAGE_REF`/`RAM_REF` to a sensible percentile;
      confirm the damage term neither saturates (always 1) nor vanishes.
- **Verify:** parity passes; the baseline policy's SPARC return *rises over training*
      and produces sane behavior (closes + strikes, doesn't flee — per SPARC); throughput
      ≥ 100× the CPU env.

## Phase 2 — Extension #1: swap proxy → trained-policy return (direct)

- [ ] **`policy_fitness(design)`** (new, in a `codesign_gpu.py`): generate body →
      **warm-start fine-tune** the baseline policy a short budget K on that body → return
      mean SPARC episode return over eval episodes.
- [ ] Wire into `optimize_design.py` behind a `--fitness {proxy,policy}` flag (keep
      `proxy_fitness` as the fast/local default; `policy` is the GPU real one).
- [ ] Run CEM over designs with `--fitness policy`.
- **Verify:** CEM climbs the **real** SPARC return; the policy-optimal design *differs
      from and out-fights* the proxy-optimal design (run both in matches). Record the
      **proxy↔real rank correlation** (the headline "how wrong was the proxy" number)
      and the per-candidate wall-clock (the cost #3 removes).

## Phase 3 — Extension #2: design-conditioned universal policy (the unlock)

- [ ] **Augment obs with the design vector** (normalized) in `mjx_env.obs` — the policy
      becomes morphology-aware. (`train_mjx.sample_dr` already samples the design space.)
- [ ] **Train ONE universal policy** with per-env design sampled from the DR ranges
      (bodies randomized across the batch — use MJX model-field randomization, Phase 8).
- [ ] Switch design-fitness to the **cheap** form: eval body x against the *fixed*
      universal policy (rollout, no training) → SPARC return. (`policy_fitness` cheap path.)
- **Verify:** (a) **generalization gap** — universal-policy return across a held-out
      body sample ≈ per-body-trained policies (within an acceptable %); (b) the cheap
      eval's **rank-correlates** with Phase-2's costly retrain eval; (c) measured
      **speedup ≫ 10×**; (d) re-running `optimize_design` with the cheap fitness finds a
      comparable-or-better design far faster.

## Phase 4 — Extension #3: unify opponent → end-to-end self-play co-evolution

- [ ] **Attacker = a generated body**, not abstract params: add `attacker.toml` (reuse
      the `robot.toml` schema, e.g. a weapon-leg/spinner variant). Delete/retire
      `coevolve.ATT_PARAMS` + `_hit_on_us`/`_our_offense`/geometric `match`.
- [ ] Replace coevolve's engagement with a real **`match_env.run_match`** between the
      two bodies + their policies (start with the universal policy for both).
- [ ] **Self-play training:** realize `match_env.selfplay()` — two policies (`pol_A`,
      `pol_B`) trained against each other on the evolved bodies (alternate/league).
- [ ] **Co-evolve (body, policy) pairs:** CEM/GA over bodies, self-play PPO over
      policies; Hall-of-Fame of *(body, policy)* pairs sampled as opponents.
- **Verify:** the arms race runs on **real matches** (SPARC net from fights, not the
      proxy); both morphologies co-adapt; the Hall of Fame is sampled; co-evolution and
      the match now share **one** morphology space (no representation split).

## Phase 5 — Extension #4a: multi-objective (NSGA-II)

- [ ] Define objectives: **max** SPARC return, **min** mass, **min** $cost; constraints:
      motor envelope satisfied (`joint_torque_limit`), SPARC weight-class limit.
- [ ] Integrate **NSGA-II** (`pymoo`, or a self-contained non-dominated-sort + crowding)
      over the design vector; fitness via the universal-policy eval (Phase 3).
- **Verify:** a **Pareto front** emerges (plot survival vs mass vs cost); identify the
      knee design; show single-objective CEM is one point on that front.

## Phase 6 — Extension #4b: topology evolution (GA over `[[leg]]`)

- [ ] A **GA over the `[[leg]]` list**: mutation (add/remove leg, jitter attachment,
      toggle `is_weapon`), crossover (swap leg sets); each genome → `build_mjcf` → eval.
- [ ] Handle the action-dim change (per `prove_robot`'s mapping): warm-start when dims
      match, transfer/from-scratch when topology changes. **Flag** a morphology-agnostic
      policy (graph/transformer over limbs, à la Transform2Act) as the clean long-term fix.
- **Verify:** topologies evolve; a *non-default* topology (5-leg / repositioned weapon)
      wins on some objective; the warm-start-vs-retrain rule is respected (no silent dim
      mismatch).

## Phase 7 — Extension #4c: differentiable co-design (MJX gradients)

- [ ] Use MJX autodiff: gradient of a short-horizon SPARC-return surrogate w.r.t. the
      design params; combine with ES (**gradient-assisted ES**) given contact-noise.
- [ ] Demo on a **smoothed sub-objective** (clearance/stand) where gradients are clean.
- **Verify:** gradient steps improve fitness faster than CEM on the smooth sub-problem;
      **honestly document** where contact discontinuities break pure gradients (the ES
      fallback).

## Phase 8 — Bottlenecks (explicit fixes)

- [ ] **Per-candidate rebuild:** for parameter-DR, randomize **`mjx.Model` fields**
      across the batch (mass, dof damping, actuator `forcerange`, geom sizes) instead of
      rebuilding XML; cache distinct *topologies* only. **Verify** throughput gain vs the
      rebuild path.
- [ ] **Disengagement/cycling:** add (a) a fixed **absolute benchmark set** of reference
      opponents to track *absolute* progress (not just relative Red-Queen), (b) population
      **diversity** (novelty/fitness-sharing), (c) keep/strengthen the Hall of Fame.
      **Verify** the arms race doesn't collapse — benchmark performance trends up, not
      flat/cyclic.
- [ ] **Hand-tuned constants:** replace `DAMAGE_REF`/`RAM_REF` (Phase 1) + coevolve
      erosion/cost weights with measured-force-derived scales; run a **sensitivity
      sweep**. **Verify** conclusions are robust to ±50% perturbation of each constant.

## Phase 9 — Integration, reproducibility, honesty

- [ ] Makefile: `gpu-mjx-train`, `gpu-codesign`, `gpu-coevolve`, `gpu-selfplay`;
      `notes/gpu-runbook.md` (provision → install → run order → expected wall-clock/cost).
- [ ] MJX↔MuJoCo **parity test** in the suite; REUSE/SPDX on all new files.
- [ ] **`notes/codesign-realization-report.md`** — what converged, the proxy↔real
      correlations, the generalization gap, the speedups, the Pareto front, and the
      honest limits (topology-policy transfer, gradient-through-contact).

## Done-when

`optimize_design` + `coevolve` rank designs by a **trained policy's SPARC return** (not
proxies); a **design-conditioned universal policy** makes design eval a cheap rollout
(speedup measured, generalization gap reported); the attacker is a **generated body** in
a **self-play co-evolution** sharing one morphology space; an **NSGA-II Pareto front**,
**topology GA**, and a **differentiable-codesign** demo exist; the three bottlenecks are
fixed with verifications; MJX↔MuJoCo parity holds; and the report states honestly what
converged and what didn't — all reproducible via `make` on a GPU box.

## What NOT to do

- Don't keep a static proxy as the *final* fitness — it must be a trained-policy return.
- Don't retrain a fresh policy per candidate once the universal policy exists (that's
  the whole point of #2).
- Don't measure co-evolution *only* relatively — without the absolute benchmark, a
  rising Red-Queen can hide zero real progress (disengagement).
- Don't re-tune `DAMAGE_REF`/weights by feel — calibrate against measured forces + a
  sweep, or the rankings are arbitrary.
- Don't claim differentiable co-design "works" through contacts without showing the
  noise and the ES fallback.
- Don't skip the MJX↔MuJoCo parity gate — a wrong port silently invalidates every
  downstream result.
- Don't fork `sparc_score`/`gen_robot_mjcf`/`robot.toml` — extend the single sources.

## Implemented (results)
_(to fill in after execution on a GPU box)_
