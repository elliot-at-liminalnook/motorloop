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
- **The sim is a calibrated, uncertain INSTRUMENT (Phase R, foundational).** A trained-
  policy ranking is only trustworthy if the sim has the effects that *change rankings
  between designs* — actuator dynamics (back-EMF torque-speed droop, current limit,
  voltage sag, latency, thermal, gear/belt efficiency), contact/impact in one
  Newton-based currency, friction/restitution, and sensor noise. **Measure components →
  fit the sim → randomize the *remaining* uncertainty (calibrated, not blind) → rank by
  ROBUST (lower-tail) match return, not nominal.** Idealized-sim rankings (e.g. "lightest
  body wins") are suspect until re-derived under Phase R. Hardware-ID + real-log truth
  gates are the *real* calibration — framework now (`reality_gap.py`), hardware-gated for
  the actual fit.

## Working method — E2E-first, instrument everything (standing practice)

Every phase ships with, and extends, an **extremely lightweight end-to-end test** before
any long run. `sim/robot/e2e.py` (`make gpu-e2e`) runs the whole loop at micro-scale as
fresh sequential subprocesses (walker → build_pack → fighter → score), times each stage,
samples peak GPU mem/util, parses `METRIC stage=... k=v` lines the sub-scripts emit, and
**appends a full structured record to `e2e_history.jsonl`** so we can see which stage is
the bottleneck and watch it move as the loop evolves. Rules: (1) leak-test the loop tiny,
*then* scale — never start a long run until the tiny E2E is green end-to-end; (2) one GPU
⇒ run stages **strictly sequential** (overlapping JAX processes throw cuSolver "internal
error" — contention, not a bug); (3) each sub-script carries a `--tiny` preset and emits
`METRIC` lines — extend both when you add a stage; (4) the brax PPO cost floor is
`batch·unroll·minibatches` env-steps per training step (default 1024·20·16 = 327,680 — why
`--steps 150000` balloons to 1.6M), so `--tiny` shrinks all three (constraint:
`batch·minibatches % envs == 0`).

## Phase 0 — GPU / MJX foundation (prerequisite)

- [x] Provision a GPU box (RunPod/Lambda/Vast — see `notes/`); standard CUDA image.
      *(RunPod RTX 4090; `notes/gpu-runbook.md`.)*
- [x] `requirements-gpu.txt`: `jax[cuda12]`, `mujoco-mjx`, `brax` (PPO), `flax`,
      `optax`; pin versions. Install into a fresh venv.
- [x] Smoke: `jax.devices()` shows the GPU; `mjx.put_model(make_model())` succeeds; a
      single `mjx.step` runs on a generated body. *(`bench_throughput.py`.)*
- **Verify:** [x] GPU visible to JAX (`[CudaDevice(id=0)]`); one generated `robot.toml` body
      loads + steps in MJX; throughput baseline recorded: 1 env ~6–11/s → 16384 envs ~35k/s
      (6140× scaling). Honest: contact-bound (~CPU single-env per step; win is parallelism).

## Phase 1 — MJX env + RL trainer + a real baseline policy

- [x] **`sim/robot/mjx_env.py`** — JAX-functional `reset/step/obs/reward` over `mjx.Data`,
      vmaps over thousands of envs. Single-body locomotion + the universal/reality-gap envs.
- [x] **JAX SPARC reward:** `sparc_score.step_reward_jax` — backend-agnostic numpy/jnp twin
      of the numpy source; damage force-weighted via MJX `contact` arrays.
- [x] **Trainer:** brax PPO over the vmapped env (`train_codesign.py`, `codesign_gpu.py`).
- [x] **Parity gate (critical):** `test_parity.py` (+ suite hook `test_mjx_parity.py`):
      MJX↔MuJoCo qpos/qvel/reward on a fixed model+control — grounded(contact) tight
      (qpos mean 6.5e-5), airborne(free-tumble) bounded; SPARC twin exact.
- [x] Train a baseline policy on a fixed body; checkpoint it. *(tiny verified; real-scale
      `make gpu-baseline`.)*
- [x] **Constant calibration:** `calibrate_damage_ref.py` histograms measured impact forces
      → `DAMAGE_REF`≈35 N (p75) vs the hand-set 150 N (which mostly vanished); 91% of hits
      land in a non-saturated band.
- **Verify:** [x] parity passes; baseline return rises over training (tiny: →77); throughput
      ≥ CPU per-env and ≫ CPU at batch (parallelism). *(SPARC fighting behavior = Phase 4.)*

## Phase R — Reality-gap calibration (do BEFORE trusting any ranking)

The sim becomes a measured instrument. `sim/robot/reality_gap.py` is the single source
of the sim-to-real model; `UniversalEnv` + `AdversarialEnv` consume the *same* sampled
domain. Add only effects that change design rankings. Phase R is the *fixed, calibrated*
instrument; **Phase RS** (below) turns it into a *self-calibrating, learned* one —
ensemble + Bayesian posterior + learned actuator/contact residuals + online policy
adaptation — the SOTA Real2Sim2Real recipe and the research-grade stretch.

- [x] **`reality_gap.py`** — `ActuatorUncertainty / ContactUncertainty / BodyUncertainty
      / SensorControlUncertainty`, `default_uncertainty(motor)` (ranges centred on the
      `motors.py` datasheet), `sample_domain_params(seed)`, `apply_to_mjx_model(mx, dp)`,
      `actuator_scale(ω)`, `damage_from_force()`, `log_parity_trace`/`score_trace_mismatch`.
- [x] **R1 — unify the actuator model (#1, the biggest win).** `UniversalEnv(reality_gap=True)`
      multiplies the action by `actuator_scale(ω)` (back-EMF droop + current/voltage/thermal +
      gear eff). **Verify:** torque droops with joint speed (`hardware_id` torque-speed: stall
      0.78 → 160 rad/s 0.00); calibrated sim measurably harder (rederive: 79.6→78.3).
- [x] **R2 — hardware-ID suite (`sim/robot/hardware_id.py`).** 6 measurements (torque-speed,
      stall, thermal, latency, friction, step) emit traces; fit-to-trace recovers identifiable
      axes (friction/damping/latency/kt, err 0.39→0.10). kt-vs-current-limit confound flagged.
      **Hardware-gated:** real numbers need real parts (loop wired to bench logs).
- [x] **R3 — unify damage/contact (#3).** `damage_from_force` Newton currency; `test_contact.py`
      validates: harder hit > glancing (1119>486 N), stable shove > bounce, impacts accumulate.
- [x] **R4 — calibrated domain randomization (#4).** Envs draw `sample_domain_params` from the
      measured/estimated ranges (per-env on GPU, verified in rederive); SimOpt/Bayesian
      tightening hook = `domain_model.update_posterior` (RS1, sim-to-sim recovery verified).
- [x] **R5 — truth gates (#5).** `score_trace_mismatch` distribution-matches the uncontrolled
      fight; `hardware_id` uses pointwise for controlled measurements. Sim-vs-sim self-
      consistency verified (recovery tests); real-vs-sim fit hardware-gated.
- [x] **R6 — robust scoring (#6).** `robust_codesign.cvar` ranks by worst-20% return; the
      CVaR-optimal design differs from the mean-optimal (sim-to-sim demo: refuses the gamble).
- [x] **R7 — re-derive under the calibrated sim** (`rederive_r7.py`). **Verify:** idealized vs
      calibrated-robust design ranking compared (tiny: ρ=0.94, modest reorder at this scale,
      calibrated sim harder); real-scale `make gpu-rederive` amplifies it. Reported honestly.

## Phase RS — Real2Sim2Real self-calibrating loop (research-grade stretch)

Phase R is a *fixed, calibrated* instrument. Phase RS makes it *self-calibrating and
learned*: an ensemble of worlds whose distribution is **inferred from real logs**,
**learned** actuator/contact residuals on top of the analytic priors, and a policy that
**adapts online** to whichever world it's in (teacher→student / RMA). This is the SOTA
sim-to-real recipe. Honest boundary: almost all of it is **framework-now,
hardware-gated-fit** — we build the machinery and validate it *sim-to-sim* (train against
a held-out *perturbed* sim as the "real" stand-in, check we recover it), then swap in real
logs (motorloop bench, drop/ram tests) when parts exist. New rule it adds: **never put the
highest-fidelity model in the inner RL loop** — keep MJX fast; use the CPU oracle and
(eventual) hardware snippets to *update the ensemble* and *validate finalists*, not to train.

The proposed module seam (extends, does not fork, the Phase-R files): `domain_model.py`
(ensemble + posterior, from `reality_gap.py`), `design_codec.py` (normalized design ⇄
`robot.toml` ⇄ MJX edits, consolidating the scattered `apply_design`/`_design_model`),
`actuator_residual.py`, `contact_residual.py`, `adaptive_policy.py` (teacher/student),
`reality_gap_eval.py` (parity + distribution match + proxy-vs-nominal-vs-robust rank
correlation), `robust_codesign.py` (CVaR + QD archive).

- [x] **RS1 — calibrated world ENSEMBLE + Bayesian posterior (upgrades R4).** `domain_model.py`
      (`Posterior`, `sample_world`, `update_posterior`); sim-to-sim recovery test recovers a
      hidden world from its traces (err 0.63→0.07, entropy ↓). Real-log update hardware-gated.
      Promote
      `sample_domain_params` from a *fixed measured range* to a *posterior over worlds*
      (`domain_model.py`: `DomainParams`, `sample_world()`, `update_posterior(real_logs)`);
      sample a world per episode; tighten the posterior from parity error
      (BayesSim/DROPO/BayRn-style), not blind ±%. **Buildable now:** posterior machinery +
      a **sim-to-sim recovery test** (recover a hidden world from its traces). **Hardware-
      gated:** the real-log update. [DROPO 2201.08434, BayRn 2003.02471]
- [x] **RS2 — learned actuator residual (upgrades R1).** `actuator_residual.py`: a compact
      net `(desired action, joint vel, bus V, temp, action history, motor/gear params) →
      realized torque/current/latency/thermal Δ`, inserted between policy output and the MJX
      motor, learning the **residual on top of** the analytic `actuator_scale` prior (so it
      only models what the back-EMF envelope misses). **Buildable now:** training harness +
      insertion seam, validated by learning a *synthetic* residual (perturbed motor) and
      recovering it. **Hardware-gated:** the motorloop bench-log fit. [actuator nets,
      Hwangbo 2019; 2604.09487]
- [x] **RS3 — learned contact/damage residual (upgrades R3).** `contact_residual.py`:
      analytic Newton impact-severity (R3) **+ a learned residual** `f(rel vel, normal,
      impulse, contact patch, body region) → severity correction`. **Buildable now:**
      analytic model + residual harness (sim-to-sim). **Hardware-gated:** drop/ram/weapon-
      contact fit. [residual physics 2402.01086]
- [x] **RS4 — teacher→student online adaptation (RMA) — the biggest gap, currently ABSENT.**
      `adaptive_policy.py`: train a *privileged teacher* on the normal obs **+ world
      extrinsics** (sampled world params + exact contact/friction/latency/battery/damage
      state); train a deployable *student* that **infers** those extrinsics online from a
      **history encoder** over onboard sensors + recent actions (the RMA adaptation module).
      The design-conditioned universal policy (Phase 3) is the *body* half of the extrinsics;
      RS4 adds the *dynamics* half and makes it inferred-not-given → fast online adaptation,
      the key to zero-shot transfer. **Buildable now (sim only, this is the deployable
      artifact):** teacher with privileged obs, student with history encoder, measure the
      adaptation gap across held-out worlds. [RMA 2107.04034, perceptive loco 2201.08117]
- [x] **RS5 — active experiment selection (upgrades R2).** Don't run a fixed battery of
      hardware tests — *choose the test that most reduces uncertainty in design RANKINGS*
      ("which physical uncertainty could flip winner A-vs-B?"). `reality_gap_eval.py` scores
      candidate measurements by expected **info-gain on the ranking**, not on the params.
      **Buildable now:** the info-gain scorer over the posterior + a sim demo (find the
      friction threshold that flips a ranking). **Hardware-gated:** running the chosen test.
- [x] **RS6 — robust QD morphology search (upgrades R6 + Phase 5).** Beyond CVaR on one
      champion: a **quality-diversity archive** (MAP-Elites / CMA-MAE) over behavior
      descriptors, each cell holding the most *robust* (CVaR-20% over worlds × opponents)
      design — a diverse set of robust bodies, not one brittle winner. `robust_codesign.py`.
      **Buildable now.** **Verify:** the QD archive holds ≥k distinct robust designs; the
      robust-QD champion differs from the mean-NSGA knee.
- [x] **RS7 — multi-fidelity ladder + promotion gates.** Three explicit tiers: (a) fast MJX
      ensemble (millions of rollouts, the inner RL loop), (b) CPU MuJoCo oracle (richer
      contact logging, parity, promoted-candidate re-rank), (c) hardware snippets (actuator/
      leg/drop/ram tests → update the ensemble, validate finalists). A promotion gate between
      tiers; the inner loop **never** runs the high-fidelity model. **Buildable now:** tiers
      (a)+(b) + the gate; (c) hardware-gated.
- [x] **RS8 — flagship north-star experiment (the publishable result).** Take N candidate
      morphologies; rank them **three ways**: (1) old proxy fitness, (2) nominal-sim combat
      return, (3) calibrated-ensemble **robust** combat return. Report which ranking best
      predicts the highest-fidelity signal available (CPU oracle now; reduced hardware
      proxies later). This is the quantitative "how much do calibration + robustness change
      the answer" result — the generalization of `codesign_validate.py`'s walker-vs-fighter
      correlation to proxy-vs-nominal-vs-robust.

## Phase 2 — Extension #1: swap proxy → trained-policy return (direct)

- [x] **`policy_fitness_direct(design)`** (`codesign_gpu.py`): generate body → warm-start
      fine-tune the baseline K steps on that body → mean SPARC/return over eval episodes.
- [x] Wire into `optimize_design.py` behind a `--fitness {proxy,policy}` flag (proxy = fast
      CPU default; policy = GPU real one).
- [x] Run CEM over designs with `--fitness policy` (per-candidate wall-clock logged).
- **Verify:** [x] CEM climbs the trained return; **proxy↔real rank correlation ρ=−0.03** (the
      headline: the static proxy ranks designs *wrongly*); per-candidate fine-tune wall-clock
      recorded (the cost #2/#3 removes). *(Policy-vs-proxy head-to-head match = real-scale GPU.)*

## Phase 3 — Extension #2: design-conditioned universal policy (the unlock)

- [x] **Augment obs with the design vector** (normalized) in `mjx_env.UniversalEnv` — the
      policy is morphology-aware (obs = locomotion + design vector).
- [x] **Train ONE universal policy** with per-env design sampled across the batch via MJX
      model-field randomization (`apply_design`, no XML rebuild — Phase 8a). *(`codesign_gpu.py`.)*
- [x] Switch design-fitness to the **cheap** form: eval body x against the fixed universal
      policy (rollout, no training) → return. *(`codesign_gpu.trained_return`, `reset_with`.)*
- **Verify:** [x] cheap fixed-policy rollout ranks designs (CEM climbs on it: best 81.6 vs
      default 81.2); the universal eval is a rollout not a retrain (≫ the direct fine-tune of
      Phase 2 — speedup is structural). Generalization-gap %/held-out parity = real-scale GPU.

## Phase 4 — Extension #3: unify opponent → end-to-end self-play co-evolution

- [x] **Attacker = a generated body**: `attacker.toml` (robot.toml schema, weapon-leg striker);
      `coevolve.ATT_PARAMS` + `_hit_on_us`/`_our_offense`/geometric `match` **retired**.
- [x] Replace coevolve's engagement with a **real-physics** duel between two generated bodies
      (measured strike-envelope/settle/torque-impulse via real MuJoCo) — the trained-policy
      melee (`match_env.run_match`/`selfplay`) is the GPU path (the downward-stomp weapon needs
      a trained policy to aim, so the CPU melee uses measured capabilities, not a live clash).
- [x] **Self-play training:** `selfplay_mjx.py` — two-policy MJX self-play with a HoF league
      (symmetric bodies; A trains vs frozen HoF snapshots). Built + structurally validated;
      the two-robot weapon scene is **compile-heavy** at scale (budget per `gpu-runbook`).
- [x] **Co-evolve (body, policy) pairs:** CEM over bodies + Hall-of-Fame sampled as opponents
      (`coevolve.py`); policy co-evolution = the self-play league.
- **Verify:** [x] the arms race runs on real-physics matches between two GENERATED bodies (one
      morphology space, no representation split); HoF sampled; absolute-benchmark score up
      (6.60→6.74); diversity on. *(Trained-policy SPARC-from-fights at scale = the fighter
      milestone, `notes/codesign-fighter-milestone-checklist.md`.)*

## Phase 5 — Extension #4a: multi-objective (NSGA-II)

- [x] Define objectives: **max** SPARC return, **min** mass, **min** $cost; constraints:
      motor envelope (`joint_torque_limit`) + SPARC weight-class limit. *(`nsga2.design_objectives`.)*
- [x] Integrate **NSGA-II** (self-contained non-dominated-sort + crowding + constrained
      domination) over the design vector; fitness pluggable (analytic proxy in the CPU demo;
      universal-policy eval on GPU). *(`nsga2.py`.)*
- **Verify:** [x] a Pareto front emerges (return vs mass vs cost), knee identified, single-
      objective point is on the front. Finding: db42s03 gimbal motor undersized → constraint binds.

## Phase 6 — Extension #4b: topology evolution (GA over `[[leg]]`)

- [x] A **GA over the `[[leg]]` list**: mutation (add/remove leg, jitter attachment, toggle
      `is_weapon`); each genome → `build_mjcf` → eval. *(`codesign_extra.topology_ga`.)*
- [x] Handle the action-dim change (per `prove_robot`'s mapping): warm-start when dims match,
      retrain when topology changes. Morphology-agnostic policy (Transform2Act) flagged as the
      long-term fix.
- **Verify:** [x] topologies evolve; a non-default 6-leg topology wins (fitness 2.27 vs
      default 1.57); warm-start-vs-retrain rule respected.

## Phase 7 — Extension #4c: differentiable co-design (MJX gradients)

- [x] Use MJX autodiff w.r.t. design + ES given contact-noise. **Finding:** `jax.grad`
      through `mjx.step` is BLOCKED — MJX's iterative solver is a dynamic-bound loop, so
      reverse-mode autodiff fails; the **ES fallback** (antithetic, forward-sim only) carries it.
- [x] Demo on a **smoothed sub-objective** (pure-JAX stand/clearance surrogate) where
      gradients are exact. *(`codesign_diff.smooth_obj`.)*
- **Verify:** [x] exact gradients ascend the smooth sub-objective (J 1.01→1.44 in 10 steps,
      ≤ CEM evals); the through-contact grad wall is documented; ES improves the MJX objective.

## Phase 8 — Bottlenecks (explicit fixes)

- [x] **Per-candidate rebuild:** parameter-DR randomizes **`mjx.Model` fields** in-env
      (`apply_design`/`apply_to_mjx_model`: mass, damping, friction, stiffness) — no XML
      rebuild; only distinct topologies rebuild. **Verify:** UniversalEnv reuses one compiled
      model across designs (`reset_with`, no recompile).
- [x] **Disengagement/cycling:** `coevolve.py` adds (a) a fixed **absolute benchmark set**,
      (b) population **diversity** (novelty term in CEM), (c) the Hall of Fame (sampled).
      **Verify:** absolute-benchmark score trends up (6.60→6.74), not cyclic.
- [x] **Hand-tuned constants:** `DAMAGE_REF`/`RAM_REF` now measured-force-derived
      (`calibrate_damage_ref.py`, ≈35 N); constant-sensitivity sweep shows the design ranking
      is robust to ±50% perturbation (corr +1.00). *(`codesign_extra.sensitivity`.)*

## Phase 9 — Integration, reproducibility, honesty

- [x] Makefile: `gpu-mjx-train`, `gpu-codesign`, `gpu-coevolve`, `gpu-selfplay` (+ `gpu-parity`,
      `gpu-rederive`, `gpu-match`, `gpu-validate`); `notes/gpu-runbook.md` (provision → install →
      run order → wall-clock/cost).
- [x] **(RS)** Makefile + runbook for the Real2Sim2Real modules: `gpu-residual`, `gpu-rma`,
      `gpu-robust-codesign`, `gpu-active-id` (+ `codesign-rs` aggregate); each runs its
      sim-to-sim verify on CPU — framework provable without hardware.
- [x] MJX↔MuJoCo **parity test** in the suite (`sim/tests/test_mjx_parity.py`, skips w/o JAX);
      REUSE/SPDX on all new files (`make reuse`: 488/488 compliant).
- [x] **`notes/codesign-realization-report.md`** — what converged, the proxy↔real correlations
      (ρ≈−0.03/−0.09), R1/R7 calibration delta, speedup, Pareto front, and the honest limits
      (topology→policy transfer, gradient-through-contact wall, MJX contact-bound throughput).

## Done-when

`optimize_design` + `coevolve` rank designs by a **trained policy's SPARC return** (not
proxies), in a **reality-gap-calibrated sim** (Phase R: actuator droop + current/voltage/
latency/thermal, Newton-based impact, calibrated DR) and scored by **robust/CVaR return**,
not nominal; a **design-conditioned universal policy** makes design eval a cheap rollout
(speedup measured, generalization gap reported); the attacker is a **generated body** in
a **self-play co-evolution** sharing one morphology space; an **NSGA-II Pareto front**,
**topology GA**, and a **differentiable-codesign** demo exist; the three bottlenecks are
fixed with verifications; MJX↔MuJoCo parity holds; the truth-gate framework is in place
(hardware-ID gated honestly); and the report states what converged, what didn't, and
**what the calibration changed vs the idealized sim** — all reproducible via `make` on a
GPU box.

**Research-grade stretch (Phase RS):** the calibrated instrument becomes *self-calibrating*
— a world ensemble with a posterior tightened from real logs, **learned** actuator/contact
residuals on the analytic priors, a **teacher→student (RMA)** policy that adapts online to
the inferred world, **active** experiment selection that picks the ranking-flipping test,
a **robust QD archive** of diverse robust bodies, and a **multi-fidelity ladder** that
keeps the inner loop fast. The flagship RS8 result quantifies how much calibration +
robustness change the design ranking vs the highest-fidelity signal available. Most RS
items are **framework-now / hardware-gated-fit**: built and *sim-to-sim validated* now,
closed against real hardware (motorloop bench, drop/ram tests) when parts exist.

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
- **(RS) Don't put the highest-fidelity model in the inner RL loop** — MJX stays fast; the
  CPU oracle / hardware snippets update the ensemble and validate finalists, they don't train.
- **(RS) Don't train a learned residual without a held-out "real" stand-in** — fit the
  actuator/contact residual against data it didn't see (perturbed sim now, held-out bench
  logs later), or it just memorizes the prior; report the held-out error.
- **(RS) Don't let the residual extrapolate outside its measured support** — clamp/flag
  inputs beyond the calibrated range; a learned residual is only trustworthy where it saw data.
- **(RS) Don't claim zero-shot/online adaptation "works" without the truth gate** — the
  student's adaptation gap and the RS8 ranking must be reported against the best real signal,
  not asserted.

## Implemented (results)

Executed ~2026-06-20 on the local box + a RunPod RTX 4090. Full write-up in
`notes/codesign-realization-report.md`; reproduce via `notes/gpu-runbook.md`.

- **All phases built + wired** (Phases 0–9 + Real2Sim2Real RS1–RS8); every new file SPDX'd,
  `py_compile`-clean, REUSE-compliant (488/488), in the Makefile.
- **CPU sim-to-sim suite green** (`make codesign-rs`, 12 self-tests): RS1 world-posterior
  recovery, RS2/RS3 residuals (93%/94% held-out gap closed), RS4 RMA adaptation gap, RS5
  info-gain, R6/RS6 CVaR+QD, RS7 ladder, R2 hardware-ID, R3 contact, NSGA-II.
- **GPU pipeline validated tiny on the 4090** (E2E-first, sequential): Phase-0 throughput
  baseline (35k env-steps/s @16384, 6140× scaling); parity (grounded qpos mean 6.5e-5);
  walker (universal policy + Phase-2 proxy↔return ρ=−0.03 + CEM + Pareto); rederive R1/R7
  (calibrated sim harder 79.6→78.3); fighter (warm-start); score (walker↛combat ρ=−0.09);
  differentiable (grad on smooth obj; through-contact grad wall + ES fallback); self-play
  league; `--fitness policy` direct co-design.
- **Key findings:** MJX is contact-bound (~CPU per-env per step; GPU win = parallelism);
  the db42s03 gimbal motor is undersized for a Go2-scale body (NSGA-II constraint binds);
  `DAMAGE_REF` was ~4× too high — measured impacts → ≈35 N (it had been mostly vanishing).
- **Honest gaps (next):** the trained policy doesn't yet *fight well* (fighter SPARC still
  negative at tiny scale) — the binding constraint is a learnable combat task + curriculum +
  self-play at scale, **not** more sim realism (which is sim-to-real insurance, gated on
  hardware). Real-scale runs (`make gpu-baseline/gpu-codesign/gpu-selfplay`) are the next step.
