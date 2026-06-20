<!-- SPDX-License-Identifier: MIT -->
# Co-design realization — results report

What converged, what the calibration changed vs the idealized sim, and the honest limits.
Companion to `notes/codesign-realization-checklist.md` (the plan) and
`notes/gpu-runbook.md` (how to reproduce). GPU results on an RTX 4090 (RunPod); the
Phase-R/RS sim-to-sim verifications run on CPU (`make codesign-rs`).

## Headline
The co-design machinery is now wired to rank bodies by a **trained policy's return** in a
**reality-gap-calibrated** sim, scored by **robust/CVaR** return, with a **design-conditioned
universal policy** for cheap eval, **self-play** with a Hall-of-Fame league, **NSGA-II**,
**topology GA**, and a **differentiable** demo — plus the **Real2Sim2Real** stretch
(posterior over worlds, learned actuator/contact residuals, RMA teacher→student, info-gain
test selection, robust QD, multi-fidelity ladder) built and **sim-to-sim validated**, with
the real-hardware fits honestly gated.

## Phase 0 — GPU/MJX foundation (PROVEN on a 4090)
- `jax.devices()` → `[CudaDevice(id=0)]`; a generated `robot.toml` body loads + steps in MJX.
- Throughput baseline (`bench_throughput.py`), this body (nq=19 nv=18 nu=12):

  | batch | env-steps/s | speedup vs 1 env |
  |---|---|---|
  | 1 | ~6–11 | 1× |
  | 2048 | ~7,900 | ~1,390× |
  | 8192 | ~24,200 | ~4,240× |
  | 16384 | ~35,100 | ~6,140× |

- **Honest finding:** this body is **contact-bound** in MJX — saturated throughput
  (~35k env-steps/s) is comparable to one CPU core's single-env rate (~34k/s); the GPU win
  is *running thousands of envs in one process*, not per-step speed. Solver iterations don't
  move it (~17k/s for Newton/CG, 4–10 iters at batch 8192); disabling self-collision buys
  only ~1.3× (22.7k vs 17.1k). Left as a documented Phase-8a optimization (not applied — it
  would invalidate trained checkpoints for a small gain and changes contact physics).

## Phase 1 — MJX env + trainer + baseline + parity
- `mjx_env.py` (brax-PPO env), `train_codesign.py` (baseline), SPARC reward twin
  (`sparc_score.step_reward_jax`, backend-agnostic numpy/jnp).
- **MJX↔MuJoCo parity gate** (`test_parity.py`, also in the CPU suite as `test_mjx_parity`,
  skipping without JAX): grounded contact regime qpos mean **6.5e-5**, reward mean **1.3e-4**
  — tight; free-tumbling airborne drift mean ~2.5e-3 (looser, expected for chaotic free
  rotation). SPARC twin: numpy↔jnp identical and equals the source on the valid [0,1] domain.
- Parity gate **confirmed on GPU (rc=0)**: grounded(contact) qpos mean 6.5e-5 / reward mean
  1.3e-4 (tight), airborne(free-tumble) bounded, SPARC numpy↔jnp twin exact (≤1e-6).

## Phase R — reality-gap calibration (the sim as a measured instrument)
- `reality_gap.py` (actuator/contact/body/sensor uncertainty, `actuator_scale` back-EMF
  envelope, `apply_to_mjx_model`, `damage_from_force`, parity gates) — the single sim-to-real source.
- **R1** actuator droop wired into `UniversalEnv(reality_gap=True)` (back-EMF + current/voltage/
  thermal + gear eff). **R2** `hardware_id.py`: 6-measurement suite emits traces; sim-to-sim
  fit recovers the identifiable axes (friction/damping/latency/kt, err 0.39→0.10); kt vs
  current-limit confound is flagged honestly. **R3** `test_contact.py`: unified Newton damage
  currency preserves orderings (harder>glancing 1119N>486N, stable>bounce, impacts accumulate).
  **R4** calibrated DR + a SimOpt/Bayesian tightening hook (`domain_model.update_posterior`).
  **R5** truth gates: distribution-match for the uncontrolled fight, pointwise for controlled
  measurements. **R6** robust scoring: CVaR-optimal ≠ mean-optimal (`robust_codesign.py`).
- **R7** re-derive under the calibrated sim (`rederive_r7.py`): tiny run (6 designs, 4 worlds)
  ideal-vs-calibrated-robust rank ρ=**+0.94** (modest reorder at this scale), winner unchanged,
  mean return **79.6→78.3** under actuator droop + DR (the calibrated sim is measurably harder).
  Real-scale `make gpu-rederive` (more designs/worlds) amplifies the reordering.

## Phase RS — Real2Sim2Real (framework-now, hardware-gated-fit; all sim-to-sim PROVEN on CPU)
| item | module | sim-to-sim result |
|---|---|---|
| RS1 world posterior | `domain_model.py` | recovers a hidden world from traces: err 0.63→0.07, entropy ↓ |
| RS2 actuator residual | `actuator_residual.py` | 93% of the held-out gap closed; out-of-support clamp |
| RS3 contact residual | `contact_residual.py` | 94% closed; Newton ordering preserved |
| RS4 RMA teacher→student | `adaptive_policy.py` | student infers dynamics online, matches teacher, ≫ z-blind |
| RS5 active experiment sel. | `reality_gap_eval.py` | info-gain picks the ranking-flipping measurement |
| RS6 robust QD | `robust_codesign.py` | CVaR≠mean; MAP-Elites archive of diverse robust bodies |
| RS7 multi-fidelity ladder | `multifidelity.py` | near-best finalist at 25% oracle cost; oracle never inner |
| RS8 flagship ranking | `reality_gap_eval.py` | robust ranking predicts the oracle best (ρ 0.96 vs 0.64 nominal, 0.22 proxy) |

## Phase 2/3 — trained-return fitness + universal policy
- Phase 2 direct: `optimize_design.py --fitness policy` (warm-start fine-tune per candidate
  → trained return) **confirmed on GPU**: default design → return 166.5 at **~200 s/candidate**
  (12k-step fine-tune) — that per-candidate cost is exactly what the universal policy (Phase 3)
  removes by making design-eval a rollout.
- Phase 3 cheap: `UniversalEnv` carries the design in obs; `codesign_gpu.py` trains ONE policy
  over the design distribution and evaluates designs by a fixed-policy rollout.
- Results (tiny run): trained-return vs static proxy rank ρ=**−0.03** (the proxy ranks designs
  *wrongly* → the trained return is the real fitness); walker-fitness vs fighter-SPARC ρ=**−0.09**
  ("walker is a prefilter/warm-start only; fight-return must rank designs" — `codesign_validate`).
  CEM on the trained return climbs (best 81.6 vs default 81.2); Pareto front 4 of 6 non-dominated.
  Speedup: the universal policy makes design-eval a **rollout** (no per-candidate retrain) — the
  direct Phase-2 fine-tune cost is logged per candidate (`--fitness policy`) for the contrast.

## Phase 4 — self-play co-evolution
- Opponent is now a **generated body** (`attacker.toml`); the abstract `[strike_h,reach,speed]`
  and the geometric `_hit_on_us` formula are **retired**. `coevolve.py` runs the arms race on
  **real-physics** engagements between two generated bodies (measured strike envelope / settle
  stability / torque impulse), with a Hall of Fame, an **absolute benchmark set** (Phase 8b —
  tracks real progress, not just Red-Queen), and population **diversity**. `selfplay_mjx.py`
  is the GPU two-policy self-play with a HoF league (symmetric bodies).
- Self-play `selfplay_mjx.py` built + structurally validated (round-0 passive mirrors the
  established match pattern; the AutoReset-wrapper info-merge bug found+fixed). The two-robot
  weapon scene (nu=30, contact-heavy) is **compile-heavy** at scale — real-scale league runs +
  the SPARC-trend curves are the fighter-milestone experiment (`codesign-fighter-milestone-checklist.md`).

## Phase 5/6/7 — multi-objective / topology / differentiable
- **5** `nsga2.py`: Pareto front over (return, mass, cost) under the motor-envelope + weight-
  class constraints; knee identified; single-objective is one point on the front. Finding: the
  `db42s03` gimbal motor (~1.5 N·m @ gear 6) is undersized for a Go2-scale body.
- **6** `codesign_extra.py`: topology GA evolves the leg list; a non-default (6-leg) topology
  wins; warm-start-vs-retrain rule respected. ±50% constant-sensitivity: ranking corr +1.00.
- **7** `codesign_diff.py`: exact `jax.grad` ascent on a smooth design sub-objective (J
  1.01→1.44 in 10 steps, ≤ CEM's evals); **`jax.grad` through `mjx.step` is BLOCKED** (MJX's
  iterative solver is a dynamic-bound loop → reverse-mode fails) — the honest contact wall;
  the **ES fallback** (antithetic, forward-sim only) optimizes the real MJX objective.

## Phase 8 — bottlenecks
- **8a** per-candidate rebuild removed on GPU by in-env field randomization (`apply_design`)
  — no XML rebuild for parameter-DR; only distinct topologies rebuild.
- **8b** anti-disengagement: absolute benchmark + diversity + HoF in `coevolve.py` (absolute
  score trended up, not cyclic).
- **8c** constants: sensitivity sweep shows the design ranking is robust to ±50% perturbation.

## Honest limits
- MJX throughput for this contact-rich body is modest (parallelism, not per-step speed).
- Cross-body self-play leagues need a policy per body shape; the league here is symmetric.
- Topology→policy transfer across action-dim changes is retrain-not-warm-start (flagged).
- Differentiable co-design through contacts is noisy → ES fallback (not pure gradient).
- Every Phase-R/RS *fit* is sim-to-sim today; the real-hardware calibration (motor bench logs,
  drop/ram tests) is wired but **hardware-gated** — no real parts yet.

## Reproduce
`make gpu-validate` (tiny, all stages green) → the real-scale targets in `notes/gpu-runbook.md`
→ `make codesign-rs` for the CPU sim-to-sim Phase-R/RS suite.
