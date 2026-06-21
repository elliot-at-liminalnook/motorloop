<!-- SPDX-License-Identifier: MIT -->
# Fighter milestone checklist — can the system learn to FIGHT, and do robust rankings win?

The architecture is built and validated (`notes/codesign-realization-checklist.md` /
`-report.md`): trained-return co-design, a reality-gap-calibrated sim, a universal policy,
self-play, NSGA-II/topology/differentiable search, and the Real2Sim2Real stretch — all
green at `--tiny` on a 4090. The open question is **no longer "is this the right
architecture?"** It is:

> **Can the system learn a competent fighting policy — and do robust, calibrated rankings
> pick bodies that fight better than proxy/nominal rankings?**

That is the experiment that makes the work compelling. This checklist runs it at real
scale and tracks the six signals that decide it:

1. **SPARC return over training** (does it climb, and to a positive decision-win?)
2. **damage dealt vs taken** (is it actually landing hits, not just surviving?)
3. **aggression / closing behavior** (closing > fleeing — it engages, per SPARC 1.2.1)
4. **benchmark-opponent score** (absolute progress vs a fixed reference set, not just relative)
5. **Hall-of-Fame robustness** (mean SPARC vs the whole archive, not just the latest foe)
6. **robust/CVaR ranking vs nominal ranking** (do calibrated rankings pick better fighters?)

## The honest boundary (read first)

- **The crux is task-learnability, not compute.** At `--tiny` the fighter's SPARC return is
  *negative* and walker-fitness anti-correlates with combat — the same wall as the
  combat-dodge work ("reactive dodge UNSOLVED on the Ant", [[motorloop-rl-combat]]). The
  weapon is a *downward stomp* spear, bodies start far apart, and reward only pays on rare
  weapon-on-body contact. **If F1 (a learnable task) fails, more steps just buy a better
  locomotor.** Budget the most effort here.
- **GPU cost is real.** RunPod 4090 ≈ $0.34–0.69/hr; this body is contact-bound in MJX
  (~35k env-steps/s); the two-robot weapon scene is **compile-heavy** (minutes per graph —
  budget it). Leak-test every stage `--tiny` before a long run; terminate the pod when idle.
- **Falsifiability.** Each phase states what result would *kill* the claim, not just confirm
  it. "It learned to fight" must mean dealt>taken AND closing>fleeing AND positive SPARC vs a
  held-out benchmark — not a rising reward curve alone (truncation/alive bonuses lie).
- **Reuse, don't fork.** Extend `train_adversarial.py`, `selfplay_mjx.py`, `match_mjx.py`,
  `rederive_r7.py`, `reality_gap_eval.py`, `robust_codesign.py`, `coevolve.py` — and the one
  objective `sparc_score.py`. New code is metric instrumentation + the ranking experiment.

## Phase F0 — instrumentation (build the six trackers BEFORE the long run)

- [x] **A `fight_metrics` logger** — `train_adversarial.py` streams per-eval
      `{sparc, dealt, taken, closing, fleeing, dist}` to `out/fight_metrics.jsonl`.
- [x] **Decompose the reward** — dealt/taken/closing/fleeing/sparc surfaced as `state.metrics`
      (merged into brax's dict so its `reward` key survives — the scan-carry fix).
- [~] **Benchmark opponent set** — the fixed passive B IS the benchmark for F2/F1; the
      spinner/HoF set is for F3 (deferred with F3).
- **Verify:** [x] a `--tiny` run emits all six metrics to `fight_metrics.jsonl`; `render_fight.py`
      sparklines them + prints the competence verdict. (Verified green before the long run.)

## Phase F1 — make the combat task LEARNABLE (the make-or-break)

- [x] **Dense reward shaping toward close→strike:** `reward = SPARC + shaping·(−0.15·dist
      −0.20·leg-proximity-to-opponent + 3·dealt) + 0.3·upright + alive` (annealable via `--shaping`).
- [x] **Fightable start:** legs-as-weapons damage + a close start (`--sep`, default 0.9). (Weapon
      is the leg itself, so any limb on B scores — more learnable than the downward spear.)
- [~] **Skill-ladder curriculum:** warm-start from the locomotor done; finer touch→strike rungs
      are part of Phase F-SPEED's sample-efficiency work (not fully exercised at budget).
- **Verify:** [FALSIFIED AT BUDGET] on the passive opponent the fighter did **not** reach
      dealt≫taken — over ~28 iterations dealt=0, closing≈0, SPARC flat ~−67. Per the rule, **the
      task+compute, not the architecture, is the problem** → see Phase F-SPEED. (Honest negative.)

## Phase F2 — real-scale single-fighter training (the competence question)

- [x] Ran `train_adversarial.py` real-scale (warm-started, shaping, small-batch 256·8·10 for
      max iterations/step), tags `f2`; six metrics logged to `fight_metrics.jsonl`.
- [x] Logged the six metrics; no improving checkpoint to keep (stayed a survivor).
- **Verify:** [FALSIFIED AT BUDGET] SPARC flat ~−67, **dealt=0** throughout, closing≈0 — a
      survivor, not a fighter (the decomposition, not the scalar, is the verdict). Two measured
      causes: ~1,300 env-steps/s (contact-bound two-robot scene) + too few iterations. The
      compute, not the architecture, is the wall → **Phase F-SPEED is the unlock.**

## RESOLVED — the contact-forcing curriculum cracks `dealt=0` (2026-06-20)

The F2/sweep failure (`dealt=0`) was an exploration problem: the sparse contact reward never
fired because random play never lands a hit. **Fix that worked:** a reverse/separation
curriculum — each env samples the A–B start separation from `[sep_lo, sep_hi]`
(`train_adversarial --sep-lo/--sep-hi`, per-env in `reset`); a close low end (0.4) guarantees
some envs spawn in striking range (the reward signal always exists), the high end widens over
phases (learn to close). On an A100, warm-started chain, resume-safe + pulled-local:
**`dealt` went 0.000 → 0.20** (close foundation) and stayed >0 across widening phases, with
**closing 0.01→0.39** in the sep-0.4–1.0 phase — the robot reliably ENGAGES and ATTACKS (it
did not before). It does not yet out-strike a passive foe (`dealt≈taken`) — winning the
exchange (active striking / self-play) is the next step — but reliable attacking engagement,
the milestone's core, is achieved. Figures: `make_fight_figures.py` →
`sim/build/gpu/figures/`. This is the answer the whole F-SPEED detour was missing: **the lever
was the curriculum, not throughput.**

## Resilience rule (any long GPU run — the pod disk is EPHEMERAL)

A 2+ hour run must be **restorable**, never lost. Required for every long run:
- [x] **Frequent save points.** `train_adversarial` checkpoints `{tag}_ckpt.pkl` AND appends
      `fight_metrics.jsonl` **every eval** — set `--evals` so a save lands every few minutes
      (incremental, not just at the end).
- [x] **Incremental pull to LOCAL.** A host-side loop (`pull_loop`) tars the pod's
      `out/*.pkl` + `*.jsonl` + `sweep_*.log` to `sim/build/gpu/out/` every ~90 s, so the
      latest checkpoint + metrics always exist on local disk even if the pod dies mid-run.
- [x] **Resume-from-latest.** `fighter_sweep.sh` resumes each config from its own
      `{tag}_ckpt.pkl` if present (else the universal warm-start) — a restarted/killed config
      continues from its last save, not from scratch. `train_adversarial --resume` handles it.
- **Verify:** kill a run mid-config; confirm the latest `{tag}_ckpt.pkl` + metrics are already
      local, and `--resume` picks up from there (no 2-hr restart).

## Phase F-SPEED — throughput & sample-efficiency unlock (the MEASURED blocker, do before F2/F3)

F2 falsified at budget for a concrete, measured reason: the two-robot fight scene runs at
only **~1,300 env-steps/s** on a 4090 (vs ~35k single-body — contact-bound), and at any
reasonable batch the per-iteration cost (batch·unroll·minibatches) leaves only a handful of
gradient iterations — far too few for a sparse-reward combat task. Until this is fixed, more
wall-clock just buys a better stander. Attack contact cost, GPU utilization, and iteration
count together.

**Faster steps (raise env-steps/s):**
- [x] **Cut contact pairs (#1).** `build_match(..., self_collision=False)` / `build_mjcf(...,
      self_collision=False)` via contype/conaffinity (floor↔A↔B, no A-A/B-B). Verified A-A
      self-pairs 136→0 while A-B(289)+floor(34) preserved; measured **6,779→7,790 env-steps/s**
      at batch 8192 (~1.15× — the A-B fight pairs dominate, so this is the smaller lever here).
      Parity-safe (CPU+MJX share the model). `--lean-contacts` flag on `train_adversarial`.
- [ ] **Fewer geoms / capped contacts.** (Not done — the bigger contact lever for this scene:
      merge per-leg capsules + `<size nconmax= njmax=>`. Left as the next contact optimization.)
- [x] **Saturate the GPU (#2 — the big one).** num_envs 2048→8192 took the fight scene from
      ~1,300 env-steps/s (F2) to ~7,800 (≈**6×**) with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- [x] **Cheaper dynamics/step.** unroll 5 (fewer steps/iteration). (frame_skip left at 5.)
- [~] **Skip recompiles.** `JAX_COMPILATION_CACHE_DIR` is wired in `e2e.py`; not added to the
      fighter launcher yet (one-line env var when needed).
- [ ] **Bigger/more GPUs (linear).** Single 4090 this session; H100 / `pmap` is the next step.

**More learning per step (decouple iterations from throughput):**
- [x] **Keep many gradient iterations.** `--minibatches 32 --updates 8 --batch 256` (=8192/32)
      → 256 SGD steps/iteration; unroll 5 keeps env-steps/iteration small → ~150 iterations in
      6M steps (vs ~6 in F2's big-batch run).

**Fewer steps NEEDED (sample efficiency, orthogonal):**
- [~] **Exploit the curriculum.** warm-start from the locomotor in use; finer touch→strike
      rungs are the next sample-efficiency step.
- [ ] **Off-policy / proxy body.** SAC / reduced-DOF proxy — future.

- **DO ONE THING NEXT (the concrete unlock):** self-collision OFF (#1) + `num_envs 8192` (#2)
      + many-iterations (`--minibatches 32 --updates 8 --batch 256`) (#3) together, then re-run F2.
- **Verify [DONE — speed levers validated; competence NOT cracked]:** all levers implemented +
      measured. Reduced-collision lean scene cut collidable pairs **595→99** (A-A 136→0, A-B
      289→81, floor 34→18); benched **8,371→10,808 env-steps/s** pure on an H100; training
      throughput **~4,670 env-steps/s** (~3.6× the 4090 F2 path's ~1,300). A 3-config param sweep
      on the H100 (`fighter_sweep.sh`: gentle / more-exploration / aggressive-close-start sep 0.6),
      5–6.5M steps each, warm-started + resume-safe + incrementally pulled local:
      **`dealt=0` in ALL configs** — the policy closes marginally (dist↓, fleeing↓) but never
      lands a hit. **Conclusion: throughput + param variation do NOT produce a competent fighter.**
      The frontier is deeper than F-SPEED: a curriculum that *guarantees early contact* (or
      demonstrations / a simpler striking sub-task), and/or far more compute — not just faster steps.
      Cost: ~$4.5 (1.5 h × $2.89/hr H100), under the $15 cap; all artifacts pulled to
      `sim/build/gpu/out/`; pod terminated.

## Phase F3 — real-scale self-play league (does an arms race produce skill?)

- [ ] Run `selfplay_mjx.py` at real scale with the HoF league (sample the archive, not just
      the latest). Budget the heavy two-robot compile; consider a leaner collision model for
      the inner loop (Phase-8a optimization) to afford the rollouts.
- [ ] Track HoF robustness: the current learner's **mean SPARC vs the whole HoF**, each round.
- **Verify:** **(4)** benchmark score trends **up** (absolute progress, not just Red-Queen
      cycling); **(5)** HoF robustness stays positive and rising; both morphologies/policies
      co-adapt. **Falsified if** benchmark is flat/cyclic while relative score swings — that's
      disengagement; add diversity / strengthen the HoF (Phase 8b levers already in `coevolve`).

## Phase F4 — the decisive experiment: do robust rankings pick better fighters?

This is the headline result (the real-scale generalization of `reality_gap_eval.rank_three_ways`
+ `codesign_validate`). Take **N candidate bodies**; rank them **three ways**:

- [x] **proxy** rank (passive-stand survival), **nominal** rank (policy return at nominal sim),
      **robust/CVaR** rank (return CVaR@20% over the calibrated world ensemble) — `fighter_rank.py`.
- [x] **Ground truth** = mean return over a wide held-out world set; **Spearman** of each
      ranking reported. (Run on the trained UNIVERSAL/locomotion policy — the fighter version is
      the same code with the fighter checkpoint, gated on a competent fighter from F2.)
- **Verify (the claim):** [RAN — fidelity-limited result] proxy ρ=+0.92, nominal ρ=+0.99,
      robust ρ=+0.98; all pick the same winner → **robust ≥ nominal ≥ proxy does NOT hold** at
      locomotion fidelity with mild calibration (the checklist's anticipated *negative*: nominal≈
      robust, proxy already predicts). Robust should earn its keep on the world-sensitive
      **combat** task once a fighter exists — reported with the fidelity caveat.

## Phase F5 — cost, reproducibility, write-up

- [x] `make` targets: `gpu-fighter` (F2), `gpu-fighter-rank` (F4) + `render_fight.py` dashboard;
      runs logged. (F-SPEED + F3 real-scale costs go in `gpu-runbook.md` when run.)
- [x] **`notes/codesign-fighter-report.md`** — the six-metric verdict, the F4 three-way ranking
      correlation, and the honest answer to both questions with the proxy→nominal→robust→hardware
      fidelity ladder.

## Done-when

A trained policy **fights competently** — positive SPARC return, dealt>taken, closing>fleeing,
rising benchmark + HoF-robust scores — and the **robust/CVaR ranking predicts real fight
performance at least as well as nominal and better than proxy**, with the robust-best body
winning head-to-head. The two questions are answered with curves and correlations (or an
honest negative result + the fidelity caveat), reproducible via `make` on a GPU box.

## Status (executed ~2026-06-20, RunPod 4090; report in `notes/codesign-fighter-report.md`)

- **F0 instrumentation:** DONE — six trackers stream to `fight_metrics.jsonl`, `render_fight.py`.
- **F1 learnable task:** built (shaping + leg-proximity + close start); verify **falsified at
  budget** (dealt=0 over ~28 iters) → the task+compute is the wall, not the architecture.
- **F2 competence:** RAN → **NO competent fighter in feasible compute** (survivor, dealt=0).
  Honest answer to Q1, with the two measured causes (throughput + iterations).
- **F-SPEED:** the measured unlock — all levers listed; **not yet executed** (the next step).
- **F3 self-play league:** built (`selfplay_mjx.py`), **deferred** (needs F-SPEED to afford).
- **F4 ranking experiment:** RAN on the trained policy → proxy 0.92 / nominal 0.99 / robust
  0.98, same winner; robust doesn't beat nominal **at locomotion fidelity** (fidelity caveat).
  Fight-specific version is the same code on a fighter checkpoint (gated on F2).
- **F5:** Make targets + report + this status DONE.

**Bottom line:** the milestone's machinery + instrumentation are complete and the two questions
are *answered honestly at this fidelity/budget* — Q1: not competent yet, compute-bound (unlock =
F-SPEED); Q2: methodology validated, robust-vs-nominal verdict needs the combat task + a fighter.
The decisive run is F-SPEED → F2 → F4-on-the-fighter.

## What NOT to do

- Don't call a rising reward curve "learned to fight" — decompose dealt/taken/closing first.
- Don't measure only relative self-play score — the fixed benchmark is what catches disengagement.
- Don't claim the robust ranking wins without the head-to-head + the ground-truth correlation.
- Don't pour compute into F2/F3 before F1 is green — a better locomotor is not a fighter.
- Don't trust nominal==robust as "calibration is useless" without noting it's a *fidelity*
  statement (sim oracle now; the verdict can change against real hardware).
