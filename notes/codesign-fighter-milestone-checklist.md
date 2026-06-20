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

- [ ] **A `fight_metrics` logger** (extend `match_mjx`/`train_adversarial`/`selfplay_mjx`):
      per-eval emit `METRIC stage=fight ... sparc=.. dealt=.. taken=.. closing=.. fleeing=..`
      so all six signals stream to CSV/JSONL during training (not just episode_reward).
- [ ] **Decompose the reward** in the env so dealt/taken/closing/fleeing are separable
      channels (already in `_sparc`/`step_reward` — surface them as `state.metrics`, which
      brax aggregates and reports per eval).
- [ ] **A fixed benchmark opponent set** (3–5 reference bodies+policies: a passive stander, a
      spinner, a HoF snapshot) checkpointed once, scored every eval for absolute progress.
- **Verify:** a `--tiny` run emits all six metrics to `fight_metrics.jsonl`; the dashboard
      (`render_fight.py` or a notebook) plots each over steps. No long run until this is green.

## Phase F1 — make the combat task LEARNABLE (the make-or-break)

- [ ] **Dense reward shaping toward close→strike:** add a shaped potential (−distance to
      opponent, + alignment of the weapon with the opponent body, + a hit bonus) on top of
      the sparse SPARC term, annealed away as competence rises (avoid reward hacking).
- [ ] **Fightable weapon geometry / start:** make the weapon reach an opponent from a real
      fighting pose (forward spear or a stomp the policy can aim), and start bodies within
      engagement range (curriculum on separation), so the gradient to "land a hit" exists.
- [ ] **Skill-ladder curriculum:** locomotor → reach-and-touch → strike-a-passive-target →
      strike-a-dodging-target → full fight. Warm-start each rung from the last (the proven
      pattern); widen difficulty only after the rung's SPARC clears a bar.
- **Verify:** on a *passive* opponent the fighter reaches **dealt ≫ taken** and
      **closing ≫ fleeing** with **positive** SPARC return. **Falsified if** SPARC stays ≤0 or
      the policy flees/idles — then the task, not the budget, is the problem (iterate F1).

## Phase F2 — real-scale single-fighter training (the competence question)

- [ ] Run `train_adversarial.py` at real scale (millions of steps) on the default body,
      warm-started from the universal locomotor, with F1's shaping + curriculum.
- [ ] Log the six metrics over training; checkpoint the best fighter.
- **Verify:** **(1)** SPARC return climbs to clearly positive; **(2)** dealt/taken ratio > 1
      and rising; **(3)** closing > fleeing throughout (it engages, never flees); **(4)** the
      fixed-benchmark score trends up. **Falsified if** the curve rises only via the alive
      bonus while dealt≈0 (a survivor, not a fighter) — report the decomposition, not the scalar.

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

- [ ] **proxy** rank (static stand/clearance/mass, `optimize_design.proxy_fitness`),
- [ ] **nominal** rank (trained-fighter SPARC at the nominal sim world),
- [ ] **robust/CVaR** rank (trained-fighter SPARC, worst-20% over the calibrated world
      ensemble, `robust_codesign.cvar` + `domain_model.sample_world`).
- [ ] **Ground truth = actual fight performance:** each body's mean SPARC vs the **held-out
      benchmark set + HoF** (the best "real" signal available; the CPU MuJoCo oracle / reduced
      hardware later). Report **Spearman(each ranking, ground truth)**.
- **Verify (the claim):** **robust ρ ≥ nominal ρ ≥ proxy ρ**, and the robust-best body
      *out-fights* the proxy-best and nominal-best in head-to-head matches. **Falsified if**
      nominal ≈ robust (calibration didn't matter at this fidelity) or proxy already predicts
      truth (the cheap fitness was good enough) — both are publishable *negative* results;
      report them honestly with the fidelity caveat.

## Phase F5 — cost, reproducibility, write-up

- [ ] `make` targets for the real-scale runs + the F4 experiment; wall-clock + $ logged in
      `notes/gpu-runbook.md`.
- [ ] **`notes/codesign-fighter-report.md`** — the six metric curves, the F4 three-way ranking
      correlation + head-to-head, and the honest verdict on both questions, with the
      proxy→nominal→robust→hardware fidelity ladder stated.

## Done-when

A trained policy **fights competently** — positive SPARC return, dealt>taken, closing>fleeing,
rising benchmark + HoF-robust scores — and the **robust/CVaR ranking predicts real fight
performance at least as well as nominal and better than proxy**, with the robust-best body
winning head-to-head. The two questions are answered with curves and correlations (or an
honest negative result + the fidelity caveat), reproducible via `make` on a GPU box.

## What NOT to do

- Don't call a rising reward curve "learned to fight" — decompose dealt/taken/closing first.
- Don't measure only relative self-play score — the fixed benchmark is what catches disengagement.
- Don't claim the robust ranking wins without the head-to-head + the ground-truth correlation.
- Don't pour compute into F2/F3 before F1 is green — a better locomotor is not a fighter.
- Don't trust nominal==robust as "calibration is useless" without noting it's a *fidelity*
  statement (sim oracle now; the verdict can change against real hardware).
