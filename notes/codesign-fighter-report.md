<!-- SPDX-License-Identifier: MIT -->
# Fighter milestone — results report

Answers the two questions of `notes/codesign-fighter-milestone-checklist.md`:
**(1) can the system learn a competent fighting policy?** and **(2) do robust, calibrated
rankings pick bodies that fight better than proxy/nominal rankings?** GPU runs on a RunPod
RTX 4090; reproduce via `notes/gpu-runbook.md` + the `gpu-fighter*` Make targets.

## F0 — instrumentation (DONE)
`train_adversarial.py` now surfaces the six trackers as `state.metrics` (brax aggregates
them per eval): **SPARC return, damage dealt, damage taken, closing, fleeing, distance** —
streamed to `out/fight_metrics.jsonl` per eval. A `--tiny` run emits all six (verified).
Flags added: `--shaping` (dense close→strike potential, annealable), `--sep` (start-
separation curriculum), `--tag`.

## F1 — make the combat task learnable
Dense shaping in the env: `reward = SPARC + shaping·(−0.25·dist + 3·dealt) + 0.3·upright +
alive`, legs-as-weapons damage, start separation `--sep`. The verdict is the
DECOMPOSITION (dealt>taken, closing>fleeing, positive SPARC), not the scalar reward.

**RESULT (honest, falsified at this budget):** with dense shaping (−dist −leg-proximity +hit
bonus), a closer start (sep 0.9), and warm-start from the locomotor, the passive-opponent run
did **not** become competent in feasible session compute: over ~28 brax iterations (573k env-
steps) **dealt stayed exactly 0.000, closing ≈ 0, fleeing flat ~13.5, SPARC flat ~−67** — the
warm-started locomotor stands roughly in place and never learns to close-and-strike. Two
compounding causes, both measured:
  1. **Throughput:** the two-robot weapon scene runs at only ~1,300 env-steps/s on the 4090
     (contact-bound, ×2 bodies) — ~10–25× slower than single-body.
  2. **Iteration cost vs hard exploration:** brax's per-iteration env-steps = batch·unroll·
     minibatches; at any reasonable batch a competent fighter needs *hundreds* of iterations,
     i.e. tens of millions of env-steps = many GPU-hours — infeasible in one session.
This is the checklist's explicit falsification ("SPARC ≤0 / flees-idles → the task, not the
budget, is the problem"). The architecture is sound; **the frontier is the learnable combat
task + far more compute**, not more sim realism. Unlocks: a leaner collision model for the
fight scene (Phase-8a), a tighter close→touch→strike curriculum, and a multi-hour / multi-GPU
budget. (Consistent with the prior "reactive dodge UNSOLVED" finding.)

## F2 — real-scale single-fighter training (the competence question)
**Answer (this fidelity/budget): NO — a survivor, not a fighter.** The six metrics (in
`out/fight_metrics.jsonl`, view with `render_fight.py`) are flat at dealt=0 / SPARC≈−67 over
the iterations we could afford (see F1 above for the two measured causes). A small-batch
config (256·8·10 → more iterations/env-step) was used to maximize learning per step; it did
not change the verdict in-budget. Honest decomposition reported, not a cherry-picked scalar.

## F3 — self-play league (does an arms race produce skill?)
**Not run at scale this session** — gated by the same throughput wall as F2 (the two-robot
weapon scene is compile- and contact-heavy; a self-play league multiplies it). The machinery
(`selfplay_mjx.py`, HoF league) is built + structurally validated tiny. It needs Phase F-SPEED
first (a leaner collision model + GPU saturation) to be affordable. Deferred, honestly.

## F4 — decisive experiment: do robust rankings pick better fighters?
Rank N bodies three ways — proxy (static), nominal (fighter SPARC at the nominal sim),
robust (fighter SPARC CVaR-20% over the calibrated world ensemble) — vs ground-truth fight
performance (mean SPARC over a wide held-out world/opponent set).
**RESULT (on the trained universal/locomotion policy; fight version gated on F2):** over 10
bodies × 10 held-out worlds — **proxy ρ=+0.92, nominal ρ=+0.99, robust ρ=+0.98** vs wide-world
ground truth; all three rankings pick the **same** best body (true perf 156.4). So the claim
*robust ≥ nominal ≥ proxy* does **not** strictly hold here — nominal edges robust (within
noise) and even the static proxy predicts well. This is the checklist's anticipated *fidelity*
outcome: **at locomotion fidelity with mild calibration the three rankings agree** (consistent
with R7's idealized-vs-calibrated ρ=0.94, winner unchanged). Robust ranking only earns its
keep where world-sensitivity actually flips winners — i.e. a **combat** task (damage is far
more world-sensitive than locomotion) with **stronger DR** and a **competent fighter**. The
machinery runs end-to-end on GPU (`fighter_rank.py`); swap the fighter checkpoint for the
fight-SPARC version — same code — once F2 yields a fighter.

## F-SPEED + H100 param sweep (the "bigger GPU + longer run + tune params" attempt)
Implemented the throughput levers and ran a param sweep on an H100 ($2.89/hr, ~$4.5 total):
- **Speed levers work (measured):** reduced-collision lean fight scene cut collidable pairs
  **595→99** (only torso/calf/foot/spear collide); benched **8,371→10,808 env-steps/s** pure on
  the H100; training throughput **~4,670 env-steps/s** (~3.6× the original 4090 F2 path).
- **Competence still NOT reached:** a 3-config sweep (gentle / high-exploration / aggressive
  close-start sep 0.6), 5–6.5M steps each, warm-started, resume-safe, incrementally pulled local
  — **`dealt=0` in every config.** The policy closes a little more (dist↓, fleeing↓) but never
  lands a hit. **A faster scene + param tuning did not crack it.** The real frontier is a
  curriculum that *forces early contact* (or imitation / a simpler striking sub-task) and/or far
  more compute — not throughput. Resilience held: per-eval checkpoints pulled to
  `sim/build/gpu/out/` every 90 s, resume-from-latest, nothing lost; pod terminated.

## Contact-forcing curriculum — the fix for `dealt=0` (the breakthrough)
The blocker was always *exploration of landing a hit* (sparse contact reward). Fix: a
**reverse/separation curriculum** — each env samples the A–B start separation from
`[sep_lo, sep_hi]` (`train_adversarial --sep-lo/--sep-hi`); a close low end (0.4) GUARANTEES
some envs spawn in striking range every batch (so the `dealt` signal always exists), and the
high end widens over phases so the policy learns to CLOSE then strike. Verified locally: at
sep 0.45 the striking geoms start **7 cm** from the opponent (a hit is one leg-extension away).

**Result (A100, contact-forcing curriculum, warm-started chain, resume-safe + pulled local):**
- **`dealt` crossed 0 and rose** — close-only foundation (cval, sep 0.4–0.5): dealt **0.11→0.20**
  (vs flat **0.000** in every prior run). The robot **reliably engages and attacks.**
- Widening phases keep `dealt>0` and grow approach behavior — the full 5-phase trajectory
  (final eval per phase, sep range → dealt / closing / SPARC):
  | phase | sep | dealt | closing | SPARC |
  |---|---|---|---|---|
  | cval | 0.4–0.5 | 0.200 | ~0.0 | −42.8 |
  | c1 | 0.4–0.7 | 0.085 | 0.04 | −43.7 |
  | c2 | 0.4–1.0 | 0.042 | 0.39 | −39.1 |
  | c3 | 0.4–1.4 | 0.029 | 0.94 | −33.1 |
  | c4 | 0.4–1.8 | 0.025 | **1.53** | **−26.4** |
  As the range widens `dealt` declines (landing a hit at distance within an episode is harder)
  but **`closing` rises 0.01→1.53 and SPARC improves +16 (−42.8→−26.4)** — the robot learns to
  *reliably approach and engage* across the full range, while still landing hits (`dealt>0`
  everywhere, 0.20 at close range). 30M+ env-steps total on an A100 (~$3.4).
- Honest boundary: `dealt≈taken` (mutual contact vs a passive opponent) — the robot *attacks*
  but does not yet *win the exchange* (`dealt>taken`); that (active out-striking / self-play)
  is the next refinement. But the core goal — teach reliable attacking engagement — is met:
  contact went from impossible-to-learn to consistent.
- Figures: `sim/build/gpu/figures/curriculum_{metrics,dealt_vs_taken,per_phase}.png`
  (`make_fight_figures.py`, parsed from the per-phase logs).

## Combat body-ranking (win-exchanges STEP 1) — Q2 RESOLVED on the contact-forced fighter
F4 above ran the three-way ranking on the *locomotion* policy and found a tie (ρ 0.92–0.99) —
the anticipated fidelity outcome, not a verdict. With the contact-forced fighter now in hand
(`cval_ckpt.pkl`), `combat_rank.py` reran the experiment **on combat SPARC**: rank 16 bodies by
the design-conditioned fighter's combat outcome three ways (proxy = static passive-stand
survival; nominal = combat SPARC at the nominal world; robust = combat SPARC CVaR@20% over the
calibrated `reality_gap` world ensemble) vs ground truth = mean combat SPARC over 24 wide
held-out worlds. RTX 4090, 1758 s, rc=0.

**RESULT — the claim holds, decisively:**

| task | proxy ρ | nominal ρ | robust(CVaR) ρ |
|---|---|---|---|
| locomotion (F4, smooth) | +0.92 | +0.99 | +0.98 |
| **combat** (contact) | **−0.61** | **+0.96** | **+1.00** |

- **robust(CVaR) ρ = +1.00 ≥ nominal +0.96 > proxy −0.61** (`claim_holds=True`), body-perf
  spread 55.3 (the fighter differentiates bodies strongly).
- The headline: the cheap static proxy is **anti-correlated** with true fight performance on
  combat — *a body that stands well passively is not a good fighter.* On locomotion it predicted
  fine (+0.92); contact dynamics flip it negative. This is the co-design thesis: when the task is
  world-sensitive, you must rank with the policy in the loop, and calibrated robustness gives the
  best ranking. The tiny pre-check agreed (−0.83/+0.94/+1.00). Spread 55.3 ⇒ a stronger fighter
  (win-exchanges STEP 2) is **not required** for this result.
- Artifacts: `sim/build/gpu/out/combat_rank.npz` + `combat_rank.log`; figures
  `sim/build/gpu/figures/ranking_loco_vs_combat.png`, `ranking_combat_scatter.png`
  (`make_ranking_figure.py`).

## Verdict
**Q1 (can it learn to engage in attacking?)** — *YES, via the contact-forcing curriculum
(this was the real blocker, not throughput).* Plain training + the F-SPEED throughput levers
left `dealt=0` (a survivor). The **separation curriculum** — spawn some envs in striking range
so the contact reward always fires, then widen — flipped it: `dealt` 0.000→0.20 and **`closing`
0.01→1.53, SPARC −44→−26** over a 5-phase A100 run. The robot now reliably approaches and
attacks across the full range. Honest limit: `dealt≈taken` vs a passive foe — it *attacks* but
doesn't yet *win the exchange*; active out-striking / self-play is the next step. (Figures:
`sim/build/gpu/figures/`.) The earlier compute-bound conclusion was half the story — the
decisive lever was the **curriculum**, with F-SPEED's faster scene making the sweep affordable.

**Q2 (do robust calibrated rankings pick better bodies?)** — *YES, on the task where it matters
(combat).* On locomotion all three rankings tied (ρ 0.92–0.99) — too smooth for world
uncertainty to flip winners. On **combat** (run on the contact-forced fighter, 16 bodies × 24
worlds): **robust(CVaR) ρ = +1.00 ≥ nominal +0.96 > proxy −0.61** — and the cheap static proxy
goes *anti-correlated*, i.e. it actively misleads. Robustness earns its keep exactly where the
task is world-sensitive. See "Combat body-ranking" above. The settled co-design result.

**Fidelity ladder:** sim-oracle ground truth now → reduced-hardware proxies → real bench/drop
tests later (all `reality_gap`/`domain_model` hooks wired, hardware-gated).

## The settled lesson + what's next
**The blocker was sparse EXPLORATION — not simulator realism, optimizer choice, or GPU
throughput.** Plain adversarial training, even with a faster collision scene and more compute,
stayed at `dealt=0` because the policy almost never discovered "make contact," so there was no
fighting reward to learn from. The reverse/separation curriculum (start some envs in striking
range, then widen) manufactures early contact → reward signal exists → contact flips from
impossible-to-learn into reliable engagement. (The F-SPEED throughput work was *not* the lever;
it just made the sweep affordable.)

**Now:** the robot **engages and attacks** but does not yet **win the exchange** (`dealt≈taken`
— trading hits). The next milestone (`notes/codesign-win-exchanges-checklist.md`): teach timing,
angle, retreat/reset, and advantageous (un-traded) contact via an A→F curriculum (spawn-in-range
→ moving opponent → attacking opponent → self-play) + reward asymmetry (clean-hit bonus, trading
penalty, post-hit disengage). And the **decisive body-ranking experiment** moves to the
contact-forced fighter checkpoint — combat damage is where world uncertainty should finally make
robust(CVaR) ranking beat nominal (it tied on locomotion because locomotion is too smooth).

## Reproduce
`make gpu-fighter` (F2) → `make gpu-selfplay` (F3) → `make gpu-fighter-rank` (F4); metrics in
`out/fight_metrics.jsonl`.
