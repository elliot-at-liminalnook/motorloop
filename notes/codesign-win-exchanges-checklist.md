<!-- SPDX-License-Identifier: MIT -->
# Win-exchanges milestone — from "engages and attacks" to "wins exchanges"

> **Document status:** Active · **Audience:** Robot-learning contributors · **Last reviewed:** 2026-07-12 · **Canonical summary:** [`locomotion-status.md`](locomotion-status.md)

The contact-forcing curriculum (`notes/codesign-fighter-*`) flipped the fighter from
`dealt=0` (can't learn) to reliable **engagement**: `dealt` 0.000→0.20 at close range,
`closing` 0→1.53, SPARC −44→−26. But `dealt≈taken` — it **trades hits**, it doesn't yet
**out-strike**. This milestone (a) answers the project's core co-design question on combat, and
(b) teaches favorable exchanges: timing, angle, retreat/reset, hit-without-being-hit.

> **The settled lesson (carry it forward):** the blocker was **sparse exploration**, not
> simulator realism, optimizer choice, or GPU throughput. The policy almost never discovered
> "make contact," so there was no fighting reward to learn from. The curriculum manufactures
> early contact so the signal exists. Every step below preserves that principle: never widen
> faster than the policy can keep landing hits.

## Goal
Two outcomes, in priority order: **(1)** does robust, calibrated **body ranking** pick better
fighters once contact dynamics matter (the co-design thesis, inconclusive on locomotion)? and
**(2)** turn "engages and attacks" into "wins exchanges" — `dealt > taken` vs opponents that
move and hit back, with clean (un-traded) hits.

## Order of attack (why this order: cheapest decisive result first, heavy RL only if needed)

### STEP 0 — free prep (local, no GPU) — unblocks the cheap decisive run
- [ ] **Add the `reality_gap` world path to `AdversarialEnv`** (mirror `UniversalEnv`: sample a
      world per episode, `apply_to_mjx_model` + actuator droop). ~30 lines; the one thing the
      decisive experiment needs that doesn't exist yet.
- [ ] **Point `fighter_rank.py` at `AdversarialEnv` + the contact-forced fighter checkpoint**
      (`cval`/`c*_ckpt.pkl`), scoring designs by **fight SPARC** (the fighter is design-
      conditioned, so it already ranks bodies) — nominal (no DR) vs robust (CVaR over the world
      ensemble) vs proxy (static).
- **Verify:** both run `--tiny` on CPU/locally where possible; ready for one short GPU run.

### STEP 1 — THE DECISIVE EXPERIMENT (cheap GPU, ~$2–3) — the project payoff, do FIRST
On locomotion, proxy/nominal/robust rankings all agreed (ρ 0.92–0.99) — *not a failure*: it
says locomotion is too smooth for world-uncertainty to flip rankings. Combat damage/contact is
where it should matter, and it **does not require a fighter that *wins*** — only one whose
performance *varies by body in a world-sensitive way*, which the current contact-forced fighter
likely already does. So this is the cheapest path to the headline result.
- [x] Run proxy vs nominal vs robust-CVaR rank correlation on combat SPARC, on the existing
      fighter checkpoint. *(`combat_rank.py`, 16 bodies × 24 worlds, RTX 4090, 1758 s, rc=0.)*
- [x] **Verify (the claim): CONFIRMED — robust(CVaR) ρ ≥ nominal ρ > proxy ρ.**
      **`proxy ρ = −0.61, nominal ρ = +0.96, robust ρ = +1.00` (spread 55.3).** Stronger than the
      tiny pre-check (−0.83/+0.94/+1.00). The headline: on **locomotion** all three tie
      (+0.92/+0.99/+0.98 — too smooth); on **combat** the cheap static proxy is *anti-correlated*
      (−0.61) — a body that stands well passively is not a good fighter — while policy-in-the-loop
      combat eval + calibrated robustness give the best ranking. Spread 55.3 ⇒ the contact-forced
      fighter differentiates bodies strongly, so **STEP 2 is NOT required for the headline result.**
      Artifacts: `sim/build/gpu/out/combat_rank.npz`, figures
      `sim/build/gpu/figures/ranking_loco_vs_combat.png` + `ranking_combat_scatter.png`.

### STEP 2 — strengthen the fighter to WIN exchanges (heavier RL; OPTIONAL — Step 1 already gave the headline)
Step 1 came back **confirmed** (robust +1.00 ≥ nominal +0.96 > proxy −0.61), so this step is no
longer needed for the co-design result — it's the separate "wins exchanges" goal (`dealt>taken`).
Follow the **proven recipe** — DeepMind 1v1 soccer (Haarnoja et al., Science Robotics 2024,
arXiv:2304.13653), the closest published analog; it independently reproduced our `dealt=0`
"degenerate optimum" and fixes it as below.

> **2·0 — RESILIENCE GATE (do BEFORE any long/expensive run; the long run is the LAST thing).**
> A long run on today's pipeline would *plateau*, not climb — the algorithm isn't yet proven to
> improve monotonically over time. The infra is resilient (resume-safe ckpts, tiny-validated,
> bounded cost); the *learning dynamics* are not. Three guards make a long run actually rise, then
> one cheap empirical check decides whether to commit:
> - [ ] **Win-reward asymmetry first (= 2b).** The current fighter sits at `dealt≈taken`; its
>       reward nets ~0 when trading, so it has **no headroom** — a long run optimizes a flat
>       objective. Land 2b (clean-hit bonus, trading penalty, post-hit disengage) so the curve has
>       somewhere to go *before* spending GPU-hours.
> - [ ] **Per-checkpoint held-out benchmark eval (the honest improvement curve).** Training reward
>       can rise while real skill stalls (and we hit "no-truncation `ep_len` lies" + per-phase
>       jsonl truncation). Evaluate every saved ckpt against a FIXED opponent set
>       (passive/mover/striker/HoF) → a benchmark-SPARC curve that is the true monotone-improvement
>       signal, not the shaped reward. (Extends 2a's benchmark-SPARC into a per-ckpt time series.)
> - [ ] **Curriculum gate + rollback (what MAKES it monotone across stages).** We *observed*
>       catastrophic forgetting at a stage transition (lethal→non-lethal attacker). So: advance a
>       curriculum stage only when its gate metric (dealt/taken ratio, stable over N evals) is met;
>       on a benchmark-SPARC drop > X%, **restore the last-good ckpt and widen slower**. Converts
>       "increasing through time" from hope into mechanism.
> - [ ] **THEN a medium ~2–4 GPU-hr learning-curve validation, NOT the full run.** Single stage
>       (≈ stage C), with the three guards live. **Verify the benchmark-SPARC curve actually rises
>       and doesn't collapse.** If it climbs → the full multi-stage long run (Steps 2c–2d) is
>       justified. If it plateaus → the reward/curriculum needs work; you've spent ~$2–4, not ~$25.

**2a. Instrument the six trackers** (extend `train_adversarial` metrics + `make_fight_figures`):
- [ ] dealt/taken ratio (headline, >1 = winning) · first-contact time · clean-hit rate
      (`dealt>τ AND taken<τ`) · mutual-contact rate (both >τ — drive DOWN) · post-hit disengage
      (outward velocity after a hit) · SPARC vs a fixed benchmark set (passive/mover/striker/HoF).
- **Verify:** a `--tiny` run emits all six before any long run.

**2b. Reward shaping for WINNING** (the `dealt≈taken` fix; current `6·(dealt−taken)+5·(clos−flee)` nets ~0 when trading):
- [ ] **Clean-hit bonus** `+w·dealt·(1−taken_norm)` — hits landed while not being hit.
- [ ] **Trading penalty** `−w·min(dealt,taken)` — punish mutual contact.
- [ ] **Post-hit disengage bonus** (outward velocity after a `dealt` event; **anneal** so it
      doesn't become fleeing). Keep the always-on upright/alive anchor + the contact signal.
- [ ] **Velocity shaping stays on early** — their ablation: no forward/approach velocity → the
      skill doesn't learn at all. Validates our `--approach-weight`; anneal it later.

**2·W. Pneumatic kinetic striker (a real attacking component — the physical enabler of clean hits).**
A fast-extending steel rod lets A deal damage from OUTSIDE B's leg reach → land a hit WITHOUT being
hit, which is exactly what `clean_weight` rewards. The body already has a rigid `_spear` weapon geom
(`gen_robot_mjcf._leg_xml`, `is_weapon`) + a legs-as-weapons damage model keyed on it; this makes it
a POWERED linear DOF. Sizing (bore 16 mm @ 8 bar → F≈161 N, moving mass ≈0.25 kg, stroke 0.10 m):
peak tip ≈11 m/s, full extension ≈18 ms, tip KE ≈16 J — out-runs a punch.
- [ ] **Body** (`robot.toml [striker]` + `gen_robot_mjcf`): per weapon-leg, a `slide` joint
      `{leg}_strike` (axis down, range 0..stroke, return spring, stiff end-stop), a steel rod geom
      (`density=7850`), and a **pneumatic actuator** (`general` constant-force `F=P·π(bore/2)²`,
      `ctrlrange 0 1`, `dyntype=filter dynprm=valve_tau` = valve/fill lag). Default `striker=off` ⇒
      body byte-identical (existing ckpts/tests unaffected).
- [ ] **Env** (`train_adversarial`): slide joints are EXCLUDED from the 38-D loco obs (hinge-only
      `_Aqa/_Ada`) but INCLUDED in the action space (`+1 fire dim/striker`); add `_rod` to the
      weapon masks; **damage scales with rod tip-speed** (`+K·pen·|slide_vel|` — fast strike = more
      damage); small **firing cost** so it fires only to connect.
- [ ] **Warm-start** grows the policy/value **action head** 12→12+n (split the final layer into
      mean/log-std halves, pad each with neutral new dims) so the fighter keeps its skills and only
      LEARNS the trigger (mirrors the obs 38→44 pad).
- **Two gotchas:** (1) +DOF ⇒ the byte-identical warm-start needs the action-head pad above
      (`robot.toml`: "from-scratch only if a leg/DOF is added"). (2) **Tunneling** — at 11 m/s a
      sub-step moves ~44 mm vs the 0.004 s timestep; B's 100 mm torso is caught but for thin/glancing
      hits drop the timestep (~0.001–0.002 s), fatten the tip, or use a swept-segment damage check.
- **Verify:** striker body builds (mujoco loads it; action_size grows by n_strikers, rod geoms
      present); env steps + `dealt` responds to a fired rod; a 12-action fighter ckpt warm-starts into
      the striker body (shared dims behave identically). Then the gated curriculum trains the trigger.

**2c. Two-stage training (NOT end-to-end — end-to-end gave them the same degenerate optima):**
- [ ] **Stage 1 — pre-train skills.** Our `cval` (strike a passive B in range) **is** their
      score-vs-untrained skill; add a **get-up/recovery skill** (episodes terminate on fall →
      train recovery to a target pose).
- [ ] **Stage 2 — distill + self-play.** Distill the skills into one agent with **adaptive-KL
      regularization** (drop reg once critic Q clears a threshold → agent surpasses the skills),
      then self-play.

**2d. Curriculum (extends `--sep-lo/--sep-hi` + `--azimuth`; warm-start chain, resume-safe, pulled):**
- [ ] **A** spawn IN range (sep 0.3–0.4) → strike · **B** near (0.4–0.6) → step-to-strike ·
      **C** medium (0.6–1.2) → close-to-strike (≈ what we have).
- [ ] **D — opponent MOVES** (scripted drift/juke; B currently passive) → timing.
- [ ] **E — opponent ATTACKS back** (scripted lunge or a frozen snapshot; `taken` now
      adversarial) → hit-without-trading.
- [ ] **F — self-play / Hall-of-Fame** (`selfplay_mjx.py`): sample opponents from the **first
      quarter** of snapshots (all-snapshots is unstable) + **condition the critic on opponent-ID**.
- **Verify (per phase):** dealt/taken ratio rises and stays >1; clean-hit up, mutual-contact
      down; benchmark SPARC up. **Falsified if** ratio≈1 (still trading) or it flees (dealt
      collapses) → retune the reward asymmetry / disengage term, not more steps.

**2e. Sim-to-real recipe (for the real motorloop battlebot):** targeted DR (friction, joint
offsets, IMU pose, ±torso mass, **latency 10–50 ms**) + **random pushes** (5–15 N) + high-freq
control + **action filter** `u_t=0.8 u_{t-1}+0.2 a_t` + safety shaping (**upright** + **knee/joint
peak-torque penalty** — ties to `joint_torque_limit`). `reality_gap`/`domain_model` cover the DR
half; add the filter + torque penalty + perturbations.

**2·C. The adaptive COACH (`arena/coach.py`) — replace brittle hand-tuned reward weights.**
Instead of fixing `clean_weight`/`fire_shaping`/... by hand, a closed-loop controller measures each
competency from the held-out decomposition and moves its reward weight (and DIFFICULTY) toward a
target: laggard ↑, satisfied ↓ (can't over-optimize into a degenerate optimum), stuck → back off
(don't pour reward into an unlearnable hole). This is the continuous-weight form of Isaac Lab's
RewardManager+CurriculumManager — but with the CLOSED LOOP they leave to you. Architecture: **shaped
coaching reward for LEARNING, sparse unshaped verdict for JUDGMENT** (so the policy can't just learn
the coach's hints).
- [x] **Sparse verdict** (the honest judgment): per-bout **win-rate** (`Σdealt−Σtaken>0`), **survival**
      (didn't fall), **actuator-safety** (didn't slam actuators). **keep-best flips onto win-rate**, not
      the dense SPARC. (Immediately exposed a hidden weakness: `survival=0` — it falls every bout.)
- [x] **Reward levers** (reward what's lagging): clean / trade / fire / approach, progress-gated.
- [x] **One CURRICULUM lever** (practice what's lagging): melee gap (`sparc_close−sparc_med`) → narrow
      the spawn (`sep_hi`, inverted) → force close combat. The first "difficulty" lever.
- [ ] **EXTENSION 1 — broaden the LEVERS** (asap): the Coach should also drive **scenario/opponent
      selection** (melee lagging → pick a *rushing* HoF snapshot) and **reset conditions**, not just
      reward + spawn-range. The `League`/`Curriculum` schedules already expose these; wire the Coach to them.
- [ ] **EXTENSION 2 — broaden the COMPETENCIES toward the REAL robot** (asap): add reward terms +
      gauges + verdicts for the real-battlebot competency set — **balance, tracking, recovery, contact
      discipline, energy use, actuator safety, sensor robustness**. (`safe_rate` + an `--energy-penalty`
      term are the first two; the rest need reward terms in the env.) This is the actual prize: the same
      coach that defends "use the rod" today defends "stay within the actuator/energy envelope" on the
      physical robot — ties to `joint_torque_limit` + the `reality_gap` sensor model (2e).
- **Verify:** each lever raises the laggard then decays it (CPU selftest ✓); the win-rate curve (not
      SPARC) is what keep-best + figures track; every coach intervention is a `coach` event in the trace.

### STEP 3 — re-run the decisive experiment on the STRONGER fighter (the definitive version)
- [ ] Repeat Step 1 with the win-exchanges fighter; this is the publishable co-design result
      (does calibrated robustness pick better fighters when contact dynamics genuinely matter).
- [ ] Analysis figures (cheap, from the paper): value-function maps + UMAP behavior embeddings
      (emergent tactic diversity).

Note: the paper used off-policy **MPO/DMPO** (distributional critic), not PPO; brax-PPO is fine,
but if sample-efficiency bites, SAC/MPO + distillation is the proven fallback. Their budgets
(scale calibration): get-up 2.4e8, soccer skill 2.0e9, full 1v1 9e8 env-steps — combat skill is
~10²–10³× locomotion's steps (why our fighter is compute-hard, and why Step 1 leads).

## Done-when
**Step 1 — DONE (2026-06-21):** combat body-ranking answered — robust(CVaR) ρ=+1.00 ≥ nominal
+0.96 > proxy −0.61 (proxy *anti-correlated* on combat), spread 55.3. The co-design headline.
**Step 2** (if pursued): a policy that **wins exchanges** — `dealt/taken > 1`, clean-hit up,
mutual-contact down, sane post-hit reset, rising benchmark SPARC vs moving + attacking opponents —
**and** the medium learning-curve validation (2·0) climbed before the full run. **Step 3:** the
decisive ranking rerun on that stronger fighter. All restorable + reproducible via `make`.

## What NOT to do
- Don't do the heavy RL (Step 2) before the cheap decisive run (Step 1) — the headline result may
  already be reachable on the fighter we have. *(Step 1 confirmed it was — robust > nominal > proxy.)*
- **Don't launch the full long run before the 2·0 medium (~2–4 GPU-hr) learning-curve validation
  CLIMBS** — with win-reward asymmetry, per-ckpt benchmark eval, and a curriculum gate/rollback
  live. A long run on a flat reward / forgetting-prone curriculum plateaus and burns ~$25 to
  prove it; the medium check costs ~$2–4 and tells you first.
- Don't conclude "robust ranking is useless" from the locomotion null — rerun it on COMBAT first.
- Don't widen separation faster than the policy keeps landing hits (the curriculum's whole point).
- Don't call `dealt>0` "winning" — it's engaging; `dealt/taken>1` + low mutual-contact is winning.
- Don't reward disengage so hard it becomes fleeing — anneal it.
- Don't put opponent-attacks (E) before strike (A) — keep the contact signal alive at each rung.
- Don't train the full task end-to-end — pre-train skills → distill → self-play (their degenerate-
  optima result == our `dealt=0`).
