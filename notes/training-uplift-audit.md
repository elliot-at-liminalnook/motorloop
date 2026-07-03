# Training-Uplift Audit — Top 10 Algorithmic Improvements for From-Scratch RL

Date: 2026-07-01. Produced by a 52-agent audit workflow: 6 code readers (fighter env,
locomotion recipe, meta-stack, body physics, failure history, bootstrap ecosystem) →
7 expert-lens proposers → 49 raw candidates consolidated to 32 → 36 adversarially
judged against the actual code (each judge opened the files and confirmed/refuted the
candidate's premises) → ranked synthesis. Judge scores shown as impact/feasibility
(1-10) and confidence (0-1).

## The headline: this was never an RL problem first

**The robot has ~8% of its design torque.** `gen_robot_mjcf.py` (~line 157) emits
`<motor ... ctrlrange="-1 1" forcerange="±tau"/>` with **no `gear` attribute**. In
MuJoCo, motor torque = gear × ctrl, ctrl clamps to ±1, and forcerange only clamps
(never amplifies) — so every hinge maxes out at ±1 N·m instead of the intended
~±13 N·m (`robot.toml` gear=12, "clearing the measured ~3× needed to stand").
A settle test shows rear knees need ~4.8 N·m just to hold stance. The body as
simulated **provably cannot hold its own stance**, which quantitatively explains the
from-scratch stall (0.18 m / 12M steps), the CPG undulation (0.083 m/s), and the
warm-start non-transfer. Independently re-verified in `model.xml` after the audit.

Second structural finding: the canonical 12M-step runs were **~37 PPO iterations**
(327,680 env-steps/iteration at default batch=1024×16×20) vs the 500–3,000 iterations
behind every published from-scratch quadruped result, under γ=0.97 (0.66 s credit
horizon — shorter than one stride) on a default 4×32 policy network fed 470-dim obs.
The stack was simultaneously underpowered, under-trained, myopic, and under-capacity —
each of which alone could explain the failure.

## Top 10 (ranked by expected uplift × probability of success)

### 1. Restore the actuator torque envelope (emit `gear`) — 10/9, conf 0.93
Add `gear="{tau:.3f}"` (tau≈12.97) to each hinge `<motor>` in `gen_robot_mjcf.py`;
verify with mj_forward (ctrl=1 → actuator_force≈13); rerun the locomotion baseline
before anything else. Then conditionally retune `robot.toml` joint_stiffness 2.0→0–0.5
(the parallel springs were tuned to prop up the accidentally weak robot), gated on
`prove_robot.py`. Effort S. **Prerequisite for everything below.** Caveats: invalidates
all checkpoints and anti-cheat calibrations; a 13× stronger robot can catapult, so
physics gates need re-validation. DR (`reality_gap.py`) never touched actuator_gear,
so no past run had real torque.

### 2. De-exploit + VALIDATE the commanded PD recipe; unify PD actions into the fighter — 9/8, 7/9
(a) Fix `commanded_env.py` rewards: clamp PROGRESS at commanded magnitude (overspeed
currently out-earns exact tracking by +3.7–4.6/step), cap air-time credit at
min(air_time, 2·AIRTIME_TARGET), body-frame velocities, add yaw-rate command +
prev_action to obs, pose-deviation and feet-slip penalties — then actually run the
pure-`pd` validation (**only cpg_pd was ever tested**). (b) Port the same PD semantics
into `AdversarialEnv`: target = stand_pose + 0.3–0.5·action, PD per-substep inside the
frame_skip fori_loop (250 Hz, replacing the 50 Hz direct-torque write at
train_adversarial.py:684-685). Striker unchanged. Direct torque at 50 Hz is the
best-documented wrong action space for legged RL (Peng & van de Panne 2017; every
Go1/Go2 MJX recipe is PD). Also structurally explains why the warm-started locomotor
never transferred (PD-semantics outputs read as torques). The pure-pd validation run
(~1 h GPU) is the cheapest decisive experiment on "can this body walk at all."
Retire old frozen opponents — shape checks pass but semantics silently changed.

### 3. PPO hygiene + a real optimization budget — 8/10 conf 0.85, 9/10 conf 0.88
γ 0.97→0.99, episode_length 300→600–1000, reward_scaling 1.0→0.1, flip the
already-plumbed `--max-grad-norm 1.0` and `--lr-schedule ADAPTIVE_KL`
(`--desired-kl 0.01`) on by default (verified: no script ever passes them); batch
512/minibatches 8/unroll 20 (81,920 steps/iter) and from-scratch `--steps` 100–150M
(~3.8 h at 7,300 steps/s). Extend benchmark rollouts to match episode length. The
highest confidence-weighted uplift per effort in the pool; the only item that
de-confounds "body can't walk" vs "training was 30× too short." Ship reward_scaling +
grad clip together (single Adam over policy+value). ADAPTIVE_KL degrades PBT's lr
dimension to initial-lr-only.

### 4. Walk-then-fight curriculum inside AdversarialEnv — 9/8, 7/9 (+retention rider)
Phase-0 reward in the fighter: exp-kernel velocity tracking of cmd = v_des·unit(opp−self)
with distance taper, v_des annealed 0.10→0.6 m/s via the existing `curriculum_drive`
phases; port upright/actrate/velz regularizers from commanded_env; add contact-based
gait terms from real MJX foot-floor contacts (capped per-foot air-time gated on positive
displacement, feet-slip penalty, pose regularization — the fighter currently has ZERO
terms distinguishing stepping from jitter). Combat terms × k_c annealed 0→1, gated on
the behavior benchmark; replace the farmable instantaneous closing term; anneal the
alive bonus. Rider: keep ~25% of episodes as pure velocity-command loco drills for the
whole combat run (task flag + jnp.where) so combat gradients can't erase the gait.
Exp-kernel tracking is maximized only at sustained matched velocity (oscillation scores
~0); air-time on genuine contact breaks is unfarmable without lifting feet. Requires 1+2.

### 5. Custom network factory: (512,256,128) MLPs + 0.01× final-layer init — 8/9, conf 0.85
Brax defaults were never overridden: 470-dim lidar obs feeds a 4×32 policy — a 15:1
first-layer bottleneck, ~10× smaller than anything shown to learn 12-DOF locomotion.
Small final-layer init starts the policy at the stand pose with healthy exploration std
instead of saturated bang-bang jitter (the exact jitter that farmed move_weight and
proprio-RND). Extend `_asymmetric_factory` (train_adversarial.py:1786) + add to
train_commanded; route ~15 checkpoint-reconstruction sites through one shared helper.
Andrychowicz et al. rank both among the highest-impact PPO choices.

### 6. Custom reset wrapper: fix the frozen-reset defect + RSI launch states + pushes — 7/6
**Verified defect**: brax's AutoResetWrapper replays the SAME cached first state per env
forever — the per-episode spawn/sep-curriculum/joint-noise/HER re-randomization the
project already built mostly doesn't exist within a run. Replace via the unused
`wrap_env_fn` hook: K≈256 distinct reset draws + "launch" states (stance pose + root
velocity U(0.1–0.5) m/s, random heading) + later mid-stride snapshots from validated
walker rollouts; periodic planar velocity kicks. Keep ~30% canonical starts; stock
resets for benchmark envs. RSI attacks the diagnosed root cause (value function never
sees mid-stride states → no stepping gradient; DeepMimic ablation). Second pass:
per-env integer curriculum level in state.info (spawn sep/bearing/opponent activity,
promoted on success) to retire curriculum_drive's 5 subprocess phases and their
recompiles. Tricky part: persist per-env RND predictor + Adam state across bank swaps.

### 7. Proprio/action history in actor obs + foot contacts in critic tail — 6/9, conf 0.75
Ring buffer of last 3–5 control steps of (qpos, qvel) + prev action appended to actor
obs (currently a single instantaneous 38-dim snapshot; only lidar is frame-stacked);
4 per-foot contact booleans into the privileged critic tail. At 50 Hz with springs and
no foot sensors, stance/swing state is unobservable from one frame — commanded_env's
air-time reward already conditions on state the policy can't see. FIFO pattern exists
(lidar); append AFTER the prefix and BEFORE the HER goal tail; don't grow _loco in
place (_rnd_feat_dim hardcoded). Land with item 2's obs changes as one layout migration.

### 8. Periodic clock-based trot-contact reward — 8/7 (escalate to trajopt-mimic only if needed)
[sin,cos](2π·phi) in obs; reward matches trot-phase swing/stance expectations against
real foot contacts (FL+RR / FR+RL), gated multiplicatively on MEASURED closing speed.
Farming the swing-phase term IS stepping. Siekmann et al. (ICRA 2021) got Cassie gaits
from scratch with exactly this, no reference motion. It supplies the gait bias the CPG
tried to give, but through reward — the policy owns the full action space, so it can't
be trapped on a prior manifold. Deploy only if items 1–7 don't produce a visible
stride. Escalation: CEM/CMA-ES trajopt over periodic spline knots on the MJX model
(vmapped harness exists in search_cpg_gait_mjx.py) → DeepMimic tracking + RSI; if
trajopt at 13 N·m finds no 0.3 m/s cycle, that's hard evidence for the body change.

### 9. Sparse zero-sum knockdown/win terminal reward, dense→sparse anneal — 7/9, conf 0.8
The trained objective and selection objective are currently disjoint: keep-best selects
on win, the judge pays 100·win, but the step reward contains no outcome term and the
episode doesn't even end when the opponent goes down. Add opp_down terminal ±W (≈10,
zero-sum) gated on the existing closing/dealt gates (the passive B sags on its springs —
gate the TERMINATION too); anneal total = α·dense + (1−α)·outcome, α 1→0.2. Bansal
et al. (quadruped sumo): dense→sparse annealing was ablation-critical for exactly this
exploit class. KO-done must be a constructor flag left OFF in the benchmark env or
survived_bout silently inverts win_rate. Near-zero value for walking → sequenced after
locomotion.

### 10. Walker-pursuer opponents + per-env PFSP mixture over the full HoF — 7/6, 6/7
`--opponent walker`: frozen commanded-env walker drives B (commanded-layout obs with
cmd = unit(A−B)·speed, same PD mapping), speed annealed 0.1→0.35 m/s — replaces the
open-loop sinusoid the repo's own notes admit "is NOT yet a competent pursuer." A
pursuer makes standing still lose STRUCTURALLY instead of via gate patches. Generalize
`--opp-ckpt` to a list: tree_map(jnp.stack) K HoF params, per-env opp_idx gathers in
step(); delete the oldest-25% sampling (selfplay_drive.py:54-57) — best-responding to
ONE deterministic opponent from the oldest quarter of the HoF is a textbook cycling
setup (likely behind the cpglong peak-then-collapse). Win-rate-prioritized weights
w∝(1−p)². Hard dependency: a validated walker checkpoint (none exists today).

## Honorable mentions
- **C25 annealed assistive-stabilization harness** (xfrc torso wrench): strong (7/8) but
  held in reserve; doubles as the fully-assisted upper-bound probe before a body change.
- **C30 predictive-sampling MPC on MJX**: best "can this body do 0.3 m/s at all"
  diagnostic if items 1–8 fail; the DAgger-distillation half fights a documented in-repo
  BC anti-correlation.
- **C10 mirror-symmetry aux loss** (aux-loss variant only; the duplication variant has an
  off-policy log-prob flaw in brax).
- **C26 low-pass action filter/CAPS**: largely subsumed by PD mode; keep α≈0.5 only.
- **C18 temporally-correlated exploration** via fold_in(key, t//k) (keeps Gaussian marginals).
- **C28/C29 hierarchical frozen-walker / kickstarting-KL**: only if monolithic
  walk-then-fight fails at the transfer boundary again.
- **C21 opponent-conditioned critic; C23(b) resample-on-reject** (league currently halts
  on rejection): riders on item 10.
- **X3 fall-recovery grace window**: every knockdown is currently an auto-loss; get-up on
  a one-way knee unverified.
- **C19 RND repair**: flip default to tactical features + expose --rnd-feature in
  pbt_train.py (near-free).

## Rejected (with reasons)
- **C11 anti-stagnation early termination** — brax EpisodeWrapper clobbers env truncation;
  reset-to-standing just stands again.
- **C14 adaptive velocity-command curriculum** — premise refuted: PROGRESS already gives a
  dense gradient at v=0.
- **C16 strict PBRS everywhere** — strictly weaker pull; the claimed farmable approach term
  actually telescopes; item 4 replaces the truly farmable term.
- **C17 PID-Lagrangian constrained RL** — headline claim wrong (penalty weights aren't in
  PBT_PARAMS); invasive brax-loss surgery for a non-blocker.
- **C22 main-exploiter agents** — wrong stage; exploiters vs a barely-walking champion
  rediscover physics exploits the gates already catch.
- **C24 live self-play arm** — L effort, kills the lidar/asymmetric line, generate_unroll
  rewrite + HER conflict, for a staleness problem not yet reached.
- **C31 AMP** — needs a ppo/train.py fork and reference data that doesn't exist (only a
  0.083 m/s CPG dataset, 3.5× below target).
- **C32 prioritized level replay** — no difficulty axis over identical standing resets.

## Recommended order
**Phase A — decisive cheap experiments (day 1–2):** (1) gear fix + mj_forward assertion +
static stance-hold check; (2) item 3's config lines; (3) item 5's network factory — all
config-level; every later baseline must run under them or its evidence is confounded.
(4) Item 2a: de-exploit commanded_env, run the pure-pd validation on the fixed body
(~1 h GPU). If the recipe walks → locomotion ground truth exists. If it fails WITH real
torque, sane hyperparameters, real capacity, and a de-exploited reward → go straight to
the C30/C25 feasibility probes and the pre-authorized body adjustment; don't burn weeks
on items 6–8.

**Phase B — fighter locomotion (week 1–2):** (5) item 2b PD port + item 7 history obs as
ONE obs-layout migration; (6) item 4's curriculum; (7) item 6's reset wrapper. Gate on a
RENDERED ≥0.3 m/s stride, not metrics — this project has been burned by metric-only wins.
(8) Item 8's clock reward only if the stride still doesn't emerge.

**Phase C — combat (after a rendered walker exists):** (9) item 9 sparse KO objective;
(10) item 10 walker-pursuer + PFSP (+C21/C23b riders); extend the reset wrapper with
per-env levels to retire the subprocess phase curriculum.

Honest uncertainty: even with items 1–7, from-scratch walking on this body is not
guaranteed — 0.3 m/s is 3.5× beyond anything ever demonstrated on it. The sequence is
designed so failure at Phase A/B yields the specific evidence (assisted and MPC upper
bounds) that justifies the body change already authorized.

## Cross-cutting integration constraints (judge-verified)
- Obs layout: HER assumes the goal occupies the LAST dims — insert any new obs before
  the goal tail; `_rnd_feat_dim` is hardcoded to LOCO_OBS — don't grow `_loco` in place.
- `validate_frozen_opponent` checks shapes only — action-semantics changes silently
  corrupt old opponents; retire them.
- PBT restores params, not optimizer state — ADAPTIVE_KL turns PBT's lr dim into
  initial-lr-only.
- Per-env state added to `state.info` must be handled in any reset-bank swap (notably
  the in-info RND predictor + Adam state).
- The `wrap_env_fn` hook in brax ppo.train is currently unused — it's the sanctioned
  entry point for custom reset/curriculum wrappers (PBT inherits via subprocess).

Full judged pool (36 candidates with per-judge reasoning):
session task output `w3neyzn3z.output`; workflow saved as
`.claude/workflows/training-uplift-audit.js`.
