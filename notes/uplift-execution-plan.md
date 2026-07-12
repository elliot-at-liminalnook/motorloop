# Uplift Execution Plan — Ordered Tasks to Retire Every Audit Finding

> **Document status:** Historical · **Audience:** Robot-learning contributors · **Last reviewed:** 2026-07-12 · **Durable verdict:** [`training-uplift-results.md`](training-uplift-results.md)

This file is the dated execution record. It intentionally preserves commands,
status logs, and superseded branches; it is not the current launch runbook.

Date: 2026-07-02. Companion to `training-uplift-audit.md` (the findings) and
`rl-verification-playbook.md` (the check designs). This is the execution artifact:
an ordered list of tasks — investigation, code changes, commands, and validation —
that, when every box is checked, means the audit is fully acted on. A coverage map
at the bottom ties every audit item (top-10, honorable mentions, rejected,
cross-cutting constraints) to the task(s) that retire it, so nothing is silently
dropped.

Conventions per task: **Investigate** (what to confirm before touching code),
**Change** (files/lines), **Run** (commands), **Done when** (validation bar).
Effort tags: S < 1 h, M ≈ half-day, L ≈ multi-day. GPU tasks are marked 💰 and
**must not start without an explicit balance top-up + authorization** (pod
terminated; balance ≈ $4). Standing rules throughout: no commits/pushes unless
asked; any sentence of the form "the robot now Xs" requires a rendered video.

Decision gates are marked ⛔ — work below a gate is conditional on its outcome;
don't pre-build past one.

**Status log (2026-07-02):** 0.1–0.2 ✅ (venv `.venv-sim` py3.12; pre-fix baseline
reproduced: gear=1.0, ctrl=1 → 1.0 N·m, artifact preserved in scratchpad).
A.1 ✅ (gear emitted; measured 12.97 N·m at every hinge — NB the outcome field is
`qfrc_actuator`, post-gear; `actuator_force` is pre-gear and reads 1.0 on a correct
model too — playbook T0 sketch corrected accordingly). A.2 ✅ (7 tests incl.
fires-on-stripped-model proof). A.3 ✅ (outcome-measured tmax, canonical-stance
absolute bar, drop test, max-effort ratio [pre-fix signature 0.08, fixed 0.99];
`--strip-gear` regression demo: fixed exits 0, stripped exits 1). A.4 ✅
(stiffness 2.0→0.5; validator identical at 2.0/0.5/0.0 — springs were pure crutch;
DR bracket → [0,1]; prove_robot green). A.5 ✅ (peak_torso_z 0.45→0.60
geometry-derived; height reward saturation 0.24→0.285; `not_moving` gate rebuilt on
a 0.5 s velocity-vector EMA — instantaneous speed went blind on the strong body;
rung-2b fixture rebuilt as a true 50-step oscillator). A.6 ✅ (γ 0.99, rs 0.1,
episode 600 as flag, grad-clip 1.0 + ADAPTIVE_KL defaults [verified: no script ever
passed them], batch 512/mb 8, steps 100M, bench 600 steps, PBT cmd updated + lr
note). A.7 ✅ (`ppo_nets.py` drop-in; 15 files re-routed; final |w|max 0.00153 =
0.01× lecun; initial mode action exactly 0, std 0.694). A.8 ✅ (`preflight.py`;
historical config aborts with 2 red lines + fan-in warning, new config passes at
1,221 iterations; `--preflight strict/warn/off`, PBT subprocesses get warn). A.9 ✅
(PROGRESS clamped at commanded magnitude, air-time capped 2×target, body-frame
obs velocities + body-frame 3-D cmd (vx,vy,yaw-rate) with 2-D caller padding,
prev_action in obs before the goal tail, pose+slip penalties, PD divisor gear not
forcerange; smoke green in pd mode). A.10 ✅ (5 exploit fixtures + honest open-loop
trot [grid-tuned: 1.5 Hz/0.35/0.30 → 0.16 m/s, return 394 vs stand-still 257];
ordering green post-A.9, VIOLATED on git-HEAD reward re-run on the fixed body —
the fixtures demonstrably fire on the historical exploit class). A.11 ✅ (mode
matrix incl. never-run `pd`; static flag-liveness 3 trainers; reset-diversity
xfail(strict) documenting the AutoReset defect until B.4; RNG-advance). **A.12 🔄
launched 2026-07-02** (pod y22mf3t2b3ye08, A100 80GB $1.39/hr; preflight green in
production: 977 iters, 2.0 s horizon; 40M steps pd-mode, tag `pdval`). B.5 ✅
(stagnation tripwires in both trainers [speed-floor commanded, bh_disp fighter],
regression tripwire ×3-consecutive with `--regression-abort`; videos-as-monitoring
via existing pbt `--video-fatal`). B.2 ✅ (ckpt_meta.py sidecars: action semantics
+ body-model hash + behavior baseline on every save; `validate_frozen_opponent`
rejects sidecar-less/mismatched artifacts [`--allow-legacy-opponent` escape];
pre-fix pulls moved to sim/build/gpu/out/retired/ with README; behavioral
warm-start acceptance wired where warm-starts happen — C.2/B.6 load paths). D.1 ✅
(RND default tactical in trainer, `--rnd-feature` exposed + forwarded in
pbt_train).

**Status log 2 (2026-07-02, evening):** A.12 training on pod y22mf3t2b3ye08
(ssh 38.128.232.232:30957, /tmp/rp.sh helper): progress/step **+0.556 m/s at
10.6M/40M steps** — above the 0.3 bar at 27% budget; video verdict pending run
completion (~50 min; sentinel watcher live). B.1 ✅ (`--action-mode pd` default:
per-substep 250 Hz PD on stance-relative targets, gear divisor; history obs H=3
×(qpos,qvel)+prev_act before HER tail; critic +4 foot contacts; obsB mirrors incl.
B history; test_obs_schema.py 4/4 green incl. zero-action stance-hold; sidecar
semantics `pd_target@250hz:scale=X` enforced by validate_frozen_opponent). B.3 ✅
code (composable knobs, defaults = legacy reward byte-identical: `--combat-scale`
k_c, `--loco-speed` exp-kernel tracking toward opponent w/ 0.5 m taper [replaces
farmable clos term when on], `--loco-drill-frac` per-episode rider, `--alive-bonus`,
`--gait-airtime-w/slip/pose` [displacement-EMA-gated, capped]). B.4 ✅ code
(reset_bank.py BankedAutoResetWrapper via wrap_env_fn: K-bank real reset draws,
70% launch states w/ 1 settle step, coprime-stride cursor, learning-info preserved;
`--diverse-resets 256` default ON; liveness tests rewritten: stock-brax-defect
documented + banked-diversity asserted — RUNNING). C.1 ✅ code (`--ko-weight/-alpha/
-done`, dealt_cum-gated zero-sum terminal, α·dense+(1−α)·outcome, ko_done never in
bench). C.3 ✅ driver level (first_quarter deprecated; pfsp_pick w∝(1−p)²+ε over
FULL HoF + hof_win bookkeeping in selfplay_drive; per-env in-step HoF stacking
deferred to when league runs resume — recorded). **Decisions recorded:** D.2
(correlated exploration) deferred — needs brax acting-loop surgery, revisit only if
B.6 shows action chatter PD doesn't fix; D.3 post-B.6 evidence; D.4 only if gait
asymmetry visible in B.6 video; D.5 reserves stay. Phase DR adopt/reject: decide
after A.12 verdict (leaning: adopt DR.1+DR.2 only if A.12 passes decisively and
co-design stays dormant, per the amendment's own advise-against). B.6 launch
command (after verdict + re-ship): `train_adversarial.py --steps 120000000
--lidar-obs --engage-obs --contact-obs --loco-speed 0.35 --combat-scale 0.3
--loco-drill-frac 0.25 --gait-airtime-w 1.0 --gait-slip-w 0.1 --gait-pose-w 0.2
--alive-bonus 0.05 --stagnation-disp-floor 0.5 --regression-abort --tag wtf1`
(~4.5 h, ~$6.5). Remaining: C.2 (needs pdval walker ckpt — pull after render),
C.4 investigation, C.5 per-env levels, A.5 rungs re-run (running), B.6+verdict.

**Status log 3 (2026-07-02 night): ⛔ A.12 VERDICT = PASS.** 40.5M steps, 71 min,
~$1.65. Final training eval progress **+0.607 m/s**; deployment square-route eval:
**0.83 m/s mean** under 1.2 commanded, survived 600/600 steps, recovered from one
stumble (z_min 0.10) without falling. Rendered video
`sim/build/gpu/out/pdval/pdval_forward.mp4` + contact sheet: visible alternating
stride, torso up throughout — the eyeball gate is met at 2.8× the bar. THE BODY
WALKS; the gear bug was the whole story. Artifacts pulled (ckpts + eval JSON +
resolved config). Rungs 4/4 ✅ (A.5 closed). **Phase V: V.1 ✅ V.2 ✅ (pydantic
RobotSpec, 5/5) V.3 ✅ (55 knobs) V.6 ✅ V.7 ✅ (goldens pinned; gait_signature.py)
V.8 ✅ — Drake sidecar LANDED: independent plant agrees 0.003% torque / machine-ε
gravity; gear-strip demo fires at 12.97× disagreement; drake==1.54 local-only.**
V.4/V.5 remain (during B.6 window, before Phase C obs changes). **B.6 LAUNCHED**
(tag wtf1, 120M steps, preflight green 1,465 iters, tripwires armed, pd+history+
banked-resets+curriculum: loco 0.35 m/s, combat 0.3, drills 25%; ~4.5 h, sentinel
watcher live). Render on pod needs `MUJOCO_GL=egl` + apt libegl1 (recorded).
Spend so far: ~$4 of $34.
**WARP LADDER (goal 2, 2026-07-03):** R1 ✅ — .venv-warp (mujoco 3.10/mujoco-warp/warp-lang 1.14);
parity 1e-4..1e-6 on single+mesh; fight-scene 3.5e-2 divergence root-caused to OUR degenerate
rod-axis geometry (fixed: rod z+2mm in gen_robot_mjcf, documented); MJX static alloc measured
778 contacts/3140 efc for 20 real; warp 13-19× faster on CPU; bench_warp_vs_mjx.py harness ready.
TL M1+M2 ✅ — sim/robot/warplayer/ 7/7 tests: analytic contacts 0.004-0.05% of MuJoCo-C (kill bar
1%); EXACT loop-coordinate joint: dt=0.004 through TDC, peak |φ̈|=1e4 (physical ceiling) vs 7e9
constraint-model — the reduced-coordinate thesis proven. M3/M4 need pod + fused obs/reward.
R2 🔄 fork debugging first-hold transient (knee −587° windup; constraint math verified correct,
consistent-pose 4e-5, late-sweep 2.1mm; mirrored-coefficient hypothesis first). R3 🔄 (clone diffs
live: mjwarp +275 lines, mjx ray +81). POD 🔄 (A100 matrix running, <$2 guard). THROTTLE: no new
agents; coordinator-only from here.

**V.4+V.5 ✅ (fork): obs_schema.py named slices (her width owned by her_goal.GOAL_DIM),
info_keys.py 27 fighter + 22 commanded keys tagged episodic/persistent with
registration-completeness tests; 4 suites green (obs_schema+info_keys 9, rnd_her 7,
lidar_env 9 — incl. fix-forward of two pre-B.1 hardcoded dims in test_lidar_env,
surfaced B.1 test debt). PHASE V COMPLETE 8/8.**

---

## Phase 0 — Local prerequisites (no GPU, do first)

### 0.1 — Local CPU venv with the pinned sim stack — S
Today `import mujoco` fails on this machine, so none of the T0–T4 checks below can
run locally. Everything in Phases A.2–A.11 depends on this.
- **Investigate:** confirm `requirements-gpu.txt` pins install cleanly on CPU
  (`jax[cuda12]` → plain `jax==0.6.2` for local).
- **Run:**
  ```bash
  python3 -m venv .venv-sim
  .venv-sim/bin/pip install jax==0.6.2 mujoco==3.9.0 mujoco-mjx==3.9.0 \
      brax==0.14.1 flax==0.10.7 optax==0.2.8 numpy==2.2.6
  .venv-sim/bin/python -c "import mujoco, brax, jax; print(mujoco.__version__, jax.default_backend())"
  ```
- **Done when:** the import line prints `3.9.0 cpu` and
  `.venv-sim/bin/python -m pytest sim/robot/test_lidar_env.py -q` passes (baseline
  sanity that the existing suite runs locally).

### 0.2 — Baseline snapshot before any physics change — S
The gear fix invalidates every checkpoint and every gate calibration. Freeze the
"before" state so later comparisons are honest.
- **Run:**
  ```bash
  git stash list && git status --short   # confirm what's uncommitted
  .venv-sim/bin/python - <<'EOF'
  import mujoco, numpy as np
  m = mujoco.MjModel.from_xml_path("sim/robot/model.xml")
  print("gear[:,0] =", m.actuator_gear[:, 0])
  print("forcerange =", m.actuator_forcerange)
  EOF
  ```
- **Done when:** the pre-fix gear values (expected: all 1.0 on hinges) are recorded
  in this file or a note — the "8% torque" claim is now locally reproduced, not
  just audit testimony.

---

## Phase A — Decisive cheap experiments (audit items 1, 3, 5, 2a + the checks they demand)

Order matters: every later baseline must run under A.1/A.6/A.7 or its evidence is
confounded (the audit's four simultaneous confounders: torque, budget, γ, network).

### A.1 — Emit `gear` on hinge motors (audit #1, 10/9, THE prerequisite) — S
- **Investigate:** `gen_robot_mjcf.py:141` computes one `tau = joint_torque_limit(spec)`
  for all hinges (`:45-50`: peak_motor × `robot.toml` gear=12 ≈ 12.97 N·m). Confirm
  the striker `<general>` actuators (`:159-162`) are left untouched — they're
  pneumatic, gain-driven, and correct as-is.
- **Change:** `gen_robot_mjcf.py:157-158` — add `gear="{tau:.3f}"` to the hinge
  `<motor>` template. With gear=tau and ctrl∈[−1,1], `forcerange="±tau"` becomes a
  true (now-reachable) clamp; keep it.
- **Run:**
  ```bash
  .venv-sim/bin/python sim/robot/gen_robot_mjcf.py --config sim/robot/robot.toml --out sim/robot/model.xml
  grep -c 'gear=' sim/robot/model.xml
  ```
- **Done when:** every hinge `<motor>` in the regenerated `model.xml` carries
  `gear≈12.97`, and A.2's outcome assertion passes. Note in the diff that **all
  prior checkpoints, frozen opponents, and anti-cheat calibrations are now invalid**
  (retired formally in B.2 / A.5).

### A.2 — T0 model-contract test (new `sim/robot/test_model_contract.py`) — S
The check that makes A.1's bug class impossible forever.
- **Change:** new pytest module that regenerates the MJCF from `robot.toml` and asserts:
  - `m.actuator_gear[hinges, 0] ≈ tau` (rtol 1%) — the attribute is emitted;
  - `ctrl=1 → mj_forward → abs(d.actuator_force[hinge] − tau) < 0.05·tau` — the
    **outcome**, exercising the same causal path training uses;
  - total mass within spec ±5%; joint ranges match `robot.toml`; `timestep ×
    frame_skip` equals the control dt the reward math assumes; foot geoms exist and
    collide with the floor under the contact mask.
- **Run:** `.venv-sim/bin/python -m pytest sim/robot/test_model_contract.py -q`
- **Done when:** suite passes on the fixed model AND fails when run against a
  gear-stripped model (verify the test can actually catch the original bug —
  temporarily revert A.1 to prove it fires, then re-apply).
- **Rider (Phase DR, optional):** DR.2 extends these assertions across an
  independently compiled engine — same outcomes, second derivation. A.2 green does
  not wait on it.

### A.3 — T1: outcome-based `validate_body.py` overhaul — M
- **Investigate:** current script reads `m.actuator_forcerange` as `tmax` and
  searches for the "tallest STABLE stand" (adaptive bar). Both patterns are on the
  playbook's banned list.
- **Change:** `sim/robot/validate_body.py` —
  - measure deliverable torque via commanded-ctrl outcomes (A.2 style), never a
    metadata field;
  - hold the **canonical training stance** (the `stand` pose both envs PD around),
    not the tallest pose that happens to work — absolute bar, no search-down;
  - torque-margin bar: max gravity-holding torque per joint ≤ 60% of measured
    deliverable torque;
  - drop test (release from spawn height → settles standing, no limb penetration,
    no explosion) and max-effort test (full ctrl on one leg → joint acceleration in
    an expected band; catches weakness AND the new 13×-stronger catapult risk).
- **Run:** `.venv-sim/bin/python sim/robot/validate_body.py` (before and after A.4).
- **Done when:** fixed body passes all bars; the pre-fix model.xml **fails** the
  stance-hold (regression-proves the validator now has teeth).
- **Rider (Phase DR, optional):** DR.5 later upgrades the settle/stance spot-checks
  to a Lyapunov-style certificate; not a Phase A dependency.

### A.4 — Conditional spring retune (audit #1 rider) — S ⛔ mini-gate
The parallel springs (`robot.toml:37`, joint_stiffness=2.0) were tuned to prop up
the accidentally weak robot.
- **Investigate:** with real torque, does the stance-hold pass with stiffness 0.5?
  0.0? Sweep locally via A.3's harness.
- **Change:** `robot.toml:37` joint_stiffness 2.0 → 0–0.5 **only if**
  `prove_robot.py` + A.3 stay green at the lower value; update the DR bracket at
  `robot.toml:86` (`joint_stiffness = [0.0, 2.0]`) to bracket the new center.
- **Run:** `.venv-sim/bin/python sim/robot/prove_robot.py` and A.3.
- **Done when:** a stiffness value is chosen with recorded pass evidence, or the
  retune is explicitly deferred with a reason (acceptable: "revisit after A.12").

### A.5 — Recalibrate physics gates for a 13×-stronger body — M
A robot that could barely twitch could not catapult; the fixed one can.
- **Investigate:** which `anti_cheat.py` metrics and `test_rungs.py` thresholds
  encode magnitudes calibrated on the weak body (airborne height 0.35 m, damage
  gates, energy scales, `AIRBORNE_PENALTY` / `HEIGHT_WEIGHT` env defaults at
  `train_adversarial.py:1243-1245`).
- **Change:** re-derive thresholds from the fixed body's measured capabilities
  (A.3's max-effort test gives the new envelope).
- **Run:** `.venv-sim/bin/python -m pytest sim/robot/test_rungs.py -q` against the
  fixed model.
- **Done when:** rungs pass with thresholds justified against the new envelope, not
  copied forward.

### A.6 — PPO hygiene + real budget (audit #3, highest confidence-per-effort) — S
- **Change:**
  - `train_adversarial.py:1800`: `discounting=0.97 → 0.99`, `reward_scaling=1.0 → 0.1`
    (ship together with grad clip — single Adam over policy+value);
  - `train_adversarial.py:1798`: `episode_length=300 → 600` (start; 1000 if
    time-to-contact demands it) — currently hardcoded, make it a flag;
  - flip the already-plumbed-but-never-passed flags to defaults:
    `--max-grad-norm` None→1.0, `--lr-schedule` None→ADAPTIVE_KL with
    `--desired-kl 0.01` (`train_adversarial.py:1209-1214`) — verified: no script in
    the repo ever passes them;
  - defaults batch 512 / minibatches 8 / unroll 20 (81,920 steps/iter) and
    from-scratch `--steps` guidance 100–150M (~3.8 h at 7,300 steps/s);
  - `train_commanded.py:174`: `discounting=0.97 → 0.99` (reward_scaling is already
    0.1 there);
  - extend benchmark/eval rollout lengths to match the new episode length
    (`eval_fighter_benchmark.py`, `selfplay_drive.py` bench args).
- **Known interaction (record in code comment):** ADAPTIVE_KL degrades PBT's lr
  dimension to initial-lr-only — note it in `pbt_train.py`'s PBT_PARAMS.
- **Done when:** A.8's preflight prints the new derived quantities (iterations ≥
  200 for from-scratch budgets; credit horizon ≥ 2× stride period) and the resolved
  config echoes the flipped defaults.

### A.7 — Custom network factory: (512,256,128) + 0.01× final-layer init (audit #5) — M
- **Investigate:** `train_adversarial.py:1786-1793` `_asymmetric_factory` currently
  falls through to brax defaults (4×32) — confirm hidden sizes are not set;
  enumerate the ~15 checkpoint-reconstruction sites (render/eval/frozen-opponent
  loaders) that must agree on architecture.
- **Change:** one shared helper (e.g. `sim/robot/networks.py`) exporting the
  factory with `policy_hidden_layer_sizes=(512,256,128)`,
  `value_hidden_layer_sizes=(512,256,128)`, and final-layer kernel init scaled
  0.01× (policy starts at the stand pose with healthy exploration σ instead of
  saturated bang-bang); route `train_adversarial.py`, `train_commanded.py`, and
  every reconstruction site through it.
- **Done when:** a grep shows zero call sites constructing brax networks with
  default sizes; a smoke rollout confirms initial actions are near-zero
  (|action| ≪ 1 at init); old checkpoints are explicitly rejected on shape
  (they're invalid anyway per A.1).

### A.8 — T2 trainer preflight (print + hard-fail) — M
- **Change:** shared `sim/robot/preflight.py` called at the top of
  `train_adversarial.py`, `train_commanded.py`, `pbt_train.py`:
  - print env-steps/iteration, total PPO iterations (**fail < 200 from-scratch**),
    credit horizon dt/(1−γ) (**fail < 2× task timescale**), episode seconds (warn
    < 2× time-to-contact), obs_dim:hidden[0] fan-in (warn > 4:1);
  - echo argv + the resolved post-defaults config; **fail on unknown/ignored
    flags**; dump resolved config JSON into the run directory.
- **Done when:** launching with the old config (γ=0.97, 12M steps, 1024×16×20)
  aborts with the two red-line failures; the new defaults pass; every run
  directory from now on contains `resolved_config.json`.

### A.9 — De-exploit `commanded_env.py` rewards (audit #2a) — M
- **Change (all in `sim/robot/commanded_env.py`):**
  - clamp PROGRESS (`PROGRESS_W`, `:40`; reward at `:545`) at the commanded
    magnitude — overspeed currently out-earns exact tracking by +3.7–4.6/step;
  - cap air-time credit: `min(air_time, 2·AIRTIME_TARGET)` at `:536-539` so long
    hops can't out-earn cadence;
  - express tracked velocities in **body frame**;
  - add yaw-rate command + `prev_action` to the actor obs (obs-layout change —
    respect the HER goal-tail constraint: insert BEFORE the goal tail; do not grow
    `_loco` in place, `_rnd_feat_dim` is hardcoded to LOCO_OBS);
  - add pose-deviation (from stand pose) and feet-slip penalties.
- **Done when:** A.10's exploit fixtures rank every cheat below the honest
  reference, and a 10-step smoke rollout in both `pd` and `cpg_pd` modes still runs.

### A.10 — T4 reward red-team fixtures (new `sim/robot/test_reward_exploits.py`) — M
- **Change:** scripted policies run through the **actual** `env.step` path:
  stand-still, in-place oscillation (max joint speed, zero displacement), torso
  jitter, single-impulse catapult, spin-in-place — plus an honest scripted
  approach/walk reference. Assert every exploit's episode return < the honest
  reference's, for both `commanded_env` and (Phase B) `AdversarialEnv`.
- **Process rule now in force:** every future discovered exploit becomes a
  permanent fixture the same day; new reward terms don't merge without a green
  fixture suite.
- **Done when:** suite passes post-A.9 and demonstrably fails pre-A.9 (run once
  against the unfixed reward to prove the fixtures would have caught stand-still
  and oscillation before any GPU was spent).

### A.11 — T3 mechanism-liveness tests — M
- **Change:** new/extended pytest modules:
  - **mode matrix:** 10-step smoke rollout for every control/obs mode any
    production script uses — critically `CMD_CONTROL_MODE=pd` (`commanded_env.py:46`
    defaults to `pd` yet only `cpg_pd` was ever exercised);
  - **flag reachability:** for each trainer CLI flag, assert it changes the
    resolved config (A.8's JSON) or a logged value — catches the next
    plumbed-but-never-passed flag;
  - **reset diversity:** step a wrapped env through 3 auto-resets; assert post-reset
    qpos/spawn separations are pairwise distinct — this FAILS today (brax
    AutoResetWrapper replays one cached state per env) and stays red until B.4
    lands; mark xfail(strict) so B.4 flipping it green is itself verified;
  - **RNG advance:** per-env noise streams differ across envs and steps
    (generalize the pattern already good in `test_lidar_env.py`).
- **Done when:** mode-matrix + flags green; reset-diversity red-as-expected
  (xfail), flipping green in B.4.

### A.12 — 💰 The decisive experiment: pure-`pd` validation run — ⛔ MAJOR GATE
**Requires: user top-up + explicit authorization. ~1–1.5 h on an A100 ($1.39/hr).**
The single cheapest experiment that decides "can this body walk at all" — with real
torque (A.1), sane hyperparameters (A.6), real capacity (A.7), and a de-exploited
reward (A.9), all preflight-verified (A.8).
- **Run (pod side):**
  ```bash
  # provision A100 80GB pod, ship code, then:
  /tmp/rp.sh 'bash /root/proj/sim/robot/setup_pod.sh'
  /tmp/rp.sh 'source /root/proj/out/env.sh && cd /root/proj/sim/robot && \
    CMD_CONTROL_MODE=pd python3 -u train_commanded.py --steps 40000000 \
      --episode-length 600 --evals 20 --tag pdval 2>&1 | tee /root/proj/out/pdval.log'
  /tmp/rp.sh 'source /root/proj/out/env.sh && cd /root/proj/sim/robot && \
    python3 -u render_commanded_video.py --tag pdval --mode forward && \
    python3 -u eval_commanded.py --tag pdval --mode square --out-json /root/proj/out/pdval_eval.json'
  # pull pdval.mp4 + eval json back before terminating the pod
  ```
- **Monitoring (T6):** watch foot air-time and displacement per eval, not just
  reward; stagnation tripwire — if displacement hasn't crossed a task floor by
  ~1/6 of budget, kill the run early and save the money.
- **Done when — PASS:** rendered video shows a visible stride ≥ 0.3 m/s (eyeball
  gate; metrics select, video confirms). Locomotion ground truth now exists; the
  checkpoint becomes the seed for C.2's walker-pursuer. **→ proceed to Phase B.**
- **Done when — FAIL:** go to A.13; **do not** burn weeks on B.4/B.7 items — the
  audit's explicit instruction.

### A.13 — 💰 Contingency probes: can this body do 0.3 m/s at all? (C30 + C25) — ⛔ gate
Only if A.12 fails.
- **C30 — predictive-sampling MPC probe:** vmapped harness exists in
  `search_cpg_gait_mjx.py`; run CEM/CMA-ES trajectory optimization over periodic
  spline knots on the fixed 13 N·m model. If trajopt at full torque finds no
  0.3 m/s cycle, that is hard evidence for the pre-authorized body change — an
  answer for minutes of compute. (Skip the DAgger-distillation half; in-repo BC
  anti-correlation is documented.) **If Phase DR is adopted, DR.3's certified
  probes supersede this sampling harness** — CEM/CMA-ES can only ever report "we
  didn't find one"; DR.3 can report "one exists" (constructive) or "none exists in
  this class" (certificate). Keep the CEM harness as fallback only.
- **C25 — annealed assistive harness:** xfrc torso wrench as the fully-assisted
  upper-bound probe (also the reserve training aid if the bound looks good).
- **Done when:** a written verdict exists: "body can (proceed to B with evidence)"
  or "body cannot (execute the body adjustment, then loop back to A.1's contract
  tests on the new body)."

---

## Phase B — Fighter locomotion (audit items 2b, 7, 4, 6; gate: a RENDERED stride)

### B.1 — One obs/action-layout migration: PD actions + history obs + critic contacts (audit #2b + #7) — L
Land as a single migration so the layout breaks once, not three times.
- **Change (`sim/robot/train_adversarial.py`):**
  - **actions:** target = stand_pose + 0.3–0.5·action; PD computed **per-substep
    inside the frame_skip fori_loop** (250 Hz), replacing the 50 Hz direct-torque
    write at `:684-686`. Striker channel unchanged. Direct torque at 50 Hz is the
    best-documented wrong action space for legged RL, and this structurally
    explains the warm-start non-transfer (PD-semantics outputs read as torques);
  - **actor obs:** ring buffer of last 3–5 control steps of (qpos, qvel) +
    prev_action (FIFO pattern already exists for lidar) — at 50 Hz with springs and
    no foot sensors, stance/swing is unobservable from one frame;
  - **critic tail:** 4 per-foot contact booleans into the privileged obs;
  - **layout discipline:** append after the proprio prefix and BEFORE the HER goal
    tail; `_rnd_feat_dim` hardcoded to LOCO_OBS — don't grow `_loco` in place;
  - **schema:** write the obs layout as named slices in one module with a test that
    HER's goal-tail and RND's feature-slice assumptions match it (today those are
    implicit positional contracts).
- **Done when:** mode-matrix smoke (A.11) extended to the fighter passes; a
  zero-action rollout holds the stand pose (the PD sanity check); obs-schema test
  green; `test_rnd_her.py` and `test_lidar_env.py` pass against the new layout.

### B.2 — T7: checkpoint semantics + retire every pre-fix artifact — M
- **Change:**
  - embed metadata in every checkpoint: MJCF model hash, action semantics
    (`torque` | `pd_target`+scale), obs-layout version, trainer config hash;
  - `validate_frozen_opponent` rejects on **semantics** mismatch, not just shape
    (shapes matched while torque-trained opponents were silently mis-driven);
  - warm-start acceptance becomes behavioral: after loading, roll the policy in the
    target env and assert ≥ 50% of source displacement — "plumbing ok" is not
    transfer;
  - **retire** all pre-A.1 checkpoints and frozen opponents (move to
    `out/retired/`, never load-able without a `--i-know-this-is-preugear` override).
- **Done when:** loading any retired artifact fails loudly; a fresh checkpoint
  round-trips with metadata verified.

### B.3 — Walk-then-fight curriculum in AdversarialEnv (audit #4) — L
- **Change (`train_adversarial.py`):**
  - Phase-0 reward: exp-kernel velocity tracking of cmd = v_des·unit(opp−self) with
    distance taper; v_des annealed 0.10→0.6 m/s via the existing `curriculum_drive`
    phases (exp-kernel is maximized only at sustained matched velocity — oscillation
    scores ~0);
  - port upright/actrate/velz regularizers from `commanded_env`;
  - contact-based gait terms from real MJX foot-floor contacts: per-foot air-time
    capped and **gated on positive displacement**, feet-slip penalty, pose
    regularization (the fighter currently has ZERO terms distinguishing stepping
    from jitter);
  - combat terms × k_c annealed 0→1, promotion gated on the behavior benchmark;
    replace the farmable instantaneous closing term; anneal the alive bonus;
  - **retention rider:** ~25% of episodes stay pure velocity-command loco drills
    for the whole combat run (per-env task flag + jnp.where) so combat gradients
    can't erase the gait.
- **Done when:** A.10's fixture suite extended with fighter-specific exploits stays
  green; reward-term unit tests confirm the exp-kernel and air-time gating behave
  on synthetic trajectories.

### B.4 — Custom reset wrapper: kill the frozen-reset defect + RSI + pushes (audit #6) — L
- **Change:** implement via the **unused `wrap_env_fn` hook** in brax ppo.train
  (the sanctioned entry point; PBT inherits it via subprocess):
  - K≈256 distinct reset draws per env-bank instead of one cached state;
  - "launch" states: stance pose + root velocity U(0.1–0.5) m/s, random heading;
    after A.12 passes, add mid-stride snapshots from validated walker rollouts
    (RSI/DeepMimic — attacks the diagnosed root cause: the value function never
    sees mid-stride states, so there's no stepping gradient);
  - periodic planar velocity kicks; keep ~30% canonical starts; stock resets for
    benchmark envs so eval comparability survives;
  - **tricky part:** per-env `state.info` contents (notably the in-info RND
    predictor + Adam state) must persist correctly across reset-bank swaps.
- **Done when:** A.11's reset-diversity xfail flips to green; an RND-enabled smoke
  run confirms predictor state survives a bank swap; benchmark envs still use
  stock resets.

### B.5 — T6 in-run tripwires + mandatory render gates — M
- **Change:** in the trainer eval loop and `pbt_train.py`:
  - log behavioral metrics every eval: bh_disp, bh_approach, per-foot air-time,
    contact-break count (reward up + air-time flat ⇒ exploit alarm);
  - stagnation tripwire: bh_disp under a task-derived floor by N iterations ⇒ stop
    the run (the 12M-step 0.18 m run should have self-terminated at ~2M);
  - regression tripwire: judge score > X% below peak for K consecutive evals ⇒
    checkpoint freeze + alert (the cpglong collapse burned ~40M steps post-peak);
  - videos ARE monitoring: `pbt_train.maybe_render_video` machinery on every
    milestone; run with `--video-fatal` so a missing video artifact fails the
    milestone rather than being cosmetic.
- **Done when:** a synthetic stagnating run (tiny steps, zero reward) demonstrably
  self-terminates; milestone without video ⇒ hard failure.

### B.6 — 💰 Fighter locomotion run — ⛔ GATE: rendered ≥0.3 m/s stride toward opponent
**Requires authorization.** First full run of B.1+B.3+B.4 under A.6/A.7/A.8
(100–150M steps ≈ 4–6 h A100).
- **Run:** `train_adversarial.py` with the new defaults; preflight must be green;
  tripwires active; render via `render_fight_video.py` at each milestone.
- **Done when:** a rendered fight-arena video shows a visible stride ≥ 0.3 m/s
  toward the opponent. Metrics select candidates; the video is the gate. This
  project has claimed "walking" from metrics twice; never again.

### B.7 — Clock-based trot-contact reward (audit #8) — ⛔ deploy ONLY if B.6 fails to stride — M
- **Change:** [sin,cos](2π·φ) in obs; reward matches trot-phase swing/stance
  expectations (FL+RR / FR+RL) against real foot contacts, gated multiplicatively
  on MEASURED closing speed — farming the swing-phase term IS stepping (Siekmann
  et al., Cassie, from scratch, no reference motion). Supplies the gait bias the
  CPG tried to give, but through reward — the policy owns the full action space.
- **Escalation if even this fails:** CEM/CMA-ES trajopt (A.13's C30 harness) →
  DeepMimic tracking + RSI; if trajopt at 13 N·m finds no 0.3 m/s cycle, that's the
  body-change evidence.

---

## Phase C — Combat (only after a rendered walker exists)

### C.1 — Sparse zero-sum KO objective + dense→sparse anneal (audit #9) — M
- **Investigate:** confirm the episode currently does NOT end on knockdown, and
  keep-best/judge select on win while the step reward contains no outcome term
  (trained vs selected objectives disjoint).
- **Change (`train_adversarial.py`):** opp_down terminal ±W (≈10, zero-sum),
  **gated on the existing closing/dealt gates — gate the TERMINATION too** (the
  passive opponent sags on its springs; ungated KO-done pays for waiting);
  anneal total = α·dense + (1−α)·outcome, α 1→0.2 (Bansal et al.: ablation-critical
  for exactly this exploit class). KO-done is a constructor flag left **OFF in the
  benchmark env** or `survived_bout` silently inverts win_rate.
- **Done when:** benchmark env verified KO-done-off; fixture suite extended with a
  "wait for the sag" exploit fixture that scores below honest aggression.

### C.2 — Walker-pursuer opponent (audit #10a) — M
Hard dependency: the validated walker checkpoint from A.12/B.6.
- **Change:** `--opponent walker`: frozen commanded-env walker drives B
  (commanded-layout obs with cmd = unit(A−B)·speed, same PD mapping), speed
  annealed 0.1→0.35 m/s — replaces the open-loop sinusoid the repo's own notes
  admit "is NOT yet a competent pursuer." A pursuer makes standing still lose
  **structurally**, not via gate patches.
- **Done when:** rendered rollout shows B actually pursuing; stand-still fixture
  vs the pursuer scores a loss.

### C.3 — PFSP mixture over the full hall of fame (audit #10b + riders) — L
- **Change:**
  - generalize `--opp-ckpt` to a list: `tree_map(jnp.stack)` K HoF param sets,
    per-env `opp_idx` gathers in `step()`;
  - **delete the oldest-25% sampling** (`selfplay_drive.py:54-57`,
    `first_quarter()`) — best-responding to ONE deterministic opponent from the
    oldest quarter is a textbook cycling setup (likely behind the cpglong
    peak-then-collapse);
  - win-rate-prioritized weights w∝(1−p)²;
  - riders: **C21** opponent-conditioned critic (opponent identity into the
    privileged critic tail only), **C23b** resample-on-reject (league currently
    halts on a rejected checkpoint instead of drawing another).
- **Done when:** a test shows per-env opponents differ within one batch; league
  soak run (small budget) shows no halt on rejection; judge-score trajectory
  tracked for reduced cycling vs the B.6 baseline.

### C.4 — Fall-recovery grace window (X3) — M
- **Investigate first:** can the body get up at all? One-way knee makes this
  non-obvious — script a get-up attempt from side-lying at full torque before
  writing any reward.
- **Change (if feasible):** knockdown starts a grace timer instead of auto-loss;
  recovery within the window resumes the bout.
- **Done when:** verdict recorded; if infeasible, the auto-loss stands with the
  investigation as evidence.

### C.5 — Per-env curriculum levels (retire the subprocess phase curriculum) — L
Second pass on B.4's wrapper: per-env integer level in `state.info` (spawn
sep/bearing/opponent activity, promoted on success), retiring `curriculum_drive`'s
5 subprocess phases and their recompiles.
- **Done when:** one continuous run reproduces what previously took 5 subprocess
  phases, with no recompile boundaries; per-env promotion visible in logs.

---

## Phase D — Cheap riders (slot in anytime after Phase A; each S)

### D.1 — RND repair (C19): tactical features by default — S
Flip the default to tactical descriptors (engagement geometry, contact events,
striker-tip speed — not proprioception) and expose `--rnd-feature` in
`pbt_train.py` (near-free; the jitter-farming fix). **Done when:** default flipped,
flag reachable per A.11's flag test.

### D.2 — Temporally-correlated exploration (C18) — S
`fold_in(key, t//k)` so exploration noise is piecewise-constant over k steps (keeps
Gaussian marginals, PPO math unchanged). **Done when:** unit test confirms noise
constant within a window, differing across windows/envs.

### D.3 — Low-pass action filter (C26) — S, only if PD actions still chatter
Largely subsumed by B.1's PD mode; keep α≈0.5 only if action-rate metrics stay
high post-B.6. **Done when:** decision recorded either way.

### D.4 — Mirror-symmetry aux loss (C10, aux-loss variant ONLY) — M, optional
The trajectory-duplication variant has an off-policy log-prob flaw in brax — do
not build it. Aux-loss variant only, if gait asymmetry is visible in B.6 videos.

### D.5 — Held in reserve (do NOT build now, keep on the list)
- **C25** assistive harness — reserved as A.13's upper-bound probe / training aid.
- **C28/C29** hierarchical frozen-walker / kickstarting-KL — only if monolithic
  walk-then-fight fails at the transfer boundary again.

---

## Phase DR — Drake sidecar (proposed amendment, 2026-07-02): independent verification + certified feasibility

**Status: proposed; adopt or reject as a unit.** CPU-only, zero GPU cost, zero
changes to the MJX/brax training loop, obs layouts, or checkpoints. Runs as a
**parallel track that never gates A.1–A.12** — its only couplings are the three
rider lines above (A.2, A.3, A.13) and the sync table below. If bandwidth ever
forces a choice between this track and Phase A, Phase A wins, always.

### Why a second engine — the argument from this project's scar tissue

The gear bug did not survive for the project's entire lifetime because nobody
checked torque. It survived because **a single simulator can never disagree with
itself**: the pre-fix `model.xml` was perfectly self-consistent — weak, but
coherently weak — so every check that consulted the model (or any artifact
compiled from it) confirmed the wrong physics. And the checks' *expectations*
share fate with the model too: `joint_torque_limit()` in `gen_robot_mjcf.py:45`
is the single torque derivation consumed by the generator AND by
`validate_body.py:53`, `nsga2.py:145`, and `coevolve.py:91`. Even A.2's
outcome-based assertion compares a measured force against an expectation computed
by the same module that emitted the MJCF — a units slip or wrong-field read in
that one function corrupts the model and the assertion identically, and the test
goes green. Finally, A.2 exercises classic MuJoCo while training consumes MJX;
that particular gap is a *runtime* gap (already `test_parity.py`'s job), but both
runtimes consume the same MjModel — so **model-level** defects are shared-fate
across every path the project currently has.

Drake (MIT/TRI's multibody toolbox) breaks the shared fate, if and only if its
model is derived independently: separate code reads the same `robot.toml` and
re-derives torque limits, inertias, and axes from `sim/tests/motors.py`
primitives. For the two engines to then agree on physical **outcomes** — force
delivered at full command, gravity-holding torque at stance, settle behavior —
the spec→model compilation has to be right twice, by two implementations that
share no code. That converts the entire class of single-simulator
self-consistency bugs from *undetectable in principle* to *red CI the same day*.
It is the playbook's Rule 3 ("every spec claim gets an assertion") applied across
engines, and it is the same cultural move the RTL layer made years ago: the
firmware is never trusted to agree with itself — it is simulated **against
independently-written models of the parts**. The robot side currently has no
equivalent referee. The four wounds, mapped:

| Documented wound | Drake answer |
|---|---|
| Gear bug: invisible to any within-engine check; expectations share code with the model (`joint_torque_limit` everywhere) | DR.2 cross-engine contract on outcomes |
| A.13/C30: stochastic search can only say "we didn't find a gait" — exactly the ambiguity that let the 0.083 m/s CPG result stand un-diagnosed (body limit? search limit? it was torque) | DR.3: collocation = constructive existence proof; convex-relaxation infeasibility = genuine impossibility certificate |
| Co-design evaluates morphologies with the most expensive, noisiest oracle available (RL training) | DR.4: seconds of CPU trajopt as fidelity-0 filter |
| "Stance is stable" = a settle test passed on N seeds | DR.5: Lyapunov/ROA certificate with explicit validity bounds |

### DR.1 — Drake toolchain + independently-derived model from `robot.toml` — M
- **Investigate:** pip `drake` wheels on Linux bundle SNOPT/IPOPT/OSQP/Clarabel —
  no solver licenses needed for collocation; pin the current release into
  `.venv-sim`. Decide floating-base representation and where the canonical stance
  pose + PD gains come from (a shared constants module — never from `model.xml`).
- **Change:** new `sim/robot/gen_robot_drake.py` building a `MultibodyPlant`
  programmatically from `robot.toml`: bodies with re-assembled inertias, revolute
  joints with limits, `JointActuator` effort limits, compliant contact on feet.
  **Independence rules, stated in the module docstring:** (1) never read
  `model.xml`; (2) never import `joint_torque_limit` or any derivation from
  `gen_robot_mjcf.py` — re-derive τ from `motors.py` primitives (kt ×
  peak_factor × rated current × gear). Sharing the raw `tomllib` load is fine;
  sharing *interpretation* is what recreates shared fate.
- **Run:**
  ```bash
  .venv-sim/bin/pip install drake            # pin exact version after first install
  .venv-sim/bin/python sim/robot/gen_robot_drake.py --config sim/robot/robot.toml --summary
  ```
- **Done when:** plant loads with the expected DoF count; the summary prints
  per-joint effort limits from the independent derivation; total mass and COM
  match spec within 1% (asserted, not eyeballed).

### DR.2 — Cross-engine contract test (rider on A.2; new `sim/robot/test_cross_engine.py`) — M
- **Investigate:** fix agreement bands per quantity class BEFORE writing asserts,
  so failures are decisions already made: contact-free quantities tight (mass/COM
  1%; per-joint gravity-holding torque at the canonical stance, feet welded,
  via inverse dynamics: 3%; deliverable torque: 1%; pinned single-joint
  acceleration under τ: 2%); contact-mediated quantities loose (settle-standing
  **boolean must match**; settled torso height within 15%). Trajectory-level
  agreement is explicitly NOT asserted — two contact solvers will never match
  step-for-step, and chasing that is the rabbit hole this design avoids.
- **Change:** the test asserts, MuJoCo-measured vs Drake-derived:
  1. **deliverable torque** — A.2's ctrl=1 → `mj_forward` measured force ==
     Drake's independently derived effort limit;
  2. **inertial outcome** — apply τ to one pinned joint in each engine → joint
     angular accelerations agree (referees mass/inertia assembly, not labels);
  3. **stance-hold feasibility** — Drake inverse-dynamics gravity torque per
     joint < deliverable torque, in both engines' numbers;
  4. **settle** — drop from spawn height in both; standing booleans agree,
     heights within band.
  Adjudication policy in the test header: a firing contract is a
  **stop-the-line** event; the tiebreaker is a hand calculation or a
  bench-measured number — never "trust the engine we train on."
- **Run:** `.venv-sim/bin/python -m pytest sim/robot/test_cross_engine.py -q`
- **Done when:** green on the fixed body, AND proven against history twice:
  (a) run against the pre-fix gear-stripped `model.xml` — assertions 1 and 3 must
  fire with the historical signature (Drake: rear knee needs ~4.8 of ~12.97 N·m
  available; MuJoCo measures 1.0 deliverable) — the gear bug as a **day-one
  detection, demonstrated, not argued**; (b) perturb a `robot.toml` mass 10%
  and regenerate only the MJCF — the contract fires on the divergence, which
  answers the standard objection head-on: yes, a second model description can
  drift, and *that drift is precisely what this test converts from silent to
  loud*. The maintenance cost buys an alarm, not a liability.

### DR.3 — Certified feasibility probes (supersedes C30 inside A.13) — M ⛔ feeds the A.13 gate
- **The epistemic upgrade, stated precisely:** CEM/CMA-ES returning nothing means
  "our sampler didn't find one." A converged direct-collocation solution is a
  **constructive existence proof** — a checkable trajectory satisfying dynamics,
  torque limits (12.97 N·m), friction cones, and periodicity at the target speed.
  An **infeasible convex relaxation** (centroidal dynamics + friction cones +
  torque images over a fixed contact schedule) is a genuine certificate: necessary
  conditions violated ⇒ no whole-body gait exists in that schedule class. Sandwich
  the truth: convex-infeasible ⇒ certified impossible; collocation-converged ⇒
  certified possible; in between, refine. Honesty rule: **NLP non-convergence
  alone is treated as no evidence** — only the convex certificate may support an
  impossibility claim.
- **Change:** new `sim/robot/drake_gait_probe.py`: (a) convex probe per
  (gait, cadence, speed) cell; (b) whole-body multi-phase direct collocation with
  prescribed contact sequences (trot / walk / bound), periodicity constraints,
  torque limits, friction cone, average speed ≥ target; grid ≈ 3 gaits × 4
  cadences × 3 speeds. Any converged solution exports as a time-series `.npz`.
- **Run:** local CPU; whole grid well under an hour (SNOPT collocation
  ~10–120 s/cell, convex probes ~0.1–2 s/cell).
- **Done when:** a written verdict artifact exists, one of:
  (i) a **reference-gait library** at ≥ 0.3 m/s — which as a side effect supplies
  B.4's mid-stride RSI states, B.7's DeepMimic escalation reference, and the very
  reference data whose absence killed C31 (AMP) in the audit; or
  (ii) an **infeasibility map with convex certificates** — the affirmative
  body-change evidence A.13 exists to produce.
  Historical-bug demo: run the grid on the pre-fix 1 N·m model — the convex probe
  must certify even static stance infeasible in seconds. That is the probe this
  project needed on day one: 12 M RL steps bought "0.18 m and confusion"; this
  buys a certificate before breakfast.
- **Sync:** A.13 keeps C25 (assisted-harness upper bound, GPU); the C30 sampling
  harness is demoted to fallback.

### DR.4 — Feasibility oracle as fidelity-0 in the co-design ladder — S–M
- **Investigate:** insertion point in `multifidelity.py` / `nsga2.py` candidate
  evaluation (nsga2 already calls `joint_torque_limit` at `:145` — the gate slots
  where that check lives, but computes outcomes, not attributes).
- **Change:** candidate params → `robot.toml` overrides → DR.1 builder → convex
  stance + single-gait probe; infeasible ⇒ cull before any policy training, with
  cull counts logged (the no-silent-caps rule).
- **The compute asymmetry:** convex probe ~0.1–2 s CPU, collocation ~10–120 s CPU,
  versus ~1–4 A100-hours (≈ $1.40–5.60) of RL training per candidate today —
  three to four orders of magnitude. At any plausible cull rate the filter pays
  for the entire sidecar within one NSGA-II generation, and it de-noises the
  Pareto front: RL-eval variance no longer decides life or death for bodies that
  were never physically viable (the pre-fix robot is the existence proof that
  such bodies enter the population).
- **Done when:** ladder logs culls; a known-good body passes; a torque-starved
  pre-fix-equivalent candidate culls; existing co-design tests stay green with
  the gate in place.

### DR.5 — Stance certificate (extends A.3; strictly AFTER the A.13 verdict) — L, lowest priority, explicitly droppable
- **Honest scope:** a full region-of-attraction for a floating-base system with
  unilateral contact is research-grade — not proposing that. The deliverable is a
  **pinned-stance certificate**: weld the feet, close the loop with the envs'
  actual PD gains, linearize at the canonical stance, take the Riccati Lyapunov
  function, and certify a level set via SOS (Clarabel/CSDP ship in the drake
  wheel) — minimum viable fallback: eigenvalue margin + falsification sampling,
  with the weaker claim stated as such. Validity boundary checked inside the level
  set: contact normal forces > 0 and friction cone respected (the certificate is
  explicitly valid only until contact would break).
- **Done when:** a certificate artifact (level-set description + margins +
  contact-validity bound) exists for the fixed body, and the pre-fix demo runs:
  at 1 N·m the equilibrium torque exceeds saturation, so **no stabilizable
  equilibrium exists and the certificate computation fails at step one** — a
  categorical statement that "the settle test passed on N seeds" (A.3's bar, and
  the old validator's) can never make. This is the same rung the RTL layer already
  climbed: hardware is verified against models; the robot side graduates from
  spot-checks to certificates, one narrow claim at a time.

### Sync points (this track never blocks Phase A)
| Sync | What happens |
|---|---|
| A.2 lands | DR.2 attaches as a rider; A.2 goes green without it |
| A.3 lands | DR.5 noted as its future upgrade; no dependency |
| A.13 opens | DR.3 runs first; C30 sampling is fallback-only |
| Co-design wing wakes | DR.4's gate is already in the ladder |
| Body redesign triggered | DR.1–DR.4 transfer as-is (spec-parameterized); rerun on the new spec before its first GPU-hour |

### Cons — and the conditions under which this amendment should be REJECTED
- **Learning curve:** pydrake is a large API; expect 2–4 focused days to the first
  trustworthy plant + collocation. That time comes out of a real budget.
- **A second model description to maintain:** every morphology change now touches
  two builders. Mitigated — not eliminated — by both consuming the same
  `robot.toml` and by DR.2 making divergence same-day-loud. The recurring cost is
  real and should be priced into every future body change.
- **Contact-model adjudication:** MuJoCo's solref/solimp and Drake's compliant
  contact WILL disagree quantitatively. That's why DR.2 pins tight bands only on
  contact-free quantities and booleans/coarse bands on settle. **Kill criterion:**
  if the settle bands can't be closed within ~2 days of tuning, downscope DR.2 to
  assertions 1–3 (mass / kinematics / actuation — where the historical bug
  actually lived) and drop settle comparison without guilt.
- **Verification for a body that may be redesigned:** A.13 can trigger a new body.
  DR.1–DR.4 survive (spec-parameterized — arguably worth *more* then, since the
  new body gets certified before its first GPU-hour). DR.5 is body-specific,
  which is exactly why it's sequenced after the verdict and marked droppable.
- **Advise AGAINST adopting** if any of the following hold: (a) A.12 passes
  decisively AND the co-design wing stays dormant — then only DR.2 carries its
  weight as regression insurance; adopt DR.1+DR.2 alone and skip the rest;
  (b) the contact-adjudication kill criterion trips AND co-design is inactive —
  the sidecar is then mostly maintenance; (c) this track would delay any Phase A
  task by even a day — it is parallel or it is nothing.

---

## Phase V — Typed contracts, shared constants, and loud-early failure (added 2026-07-02)

**Why, from today's own session:** three near-misses in eight hours of implementation.
(1) The T0 verification sketch read `actuator_force` (pre-gear) instead of
`qfrc_actuator` — caught by suspicion, would have been caught by a disagreeing
second engine. (2) The A.12 run trained in `cardinal` mode with yaw commands
silently dormant — half the system's config flows through `CMD_*` env vars that
preflight never sees. (3) kp=30/kd=1.0 now live in FOUR places (commanded_env,
validate_body default, fighter `_pd_kp_act`, and the env-var fallback strings) —
today's edits duplicated 0.02/0.004/0.04/0.09 magic numbers again because there
is no shared home for them. Ordering principle: contracts closest to the metal
first — a constant that's wrong breaks everything above it, so it must be the
first thing that can refuse to build. Pydantic is adopted for OFFLINE validation
only (spec load, config resolution, preflight) — never inside jit paths; add
`pydantic==2.*` to requirements-gpu.txt and `.venv-sim`.

### V.1 — `sim/robot/constants.py`: one home for every load-bearing number — M
- **Investigate:** inventory the duplicates: TIMESTEP 0.004, FRAME_SKIP 5,
  CONTROL_DT 0.02, FOOT_CONTACT_Z 0.04, FALL_Z 0.09, TOPPLE_UP_Z 0.3, PD_KP 30 /
  PD_KD 1.0, VMAX 1.2, AIRTIME_TARGET 0.2 (+cap 2×), DAMAGE_REF 0.05, LOCO_OBS 38,
  stance sag bars, launch-speed range (0.1, 0.5), EMA β 0.04, move_eps 0.1.
- **Change:** constants module with a one-line derivation comment per constant
  (where it comes from, what breaks if it moves); consumers import — grep-zero for
  the raw literals in env/validator/test files afterwards. Env-var overrides stay
  but route THROUGH the module (`CMD_PD_KP` read once, in one place).
- **Done when:** `test_model_contract.py` asserts the compiled model agrees with
  the module (timestep, dt); a lint test fails on new bare occurrences of the
  inventoried literals in sim/robot/*.py reward/obs code.

### V.2 — Pydantic `RobotSpec`: robot.toml validated at load — S/M
- **Change:** typed model (units in field names, trust-tag enum) with cross-field
  validators: gear > 0, knee range one-way, DR brackets CONTAIN their center
  (the joint_stiffness bracket drifted from its center once already), masses > 0,
  spawn_height > kinematic stance height. `load_spec` returns the validated model
  (dict-compatible); a malformed/missing/typo'd field dies at BUILD, not 40 min
  into a training run.
- **Done when:** mutating any robot.toml field to nonsense fails the load with a
  named error; the DR-bracket validator fires on a deliberately off-center bracket.

### V.3 — Preflight env-var contract: the `CMD_*` constellation made visible — S ⚡ do first
- **Change:** a declared registry (pydantic Settings) of every `CMD_*` /
  `AIRBORNE_*` / `HEIGHT_*` env knob with type + default + consumer; preflight
  dumps resolved values into the run's resolved-config JSON and **fails on any
  unknown `CMD_*` var in the environment** (typo detection — `CMD_TRIAN_MODE`
  today silently does nothing).
- **Done when:** the resolved-config JSON for a run shows the full constellation
  (the cardinal/yaw surprise becomes impossible to miss); an injected typo'd var
  aborts strict preflight.

### V.4 — Named-slice obs schema (single source of truth) — M, BEFORE any Phase C obs change
- **Change:** `obs_schema.py`: ordered named slices (loco, lidar, priv, hist,
  prev_act, contacts, her_goal) computed from env config; env `_obs`/`_lidar_obs`
  assemble FROM the schema; HER's goal-tail and RND's feature slice read from it;
  test_obs_schema asserts the schema's total == observation_size and round-trips
  each slice. Positional conventions become named lookups.
- **Done when:** her_goal.py and rnd path contain no hardcoded slice arithmetic;
  an obs-layout change is one schema edit + green tests.

### V.5 — Info-key registry: episodic vs persistent — S
- **Change:** declare every `state.info` key with a lifetime tag
  (`episodic` = reset each episode: prop_hist, air_time, vel_ema, dealt_cum;
  `persistent` = survives episode swaps: rnd_predictor, rnd_opt_state, lidar_rng).
  reset_bank + any future wrapper asserts it handled every registered key —
  the audit's bank-swap caveat becomes checkable instead of a comment.
- **Done when:** adding an unregistered info key fails a test; reset_bank's
  preserve-list is generated from the registry.

### V.6 — Import & pattern guardrails (CI greps, ~5 lines each) — S
- no file outside ppo_nets.py imports brax's ppo networks factory;
- no new `actuator_forcerange` used as a torque DIVISOR (the tmax anti-pattern);
- no `mujoco.MjModel.from_xml_path("model.xml")` in env code (envs build from
  spec — the on-disk artifact is for humans);
- **Done when:** each grep-test fails on a planted violation, passes on the tree.

### V.7 — Golden trajectories + gait signatures — M
- **Change:** (a) per env-config, a pinned 50-step zero-action + scripted-action
  rollout hash with tolerance — any physics/obs semantic drift diffs loudly
  (today's height-constant and EMA-gate edits had only indirect coverage);
  (b) gait-signature extraction on rendered rollouts (stride frequency from foot
  contacts, duty factor, diagonal phase offset) written next to every video —
  the eyeball stays the GATE, the numbers make it a characterization.
- **Done when:** goldens exist for pd/cpg_pd/fighter configs; B.6's video lands
  with a gait signature JSON beside it.

### V.8 — Drake adoption decision + ordering (executes Phase DR) — decision now, work M
- **Decision (2026-07-02): ADOPT DR.1 + DR.2** (independent model + cross-engine
  contract — regression insurance for the whole class of self-consistent model
  bugs, twice-demonstrated); **ADD DR.3** repurposed as speed-envelope calibration
  (trajopt ceiling → is commanding 1.2 m/s physically honest? feeds CMD_VMAX);
  **DEFER DR.4** (co-design dormant) and **DR.5** (revisit only for push-recovery
  work; its guessed launch-speed range is served well enough by V.1 constants +
  B.4 empirics for now). Sequencing: after B.6 launches (CPU-parallel to the run).
- **Done when:** DR.1/DR.2 land per their own done-whens; DR.3 produces the
  envelope verdict; deferrals recorded here.

**Ordering:** V.3 immediately (1 h, closes today's live near-miss) → V.1 → V.2
(both before the next GPU launch after B.6) → V.6 same day → V.4 + V.5 before any
Phase C obs/self-play change → V.7 alongside B.6's analysis → V.8 after B.6 launch.

---

## Explicitly rejected (audit-verified; do not resurrect without new evidence)

| Item | Why it stays dead |
|---|---|
| C11 anti-stagnation early termination | brax EpisodeWrapper clobbers env truncation; reset-to-standing just stands again |
| C14 adaptive velocity-command curriculum | premise refuted — PROGRESS already gives a dense gradient at v=0 |
| C16 strict PBRS everywhere | strictly weaker pull; the claimed farmable term actually telescopes; B.3 replaces the truly farmable one |
| C17 PID-Lagrangian constrained RL | headline claim wrong (penalty weights aren't in PBT_PARAMS); invasive for a non-blocker |
| C22 main-exploiter agents | wrong stage; exploiters vs a barely-walking champion rediscover physics exploits the gates already catch |
| C24 live self-play arm | L effort, kills the lidar/asymmetric line, generate_unroll rewrite + HER conflict, for a staleness problem not yet reached |
| C31 AMP | needs a ppo/train.py fork + reference data that doesn't exist (only a 0.083 m/s CPG dataset, 3.5× below target) |
| C32 prioritized level replay | no difficulty axis over identical standing resets (moot anyway once B.4 lands) |

---

## Cross-cutting constraints — verify at every relevant task

- [ ] Obs layout: HER assumes the goal occupies the LAST dims — new obs inserted
      before the goal tail (A.9, B.1); `_rnd_feat_dim` hardcoded — never grow
      `_loco` in place. Retired by B.1's explicit schema + test.
- [ ] `validate_frozen_opponent` shape-only check — retired by B.2.
- [ ] PBT restores params, not optimizer state; ADAPTIVE_KL ⇒ PBT's lr dim is
      initial-lr-only (documented in A.6).
- [ ] Per-env `state.info` (RND predictor + Adam) handled across reset-bank swaps
      (B.4).
- [ ] `wrap_env_fn` is the sanctioned custom-wrapper entry point (B.4); PBT
      inherits via subprocess.
- [ ] Every GPU run: preflight green (A.8), tripwires on (B.5), video artifacts
      mandatory, resolved-config JSON in the run dir.
- [ ] One variable per run: never launch a run that changes more than one
      hypothesis-bearing thing (the audit found four confounders precisely because
      runs changed many things at once).

---

## Coverage map — audit item → task(s)

| Audit item | Task(s) |
|---|---|
| #1 gear fix (+ spring retune, gate recalibration, DR note) | A.1, A.2, A.4, A.5 |
| #2a de-exploit + validate commanded PD | A.9, A.10, A.11, A.12 |
| #2b PD actions in fighter (+ retire opponents) | B.1, B.2 |
| #3 PPO hygiene + budget | A.6, A.8 |
| #4 walk-then-fight curriculum + retention rider | B.3 |
| #5 network factory | A.7 |
| #6 reset wrapper / RSI / pushes / per-env curriculum | B.4, C.5 |
| #7 history obs + critic contacts | B.1 |
| #8 clock trot reward + trajopt escalation | B.7 (conditional) |
| #9 sparse KO + anneal | C.1 |
| #10 walker-pursuer + PFSP + riders C21/C23b | C.2, C.3 |
| HM: C25 harness / C30 MPC probes | A.13 (C25); C30 → superseded by DR.3 if Phase DR adopted |
| HM: C10, C26, C18, C19, C28/C29, X3 | D.4, D.3, D.2, D.1, D.5, C.4 |
| Rejected: C11/C14/C16/C17/C22/C24/C31/C32 | rejected table (tracked, not built; DR.3(i) would un-block C31's missing reference data — revisit only then) |
| Verification stack the audit demands (T0–T7) | A.2, A.3, A.8, A.10, A.11, B.2, B.5 |
| DR (optional): cross-engine model referee | DR.1, DR.2 |
| DR (optional): certified gait feasibility | DR.3 |
| DR (optional): co-design feasibility oracle | DR.4 |
| DR (optional): stance certificate | DR.5 (deferred per V.8) |
| Phase V: shared constants / typed spec / env-var contract | V.1, V.2, V.3 |
| Phase V: obs schema / info registry / guardrails | V.4, V.5, V.6 |
| Phase V: goldens + gait signatures / Drake execution | V.7, V.8 |

## Definition of done for the whole plan

1. All Phase 0/A tasks checked; preflight + contract/fixture/liveness suites green
   in CI.
2. A.12 verdict recorded with a rendered video either way.
3. If A.12 passed: Phase B complete with a rendered ≥0.3 m/s stride **in the
   arena** (B.6), then Phase C complete with the PFSP league running on
   post-fix checkpoints only.
4. Every 💰 task has its authorization noted, cost logged, and artifacts (video +
   resolved config + eval JSON) pulled off the pod before termination.
5. Coverage map fully ticked — every audit row points at a completed task or a
   recorded decision not to build it.
6. Phase DR: an explicit adopt/reject decision is recorded. If adopted: DR.2
   green in CI (including both historical-bug firing demos), and any A.13 verdict
   cites DR.3 artifacts (gait library or infeasibility certificates), not
   sampler silence.

### STATUS LOG — WARP LADDER ROUND 2 (2026-07-03, user directive "work through the rest of §10")
- **§10(b)4 #868**: rank-1 cholUpdate drafted in worktree ~/Projects/mujoco_warp-868
  (branch incr-hessian-868; suite 1013 passed). HONEST MIXED FINDING: factor reuse
  (64% of solver events) = −55% kernel time and is the mergeable core; the rank-1
  update ITSELF is +4–26% slower than warp's cooperative tile refactorization on
  CPU. GPU verdict: bench_cholupdate_868.py queued on pod.
- **§10(b)5 Newton lidar**: built (12/12 tests, docs+example+CHANGELOG), published
  proposal-first per their CONTRIBUTING: issue #3346 + DRAFT PR #3347. EasyCLA
  blocked pending user: verify elliot@liminalnook.com on GitHub, sign LFX ICLA.
- **§10(c) thin layer M3/M4**: (iii) lidar kernel + (iv) obs/reward kernels +
  fused.py/bench_m3.py/m4_train_demo.py landed; 26/26 tests; physics bit-identical;
  CPU proxy 1.70× (lidar dedup). ≥2× GPU kill criterion queued on pod.
- **meshwalk1** (first mesh-robot training run): healthy to ~26M steps; reward
  1.03→5.26 then plateau at lean-and-creep (progress ~0.05 m/s); full 40M will
  complete in-timeout; render gate pending.

### STATUS LOG — LADDER ROUND 2 VERDICTS + meshwalk1 FINAL (2026-07-03 evening)
- meshwalk1 COMPLETE: 40.5M steps, 75 min, final eval reward 5.30. RENDER GATE:
  duty_factor 0.994 (feet planted 99.4%), 0.016 m/s vs 0.36 commanded — the
  lean-and-creep local optimum, coordination-trap diagnosis CONFIRMED (yaw 1.2 N·m
  cannot slew loaded feet vs ~4 N·m friction; unload-then-yaw never discovered).
  Run-2 recipe: AIRTIME_TARGET 0.45s/cap 1.0, yaw authority 0.9, clock-bonus
  bootstrap annealed, warm-start meshwalk1.pkl. Artifacts:
  sim/build/gpu/out/meshwalk1/ (video, gait json, ckpts, all bench logs).
- M3 GPU VERDICT: fused 1.22× (lidar) / 0.92× (no lidar) vs ≥2× bar → thin layer
  KILLED by its own criterion. Salvage: lidar dedup ~22% (wrapper-portable); M1/M2
  mechanism findings stand. M4 GPU demo blocked by demo-script njmax (moot).
- #868 GPU VERDICT: reuse +1.9%, rank-1 +3.2% slower at solver_niter 1.9 — honest
  negative for upstream; CPU −55% kernel win only pays at high iteration counts.
- Pod terminated (0 remaining), balance $19.70. §10 fully closed: 5/5 contribution
  targets actioned (3 Google PRs CLA-green, newton issue #3346 + draft PR #3347
  EasyCLA-green, #868 draft with negative finding), ladder rungs 1-3 validated,
  thin layer adjudicated.

### STATUS LOG — SERVO-TRUE MODEL + RUNS (2026-07-03/04 night, "get it right and send it")
- MODEL: goBILDA 2000 servo on all 12 joints (gen_mesh_robot_mjcf: GEARS/WFREE/
  ARMATURE servo-true; belt ratio 4 DESIGN DECISION; masses +60g×12, total 9.06 kg;
  worm frictionloss 2→4 resized; knee strike gate recalibrated −25→−15 mm, lift>12
  still passes). Env: torque-speed derating (drive derates to 0 at WFREE, braking
  full). 18/18 suites green. mesh_robot.xml regenerated.
- GATE meshsrv1 (8M, warm from meshval2): PASS — transferred policy knocked to
  progress 0.020 by real physics (was exploiting impossible speeds), relearned to
  0.029, reward climbing, stable. → committed long run.
- LONG meshsrv2 (40M resume): reward 5.13→5.45, align 0.51, progress 0.036 (train,
  stochastic) BUT deterministic render duty 0.993, 0.008 m/s, diag_sync 0.0 —
  CREEP, rhythm lost. ROOT CAUSE (owned): CLOCK_HZ=1.1/AIRTIME_TARGET=0.45 derived
  from the YAW sweep; the PITCH axis that lifts feet is 0.76 rad/s (5× slower) —
  the requested cadence is physically unfollowable, so the clock income is
  unreachable and creep wins. NEXT LEVER (derived, not guessed): CLOCK_HZ≈0.35,
  MESH_AIRTIME_TARGET≈1.2s, episode-length 800 (16s ≈ 5 slow cycles), anneal clock
  after breakout; consider belt ratio 6 and/or lighter torso to cut the friction
  threshold. Artifacts: sim/build/gpu/out/meshsrv/.
- Pods 0, balance $17.05 (day total ≈ $7.9 across 4 pod sessions).

### STATUS LOG — GAIT FEASIBILITY VERDICT (2026-07-04)
- The 10-upgrade retrain surfaced the real blocker via the gait-feasibility
  workstream: THE MESH LEG CANNOT WALK AS BUILT. notes/gait-feasibility-verdict.md
  has the measured proof: (1) level stance only at all-legs pitch -10/knee -50
  (z=0.448); (2) stability cliff beyond -12 deg pitch; (3) NO single-leg actuation
  lifts a foot clear (stays 4-11mm, radius 12mm — always contact); (4) 58mm
  kinematic clearance exists but only at pitch+8 which destabilizes; (5) yaw stride
  is 23mm mostly LATERAL; (6) COM-shift authority ~1-2cm vs ~3-4cm needed.
- ROOT CAUSE: leg is a stomp/strike weapon (worm pitch + slider-crank blade),
  strong DOWN authority, none UP — a fighting leg, not a walking leg. The RL creep
  result across all runs is the CORRECT gait this hardware affords, not a failure.
- reference_gait.json marked VALID=false; imitation hook has nothing valid to eat.
- Software ladder (feasibility->reference->imitation+RSI RL, warp trainer 36/36)
  is BUILT and waits on a hardware change (add foot-lift DOF / widen stance / accept
  shuffle). Re-run needs only the leg to change, not the code.
- Upgrades status: #1 feasibility DONE (verdict). #3 cadence, #5 belt6+diet, #6 SEA
  option, #4/7/8/9/10 warp trainer with curriculum/filter/entropy/asym-critic/
  telemetry+tripwire ALL BUILT & validated. #2 imitation built but moot until a
  valid reference exists (post-hardware).
