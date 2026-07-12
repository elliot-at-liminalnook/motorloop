# Locomotion Bootstrap Teacher Checklist

> **Document status:** Active · **Audience:** Robot-learning contributors · **Last reviewed:** 2026-07-12 · **Canonical summary:** [`locomotion-status.md`](locomotion-status.md)

This checklist describes the path from the current CPG/PD scaffold to a less
hand-authored locomotion system:

> Stop hand-authoring the gait. Hand-author only the interface and objectives,
> then let search, RL, and imitation discover the gait parameters.

## Current Baseline

- [x] Fix CPG search so it matches `CommandedEnv`.
- [x] Use `robot.toml` stance, not stale hard-coded stance.
- [x] Clip CPG actions to deployed motor range.
- [x] Penalize falling candidates strongly.
- [x] Verify cardinal priors through `eval_commanded.py`.
- [x] Confirm square command sequence survives.
- [x] Confirm pure CPG does not yet solve checkpoint navigation.
- [x] Confirm cardinal residual PPO does not yet solve checkpoint navigation.

## Current Verified State

- Pure CPG cardinal priors survive forward, backward, left, right, and square command evals.
- Backward, left, and right have useful net displacement.
- The original fixed sustained-forward CPG is weak; rich CPG search found stronger forward priors.
- Pure CPG checkpoint navigation reaches `0/4` waypoints.
- Cardinal residual checkpoint navigation still reaches `0/4` waypoints.
- Forward-only residual training reached `0/4` waypoints and did not solve sustained forward motion.
- Expanded rich CPG search improved forward from `x_delta ~= 0.015 m` to `x_delta ~= 0.567 m` over 720 steps.
- The best rich forward prior reaches `1/4` waypoints without falling, but is not promotable alone.
- Rich forward + rich left with tighter radius reaches `2/4` waypoints without falling.
- Rich backward and rich right priors exist, but blending all four rich priors regresses to `1/4`.
- Env-controlled router experiments are implemented, but tested router variants did not solve composition:
  `softmax6` reached `0/4`; all-rich `winner + segment-axis` reached `1/4`.
- Route-aware rich CPG search is implemented in `sim/robot/search_cpg_route_mjx.py`.
- The route search now scores the same `cpg_pd_step_target` path used by `CommandedEnv`, carries state
  through waypoint changes, saves deployable raw vectors, and avoids `jax.vmap` for route scoring because
  vmapped MJX route stats disagreed with single-rollout traces.
- Focused third-segment search is implemented with route mutation masks, frequency freeze controls, and
  extra progress/distance/cross-axis terms for the active failing segment.
- Full 2D command scoring is implemented for route search, along with deployable router settings for
  `blend`, `softmax`, and `winner` routing.
- Best route-aware candidate currently replays at `2/4` waypoints without falling; it still fails the `3/4`
  teacher promotion gate.
- Latest full-command softmax replay reached `2/4` without falling. It got closest to waypoint 2 at
  `0.181720 m` from `(0.126674, 0.480291)`, so the remaining failure is still excess Y during the
  backward/left transition.
- `cpg_route_full_softmax2_wp2_deploy_r07` also reaches `2/4` without falling. Its calibration report shows
  waypoint-2 residual burden `0.189471 m`, with required correction `[-0.168579, -0.086490]`.
- Transition-aware dataset generation is implemented and verified on the pod for the current scaffold:
  `38,560` total samples: `24,960` route-derived samples, `6,400` command-transition fallback samples, and
  `7,200` steady command-family samples.
  The route-derived set includes `5,664` waypoint-2-focused samples and `2,816` positive-progress
  waypoint-2 samples.
- Methodology update: do not require a pure CPG route teacher to reach `3/4` before BC. Treat the current
  `2/4` no-fall route as a viable scaffold, then collect transition-aware data and let BC/residual learning
  solve steering, switching, timing, and recovery.
- The current failure is now classified as non-composable priors, not simply insufficient cardinal gait search.
- Continuous-prior update: `sim/robot/cpg_teacher.py` now has `param_rbf` / `continuous` parameter-space
  routing with command direction, speed, and previous-command smoothing. `CommandedEnv` and
  `search_cpg_route_mjx.py` now pass `prev_cmd` into the shared teacher, so transition-aware prior behavior
  matches deployment.
- `out/continuous_prior_v1_gp2_continuous_prior_compare.json`: the saved `softmax` scaffold with gate power `2`
  reproduces `2/4` waypoints, `final_dist=0.188458 m`, and `focus_dist=0.177812 m`. Direct `blend` and
  `winner` replay both fall to `0/4`. The new `param_rbf` continuous parameter blend survives but reaches
  `0/4`; it improves some isolated transition probes (`left -> backward` survival `1.0` vs `0.675` and net
  cross `0.034 m` vs `0.289 m`), but breaks `forward -> left` (`net_progress=-0.319 m`) and is not a
  promotable full-route replacement.
- Transition-control update: `sim/robot/cpg_teacher.py` now exposes transition-only controls for
  `phase_reset_amount`, `phase_speed_boost`, `stance_height_boost`, `turn_bias`, and
  `cross_axis_correction`. `sim/robot/search_transition_controls.py` searches only those five controls over a
  frozen route scaffold, so transition tuning is separate from steady-state gait-vector tuning.
- `out/transition_controls_smoke_v2_transition_controls.json`: the separate transition-control search path ran
  on the pod with `pop=8`, `gens=2`. It did not pass the `3/4` route gate; the best smoke result was still
  `2/4`, so this is mechanism evidence only, not a promoted walker.
- Prior-strength observation update: `CommandedEnv` can append current prior strength before the command tail
  when `CMD_OBS_PRIOR_STRENGTH=1`; the command remains the final two observation entries. Pod smoke confirmed
  the opt-in path has `obs_size=41`, preserves shape across a step, and reports `prior_strength=0.76923`.
- Strict walker-gate update: checkpoint navigation now defaults to radius `0.07` in
  `eval_checkpoint_navigation.py`, `train_bootstrap_bc.py`, and `train_residual_locomotion.py`.
  `scripts/run_validated_training_pipeline.sh` passes `--radius "$NAV_RADIUS"` with default `0.07` and gates
  `validate_commanded.py` at `--min-nav-frac 0.75`, so future BC/residual walkers must reach at least `3/4`
  waypoints at the strict radius before the pipeline can continue to self-play.
- Waypoint-2 signed-reward update: `CommandedEnv` route reward now exposes opt-in
  `CMD_ROUTE_WP2_Y_OVERSHOOT_W`, `CMD_ROUTE_WP2_Y_VEL_W`, `CMD_ROUTE_WP2_X_PROGRESS_W`, and
  `CMD_ROUTE_WP2_X_REMAINING_W`, and `train_residual_locomotion.py` passes them via CLI flags. Pod smokes
  from known waypoint-2 miss states reported `wp2_y_overshoot=0.086574` for the original scaffold miss and
  `wp2_x_remaining=0.223525` for the residual miss near `(0.223, 0.415)`, matching the diagnosed drift and
  remaining X error.
- `out/residual_wp2_signed_v1_residual_report.json`: aggressive signed-drift residual with
  `CMD_OBS_PRIOR_STRENGTH=1` passed fixed-direction survival but regressed checkpoint nav to `1/4`;
  waypoint-2 miss worsened to `0.271604 m`, so it was rejected.
- `out/residual_wp2_signed_v2_gentle_residual_report.json`: gentler signed-drift residual kept fixed-direction
  and square survival and restored `2/4` checkpoint nav. It reduced the waypoint-2 positive-Y overshoot from
  the scaffold's about `0.08649 m` to `0.064815 m`, but waypoint-2 distance still worsened to
  `0.232435 m` because X progress was not recovered. This does not promote; the next residual target is
  preserving/recovering negative-X progress while keeping Y drift bounded.
- `out/residual_wp2_xremain_v1_residual_report.json`: direct waypoint-2 X-remaining pressure kept all fixed
  directions and square survival alive, but checkpoint navigation regressed to `0/4`; waypoint-2 miss worsened
  to `0.347302 m` and final distance to `0.536118 m`. This rejects the run and suggests the direct X-distance
  penalty was too broad/aggressive for origin-start route composition.
- Promotion-gate fix: `train_residual_locomotion.py` now resolves JSON paths from the repo root and `CODESIGN_OUT`
  and raises if `--baseline-nav` is provided but missing. This fixed a false promotion marker from the
  `residual_wp2_xremain_v1` run, where `out/...` was previously interpreted relative to `sim/robot`.
- Route-context observation update: `CMD_OBS_ROUTE_CONTEXT=1` appends a 9-D context block before the command tail:
  route-active flag, normalized active waypoint, waypoint-2 flag, goal-relative delta/distance, transition amount,
  and previous command. Pod smoke confirmed the default path remains `40` obs dims, the context path is `49` dims,
  warm-start pads `40->49`, and the command remains the final two observation entries.
- `out/residual_routectx_v1_residual_report.json`: route-context residual training with moderate waypoint-2
  penalties also kept fixed-direction and square survival alive, but checkpoint navigation regressed to `0/4`;
  waypoint-0 closest distance was `0.255443 m`, waypoint-2 miss was `0.336586 m`, and final distance was
  `0.531675 m`. This shows route context alone is not enough; residual freedom itself needs to be gated/localized
  so waypoint-2 correction pressure cannot rewrite the early route.
- Waypoint-2 residual-scale gate update: `CMD_CPG_RESIDUAL_SCALE_WP2` / `--wp2-residual-scale` can now enforce
  a higher residual scale only while active route waypoint `2` is being corrected. Pod smoke confirmed waypoint
  `0` used residual scale `0.005` while waypoint `2` used `0.03`, both with 49-D route-context observations.
- Reproducible scaffold-env update: `train_residual_locomotion.py` now accepts `--route-candidate` and merges the
  candidate JSON's `deploy_env` before training/eval. Pod smoke resolved
  `out/cpg_route_full_softmax2_wp2_focuscross_route_gait.json` to 8 deploy keys, including the raw CPG vectors,
  `CMD_CPG_ROUTER=softmax`, `CMD_CPG_GATE_POWER=2.0`, and `CMD_RESET_NOISE=0.0`.
- Explicit route-candidate replay check: with the saved route `deploy_env`, pure CPG replay still reaches `2/4`
  with waypoint-2 miss `0.189471 m`, but `residual_mixed_route_v1_r0p03.pkl` replays at only `1/4`
  with waypoint-2 miss `0.278214 m`. Future residual runs should always record/load the route candidate rather
  than relying on ambient CPG defaults.
- `out/residual_routectx_wp2gate_v1_residual_report.json`: route context plus a low global residual scale
  (`0.005`) and waypoint-2 gate (`0.03`) kept fixed-direction and square survival alive but still regressed
  checkpoint navigation to `0/4`; waypoint-2 miss was `0.350059 m`. This run is rejected.
- `out/residual_wp2only_routecandidate_v1_residual_report.json`: starting from the base walker, loading the
  explicit route candidate, and applying residual freedom only at waypoint 2 preserved scaffold navigation at
  `2/4`, but did not improve it: waypoint-2 miss was `0.189844 m` versus baseline `0.189471 m`, final distance
  was `0.198624 m`, and the run was rejected.
- Constant waypoint-2 residual diagnostic: `sim/robot/search_wp2_constant_residual.py` now reproduces the
  official remote waypoint loop, synchronizes route bookkeeping so waypoint-2 residual gates apply, then searches
  a constant 12-D residual action used only while active waypoint 2 is being corrected. The initial internal-route
  smoke was misleading because it never reached active waypoint 2; the corrected remote-loop smoke reached the
  waypoint-2 prefix at step `352`.
- Constant residual scale sweep results: the route scaffold baseline waypoint-2 miss is `0.189471 m`. Constant
  residual search improved it to `0.162087 m` at scale `0.08`, `0.150805 m` at scale `0.16`, `0.120824 m` at
  scale `0.32`, and `0.119121 m` at scale `0.64`, all without falling but still at only `2/4` waypoints. This
  proves the residual action interface can reduce the miss, but a single constant residual saturates above the
  strict `0.07 m` gate; the next target should be a phase/time-varying or state-conditioned waypoint-2 residual
  initialized from these searched labels.
- Segmented waypoint-2 residual update: `search_wp2_constant_residual.py` now supports `--segments` and
  `--init-action-json`, and `CommandedEnv` can load a searched schedule through `CMD_CPG_WP2_ACTION_JSON`.
  A 3-segment residual at scale `0.64` reduced the diagnostic waypoint-2 miss to `0.064998 m` and the official
  `eval_checkpoint_navigation.py` replay reached `3/4` waypoints at radius `0.07` with no fall. In the standard
  260-step budget, waypoint-2 closest distance was `0.043081 m`; with a 400-step budget it improved to
  `0.020456 m`. The remaining failure is now the final return-to-origin segment, not the waypoint-2 transition.
- Waypoint-3 residual update: `CommandedEnv` now has separate `CMD_CPG_RESIDUAL_SCALE_WP3` and
  `CMD_CPG_WP3_ACTION_JSON` hooks, and `search_wp2_constant_residual.py` can keep the waypoint-2 prefix at one
  residual scale while searching a different active target scale. A waypoint-3 smoke search at 400 steps improved
  active return distance only from `0.270603 m` to `0.260913 m`; a higher `1.5` target scale was worse
  (`0.268204 m`). Official replay with both WP2 and WP3 schedules stayed at `3/4`, no fall, but improved final
  distance from `0.285055 m` to `0.271778 m`. This confirms the WP3 hook works, but also confirms that blind
  scheduled residual search is not yet enough for the final return.
- Route-schedule search update: `sim/robot/search_route_residual_schedule.py` now searches waypoint-2 and
  waypoint-3 residual schedules jointly against the full route. The first standard-budget joint search,
  `route_sched_joint_v1_s260`, preserved only `3/4` and did not improve final distance; its seeded best ended at
  `0.384563 m`, worse than the prior WP2-only replay. This rejects high-dimensional blind joint perturbation as
  the next promotion path.
- Evaluator-sync update: `eval_checkpoint_navigation.py` now synchronizes `route_wp` and `route_prev_dist` to
  the external checkpoint controller before each env step when those fields exist. This removes an ambiguity
  where scheduled residual gates could use env-internal route bookkeeping that differed from the evaluator's
  active waypoint.
- Seed/timing update: standard-budget WP3 refinement on non-validation seeds improved diagnostic final distance
  from `0.369084 m` to `0.329068 m`, but did not transfer to the seed-53 validation gate. A seed-53 waypoint-2
  early-handoff search reduced diagnostic target steps to `601` with waypoint-2 closest `0.044487 m`, but when
  loaded as a schedule for the WP3-prefix run it still missed waypoint 2 at `0.070404 m`. Official synchronized
  replay with the early-WP2/refined-WP3 pair stayed `3/4`, no fall, and final distance `0.370650 m`, so these
  variants are rejected. The remaining issue is robust, validation-seed-stable handoff timing, not just a better
  return action after a brittle handoff.
- Feedback-corrector pivot: the corrected seed-53 schedule search now matches evaluator timing. The validated
  `wp2_segment3_s260_seed53_early_sync_v1` schedule officially reaches `3/4`, no fall, hits waypoint 2 at step
  `956`, and slightly improves final distance to `0.366889 m`. A matched seed-53 WP3 schedule improves official
  final distance further to `0.308910 m` but still remains `3/4`. This proves the residual channel can correct
  hard transitions, but fixed schedules are still brittle and should now be treated as oracle labels for a
  state-conditioned correction policy, not as the final controller.
- Inflection-point decision: the residual interface is no longer the main blocker. The WP2 schedule solved the
  hard left-to-backward transition at strict radius, and the matched WP3 schedule improved return distance. The
  remaining blocker is return-to-origin / waypoint-3 handoff robustness under timing, state, and validation
  variation. Do not spend more mainline pod cycles trying to make route search or blind residual schedules solve
  the full route directly; use those searches only as diagnostics and oracle-label generators.
- Methodology inflection update: treat the current `3/4` fixed-schedule result as proof that the motion is
  physically correctable, not as a controller architecture to keep elaborating. The mainline should now be
  closed-loop correction learning: roll out the scaffold plus corrector, collect the states where WP2/WP3 miss,
  relabel those states with a local corrective oracle, retrain, and promote only by strict replay.
- Closed-loop correction target: the next promotable controller should be a low-dimensional, state-conditioned
  feedback corrector over the frozen scaffold:
  `[route_state, goal_error, velocity, phase, previous/current command, time_since_switch, recent_progress]`
  to `[residual_strength, x_correction, y_correction, turn_bias, phase_speed_adjust, handoff_timing_bias]`.
  The CPG/PD scaffold should keep survival and rhythm; the corrector should handle steering, transitions,
  return timing, and recovery.
- Single continuous policy target: the final walker should not be four cardinal policies plus schedule patches.
  It should be one continuous route policy where the frozen CPG/PD prior supplies stable rhythmic motion and a
  compact feedback head continuously modulates correction strength, XY steering, turn bias, phase speed, and
  handoff timing from the current waypoint-relative state.
- Latest control-representation decision: the project is not blocked globally; it is blocked by a too-narrow
  representation of the final return problem. The fixed schedules prove the motion is physically possible, but
  the learned correctors are still copying timing-specific patches instead of learning why the patch works.
  Stop expanding residual schedules, stop cloning raw 12-D residual actions as the mainline, and stop asking one
  objective to solve the whole square route. Isolate return-to-origin as a local closed-loop skill, then gate it
  back into the route only after local competence is proven.
- Validation caveat: `route_corrector_seed53_7_v1` contains seed labels, but `CMD_RESET_NOISE=0.0` makes those
  runs too similar to count as true held-out robustness. Future corrector validation must add reset/start
  perturbations or other state variation before treating multi-seed results as a generalization signal.
- Fitted-map update: `sim/robot/fit_cpg_param_map.py` fits a weighted ridge map from dataset context
  `[cmd, previous_cmd, speed, command_change, active_waypoint2]` to cardinal CPG weights, and
  `CMD_CPG_ROUTER=param_linear CMD_CPG_MAP_JSON=<map.json>` loads it in `cpg_teacher.py`.
- `out/cpg_param_map_fit_v1_cpg_param_map.json`: fitted from `99,052` samples across `out/gait_dataset` and
  `out/gait_dataset_phase_oracle_v1`, with weighted MSE `0.101914`.
- `out/cpg_param_map_fit_v1_continuous_prior_compare.json`: fitted `param_linear` survived but reached `0/4`
  with final distance `0.258351 m`; the saved `softmax` scaffold remained better at `2/4`. The fitted map is
  therefore useful infrastructure/evidence, not a promotable router.

## 1. Stabilize Current Baseline

- [x] Run `eval_checkpoint_navigation.py` with corrected CPG defaults.
- [x] Record whether pure CPG reaches `0.35,0 -> 0.35,0.35 -> 0,0.35 -> 0,0`.
- [x] Save navigation traces and inspect failure mode: undershoot, drift, command switching, or fall.
- [x] Run a cardinal residual PPO pass on top of the corrected CPG priors.
- [x] Re-test sustained forward motion and checkpoint navigation.
- [x] Finish targeted forward-residual curriculum.
- [x] Re-test sustained forward motion for 720 steps.
- [x] Re-test checkpoint navigation.
- [x] If forward remains weak, expand the CPG teacher parameterization instead of continuing fixed-prior tuning.
- [x] Do not start self-play until checkpoint navigation works.

## 2. Turn CPG Into A Teacher Interface

Create `sim/robot/cpg_teacher.py`.

- [x] Define `CPGParams`.
- [x] Move CPG action generation out of `commanded_env.py`.
- [x] Expose `decode_params(raw)`.
- [x] Expose `cpg_action(phase, params)`.
- [x] Expose `cpg_pd_step_target(obs/state, command, params)`.
- [x] Keep `CommandedEnv` using the same teacher implementation.
- [x] Add a unit check script: `sim/robot/check_cpg_teacher_equivalence.py`.
- [x] Run `check_cpg_teacher_equivalence.py` on the pod after the active training process exits.
- [x] Confirm default blend equivalence: `teacher_action_max_err=0.000e+00`, `teacher_target_max_err=0.000e+00`.
- [x] Confirm `CMD_CPG_ROUTER=softmax CMD_CPG_GATE_POWER=6` equivalence on a diagonal command.

## 3. Replace Fixed Cardinal Constants With Searchable Params

Refactor `sim/robot/search_cpg_gait_mjx.py`.

- [x] Search over `CPGParams`, not one-off constants.
- [x] Support continuous command input `[vx, vy]` first, yaw later.
- [x] Score against the actual deployed MJX env.
- [x] Keep survival as a hard gate.
- [x] Reject candidates with action clipping saturation above a threshold.
- [x] Store each successful candidate as `{command, params, score, rollout_stats}`.
- [x] Run at least one post-refactor MJX CPG search on the pod.
- [x] Add full-vector env overrides such as `CMD_CPG_FWD_RAW` for deploying rich CPG candidates.
- [x] Run a longer-horizon rich forward search that can reach the first checkpoint.
- [x] Deploy-test the rich forward candidate through `CommandedEnv`.
- [x] Re-test checkpoint navigation with the rich forward candidate.
- [x] Search rich left, backward, and right priors.
- [x] Test rich forward + rich left checkpoint navigation.
- [x] Test all-rich-cardinal checkpoint navigation.
- [x] Add env-controlled route selection experiments: `CMD_CPG_ROUTER=blend|softmax|winner`.
- [x] Test sharpened `softmax` routing on the previous best rich forward + rich left setup.
- [x] Test `winner` routing with `--segment-axis` using all four rich cardinal priors.
- [ ] Fix command/prior blending or route selection so searched cardinal priors compose.
- [ ] Reduce drift/composition errors enough to navigate all four checkpoints.
- [x] Add route-aware search script: `sim/robot/search_cpg_route_mjx.py`.
- [x] Make route search call the same shared `cpg_pd_step_target` path as deployment.
- [x] Avoid vmapped route scoring after confirming `jax.vmap` disagreed with single-route traces.
- [x] Add `--init-route` so route search can continue from a saved route candidate.
- [x] Add `--mutate-route` so focused searches can touch only selected directional priors.
- [x] Add `--freeze-route-freq` so backward-transition refinement does not damage shared phase timing.
- [x] Add `--focus-active`, `--focus-progress-w`, `--focus-dist-w`, and `--focus-cross-w` for
      third-segment/backward-transition refinement.
- [x] Add `--command-mode full` and deployable router settings for full 2D command/router tests.
- [x] Validate a saved route candidate through `eval_checkpoint_navigation.py`.

## 3A. Make The Prior Continuous And Transition-Aware

The CPG should become a smooth, goal-conditioned prior, not four independent limit cycles plus a hard router.

- [x] Add a command-conditioned CPG parameter map prototype:
      `[goal_dx, goal_dy, speed, phase, previous_command] -> CPGParams`.
- [x] Start with a small fitted map, such as an RBF table, spline, or tiny MLP.
- [x] Train or fit the map from cardinal rollouts, partial route rollouts, and transition buckets.
- [x] Compare the map against `winner`, `softmax`, and `blend` routing on held-out waypoint transitions.
- [x] Add transition controls that are only active at command switches:
  - [x] `phase_reset_amount`
  - [x] `phase_speed_boost`
  - [x] `stance_height_boost`
  - [x] `turn_bias`
  - [x] `cross_axis_correction`
- [x] Search transition controls separately from steady-state gait parameters.
- [ ] Add a prior-strength schedule:
  - [x] strong CPG prior for stable cardinal walking
  - [x] weaker CPG prior for route transitions
  - [ ] weakest CPG prior for contact-heavy combat moments
- [x] Expose prior strength as an observation or curriculum variable for residual training.

## 4. Build Gait Library

Create `sim/robot/collect_gait_dataset.py`.

- [x] Create `sim/robot/collect_gait_dataset.py`.
- [x] Sample command buckets in code:
  - [x] forward
  - [x] backward
  - [x] left
  - [x] right
  - [x] diagonals
  - [x] slow/medium/fast
  - [x] hold
- [x] Run CEM for each command bucket in the collector.
- [x] Keep only rollouts that pass:
  - [x] survived full rollout
  - [x] positive command-axis progress
  - [x] low cross-axis drift
  - [x] no fall
  - [x] bounded saturation
- [x] Save dataset to `out/gait_dataset/*.npz`.
- [x] Save manifest to `out/gait_dataset/manifest.json`.
- [x] Add transition-bucket collector: `sim/robot/collect_transition_dataset.py`.
- [x] Add transition buckets:
  - [x] forward to left
  - [x] left to backward
  - [x] backward to right
  - [x] right to forward
  - [x] diagonal correction toward waypoint
  - [x] overshoot recovery
  - [x] cross-axis drift correction
- [x] Add a focused waypoint-2 transition generator:
  - [x] sample start states near the current miss region after waypoint 1
  - [x] set goal to `(0, 0.35)`
  - [x] label distance reduction to waypoint 2
  - [x] label excess positive Y drift
  - [x] label losing too much X progress
  - [x] reject/label falls and high saturation
- [x] Save failed-but-informative traces with labels for drift direction, distance reduction, X loss,
      saturation, and fall state.
- [x] Run the full scaffold gait-library collection on the pod.
- [x] Confirm the manifest has kept examples for every command family and route-transition family.

## 5. Behavior Clone The Teacher

Create `sim/robot/train_bootstrap_bc.py`.

- [x] Create `sim/robot/train_bootstrap_bc.py`.
- [x] Load gait dataset.
- [x] Train policy: `obs + command -> CPG/PD motor action`.
- [x] Validate BC on held-out commands.
- [x] Add rollout-eval path for fixed directions and checkpoint route.
- [x] Gate on fixed-direction survival in rollout eval.
- [x] Gate on at least one checkpoint route in rollout eval.
- [x] Gate on no survival regression below teacher in rollout eval.
- [x] Train BC from cardinal, partial-route, focused-transition, and failed-but-labeled traces.
- [x] Include goal-relative features, previous command, active waypoint/transition bucket, and prior strength
      in the BC observation.
- [x] Add context corruption support for transition-feature BC:
  - [x] previous-action noise
  - [x] command jitter
  - [x] phase jitter
  - [x] delayed/stale command
  - [x] wrong previous-action dropout/substitution
- [x] Add lightweight DAgger-style learner-state collector: `sim/robot/collect_dagger_dataset.py`.
- [x] Collect learner rollout states from the current best BC policy.
- [x] Relabel learner states with the CPG scaffold teacher.
- [x] Add a stronger corrective relabel mode to the DAgger collector:
      `--label-mode one_step_oracle`.
- [x] Collect learner rollout states with `--label-mode one_step_oracle`.
- [x] Relabel a first learner-state batch with a stronger oracle/corrective target, not only the same CPG prior.
- [x] Add explicit BC promotion reporting based on closed-loop nav improvement, regardless of BC loss.
- [x] Reject BC variants unless closed-loop nav improves, regardless of BC loss.
- [x] Train a BC candidate to imitate useful teacher actions while correcting labeled drift where the teacher misses.
- [x] Confirm BC fixed-direction survival rollout gate passes.
- [ ] Confirm BC checkpoint navigation gate passes.
- [x] Evaluate whether BC reduces the waypoint-2 miss from about `0.18 m` toward the `0.07 m` radius.

## 6. Train Residual Locomotion

Update or create `sim/robot/train_residual_locomotion.py`.

- [x] Create `sim/robot/train_residual_locomotion.py`.
- [x] Use action formula: `action = teacher_action + residual_scale * policy_action`.
- [x] Start with `residual_scale = 0.1`.
- [x] Ramp only after validation passes:
  - [x] `0.1 -> 0.25`
  - [x] `0.25 -> 0.5`
  - [x] `0.5 -> 1.0`
- [x] Reward velocity tracking.
- [x] Reward checkpoint progress.
- [x] Reward uprightness.
- [x] Penalize cross-axis drift.
- [x] Penalize energy.
- [x] Penalize action jerk.
- [x] Penalize falls.
- [x] Save best checkpoint only when held-out navigation improves.
- [x] Add support for stronger residual freedom during command-transition windows:
  `CMD_CPG_RESIDUAL_SCALE_TRANSITION`, `CMD_CPG_TRANSITION_HOLD_STEPS`, and
  `CMD_CPG_TRANSITION_DELTA_THRESH`.
- [x] Add closed-loop route-training mode: `CMD_TRAIN_MODE=route` with `CMD_REWARD_MODE=route`.
- [x] Add residual-action penalty for route reward so the learned policy stays a correction over the scaffold.
- [x] Run a first residual ramp from the CPG scaffold once cardinal and transition buckets exist.
- [x] Use stronger residual freedom during route transitions than during stable cardinal walking.
- [x] Gate each residual ramp on reduced waypoint-2 miss distance, not just survival.

## 6A. Train Closed-Loop Route Corrector

Use the successful WP2/WP3 schedules as oracle data, not as the final controller.

- [x] Stop treating fixed residual schedules as promotable controllers after they top out at `3/4`.
- [x] Record the inflection-point decision: the residual channel has proven WP2 is physically correctable; the
      active problem is feedback robustness for WP3 / return-to-origin handoff.
- [x] Treat successful schedules as teacher labels and diagnostics, not as the deployed control strategy.
- [x] Define the single-policy target as scaffold plus compact state-conditioned correction head, not cardinal
      gait routing plus fixed schedule patches.
- [x] Classify the current learned-corrector gap as a label/control-representation problem: the learner has seen
      timing-specific patches, but it has not yet learned the state-based reason those patches work.
- [x] Reject schedule-index imitation as the primary controller representation for the final return segment.
- [x] Add route-state correction dataset collector: `sim/robot/collect_route_correction_dataset.py`.
- [x] Collect a first correction dataset from schedule-assisted rollouts:
      `out/route_correction_dataset/route_corrector_seed53_7_v1.npz`.
- [x] Convert the best fixed WP2/WP3 schedules into an explicit local oracle-label source:
  - [x] use fixed schedules to prove which route states are physically correctable
  - [x] store schedule actions and compact coefficients as supervised labels
  - [x] add local corrective search/labeling for learner-visited WP3 miss states
  - [x] label command/residual corrections, not only raw 12-D motor residuals
- [x] Add true route-state variation before claiming held-out generalization:
  - [x] nonzero reset noise or sampled route-start perturbations
  - [x] early and late waypoint-handoff states
  - [x] off-nominal velocity states
  - [x] phase-offset variation at waypoint switches
- [ ] Expand correction data around scaffold failure states:
  - [ ] near waypoint 2 after the left-to-backward handoff
  - [x] near waypoint 3 / return-to-origin after waypoint-2 handoff
  - [ ] borderline waypoint-2 misses in the `0.070-0.09 m` band
  - [ ] excess-Y drift states
  - [ ] insufficient-X recovery states
- [x] Represent corrections with a compact low-dimensional basis instead of direct 12-D motor residuals.
- [x] Add compact correction trainer: `sim/robot/train_route_corrector.py`.
- [x] Train a first supervised state-conditioned correction policy smoke:
      `out/route_corrector_seed53_7_v1.pkl`.
- [x] Prefer low-dimensional correction heads over whole-gait imitation:
  - [x] compact residual basis coefficients
  - [x] residual strength
  - [x] x/y correction command
  - [x] turn bias
  - [x] phase-speed adjustment
  - [x] handoff-timing bias
- [x] Make WP3 / return-to-origin the primary failure-state curriculum:
  - [x] collect nominal WP3-long schedule-assisted data
  - [x] collect targeted return-handoff states
  - [x] collect learner-generated WP3 misses from the current gated corrector
  - [x] relabel those misses with a stronger local return oracle
  - [x] preserve the proven WP2 correction while training WP3-only artifacts
- [x] Include correction-policy features:
  - [x] active waypoint id
  - [x] goal-relative delta and distance
  - [x] base velocity
  - [x] CPG phase
  - [x] previous/current command
  - [x] time since waypoint switch
  - [x] prior strength / residual scale
  - [x] recent waypoint progress
- [x] Add closed-loop compact-corrector evaluator: `sim/robot/eval_route_corrector.py`.
- [x] Replay a compact corrector through the CPG/PD residual path.
- [x] Promote compact correctors only by closed-loop replay, not supervised loss.
- [x] Require the corrector to preserve early-route behavior while improving WP3 final distance.
- [x] Compare each corrector against:
  - [x] pure scaffold baseline
  - [x] WP2-only fixed schedule
  - [x] WP2+WP3 fixed schedule
- [ ] Improve compact corrector rollout beyond the fixed-schedule baseline.
- [ ] Integrate the compact corrector into the main training/promotion pipeline.
- [x] Add a safer waypoint-specific/gated corrector evaluation path.
- [x] Run a lightweight DAgger/RoaD-style loop:
  - [x] roll out scaffold plus current corrector
  - [x] collect states where WP2/WP3 miss
  - [x] relabel with local schedule/oracle correction
  - [x] retrain corrector
  - [x] promote only by closed-loop route improvement
- [ ] Replace schedule-index dependence with state-conditioned feedback:
  - [ ] condition on waypoint-relative error and recent progress
  - [ ] condition on time since waypoint switch without hard-coding action timing
  - [ ] tolerate early/late WP2 handoffs
  - [ ] tolerate validation-seed phase and reset variation
- [ ] Treat full-route corrector training as integration only after local WP3 / return-to-origin competence is
      demonstrated in the isolated skill env.
- [ ] Validate the corrector across validation and held-out seeds:
  - [ ] seed `53`
  - [ ] seed `7`
  - [ ] at least three additional held-out seeds
- [ ] Reject if fixed-direction survival or square-command survival regresses.

## 6B. Train Local Return-To-Origin Skill

Isolate the remaining route problem as its own skill before trying to solve the whole square again.

- [x] Stop treating the final return miss as another full-route schedule-imitation problem.
- [x] Reframe the blocker as a too-narrow control representation, not as proof the whole project is stuck.
- [x] Reject these as mainline next moves:
  - [x] bigger residual schedules
  - [x] cloning 12-D residual actions
  - [x] solving the whole square route in one learner objective
- [x] Add local return skill env: `sim/robot/return_skill_env.py`.
- [x] Use first reset states near active WP3 / return handoff:
  - [x] nominal WP3 states from current gated-corrector replay
  - [x] small XY start jitter
  - [x] phase jitter
  - [x] velocity jitter
- [ ] Expand return-skill resets to include:
  - [ ] states after successful WP2
  - [ ] states near active WP3
  - [ ] states from failed learned-corrector rollouts
  - [ ] small XY / phase / velocity perturbations
- [x] Expose return-skill observation:
  - [x] goal-relative `dx, dy`
  - [x] distance to origin
  - [x] current `vx, vy`
  - [x] CPG phase
  - [x] previous/current command
  - [x] time since handoff
  - [x] recent progress
  - [x] prior strength / body stability context
- [x] Expose low-dimensional return action:
  - [x] x correction
  - [x] y correction
  - [x] turn bias
  - [x] phase speed adjustment
  - [x] speed/residual-strength modulation
- [ ] Prefer command/correction labels over raw motor-residual labels:
  - [ ] move-x-toward-origin
  - [ ] move-y-toward-origin
  - [ ] heading / turn bias
  - [ ] slow down / speed up
  - [ ] residual strength
  - [ ] handoff earlier/later
- [x] Add local reward:
  - [x] reduce distance to origin
  - [x] preserve uprightness
  - [x] penalize moving away from origin
  - [x] penalize action size and jerk
  - [x] bonus for entering the `0.07 m` radius
- [x] Add return-skill PPO trainer: `sim/robot/train_return_skill.py`.
- [x] Add return-skill evaluator: `sim/robot/eval_return_skill.py`.
- [x] Run return-skill reset/step smoke on the pod.
- [x] Run tiny return-skill PPO smoke and save an artifact.
- [x] Evaluate the smoke artifact on local WP3 starts.
- [x] Run the first medium return-skill PPO attempt and reject it because it produced `0/8` local hits with no
      fall; the mechanism works, but the reward/curriculum is not yet competent.
- [ ] Add an easier-to-harder return curriculum:
  - [ ] start closer to the origin before using the full WP3 handoff distribution
  - [ ] increase start distance once hit rate passes the local gate
  - [ ] mix nominal WP3 handoff starts with learner-failure starts
  - [ ] keep phase and velocity perturbations in the validation set
- [ ] Add or tune return-skill reward terms for:
  - [ ] radial velocity toward the origin
  - [ ] reducing lateral drift around the origin vector
  - [ ] slowing down near the `0.07 m` hit radius
  - [ ] preserving uprightness and action smoothness as hard gates
- [ ] Train a competent return skill until local eval enters the `0.07 m` origin radius without falling.
- [ ] Gate the competent return skill behind WP2/WP3 handoff in full-route replay.
- [ ] Require the gated return skill to preserve the proven WP2 transition.
- [ ] Validate the gated return skill across held-out start perturbations/seeds.

## 6C. Recompose The Route From Local Skills

Only integrate after the isolated return skill works locally.

- [ ] Freeze the CPG/PD scaffold as the survival/rhythm layer.
- [ ] Freeze or gate the proven WP2 transition corrector so it handles the left-to-backward transition.
- [ ] Route active-WP3 / return-to-origin states to the local return skill.
- [ ] Convert the return-skill output through the low-dimensional command/correction interface, not direct 12-D
      motor residual cloning.
- [ ] Add a router/gate that blends by state:
  - [ ] normal scaffold for stable route segments
  - [ ] WP2 corrector for the hard transition
  - [ ] return skill after WP2/WP3 handoff
  - [ ] low correction when already lined up with the origin
- [ ] Validate recomposed full-route replay:
  - [ ] preserve fixed-direction survival
  - [ ] preserve square-command survival
  - [ ] preserve the proven WP2 `3/4` prefix
  - [ ] improve final distance beyond the best fixed-schedule oracle
  - [ ] promote only on `4/4` strict-radius replay or a clearly better full-route gate defined in advance
- [ ] Run held-out route perturbation validation before calling the recomposed policy robust.

## 7. Validate Walker

For every candidate walker:

- [x] Add evaluator support for fixed directions: forward, backward, left, right.
- [x] Add square command eval.
- [x] Add checkpoint navigation eval.
- [x] Add random command sweep eval.
- [x] Add rendered rollout support.
- [x] Extend `validate_commanded.py` to gate fixed modes, nav, render, and nav improvement.
- [x] Update `scripts/run_validated_training_pipeline.sh` to use the stricter walker gate.
- [x] Add `sim/robot/prior_calibration.py`.
- [x] Report `prior_survival`: whether the CPG keeps the body alive.
- [x] Report `prior_progress`: whether the CPG moves roughly in the commanded direction.
- [x] Report `prior_residual_burden`: estimated correction needed to hit the waypoint.
- [x] Report `prior_composability`: degradation at command switches.
- [x] Report `prior_overconstraint`: whether stronger CPG weight prevents residual correction.
- [ ] Run the full validation suite on a candidate walker.
- [ ] Promote only if it beats the previous walker on navigation.
- [ ] Reject promotion if survival regresses.

## 8. Connect Walker To Fighter

- [ ] Use validated walker as base locomotion checkpoint.
- [ ] Add combat commands: approach, retreat, circle, hold.
- [x] Keep fighting reward separate from locomotion survival.
- [x] Start with preliminary scripted/passive opponent curriculum.
- [x] Keep the old unattended self-play path behind the walker validation gate in
      `scripts/run_validated_training_pipeline.sh`.
- [x] Add a scaffold-prior combat path that does not block passive-opponent contact learning on perfect
      checkpoint navigation: `scripts/run_scaffold_combat_curriculum.sh`.
- [x] Run the scaffold-prior combat path on the pod and require the benchmark decomposition to trend in the
      right direction: SPARC, dealt/taken, closing/fleeing, fall/survival, and benchmark score.
- [x] Render before/after fight rollouts from the same command.
- [ ] Require fighter to still pass locomotion/navigation smoke tests after combat training.

## 9. Self-Play Launch Gate

Before leaving the pod unattended:

- [ ] Walker passes checkpoint navigation.
- [ ] Fighter can move without falling.
- [ ] Fighter has nonzero dealt-damage signal.
- [x] Robust selector rejects regressions in `sim/robot/robust_fighter_select.py`.
- [x] Hall-of-fame promotion is fail-closed in `sim/robot/selfplay_drive.py`.
- [ ] Rendered samples show plausible walking and fighting.
- [x] Long-run scripts write metrics, checkpoints, and validation JSON.
- [ ] No GPU/process contention during the actual launch.
- [ ] Resume/restart path is tested after the new walker gate.

## Current Next Gate

Do not continue the `walk22_forward_resid` line as the main path. It already
failed sustained-forward and checkpoint navigation gates.

Do not keep treating pure CPG route search or fixed residual schedule search as
the only unlock. The current blocker is no longer whether the residual channel
can make the transition; the WP2 schedule has proven that it can. The blocker is
making that correction feedback-based and robust to timing, state, and seed
variation. The compact route-corrector line has also shown the limit of cloning
schedule patches: it can preserve `3/4`, but it has not learned a robust final
return-to-origin behavior. Use schedules as oracle/debug labels, not as the
main controller representation.

The current policy direction is:

```text
frozen CPG/PD scaffold
+ WP2 transition corrector / schedule gate
+ local return-to-origin skill
+ full-route router/gate after the local skill works
+ promotion by strict replay across perturbed validation states
```

For combat, the scaffold is used differently:

```text
combat-compatible scaffold seed
+ contact-forcing passive-opponent curriculum
+ benchmark decomposition selected by closed-loop combat metrics
+ rendered scaffold/trained rollouts
```

Do not imitate the whole gait as the main objective. Do not search another
high-dimensional full-route schedule as the main objective. Do not keep cloning
12-D residual actions or command-head labels from one brittle trajectory. The
next useful unit of progress is a competent local return-to-origin skill trained
with its own closed-loop objective. It must reach the origin from active-WP3
handoff states without falling before being asked to preserve the whole square
route.

Do not block scaffold-prior combat learning on the final route-return issue. Route promotion still requires
strict waypoint replay, while combat promotion requires closed-loop engagement metrics: SPARC, dealt/taken,
closing/fleeing, fall/survival, benchmark-opponent score, and rendered before/after rollouts.

Immediate next work:

- improve the `return_skill_env.py` curriculum/reward after `return_skill_v1` reached `0/8` local hits;
- train the local return skill until eval reaches the `0.07 m` origin radius without falling;
- validate the local return skill across perturbed active-WP3 starts and held-out seeds;
- plug the return skill behind the existing WP2 handoff gate only after local eval works;
- promote only if full-route strict replay improves beyond the fixed-schedule oracle or reaches `4/4`.
- run `scripts/run_scaffold_combat_curriculum.sh` on the pod to train/evaluate/render scaffold-prior combat
  engagement without the old waypoint-navigation blocker.

Latest evidence:

- `scaffold_combat_contact_short`: the scaffold-prior combat command ran on the pod for two curriculum phases
  (`strk0`, then resumed into `cval`) from `walk22_forward_resid.pkl`. It converted the locomotion checkpoint
  into a combat-compatible seed (`obs 40->44`, action head `12->14`), trained `819,200` env steps total, saved
  `curriculum_best.pkl`, benchmark JSONL, a benchmark curve, and three top-down fight videos. Baseline vs trained
  held-out benchmark: SPARC `-18.465 -> -12.135`, closing `1.375 -> 1.605`, fleeing `5.080 -> 4.035`, distance
  `71.99 -> 69.04`, survival `1.0 -> 1.0`, safe rate `0.0 -> 1.0`, and trained dealt damage stayed nonzero
  (`0.00215`). Because the scaffold baseline had unsafe early clash contact, the summary also reports safe-row
  trends: safe-contact dealt `0.0004 -> 0.0021`, SPARC `-18.224 -> -12.135`, and fleeing `4.877 -> 4.035`.
  Artifacts are under `out/scaffold_combat_contact_short/`.
- `walk22_forward_resid_forward720`: survived, but only reached about `0.04 m/s`
  on a `0.35 m/s` forward command.
- `walk22_forward_resid_nav`: reached `0/4`.
- `cpg_rich_fwd_left_nav_r07`: reached `2/4`, no fall. This is the current
  high-water result.
- `cpg_rich_all_nav_phasefix_r07`: reached `0/4`, no fall.
- `cpg_rich_fwd_left_nav_softmax6_r07`: reached `0/4`, no fall; drifted in +Y
  while trying to reach the first forward checkpoint.
- `cpg_rich_all_winner_segment_r14`: reached `1/4`, no fall; moved upward after
  checkpoint 1 but lost too much X to reach checkpoint 2.
- `cpg_route_smoke_exactteacher_deploy_segment_r07`: reached `2/4`, no fall.
- `cpg_route_refine_from_2wp_deploy_segment_r07`: reached `2/4`, no fall.
- `cpg_route_progress_push` did not break through the third waypoint; the route still fails
  on the backward segment after waypoint 2 by drifting upward in Y.
- `cpg_route_focus_back_wp2` showed why unconstrained backward refinement is risky: it improved the focused
  internal score but deployment regressed to `1/4` because changed backward frequency damaged earlier route
  composition.
- `cpg_route_focus_back_wp2_freezefreq_deploy_segment_r07`: reached `2/4`, no fall, preserving the earlier
  route, but did not improve the deployed closest approach to waypoint 2.
- `cpg_route_full_softmax_wp2_deploy_r07`: reached `2/4`, no fall, final distance `0.189236894 m`.
  Closest waypoint-2 approach was `0.181720 m` at `(0.126674, 0.480291)`, so full-command softmax routing is
  better than the worst composition failures but still misses the `0.07 m` waypoint radius.
- `cpg_route_full_blend_wp2_focuscross`: rejected before deployment replay. It reached `2/4` in route search
  with route progress `2.388556`, final distance `0.261033 m`, and focused distance `0.238409 m`; this is
  worse than the previous full-command/softmax high-water mark.
- `cpg_route_full_softmax2_wp2_focuscross`: reached `2/4` in route search with route progress `2.542018`,
  final distance `0.186107 m`, and focused distance `0.177812 m`. This is the best internal waypoint-2
  focus metric so far, but it still misses the pure CPG `3/4` gate and should be treated as scaffold evidence,
  not a reason to delay transition-aware BC.
- `cpg_route_full_softmax2_wp2_deploy_r07`: deployment replay reached `2/4`, no fall.
- `cpg_route_full_softmax2_wp2_transition_manifest.json`: generated `38,560` scaffold samples on the pod.
  The route-derived waypoint-2 bucket has `5,244` samples and `2,585` positive-progress samples. The manifest
  also includes `6,400` command-transition fallback samples, including `1,600` `backward_to_right` samples,
  and `7,200` steady command-family samples covering forward/backward/left/right/diagonals/hold.
- `cpg_route_full_softmax2_wp2_prior_calibration.json`: `bc_scaffold_ready=true`; survival and progress are
  positive, waypoint-2 residual burden is `0.189471 m`, composability still fails before the full route, and
  prior-overconstraint remains unmeasured until a BC/residual rollout exists. The report counts `38,560`
  samples, no missing transition families, and no missing command families.
- `bc_scaffold_v1`: first BC smoke trained for `8,000` supervised steps on the scaffold dataset.
  Validation loss reached about `0.048873`. Closed-loop rollout survived all fixed directions and reached
  `2/4` checkpoints, but waypoint-2 miss worsened to `0.260846 m`, so BC is stable enough as a scaffold
  policy but does not yet correct the failing transition.
- `bc_scaffold_transition_v1`: transition-feature BC smoke trained for `8,000` supervised steps with previous
  command, command delta, previous action, prior strength, and active context. Validation loss improved to
  about `0.001596`, but closed-loop navigation regressed to `0/4`; low BC loss is therefore not sufficient,
  and architecture/input changes must be selected by rollout metrics.
- `bc_scaffold_weighted_v1`: weighted base-feature BC trained with positive-progress/upweighted waypoint-2
  samples and downweighted bad drift/fall/saturation samples. Closed-loop rollout survived fixed directions
  but regressed to `1/4`, waypoint-2 miss `0.263474 m`.
- `bc_scaffold_transition_corrupt_v1`: transition-feature BC with command jitter, stale command, previous-action
  noise, previous-action dropout, and wrong previous-action substitution. It survived fixed directions but
  still reached `0/4`, waypoint-2 miss `0.262617 m`.
- `dagger_bc_scaffold_v1`: first learner-state relabel pass collected `4,160` states from `bc_scaffold_v1`
  and relabeled them with the same CPG scaffold teacher. Retraining as `bc_scaffold_dagger_v1` survived fixed
  directions but reached `0/4`, waypoint-2 miss `0.291767 m`. This shows learner-state data helps only if the
  relabeler is a stronger corrective oracle than the current CPG prior.
- Implementation update after `bc_scaffold_dagger_v1`: `train_bootstrap_bc.py` now adds transition phase
  features and supports `--phase-jitter-std`; `collect_transition_dataset.py` now saves actual oscillator
  phase for regenerated transition datasets; `collect_dagger_dataset.py` now supports
  `--label-mode one_step_oracle`; BC rollout reports now include a `promotion` block comparing closed-loop
  navigation against a supplied baseline. These mechanisms are ready for the next pod run, but no new
  phase-aware/oracle-labeled model has passed promotion yet.
- Pod smoke: `collect_dagger_dataset.py --label-mode one_step_oracle` ran for one tiny episode with
  `24/24` oracle-labeled samples and `17` positive-progress samples, then the smoke `.npz` was removed so it
  will not contaminate full BC training.
- `gait_dataset_phase_oracle_v1`: regenerated the scaffold transition dataset with actual oscillator phase
  saved per sample. It contains `29,120` samples: `8,320` route-derived samples, `6,400`
  command-transition fallback samples, and `14,400` steady command-family samples. The waypoint-2-focused
  bucket has `1,991` samples and `875` positive-progress samples. All transition families and command
  families are covered.
- `dagger_bc_scaffold_v1_oracle_v1`: reduced first real oracle-labeled learner-state batch collected `1,280`
  samples from `bc_scaffold_v1`, all labeled by `--label-mode one_step_oracle`; `730` samples had positive
  progress. The larger `4`-episode, `9`-candidate oracle run was stopped because it was too slow for
  iteration.
- `bc_scaffold_phase_oracle_v1`: trained for `8,000` supervised steps with transition features, phase jitter,
  useful-sample weighting, context corruption, and the oracle-labeled batch. It survived fixed-direction
  rollout but regressed to `1/4` checkpoints. Waypoint-2 closest distance worsened from the refreshed
  `bc_scaffold_v1` baseline `0.260846 m` to `0.291785 m`, so the promotion block correctly rejected it.
- `cpg_route_full_softmax2_wp2_phase_oracle_v1_prior_calibration.json`: prior-overconstraint is now measured.
  The CPG prior reached `2/4` with waypoint-2 miss `0.189471 m`; the phase/oracle BC learner reached `1/4`
  with waypoint-2 miss `0.291785 m`. The report marks `overconstraint_suspected=true`, meaning this
  supervised correction path is currently making the learner worse than the scaffold.
- Residual/validation update: `eval_checkpoint_navigation.py` now writes `closest_waypoints` and
  `waypoint2_min_dist`; `train_residual_locomotion.py` and `validate_commanded.py` score nav improvements by
  reached waypoints first, then waypoint-2 miss, then final distance. Pod smoke `nav_metrics_smoke` confirmed
  the new summary fields are emitted.
- Residual transition-scale update: `CommandedEnv` now tracks `prev_cmd` and a short `transition_timer`.
  `train_residual_locomotion.py` can set a separate transition residual scale. Pod smoke
  `transition_scale_smoke` confirmed the runtime path executes without state-shape/type errors.
- `residual_route_v1`: first closed-loop route-residual ramp over the frozen scaffold used
  `CMD_TRAIN_MODE=route`, `CMD_REWARD_MODE=route`, baseline nav
  `cpg_route_full_softmax2_wp2_deploy_r07_v2_nav_summary.json`, residual scale `0.1`, and transition
  scale multiplier `2.5`. It trained for `245,760` env steps and was rejected. Fixed forward/left/right/square
  survived, but backward fell at step `100` (`101/220` survival), so the hard survival gate failed.
  Checkpoint navigation regressed to `1/4`; waypoint-2 miss worsened to `0.262959 m` versus the scaffold
  baseline `0.189471 m`. The driver correctly left `best` at `out/walk22_forward_resid.pkl`.
- `residual_route_v2_safe`: safer route-residual pass used base residual scale `0.03`, transition scale `0.09`,
  `CMD_ROUTE_ACTION_W=0.05`, `122,880` env steps, and the same scaffold baseline. It was also rejected:
  backward survival improved but still failed at `123/220`, checkpoint nav stayed `1/4`, and waypoint-2 miss
  worsened to `0.270991 m`. This suggests the residual learner needs a focused failure-state curriculum before
  full-route residual training.
- Focused route-start update: `CommandedEnv` and `train_residual_locomotion.py` now support
  `CMD_ROUTE_START_WP` / `--route-start-wp` and `CMD_ROUTE_START_XY` / `--route-start-xy`, so PPO can train
  residual corrections directly near the waypoint-2 miss state distribution instead of always starting at the
  route origin.
- `residual_wp2_focus_v1`: waypoint-2-focused route residual started training episodes at active waypoint `2`
  near `(0.168579, 0.436490)`. Fixed forward/backward/left/right/square survival all passed, including
  backward `220/220`, so focused residual training avoided the full-route residual survival regression.
  It still did not promote: checkpoint nav reached only `1/4`; waypoint 1 closest distance was `0.084532 m`
  just outside the `0.07 m` gate, and waypoint-2 miss was `0.249643 m`, worse than the scaffold baseline
  `0.189471 m`. This points to mixed-start route residual training: origin plus waypoint-1/waypoint-2 states,
  not a single focused miss state.
- Mixed route-start update: `CommandedEnv` now supports `CMD_ROUTE_STARTS`, and
  `train_residual_locomotion.py` exposes `--route-starts`, allowing reset mixes like
  `0,0,0;1,0.296197,0.355607;2,0.168579,0.436490`.
- `residual_mixed_route_v1`: mixed-start residual used origin, waypoint-1, and waypoint-2 resets. It passed
  fixed forward/backward/left/right/square survival and restored full-route checkpoint navigation to `2/4`.
  It still did not promote: waypoint-2 miss was `0.220587 m`, worse than the scaffold baseline
  `0.189471 m`. This is the best residual shape so far, but it needs more waypoint-2 pressure without losing
  the waypoint-1 and survival gains.
- `residual_mixed_route_v2_wp2heavy`: refinement resumed from `residual_mixed_route_v1_r0p03.pkl` and
  oversampled waypoint-2 starts, including the new residual miss at `(0.207456, 0.424972)`. Fixed-direction
  survival still passed, but full-route nav regressed to `1/4`; waypoint-1 miss grew to `0.294201 m` and
  waypoint-2 miss was `0.239290 m`. This rules out naive waypoint-2 oversampling as the next unlock.
- `residual_wp2_xremain_v1`: added explicit waypoint-2 X-remaining reward pressure and resumed from
  `residual_mixed_route_v1_r0p03.pkl`. Fixed forward/backward/left/right/square survival all passed, but
  checkpoint navigation regressed to `0/4`; closest waypoint 0 was `0.259385 m`, waypoint 2 was
  `0.347302 m`, and final distance was `0.536118 m`. The run is rejected. The first generated report had a
  false `promoted=true` because `--baseline-nav out/...` did not resolve from `sim/robot`; the driver now
  resolves repo-root/`CODESIGN_OUT` paths and fails fast when a requested baseline is missing.
- `residual_routectx_v1`: added opt-in route-context observation (`CMD_OBS_ROUTE_CONTEXT=1`) and resumed from
  `residual_mixed_route_v1_r0p03.pkl` with moderate waypoint-2 penalties. Warm-start padded the policy from
  `40` to `49` observation dims, and fixed forward/backward/left/right/square survival all passed. Checkpoint
  navigation still regressed to `0/4`; closest waypoint 0 was `0.255443 m`, waypoint 2 was `0.336586 m`, and
  final distance was `0.531675 m`. The next residual mechanism should gate residual scale by route state
  instead of only giving the policy passive route context.
- `residual_routectx_wp2gate_v1`: added `CMD_CPG_RESIDUAL_SCALE_WP2` / `--wp2-residual-scale` and used
  residual scale `0.005` globally with `0.03` while active waypoint 2 was being corrected. Fixed
  forward/backward/left/right/square survival all passed, but checkpoint navigation still regressed to `0/4`;
  waypoint 0 closest distance was `0.251526 m` and waypoint 2 was `0.350059 m`. This rejects the run and shows
  the previously trained residual checkpoint is not a good base for explicit route-scaffold replay.
- Reproducible route-scaffold update: `train_residual_locomotion.py` now supports `--route-candidate` and merges
  the candidate `deploy_env` before residual/reward settings. Direct replay with
  `out/cpg_route_full_softmax2_wp2_focuscross_route_gait.json` preserved the pure scaffold at `2/4`,
  waypoint-2 miss `0.189471 m`; the old `residual_mixed_route_v1_r0p03.pkl` checkpoint reached only `1/4`
  under the same explicit scaffold, waypoint-2 miss `0.278214 m`.
- `residual_wp2only_routecandidate_v1`: resumed from `walk22_forward_resid.pkl`, loaded the explicit route
  candidate, used global residual scale `0.0`, and allowed residual scale `0.03` only at active waypoint 2.
  Fixed-direction and square survival passed and checkpoint navigation returned to `2/4`, but did not improve:
  waypoint-2 miss `0.189844 m` versus scaffold `0.189471 m`, final distance `0.198624 m`. This run is rejected
  but is the cleanest scaffold-preserving residual setup so far.
- `wp2_const_remote_*`: added a remote-loop constant residual diagnostic for active waypoint 2. After fixing the
  diagnostic to match `eval_checkpoint_navigation.py` instead of internal route mode, searched constant residuals
  improved the waypoint-2 miss monotonically with residual scale: `0.162087 m` at scale `0.08`, `0.150805 m`
  at `0.16`, `0.120824 m` at `0.32`, and `0.119121 m` at `0.64`. This does not pass the `0.07 m` gate, but it
  gives a concrete corrective residual target and shows the action scale used in PPO (`0.03`) was likely too
  weak for the remaining waypoint-2 correction.
- `wp2_segment3_v1` / `wp2_segment3_schedule_nav`: extended the residual diagnostic to piecewise-constant
  waypoint-2 actions and loaded the best 3-segment schedule through `CMD_CPG_WP2_ACTION_JSON` in
  `CommandedEnv`. The searched diagnostic reached waypoint 2 with closest distance `0.064998 m`; official
  checkpoint replay reached `3/4` with waypoint-2 closest distance `0.043081 m` in the standard budget and
  `0.020456 m` with `400` steps per waypoint. The candidate is not a full walker promotion because final
  return-to-origin still fails (`3/4`, not `4/4`).
- `wp3_segment3_*` / `wp2wp3_segment3_schedule_nav_400_smoke`: added separate waypoint-3 scheduled residual
  support (`CMD_CPG_RESIDUAL_SCALE_WP3`, `CMD_CPG_WP3_ACTION_JSON`) and scale-separated target search. The
  waypoint-2-only 260-step regression check still reaches `3/4`, no fall, with waypoint-2 distance
  `0.043081 m`. In 400-step replay, the waypoint-3 smoke schedule improves final distance from `0.285055 m` to
  `0.271778 m` but remains `3/4`; higher target scale `1.5` did not help. The remaining route issue is now
  early enough handoff plus a stronger return-specific/state-conditioned residual, not a missing WP3 schedule
  hook.
- `route_sched_joint_v1_s260`: added `sim/robot/search_route_residual_schedule.py` to optimize WP2 and WP3
  residual schedules together under the full standard route score. The first joint CEM run did not improve the
  route: best remained `3/4`, no fall, with final distance `0.384563 m`. This rejects broad joint random
  schedule perturbation as a useful next move without a lower-dimensional/state-conditioned parameterization.
- `eval_checkpoint_navigation.py` now syncs `route_wp` and `route_prev_dist` from the external waypoint
  controller before stepping the env. This makes validation route bookkeeping explicit for scheduled residual
  gates and route-context observations.
- `wp3_segment3_s260_refine_v1/v2` and `wp3_segment6_s260_refine_v1`: standard-budget WP3-only refinements on
  non-validation seeds reduced diagnostic final distance from `0.369084 m` to `0.329068 m`, but 6 segments
  plateaued around `0.333231 m`, so more schedule granularity alone did not solve the return segment.
- `wp2_segment3_s260_early_v1/v2`: target-step penalties can find earlier diagnostic waypoint-2 handoffs, but
  the first aggressive schedule missed official validation by a small margin (`0.072899 m` at waypoint 2), and
  the gentler schedule still failed to improve official final distance. These are rejected.
- `wp2_segment3_s260_seed53_early_v1` / `wp3_segment3_s260_seed53_refine_v1`: rerunning the timing search on the
  validation seed exposed seed sensitivity. The WP2 diagnostic reached `3/4` with `target_steps=601`, but the
  same schedule loaded through the env hook for the WP3-prefix run reached only `2/4` with waypoint-2 closest
  `0.070404 m`. Do not promote borderline schedule-search results unless they are replayed through
  `eval_checkpoint_navigation.py` on the validation seed.
- `wp2_segment3_s260_seed53_early_sync_v1` / `wp3_segment3_s260_seed53_sync_v1`: after fixing the search/eval
  off-by-one, the WP2 schedule transfers through official replay at seed `53`, reaching waypoint 2 at step `956`
  and final distance `0.366889 m`. The matched WP3 schedule improves official final distance to `0.308910 m`
  but still remains `3/4`. This is the handoff point from fixed schedule search to closed-loop correction
  learning.
- `collect_route_correction_dataset.py`: added a route-state data collector that records goal-relative state,
  velocity, CPG phase, previous/current command, time since switch, progress, prior strength, full residual
  schedule action, and compact PCA coefficients from WP2/WP3 schedules. Use this to train the next compact
  feedback corrector.
- `route_corrector_seed53_7_v1`: first route-correction dataset and supervised trainer smoke. The dataset has
  `2,080` samples, `20` route-state features, `4` compact correction coefficients, and a `12`-D reconstruction
  basis. The trainer `sim/robot/train_route_corrector.py` produced `out/route_corrector_seed53_7_v1.pkl`; final
  supervised validation metrics were `val_coeff_mse=0.013717` and `val_action_mse=0.004573`. This is not a
  walker promotion because the compact corrector is not yet integrated into closed-loop replay.
- `route_corrector_seed53_7_v1_eval_s53`: first closed-loop compact-corrector replay through
  `sim/robot/eval_route_corrector.py`. It reached `3/4`, no fall, and corrected waypoint 2 to `0.036505 m`, but
  final distance regressed to `0.376396 m`. This rejects the first supervised compact corrector as a walker
  promotion and confirms the next feedback loop needs more return-to-origin / WP3 failure-state data, not just
  imitation of the current schedule traces.
- `route_corrector_seed53_7_active_wp3w_v1`: active-only retrain with waypoint-3 samples weighted `8x` drove
  supervised action MSE down to `0.000702`, but closed-loop replay still reached only `3/4`. It improved over
  the first compact corrector only slightly: waypoint-2 distance `0.048594 m`, final distance `0.360580 m`.
  This confirms reweighting the sparse WP3 tail is not enough.
- `route_corrector_seed53_7_wp3long_v1`: collected a longer schedule-assisted dataset at `400` steps per
  waypoint using the same synced WP2/WP3 schedules. Dataset size increased to `3,200` samples and active
  waypoint-3 coverage increased from `172` to `1,292` samples while preserving the same compact basis.
  Active-only training with WP3 weight `2x` reached supervised action MSE `0.000262`. Closed-loop replay still
  reached only `3/4`, but it improved the feedback-corrector line: waypoint-2 distance `0.015134 m`, final
  distance `0.325241 m`, no fall. It is still rejected because it does not beat the fixed WP2+WP3 schedule
  final distance `0.308910 m` and does not reach `4/4`.
- `route_corrector_seed53_7_wp3long_v1_eval_s53_wp3s08`: increasing only WP3 residual scale from `0.64` to
  `0.8` did not help; replay stayed `3/4`, waypoint-2 distance improved to `0.007906 m`, but final distance
  slightly worsened to `0.326078 m`. This points away from simple scale-up as the return-to-origin fix.
- Perturbation-data update: `collect_route_correction_dataset.py` now supports `--reset-noise`, `--xy-noise`,
  and `--phase-jitter`, and records those settings in the manifest. A remote smoke with nonzero perturbations
  wrote a valid manifest and was removed so it cannot contaminate training.
- `route_corrector_perturb_wp3long_v1`: stronger perturbation
  (`reset_noise=0.01`, `xy_noise=0.005`, `phase_jitter=0.1`) produced `8,000` samples across five seeds, but it
  was too disruptive for WP3 training. Active counts were `{0: 4540, 1: 2541, 2: 919}` and `0` active WP3
  samples; most seeds failed before the return segment. Keep this as robustness evidence, not as WP3-label data.
- `route_corrector_perturb_mild_wp3long_v1`: milder perturbation
  (`reset_noise=0.002`, `xy_noise=0.001`, `phase_jitter=0.02`) produced `8,000` samples with useful but uneven
  variation: active counts `{0: 1992, 1: 2791, 2: 2357, 3: 860}`. Seeds `53` and `307` reached active WP3, seed
  `101` reached active WP2, and seeds `7`/`203` failed earlier. This is useful evidence that route robustness is
  sensitive to tiny reset/phase changes.
- `route_corrector_wp3long_mild_v1`: mixed nominal WP3-long data with mild perturbed data and trained an
  active-only corrector. Supervised validation action MSE was `0.000640`, but strict nominal closed-loop replay
  regressed to `3/4`, waypoint-2 distance `0.012680 m`, final distance `0.357819 m`. Directly mixing broad
  full-route perturbation data is therefore rejected; the next data iteration should target handoff states
  directly instead of perturbing whole-route starts.
- Targeted handoff-data update: `collect_route_correction_dataset.py` now supports `--start-states` and
  `--start-jitter`, allowing direct collection from route states such as active waypoint 3 near the
  return-to-origin handoff. Remote smoke from `3,0.012,0.34` produced active WP3 samples and was removed so it
  cannot contaminate training.
- `route_corrector_targeted_wp3_v1`: targeted WP3 collection used real handoff states from recent replays:
  `(0.0544,0.3864)`, `(0.0595,0.3765)`, `(0.0428,0.4009)`, and `(0.0255,0.3218)`, with `0.02 m` start jitter
  and phase jitter. It produced `8,320` samples, all active waypoint 3, across eight seeds. This satisfies the
  targeted return-state data mechanism, but the schedule oracle itself is weak in these states; active WP3
  distance ranged from `0.240283 m` to `0.692715 m`.
- `route_corrector_wp3target_v1`: mixed nominal WP3-long data with targeted WP3 handoff data. Supervised
  validation action MSE was much higher (`0.003369`) and strict nominal replay regressed to `1/4`, no fall,
  waypoint-2 distance `0.217309 m`. This branch is rejected. The next corrector needs a safer waypoint-specific
  or gated training path so targeted WP3 data cannot damage early-route behavior.
- Gated-corrector evaluator update: `eval_route_corrector.py` now supports `--wp2-artifact` and
  `--wp3-artifact`. This lets WP2 use the proven `route_corrector_seed53_7_wp3long_v1` artifact while WP3 uses
  a separate return-specific artifact. Rerunning the current best under the updated evaluator reproduced the
  prior result exactly: `3/4`, waypoint-2 distance `0.015134 m`, final distance `0.325241 m`.
- `route_corrector_gated_wp2long_wp3target_v1_eval_s53`: two-artifact replay used
  `route_corrector_seed53_7_wp3long_v1` for WP2 and `route_corrector_wp3target_v1` only after active WP3. It
  preserved early-route behavior and reached `3/4`, but worsened final distance to `0.331941 m`. This proves the
  waypoint-specific gate works, but also rejects the targeted WP3 artifact as a better return policy.
- `route_corrector_wp3only_nominal_v1`: `train_route_corrector.py` now supports `--active-filter`, enabling a
  return-only artifact trained on active waypoint 3 samples. The nominal WP3-only model trained on
  `1,292` active-WP3 samples and reached validation action MSE `0.000121`.
- `route_corrector_gated_wp2long_wp3only_nominal_v1_eval_s53`: gated replay used the current best WP2 artifact
  for waypoint 2 and the WP3-only nominal artifact for waypoint 3. It is the best feedback-corrector result so
  far: `3/4`, no fall, waypoint-2 distance `0.019491 m`, final distance `0.315691 m`. It still does not beat
  the fixed WP2+WP3 schedule final distance `0.308910 m` and still does not reach `4/4`, so it is not promoted.
- `route_corrector_gated_wp2long_wp3only_nominal_v1_eval_s53_wp3s08`: increasing the WP3-only artifact scale to
  `0.8` regressed final distance to `0.347186 m`, so the best gated WP3-only setting remains scale `0.64`.
- `route_corrector_gated_wp2long_wp3only_nominal_v1_eval_s53_wp3s05`: lowering the WP3-only artifact scale to
  `0.5` also regressed final distance to `0.376473 m`. The local scale sweep rejects both lower and higher WP3
  scales; improving return now needs a better WP3 oracle/objective, not simple gain tuning.
- WP3 oracle-search fix: `search_wp2_constant_residual.py` now accepts `--prefix-wp2-action-json`. The first
  attempted continuation search accidentally omitted the WP2 schedule and never reached active WP3
  (`prefix_steps=1040`, `target_steps=0`), so it was invalid. With the explicit WP2 prefix schedule loaded,
  `wp3_segment3_s260_seed53_sync_refine_v3` improved the diagnostic WP3 target distance from `0.308348 m` to
  `0.298490 m` and final distance to `0.300479 m`.
- `wp2earlysync_wp3refinev3_s260_nav`: official checkpoint replay with the known WP2 schedule plus refined WP3
  schedule stayed `3/4`, no fall, but improved final distance to `0.299545 m`, with waypoint-2 distance
  `0.037776 m`. This is the best fixed-schedule WP3 oracle so far, but still not a walker promotion because it
  does not reach the final waypoint.
- `route_corrector_seed53_7_wp3refinev3_long_v1`: collected a long feedback dataset from the refined WP3
  schedule. It has `3,200` samples with active counts `{0: 586, 1: 116, 2: 1206, 3: 1292}`; active-WP3 mean
  distance improved to `0.276999 m` versus the previous oracle's `0.292862 m`.
- `route_corrector_wp3only_refinev3_v1`: trained a WP3-only artifact from the refined-oracle dataset. Validation
  action MSE was `0.000168`.
- `route_corrector_gated_wp2long_wp3only_refinev3_v1_eval_s53`: gated replay with the refined WP3-only artifact
  reached `3/4`, no fall, waypoint-2 distance `0.013353 m`, and final distance `0.314993 m`. This is the best
  learned feedback-corrector result so far, slightly ahead of the previous gated WP3-only result `0.315691 m`,
  but it still trails the fixed refined schedule oracle `0.299545 m` and does not reach `4/4`.
- `wp3_segment3_s260_seed53_sync_refine_scale1_v1`: increasing the searched WP3 fixed-schedule target scale to
  `1.0` and continuing from `wp3_segment3_s260_seed53_sync_refine_v3` improved the diagnostic target distance
  to `0.294800 m`, with final distance `0.294970 m`.
- `wp2earlysync_wp3scale1v1_s260_nav`: official replay with the known WP2 schedule plus the scale-1 WP3 schedule
  stayed `3/4`, no fall, but improved final distance to `0.294959 m` and waypoint-2 distance to `0.033400 m`.
  WP3-only schedule search is still far from the `0.07 m` final waypoint gate, so the next lever is earlier WP2
  handoff timing.
- `wp2_segment3_s260_seed53_earlier_fast_w0p2_v1`: a faster stop-after-target-hit WP2 timing search with
  stronger target-step penalty found a diagnostic handoff with `target_steps=581` versus the previous `603`.
  Official replay combined with the best scale-1 WP3 schedule stayed `3/4`, no fall, but final distance regressed
  to `0.312804 m`. This candidate is rejected; earlier WP2 timing alone did not transfer into a better full
  route.
- `wp3_segment4_s260_seed53_scale1_v1`: a four-segment WP3 schedule at target scale `1.0`, initialized from the
  best three-segment scale-1 schedule, improved diagnostic target distance to `0.291956 m`.
- `wp2earlysync_wp3seg4scale1_s260_nav`: official replay with the known WP2 schedule plus the four-segment
  scale-1 WP3 schedule stayed `3/4`, no fall, and improved final distance to `0.291956 m`. This was an
  improvement over the previous fixed-schedule oracle, but still far from the `0.07 m` final waypoint gate.
- `wp3_segment4_s260_seed53_scale1p5_v1`: continuing the four-segment WP3 search at target scale `1.5` improved
  diagnostic final distance to `0.290849 m`.
- `wp2earlysync_wp3seg4scale1p5_s260_nav`: official replay with the known WP2 schedule plus the scale-1.5
  four-segment WP3 schedule stayed `3/4`, no fall, and improved final distance to `0.290866 m`. This improved
  the fixed-schedule oracle line, but it still did not reach the final checkpoint.
- `wp3_segment5_s260_seed53_scale1p5_v1`: a five-segment WP3 schedule at target scale `1.5` improved the
  diagnostic target distance to `0.285181 m`, with waypoint-2 closest distance `0.049024 m`.
- `wp2earlysync_wp3seg5scale1p5_s260_nav`: official replay with the known WP2 schedule plus the scale-1.5
  five-segment WP3 schedule stayed `3/4`, no fall, and improved final distance to `0.285226 m`; waypoint-2
  closest distance was `0.049024 m`. This is the current best fixed-schedule oracle, but it remains far from
  the `0.07 m` final waypoint gate and should be used as label-generation evidence, not as a reason to keep
  expanding fixed schedules.
- Learner-state oracle-label update: `collect_route_correction_dataset.py` now supports
  `--rollout-policy corrector` with separate rollout artifacts for WP2/WP3. In this mode the fixed schedules are
  not installed into `CommandedEnv`; the learned gated corrector generates the visited states, and the fixed
  WP2/WP3 schedules are used only as oracle labels.
- `route_corrector_learner_wp3seg5_v1`: collected `8,320` active-WP3 samples from the current gated corrector's
  return-to-origin miss trajectory, using eight start states sampled from
  `route_corrector_gated_wp2long_wp3only_refinev3_v1_eval_s53` plus `0.015 m` start jitter and `0.03` phase
  jitter. All eight starts avoided falls; closest final-origin distances ranged roughly `0.271310-0.296128 m`.
- `route_corrector_seed53_7_wp3seg5_long_v1`: collected a compatible nominal five-segment schedule dataset with
  `3,200` samples and the same correction basis. The schedule-assisted replay stayed at `3/4`, no fall, with
  waypoint-2 closest distance `0.049024 m`.
- `route_corrector_wp3only_learnerseg5_v1`: trained a WP3-only artifact on the learner-state dataset with
  held-out seed `601`; final validation action MSE was `0.006172`. Closed-loop gated replay preserved `3/4` and
  no fall, but regressed final distance to `0.321325 m`, so it is rejected.
- `route_corrector_wp3only_learnerseg5_mix_v1`: mixed nominal five-segment schedule data with learner-state WP3
  data. Final validation action MSE was `0.014495`. Closed-loop gated replay reached `3/4`, no fall, but final
  distance was `0.320124 m` at WP3 scale `0.64` and `0.325513 m` at WP3 scale `1.5`; both are rejected. This
  means naive supervised imitation of the five-segment oracle on learner-visited states still does not beat the
  older learned corrector (`0.314993 m`) or the fixed schedule oracle (`0.285226 m`).
- Command-head update: `collect_route_correction_dataset.py` now stores command-style labels
  (`command_target`, `command_correction`, and 6-D `head_target`) in addition to residual action labels.
  `train_route_command_corrector.py` trains the low-dimensional head
  `[cmd_dx, cmd_dy, residual_strength, turn_bias, phase_speed_adjust, handoff_timing]`, and
  `eval_route_corrector.py` can apply a WP3 command-head artifact behind the existing waypoint-specific gate.
- `route_command_corrector_wp3seg5_v1`: trained on `1,292` active-WP3 samples from
  `route_corrector_seed53_7_wp3seg5_head_v1`; final held-out head MSE was `0.000141`. Closed-loop replay with
  `wp3_command_blend=0.5` stayed `3/4`, no fall, but final distance regressed to `0.396910 m`. This rejects
  full-route command-head cloning as a promotion path and supports the narrower return-skill pivot.
- Return-skill implementation update: `return_skill_env.py` wraps the same CPG/PD `CommandedEnv` but exposes a
  local 16-D observation and 5-D action for active-WP3 return-to-origin states. `train_return_skill.py` and
  `eval_return_skill.py` provide the local PPO and evaluation loop.
- `return_skill_smoke_v1`: tiny PPO smoke over the local return skill trained for `16k` requested timesteps
  (`20,480` reported env steps with the PPO batch settings), saved `out/return_skill_smoke_v1.pkl`, and verified
  the train path. Local eval over eight perturbed WP3 starts had `0/8` hits, `0` falls, mean final distance
  `0.332790 m`, and mean closest distance `0.290389 m`. This is mechanism evidence only; it is not a competent
  return skill yet.
- `return_skill_v1`: medium local return-skill run requested `100k` timesteps (`122,880` reported env steps with
  PPO batch settings). It trained from scratch because the first resume check incorrectly read a hidden-layer
  width as the action width; that resume check is now fixed to accept policy output widths of `action_dim` or
  `2 * action_dim`. Local eval over eight perturbed WP3 starts had `0/8` hits, `0` falls, mean final distance
  `0.335472 m`, and mean closest distance `0.312183 m`, so the run is rejected. The return-skill mechanism is
  in place, but the reward/curriculum still needs work before it is a competent origin-return skill.
- Current corrector-data caveat: WP3 / return-to-origin data now exists in nominal, targeted, and learner-state
  forms, but supervised imitation of the fixed oracle still does not produce a better closed-loop corrector.
  Command-style label cloning also regressed in full-route replay. The next improvement should come from training
  the local return-to-origin skill with its own closed-loop objective, then gating that skill into the route.
  Perturbation-based held-out validation is still required before treating any multi-seed corrector result as
  robust.

## SPARC Combat Branch Update

Current best combat artifact:

```text
/root/proj/out/sparc_current_best.pkl
source checkpoint: out/sparc_engage_kernel_r1_smoke0_best.pkl
previous checkpoint backup: out/sparc_current_best_before_engage_kernel_r1_smoke0.pkl
current physics: robot.toml contact friction [2.8, 0.28, 0.28]
required eval flags: --engage-obs --lean-contacts
```

Earlier clean strict-search promotion:

```text
benchmark:
  sep: 0.25-0.70 m
  azimuth: full ±pi
  seeds: 20240601, 20240611, 20240621
  episodes: 8 per seed
  steps: 80
  lean_contacts: true
  engage_obs: true

result:
  sparc: +4.241
  win_rate: 0.333
  survival_rate: 1.00
  safe_rate: 1.00
  dealt: 1.7899
  taken: 0.1605
  margin: +1.6294
  closing: 1.3080
  fleeing: 2.4151
  peak_penetration: 0.04883
  judge: 70.16
  early_damage_fraction: 0.1311
```

Held-out two-seed check:

```text
seeds: 20240631, 20240641
episodes: 8 per seed

result:
  sparc: +10.074
  survival_rate: 1.00
  safe_rate: 1.00
  dealt: 2.1348
  taken: 0.6435
  margin: +1.4914
  closing: 1.7086
  fleeing: 1.4834
  peak_penetration: 0.04435
  early_damage_fraction: 0.0347
```

Broader strict replay of the promoted checkpoint:

```text
seeds: 20240601, 20240611, 20240621, 20240631, 20240641
episodes: 8 per seed

result:
  sparc: +6.58
  survival_rate: 1.00
  safe_rate: 1.00
  dealt: ~1.93
  taken: ~0.35
  margin: ~+1.58
  closing: ~1.47
  fleeing: ~2.04
  peak_penetration: ~0.0471
  early_damage_fraction: ~0.0925

weakest seed:
  seed: 20240601
  sparc: ~-6.66
  issue: high fleeing / poor medium-range conversion, not a safety or penetration failure
```

Fresh held-out replay of the promoted checkpoint:

```text
seeds: 20240651, 20240661
episodes: 8 per seed

result:
  sparc: +2.571
  survival_rate: 1.00
  safe_rate: 1.00
  dealt: 2.1806
  taken: 0.8636
  margin: +1.3170
  closing: 1.0644
  fleeing: 2.1305
  peak_penetration: 0.04415
  early_damage_fraction: 0.1697
  weakest_seed_sparc: -6.214
```

Current rank-1 engagement-kernel promotion:

```text
source: out/sparc_engage_kernel_r1_smoke0_best.pkl
method:
  action-head bias
  + rank-1 low-rank adapter on hidden_0 weights for the 8 engagement observation features

strict five-seed replay:
  seeds: 20240601, 20240611, 20240621, 20240631, 20240641
  episodes: 8 per seed
  sparc: +7.196
  survival_rate: 1.00
  safe_rate: 1.00
  dealt: 2.0208
  taken: 0.3435
  margin: +1.6773
  closing: 1.4889
  fleeing: 2.0625
  peak_penetration: 0.04704
  early_damage_fraction: 0.0946
  weakest_seed_sparc: -7.741
  weakest_seed_dealt: 1.0502

held-out replay:
  seeds: 20240651, 20240661
  episodes: 8 per seed
  sparc: +3.185
  survival_rate: 1.00
  safe_rate: 1.00
  dealt: 2.2556
  taken: 0.8633
  margin: +1.3922
  closing: 1.0927
  fleeing: 2.1265
  peak_penetration: 0.04368
  early_damage_fraction: 0.1686

fresh weak-pair check:
  seeds: 20240671, 20240681
  current_before_promotion_sparc: -0.717
  promoted_candidate_sparc: -0.602
  status: still invalid because both checkpoints produce almost no damage on this pair
```

Decision points from the SPARC push:

- [x] Add explicit training knobs for `flee_penalty`, `close_bonus`, `close_radius`, and `damage_bonus`.
- [x] Add configurable contact friction in `robot.toml` / `gen_robot_mjcf.py`.
- [x] Test stochastic deployment; reject it because stochastic held-out SPARC regressed.
- [x] Test face-opponent spawn; reject it because it regressed the existing world-frame policy.
- [x] Add `--engage-obs`; keep it enabled because the 52-wide observation checkpoint improved robust SPARC.
- [x] Add `--max-keep-peak-pen` and reject contact-solver exploits instead of selecting by headline SPARC alone.
- [x] Sweep contact friction; keep `2.8` as the current clean cap because `2.9+` improved headline SPARC but
      exceeded the `peak_pen <= 0.05` gate.
- [x] Add direct policy-head bias search: `sim/robot/search_policy_bias.py`.
- [x] Reject anti-flee PPO/head-only reward refinement because it preserved the baseline but did not reduce
      fleeing or improve held-out SPARC.
- [x] Reject one-seed hinge-bias polish as a promotion by itself: it improved seed `20240601` to `-5.796`, but
      was worse than the current checkpoint on the broader three-seed aggregate.
- [x] Promote `sparc_bias_multiseed_strict0_best.pkl`: strict CEM search over the 12 hinge-action mean biases
      produced positive SPARC with survival/safe/penetration gates intact.
- [x] Validate the promoted checkpoint on two unseen seeds before overwriting `sparc_current_best.pkl`.
- [x] Add per-seed gates and per-seed report rows to `sim/robot/search_policy_bias.py` so search cannot hide a
      bad seed behind aggregate SPARC.
- [x] Reject five-seed `min_sparc` bias search as a promotion: it barely improved the weakest seed and reduced
      aggregate SPARC.
- [x] Reject five-seed aggregate hinge-bias search as a promotion: it only improved the search aggregate by a
      tiny amount and slightly worsened fresh held-out SPARC.
- [x] Reject striker-inclusive five-seed bias search as a promotion: search-seed SPARC improved to `+11.310`,
      but the fresh held-out pair regressed from `+2.571` to `+2.166`, so `sparc_current_best.pkl` stays on the
      previous strict checkpoint.
- [x] Add a batched multi-seed evaluator to `sim/robot/search_policy_bias.py`; one compiled benchmark now returns
      both aggregate and per-seed rows, avoiding one JAX compile per seed during search.
- [x] Reject engagement-normalizer/gain-only tuning as a promotion: smoke SPARC improved slightly, but strict
      five-seed replay regressed to `+6.04`.
- [x] Add low-rank engagement input adapters to `sim/robot/search_policy_bias.py`.
- [x] Promote `sparc_engage_kernel_r1_smoke0_best.pkl`: strict five-seed SPARC improved `+6.58 -> +7.196`, and
      held-out `20240651,20240661` improved `+2.571 -> +3.185` with survival/safe/penetration gates intact.
- [x] Sweep older fighter checkpoints with one compiled benchmark; select `sparc_push_polish0_best.pkl` as the
      best seed before the final polish run.
- [x] Promote the current clean `sparc_current_best.pkl` line under friction `2.8` / engage-observation eval.
- [x] Render a headless top-down trace:
      `out/sparc_artifacts/sparc_current_strict0_passive_trace.mp4`.

Remaining SPARC blockers:

- [x] Break the negative-SPARC barrier under clean contact gates.
- [ ] Weak seeds still go negative despite clean aggregate SPARC; reduce seed-specific fleeing / medium-range
      non-conversion without losing damage margin or survival.
- [ ] Fresh seeds `20240671,20240681` still produce almost no damage. The promoted rank-1 adapter improved SPARC
      slightly there, but the pair remains invalid and needs targeted approach/contact acquisition.
      - Rejected local follow-ups: targeted rank-1 adapter search improved only to `-0.316` with dealt `0.018`;
        short full-policy PPO improved movement SPARC to `-0.21` but final benchmark damage was zero; margin search
        only raised dealt-minus-taken to about `+0.024`.
      - Rejected curriculum follow-ups: close-contact training and medium-start PPO created training-distribution
        damage or better movement, but did not cross the benchmark damage gate on `20240671,20240681`; the
        narrow-bearing seed-`20240681` run reached training dealt `~1.05-1.93` yet benchmark ratio stayed `0.00`.
        Next work should target benchmark-state strike-angle/contact conversion, not more generic approach reward.
      - New strongest lever: striker morphology. `rod_len=0.22`, `stroke=0.10`, `rod_radius=0.0152` lifts the fresh
        weak pair to SPARC `+2.612` with dealt `0.4539`, taken `0.0052`, and peak pen `0.0446`, proving the old
        striker was contact-starved. Do not promote it yet: strict five-seed replay is only `+6.041` and peak pen
        reaches `0.0510`.
      - Rejected representation/control follow-ups: fire-only overlay and geometric residual overlay did not produce
        damage; contact-observation PPO improved fire/movement but still left seed `20240681` at zero dealt on the
        old morphology.
      - New control lever: `--face-weight` close-range alignment shaping. With the `rod_len=0.22`, `stroke=0.10`,
        `rod_radius=0.0152` morphology and contact observations, seed `20240681` reached SPARC `+1.255`, dealt
        `0.1956`, taken `0.0`, peak pen `0.0469`.
      - Do not promote the face/morphology checkpoint yet: weak pair improves to SPARC `+4.343`, but strict five-seed
        replay remains below the promoted baseline and fails gates (`+4.919`, survival `0.975`, peak pen `0.0508`;
        smaller radii still fail strict).
- [ ] Fleeing is still larger than closing on broader strict and fresh held-out replay; reduce it without
      sacrificing clean damage.
- [ ] Early damage is nonzero (`~0.09` on broader strict replay and `~0.17` on the fresh held-out pair); reduce it
      without losing the damage margin.
- [ ] Do not raise contact friction beyond `2.8` unless the peak-penetration gate is also tightened and passed.
- [ ] Add a real approach/reorientation skill instead of relying on a forward-biased fighter policy.
- [ ] For all future fighter keep-best runs, set per-seed gates: `--min-keep-margin 0.0`,
      `--min-keep-survival 1.0`, and `--min-keep-safe 1.0` in addition to dealt/penetration/early-damage gates.
- [ ] Promote the next SPARC checkpoint only if 8-episode held-out SPARC improves and `survival_rate`,
      `safe_rate`, nonzero clean damage, and `peak_pen <= 0.05` remain intact.

Run this before any new pod work:

```bash
/tmp/rp.sh 'cd /root/proj && ps -eo pid,etimes,cmd | grep -E "python .*sim/robot/(search|eval|train|collect|compare|check|render)|python .*eval_commanded|python .*train_commanded|python .*eval_checkpoint|python .*train_residual|jax|mjx" | grep -v grep || true'
```

Then do the next gate in this order:

- [x] Add a route-aware CPG search path that scores the actual checkpoint route,
      not only isolated cardinal rollouts.
- [x] Seed route-aware search from:
  - [x] `out/cpg_rich_long_forward_gait.json`
  - [x] `out/cpg_rich_left_left_gait.json`
  - [x] `out/cpg_rich_back_backward_gait.json`
  - [x] `out/cpg_rich_right_right_gait.json`
- [x] Carry simulator state through direction changes during search.
- [x] Penalize cross-axis drift at each segment, not only final displacement.
- [x] Penalize loss of previously reached segment-axis progress.
- [x] Keep survival, uprightness, torque, saturation, and joint-limit penalties.
- [x] Save route candidates with the full raw parameter vectors and trace stats.
- [x] Validate route candidates through `eval_checkpoint_navigation.py`.
- [x] Add a focused third-segment/backward-transition search objective.
- [x] Add full-command route scoring and deployable router settings.
- [x] Generate `5k-50k` short transition rollouts around the waypoint-2 miss region.
- [x] Label each transition rollout with distance reduction, Y drift, X loss, fall state, and saturation.
- [x] Build a transition-aware gait dataset from:
  - [x] stable cardinal rollouts
  - [x] partial `2/4` route rollouts
  - [x] focused waypoint-2 transition rollouts
  - [x] failed-but-informative drift traces
- [x] Move to BC dataset collection once:
  - [x] fixed cardinal survival passes
  - [x] at least one route reaches `2/4` without falling
  - [x] focused transition buckets produce nonzero progress toward the failing waypoint
  - [x] the dataset covers every command family and route-transition family
- [x] Train BC/residual to reduce Y drift during the backward segment after waypoint 2.
- [x] Make `3/4` at radius `0.07` a BC/residual walker validation gate, not a pure CPG teacher prerequisite.
- [x] Train the next compact route corrector from the successful WP2/WP3 schedules plus extra WP3 failure states.
- [x] Evaluate the compact route corrector against fixed-schedule baselines on strict-radius replay.
- [ ] Add perturbation-based held-out validation before calling a multi-seed corrector robust.
- [x] Replace broad full-route perturbation mixing with targeted handoff-state collection near WP2/WP3.
- [x] Add a safer waypoint-specific/gated corrector path after targeted WP3 mixing regressed early route replay.
- [x] Find or train a WP3-specific artifact that improves final distance when used behind the gate.
- [ ] Improve the gated WP3-specific artifact beyond the fixed WP2+WP3 schedule baseline or reach `4/4`.
- [x] Use the current best five-segment WP3 fixed schedule as an oracle-label source for WP3-only feedback data.
- [x] Collect learner-visited WP3 miss states from the current best gated corrector, not only schedule-assisted
      nominal states.
- [x] Retrain/evaluate a WP3-only compact corrector from those learner states before running more broad route
      schedule search.

Minimum validation command for every new route candidate:

```bash
eval $(/root/proj/venv/bin/python - <<'PY'
import json, shlex
j = json.load(open("out/<candidate_route_gait>.json"))["deploy_env"]
for k, v in j.items():
    print(f"export {k}={shlex.quote(v)}")
PY
)
CODESIGN_OUT=/root/proj/out \
CMD_CONTROL_MODE=cpg_pd \
CMD_CPG_RESIDUAL_SCALE=0.0 \
CMD_TRACK_SIGMA=0.05 \
/root/proj/venv/bin/python sim/robot/eval_checkpoint_navigation.py \
  --tag <candidate_nav_tag> \
  --ckpt out/walk22_forward_resid.pkl \
  --waypoints "0.35,0;0.35,0.35;0,0.35;0,0" \
  --radius 0.07 \
  --gain 2.0 \
  --steps-per-waypoint 260 \
  --seed 7
```

Use the candidate JSON's `deploy_eval_args` when present. Older segment-axis candidates need
`--segment-axis`; full-command/router candidates intentionally omit it.

Promotion rule:

- [x] Reject a standalone pure-CPG teacher if it reaches fewer than `3/4` waypoints.
- [x] Allow a scaffold teacher into BC if it reaches `2/4` without falling and transition buckets cover the
      failing route states.
- [x] Treat fixed residual schedules that stay at `3/4` as oracle-label sources, not promoted walkers.
- [x] Reject if it falls or violates uprightness gates.
- [x] Reject if fixed forward/backward/left/right survival regresses.
- [x] Promote a BC/residual variant only if fixed-direction evals survive and closed-loop checkpoint navigation
      improves against the current baseline, regardless of BC validation loss.
- [x] Fail fast if a requested residual-promotion baseline JSON cannot be resolved.
