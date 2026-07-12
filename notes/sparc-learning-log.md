# SPARC Learning Log

> **Document status:** Active · **Audience:** Robot-learning contributors · **Last reviewed:** 2026-07-12 · **Canonical summary:** [`locomotion-status.md`](locomotion-status.md)

This is a detailed experiment log. Use the linked summary for the current
project-level verdict; entries below retain exact intermediate outcomes.

Purpose: keep the working lessons from SPARC experiments explicit, including rejected paths.

## Current Baseline

- Promoted checkpoint is `/root/proj/out/sparc_current_best.pkl`.
- Required eval flags are `--engage-obs --lean-contacts`.
- Contact friction remains `[2.8, 0.28, 0.28]`; higher friction improved headline SPARC but tripped the contact-penetration gate.
- Current promoted checkpoint source is `/root/proj/out/sparc_engage_kernel_r1_smoke0_best.pkl`.
- Current best router candidate is `/root/proj/out/sparc_current_best_router.json`.
- Current promoted checkpoint is valid and positive, but not robustly high:
  - 5-seed strict replay: about `+7.196 SPARC`, dealt `2.0208`, taken `0.3435`, margin `+1.6773`, peak penetration `0.04704`.
  - fresh heldout `20240651,20240661`: about `+3.185 SPARC`, dealt `2.2556`, taken `0.8633`, margin `+1.3922`, peak penetration `0.04368`.
  - fresh weak pair `20240671,20240681` can still be poor; seed `20240681` is the recurring zero-damage blocker.

## Status Checkpoints

### 2026-06-26 06:23 EDT

- Remote pod is idle; no active SPARC train/search/eval jobs were running.
- Latest remote artifact is still `sparc_mix_blocker_heldout0_checkpoint_interpolation.json`.
- Local `sim/robot/search_policy_router.py` exists and compiles.
- Next experiment should be action-level observation-gated routing between the all-seven branch and heldout branch, not parameter interpolation or another single global adapter.

### 2026-06-26 Router Experiments

- Added `sim/robot/search_policy_router.py` support for:
  - saved-router replay via `--eval-router-json`;
  - nonzero per-seed dealt gates via `--per-seed-min-dealt`;
  - initialized CEM from a saved vector via `--init-router-json`.
- Smoke router between the all-seven branch and heldout branch worked mechanically, but all-nine replay was not promotable:
  `sparc_router_blocker_heldout_judge0_all9_eval0` scored SPARC `+4.855`, dealt `1.3227`, taken `0.3433`, margin `+0.9794`, valid. It still left seed `20240681` at zero damage.
- Router between the all-seven branch and the seed-81 bias checkpoint at the safer `rod_len=0.20`, `rod_radius=0.0135` was also not promotable:
  `sparc_router_blocker_seed81bias_judge0_all9_eval0` scored SPARC `+5.993`, dealt `1.5989`, taken `0.3843`, margin `+1.2146`, valid, but seed `20240681` still had zero dealt damage.
- Important correction: the earlier seed-81 bias fix was geometry-dependent. It used `rod_len=0.22`, `rod_radius=0.0152`; under the safer `0.20/0.0135` geometry, pure branch B replayed seed `20240681` at zero damage.
- Under the larger `0.22/0.0152` geometry, pure seed-81 branch B does reproduce the seed-81 contact fix:
  `sparc_router_seed81bias_bonly_seed81_len022_eval0` scored seed `20240681` at dealt `0.2458`, taken `0.0`, valid.
- Larger-geometry all-nine router search briefly looked strong at 4 episodes (`SPARC +7.727`, valid), but 8-episode replay rejected it:
  `sparc_router_len022_blocker_seed81bias_judge0_all9_eval0` scored SPARC `+5.119`, dealt `1.4750`, taken `0.4463`, margin `+1.0287`, valid; seed `20240681` returned to zero dealt damage.
- Initialized 8-episode refinement did not materially improve the larger-geometry router:
  `sparc_router_len022_seed81bias_all9_epis8_refine0` scored SPARC `+5.135`, dealt `1.4773`, taken `0.4457`, margin `+1.0317`, valid; seed `20240681` still had zero dealt damage.
- Worst-seed/per-seed-dealt search on blocker seeds did force nonzero seed-81 damage in-sample (`20240681` dealt `0.1685`), but failed the per-seed dealt gate because seed `20240621` sampled zero damage. Pure A and pure B both produce nonzero seed21 damage at 8 episodes, so that zero was sample variance plus a too-small 4-episode search budget.
- Current router conclusion: action-level gating is useful, but a single linear logistic gate over the current features is not sharp enough. The next representation should be a nonlinear/stateful router, a direct low-dimensional residual correction, or a three-way router with explicit seed-81/contact-acquisition behavior, evaluated directly on the 8-episode all-nine objective.

### 2026-06-26 Hybrid Router Results

- Added feature-mean reporting to `search_policy_router.py`. Seed `20240681` is distinctive in the failing mixed routers: high `min_rod` and low `front`, but average feature thresholds alone are too crude.
- Added nonlinear hidden router support. `router_hidden=2` improved the 8-episode all-nine objective only modestly:
  `sparc_router_len022_seed81bias_mlp2_all9_epis8_0` scored SPARC `+5.236`, dealt `1.4522`, taken `0.3769`, margin `+1.0753`, valid, but seed `20240681` still had zero damage.
- A wide linear state gate with a hard nonzero per-seed dealt gate proved all-seed contact is feasible:
  `sparc_router_len022_linear_allseed_dealt0` scored SPARC `+4.969`, dealt `1.3622`, taken `0.3870`, seed-min dealt `0.0480`, valid. It fixed seed `20240681` (`dealt=0.2777`) but sacrificed aggregate score.
- Constant per-action mixing alone was not enough. Best action-gate candidate scored SPARC `+5.557`, but was invalid because seed `20240681` still had zero damage.
- The useful representation is action mixing plus a state-conditioned branch-B boost:
  `sparc_router_len022_actionboost_allseed_dealt0` is now the best router candidate:
  SPARC `+6.497`, dealt `1.6106`, taken `0.3820`, margin `+1.2286`, win rate `0.3194`, survival `1.0`, safe `1.0`, peak penetration `0.0153`, seed-min dealt `0.0477`, valid.
- Per-seed rows for that best hybrid:
  `20240601`: SPARC `+1.004`, dealt `1.7672`, taken `0.0451`;
  `20240611`: SPARC `+24.124`, dealt `4.8917`, taken `0.6405`;
  `20240621`: SPARC `+3.402`, dealt `0.0477`, taken `0.0`;
  `20240631`: SPARC `+9.954`, dealt `0.7092`, taken `0.0`;
  `20240641`: SPARC `-2.378`, dealt `1.1175`, taken `1.4355`;
  `20240651`: SPARC `+6.634`, dealt `1.9682`, taken `0.6301`;
  `20240661`: SPARC `-5.207`, dealt `0.7250`, taken `0.6868`;
  `20240671`: SPARC `+18.228`, dealt `2.9981`, taken `0.0`;
  `20240681`: SPARC `+2.713`, dealt `0.2712`, taken `0.0`.
- Adding the heldout-specialized checkpoint as a third state-gated branch did not improve the result:
  `sparc_router_len022_actionboost_heldoutc0` scored SPARC `+6.492`, essentially tied/slightly worse.
- Worst-seed optimization was rejected as lead candidate:
  `sparc_router_len022_actionboost_minjudge0` improved seed-min judge only from about `8.055` to `8.674`, while dropping aggregate SPARC to `+4.353`.
- Current decision point: the best direction is no longer scalar routing. The strongest working control representation is `per-action A/B mix + state-conditioned boost toward seed81-contact branch`. Remaining blockers are seed `20240641` negative margin and seed `20240661` low SPARC/low margin.

### 2026-06-26 Follow-up Router/Geometry Checks

- Exact larger-geometry replay of alternate pure branches showed they are not good sources for the remaining weak seeds:
  - `sparc_router_strictheldout_pure_len022_all9_eval0`: SPARC `+2.787`, margin `+0.6575`, valid; seed `20240641` margin `-0.3668`, seed `20240661` margin `+0.1311`, seed `20240681` zero dealt.
  - `sparc_router_heldout_pure_len022_all9_eval0`: SPARC `+4.115`, margin `+0.7909`, valid; seed `20240641` margin `-0.3059`, seed `20240661` margin `+0.1623`, seed `20240681` zero dealt.
- Nearby geometry sweep of the best hybrid router on seeds `20240641,20240661,20240681` did not improve the weak subset:
  - `rod_len=0.210`, `rod_radius=0.0145`: invalid, focused SPARC `-2.077`, margin `-0.0548`.
  - `rod_len=0.210`, `rod_radius=0.0152`: invalid, focused SPARC `-2.094`, margin `-0.0593`.
  - `rod_len=0.220`, `rod_radius=0.0145`: invalid, focused SPARC `-1.922`, margin `-0.0611`.
  - `rod_len=0.220`, `rod_radius=0.0152`: valid baseline, focused SPARC `-1.585`, margin `+0.0013`.
- Focused weak-seed refinement on seeds `20240641,20240661,20240681` gave only a small local improvement and hurt all-nine replay:
  `sparc_router_len022_actionboost_weak416181_refine0_all9_eval0` scored SPARC `+5.101`, dealt `1.3244`, taken `0.3413`, margin `+0.9831`, valid. It improved seed `20240681` dealt to `0.3028`, but remained worse globally than the lead hybrid.
- Relaxing the hard nonzero per-seed dealt requirement did not raise the all-nine ceiling:
  `sparc_router_len022_actionboost_unconstrained0` scored SPARC `+6.497`, dealt `1.6077`, taken `0.3800`, margin `+1.2278`, seed-min dealt `0.0477`, valid, effectively tied with the lead hybrid.
- Current lead remains `sparc_router_len022_actionboost_allseed_dealt0`. Next useful step is probably not more scalar/branch routing. The open problem is a seed41/seed61 local correction or training branch that reduces taken damage/mutual trade while preserving seed81 contact.

### 2026-06-26 Seed41/61 Specialist Attempts

- Built a seed41/61 specialist branch from the seed81-contact checkpoint using a rank-1 output adapter:
  `sparc_seed4161_len022_output_r1_0_best.pkl`.
  It improved the target pair slightly but did not solve trade:
  target aggregate margin moved from `-0.1100` to `-0.0466`;
  seed `20240641` margin improved to `-0.2272`;
  seed `20240661` margin improved to `+0.1341`.
- Using that specialist as a third router branch did not help:
  `sparc_router_len022_actionboost_seed4161c0` stayed effectively C-off and scored SPARC `+6.466`, below the lead.
- Continued the specialist with a rank-2 margin-focused output adapter:
  `sparc_seed4161_len022_margin_r2_0_best.pkl`.
  This was the first standalone specialist with positive average target margin:
  target aggregate margin `+0.0533`;
  seed `20240641` margin `-0.0377`;
  seed `20240661` margin `+0.1442`.
- Routing that margin specialist as branch C produced only a numerical tie/micro-improvement:
  `sparc_router_len022_actionboost_margin4161c0` scored SPARC `+6.500`, dealt `1.6108`, taken `0.3823`, margin `+1.2285`, seed-min dealt `0.0477`, valid.
  It did not fix seed41/61 in the full router: seed `20240641` margin stayed about `-0.318`, seed `20240661` margin stayed about `+0.038`.
- Practical conclusion: the seed41/61 specialist can improve those seeds as a standalone branch, but the current state-gated branch selector does not identify a useful window to apply it. A future fix likely needs either time/contact-phase features in the router or a local residual/correction policy trained on seed41/61 trade states, not just another static branch.

### 2026-06-26 Time-Feature Router

- Added optional `--time-features` to `search_policy_router.py`, appending normalized episode `time` and `time_sq` to router feature vectors. Old router JSONs still work without this flag; time-enabled runs use padded vectors.
- Padded the previous current-best actionboost router into `sparc_current_best_router_time_init0.json` and searched with the seed41/61 margin specialist as branch C.
- New lead:
  `sparc_router_time_actionboost_margin4161c0` scored SPARC `+6.599`, dealt `1.6233`, taken `0.3834`, margin `+1.2399`, win rate `0.3194`, survival `1.0`, safe `1.0`, peak penetration `0.0153`, seed-min dealt `0.0482`, seed-min judge `8.1924`, valid.
- Per-seed changes versus the previous current-best were small but consistent:
  seed `20240641` margin improved from about `-0.318` to `-0.301`;
  seed `20240661` margin improved from about `+0.038` to `+0.041`;
  seed `20240681` stayed solved with dealt about `0.284`.
- A follow-up lower-sigma refinement did not improve it:
  `sparc_router_time_actionboost_margin4161c_refine1` scored SPARC `+6.543`, margin `+1.2323`, valid.
- Updated `/root/proj/out/sparc_current_best_router.json` and local `out/sparc_artifacts/sparc_current_best_router.json` to the time-feature lead.

## Lessons So Far

- Bias-only polishing can break the negative-SPARC barrier, but constant action bias is reaching diminishing returns.
- The current failure mode is mostly weak-seed medium-range conversion: too much fleeing / too little closing, not falling or contact exploitation.
- Striker-inclusive action-bias search improved five search seeds strongly (`+6.58 -> +11.31`) but regressed fresh heldout (`+2.57 -> +2.17`), so it was not promoted.
- Per-seed gates are necessary; aggregate SPARC can hide a bad seed.
- Promotion should require strict contact gates plus fresh heldout improvement, not just search-seed improvement.
- A better next lever is state-conditioned approach/reorientation, because the SPARC term directly rewards `closing - fleeing`.

## Current Experiment

- Added optional engagement-normalizer search to `sim/robot/search_policy_bias.py`.
- It tunes the last 8 engagement observation features:
  `[dist, unit_x, unit_y, radial, lateral, rel_radial, closing, fleeing]`.
- This keeps the checkpoint interface unchanged while changing how strongly the existing policy reacts to opponent state.
- First large seven-seed attempt was stopped because JAX spent too long compiling before producing a score.
- Current run: `sparc_engage_norm_5seed0`, five strict seeds, action bias plus engagement normalizer shift/gain.
- The five-seed strict version also spent too long compiling before a first score. Lesson: use cheap
  smoke-search settings for adapter exploration, then run the full strict benchmark only on candidates.
- Patched `search_policy_bias.py` with a single batched multi-seed benchmark. This made the smoke search complete
  instead of spending minutes compiling one benchmark per seed.
- First smoke adapter search (`sparc_engage_norm_smokesearch0`) only nudged short-horizon SPARC from `-4.33` to
  `-4.20` and stayed below the smoke dealt gate. Next search should rank by raw smoke SPARC and leave strict gates
  for promotion replay.
- Second smoke adapter search (`sparc_engage_norm_smokesearch1`) improved smoke SPARC only slightly (`-4.33` to
  `-4.17`). Strict five-seed replay scored `+6.04`, worse than current `+6.58`, so normalizer/gain-only tuning is
  rejected for promotion.
- Next representation: low-rank first-layer adapter on the 8 engagement features. This is still checkpoint-compatible
  but can create new state-conditioned approach/reorientation responses rather than only rescaling existing ones.
- Rank-1 engagement-kernel smoke search (`sparc_engage_kernel_r1_smoke0`) improved smoke SPARC from `-4.33` to
  `-4.12`. Strict five-seed replay improved from the promoted baseline `+6.58` to `+7.20` while staying valid.
  This is the first adapter result worth held-out testing.
- The rank-1 adapter was promoted after held-out `20240651,20240661` improved from `+2.57` to `+3.18`.
- Fresh `20240671,20240681` remains the next blocker: current promoted score is about `-0.60` with almost no
  damage. A targeted rank-1 follow-up improved it to `-0.32`, but dealt damage only rose from `0.007` to `0.018`
  and one seed still has zero damage. Not promotable; this needs actual approach/contact acquisition, not just
  small post-hoc adapter tweaks.
- Short full-policy PPO refinement with strong approach/flee/contact shaping improved `71/81` benchmark SPARC
  from `-0.60` to `-0.21`, but held-out dealt damage never crossed the keep gate and final benchmark damage was
  zero. Rejected: the shaping improved movement without producing contact inside the 80-step benchmark.
- Margin-focused local search on `71/81` improved dealt-minus-taken only from about `+0.007` to `+0.024`.
  Rejected: optimizing damage margin locally still cannot discover enough contact from those starts.
- Current conclusion for `71/81`: the policy can improve approach metrics but does not reach/strike fast enough
  within 80 steps. Next useful work should create an explicit contact-acquisition curriculum or reset distribution
  near the problematic starts, then distill/merge only if broader SPARC does not regress.
- A 120-step replay still dealt only `0.007`, so the failure is not fixed by a modestly longer horizon.
- Top-down render of weak seed `20240681` episode 0 at step 81 shows A still separated from B by roughly half a
  meter, with weapon points moving around A rather than reaching into B. Visual diagnosis matches the metrics:
  approach geometry/contact acquisition is the blocker.
- Close-contact acquisition training (`sep 0.12-0.32`) produced strong close-range training damage, but did not
  transfer to `71/81`: benchmark SPARC worsened from about `-0.60` to `-1.07` and dealt damage stayed below the
  keep gate. Rejected: close-contact skill alone specializes without solving medium-start approach geometry.
- Added direct `dealt` / `min_dealt` objectives to `search_policy_bias.py`.
- Rank-2 engagement adapter search optimized directly for dealt damage on `71/81`, but best dealt only reached
  about `0.017`. Rejected: even stronger local adapters cannot discover the missing contact behavior from these
  starts.
- Medium-start PPO curriculum (`sep 0.45-0.70`, full azimuth, high approach/flee shaping) matched the weak reset
  geometry better than close-contact training. It improved `71/81` benchmark SPARC from about `-0.60` to `-0.09`,
  but dealt damage still stayed below the `0.05` keep gate and no best checkpoint was saved. Rejected for promotion:
  approach improved, contact acquisition still failed.
- Added low-rank output-mean-kernel adapters to `search_policy_bias.py`, effectively a small LoRA on the action
  head while keeping checkpoint compatibility.
- Rank-2 output adapter optimized for dealt damage on `71/81` produced the best local contact so far: average dealt
  rose to `0.032`, with one seed at `0.063`, but the other seed stayed at zero and the candidate remains invalid
  for promotion.
- Follow-up `min_dealt` search from that output-adapter candidate could not lift the zero-damage seed; min dealt
  stayed `0.000`. The remaining failure is specific-start contact geometry, not just insufficient adapter capacity.
- Isolating seed `20240681` alone from the one-seed-hit output-adapter candidate still produced zero dealt damage
  after five generations. Movement SPARC improved, but no contact appeared. This seed needs a qualitatively new
  low-level approach/turn/strike behavior or reset-aware data, not another small adapter polish.
- Narrow-bearing PPO curriculum for seed `20240681` (`sep 0.45-0.70`, azimuth `1.10`, high approach/contact shaping,
  `rod_reach 0.42`) created real damage in the training reset distribution (`dealt` about `1.05-1.93` during train
  reports), but benchmark replay on the hard seed stayed at `ratio 0.00`, margin near zero, and never crossed the
  `min_keep_dealt` gate. Rejected: training-distribution contact is not enough; the missing piece is still
  bearing-specific contact conversion under the benchmark reset, likely requiring a more explicit strike-angle/contact
  oracle or targeted action-space adapter rather than more generic approach reward.
- Contact-geometry diagnostic for seed `20240681` showed the policy is frequently side-on or backward-facing at
  closest approach, and often stops firing exactly when a rod is closest. A fire-only geometric overlay, including
  always-fire, still produced zero dealt damage. A compact geometric residual overlay improved movement SPARC
  (`-0.44 -> +2.06`) but also produced zero dealt damage. Rejected: the issue is not just striker fire timing; it is
  reach/orientation/contact geometry.
- Added optional `--contact-obs` features to `train_adversarial.py`: rod distances, min limb distance, front/side
  alignment, rod-close, and body-close. Warm-start from the promoted `52`-obs checkpoint to `60` obs works cleanly.
  Contact-observation PPO on the old morphology increased fire and movement SPARC on seed `20240681`, but benchmark
  dealt damage stayed `0.0`. Rejected for promotion.
- Striker morphology sweep found the first real weak-pair contact lever. With `rod_len=0.22`, `stroke=0.10`, and
  `rod_radius=0.0152`, the fresh weak pair `20240671,20240681` improved to SPARC `+2.612`, dealt `0.4539`, taken
  `0.0052`, peak penetration `0.0446`. This is the strongest evidence that the previous design was contact-starved.
- The same morphology is not promotable yet: strict five-seed replay scored SPARC `+6.041`, dealt `1.8562`, taken
  `0.6049`, and peak penetration `0.0510`, below the promoted strict SPARC `+7.196` and slightly over the
  penetration gate. Radius `0.0158` was similar/worse on strict (`+5.755`, peak pen `0.0514`).
- PPO on the `rod_len=0.22`, `radius=0.0152` morphology with contact observations kept strong aggregate benchmark
  damage but still had `bench_min_dealt=0.0`; one seed remains contact-starved. A focused `radius=0.016` seed-81 run
  crossed a low dealt gate only by taking much more damage (negative margin), so it was rejected.
- Patched `train_adversarial.py` keep-best logic with `--min-keep-margin`, `--min-keep-survival`, and
  `--min-keep-safe`. Lesson: `min_keep_dealt` alone can save a bad checkpoint with negative per-seed margin; future
  runs should require per-seed positive margin and full survival/safety when selecting best.
- Added `--face-weight` close-range alignment shaping. On the `rod_len=0.22`, `stroke=0.10`, `rod_radius=0.0152`
  morphology, face-alignment PPO produced the first clean learned seed-`20240681` fix: seed-81 benchmark SPARC
  `+1.255`, dealt `0.1956`, taken `0.0`, win `0.125`, peak pen `0.0469`, saved at step `573440` under stricter
  positive-margin/survival/safety gates.
- The face-alignment checkpoint strongly improves the fresh weak pair when evaluated with the same morphology and
  contact observations: `20240671,20240681` SPARC `+4.343`, dealt `0.7757`, taken `0.0`, peak pen `0.0440`.
  Smaller radii `0.0145-0.0150` preserve most of that weak-pair gain.
- The face-alignment checkpoint is still not promotable: strict five-seed replay is only SPARC `+4.919` at
  `rod_radius=0.0152`, with survival `0.975` and peak pen `0.0508`; radius `0.0150` and `0.0145` still fail strict
  (`+4.756` / `+5.069`, survival `0.975`, peak pen `~0.0506-0.0508`). Current promoted rank-1 engagement adapter
  remains the production baseline.
- Shortening the face-alignment morphology to `rod_len=0.20`, `stroke=0.10`, `rod_radius=0.0145` recovered much of
  the strict five-seed aggregate score: SPARC `+7.194`, dealt `2.0576`, taken `0.4998`, margin `+1.5578`. It still
  is not promotable because seed-level inspection shows multiple peak-penetration violations above `0.05` and seed
  `20240611` survival only `0.875`. Aggregate averages can hide gate failures.
- Continuing from that candidate with a penetration penalty (`penetration_penalty=120`, tolerance `0.045`) over the
  seven-seed bench improved aggregate SPARC to about `+7.24` with margin about `+1.40`, but did not clear the hard
  gates: survival stayed `0.9821` and max peak penetration stayed around `0.0535-0.0539`. No promoted best was saved.
- A safer broad geometry sweep from the penalty checkpoint found a strong aggregate candidate at
  `rod_len=0.20`, `rod_radius=0.0135`: seven-seed SPARC `+7.484`, dealt `1.7534`, taken `0.3090`, margin `+1.4444`.
  It is still invalid because strict seed rows keep peak penetration around `0.052-0.053` and seed `20240611`
  survival remains `0.875`.
- A seed-`20240611` survival/penetration fine-tune with higher penalty (`300`) did not improve the failure mode:
  benchmark survival stayed about `0.88` and peak penetration stayed about `0.053`. This branch is rejected. The
  failure now looks like a deterministic contact/posture geometry issue, not something reward magnitude fixes.
- Current SPARC decision point: the promoted baseline remains `/root/proj/out/sparc_current_best.pkl`
  (`sparc_engage_kernel_r1_smoke0_best.pkl`). The next useful step is contact-pair diagnostics for seed `20240611`
  under the strong morphology candidates, because we need to know whether the hard gate violation comes from rod/body,
  foot/floor, torso/floor, or another contact pair before training more.
- Contact-pair diagnostic for seed `20240611` on `sparc_face_len020_penalty0_ckpt.pkl` with
  `rod_len=0.20`, `stroke=0.10`, `rod_radius=0.0135` showed the hard peak-penetration issue is floor/calf, not
  striker/body. Worst aggregate pair was `floor | A_RL_calfg` with max penetration `0.07183`; next were
  `floor | A_RR_calfg` (`0.06200`), `floor | B_RR_calfg` (`0.05868`), and `floor | B_RL_calfg` (`0.05807`).
  The one falling episode also had dominant rear-calf/floor contact. This points at stance/contact physics, not
  weapon reach.
- Added optional contact `solref` support in `gen_robot_mjcf.py` and a `--contact-timeconsts` sweep path in
  `sweep_striker_geometry.py`. Seed-`20240611` stiffness sweep found:
  `solref=0.012` fixes penetration (`0.0234`) but not survival (`0.88`);
  `solref=0.010` fixes survival and penetration (`survival=1.00`, `pen=0.0182`) but drops damage;
  `solref=0.008/0.006` also survive cleanly but lose still more damage.
- Full seven-seed replay with the safer `solref=0.010` is clean but not promotable: SPARC `+1.252`, dealt `0.4388`,
  taken `0.1167`, margin `+0.3220`, survival `1.00`, peak pen `0.0192`. It regresses seed `20240681` to zero dealt
  damage and seed `20240641` to negative margin. Conclusion: firmer contact is a useful safer adaptation environment,
  not a no-retrain promotion path.
- Added `--contact-solref-timeconst` / `--contact-solref-dampratio` to `train_adversarial.py` so PPO adaptation can
  run on the same safer contact physics used by the sweep.
- Short blocker-seed adaptation under `solref=0.010` (`sparc_solref010_blocker_adapt0`, seeds
  `20240611,20240641,20240681`) stayed physically clean but did not open the keep gate. Initial blocker eval:
  SPARC `-0.124`, margin `+0.1286`, min margin `-0.2917`, min dealt `0.0`, survival `1.0`, peak pen `0.0207`.
  Best-looking midpoint at step `327680`: SPARC `-0.053`, margin `+0.1418`, min margin `-0.2431`, min dealt `0.0`.
  Final step `655360` regressed to SPARC `-0.470`, margin `+0.0710`, min margin `-0.3752`. No best checkpoint was
  saved. Conclusion: generic reward adaptation under stiff contacts does not recover the missing seed-`20240681`
  contact or seed-`20240641` margin quickly enough.
- Stiff-contact seed-`20240681` morphology sweep with larger/longer rods also failed to recover damage:
  `rod_len=0.22/0.24` and `rod_radius=0.0152/0.0160` all had dealt `0.0` with clean penetration around `0.0184`.
  Conclusion: once contacts are made stiff enough to clear floor/calf penetration, this checkpoint loses the
  seed-`20240681` strike mechanism; the next useful branch needs a targeted seed-81 contact-acquisition policy or a
  different action/stance correction, not just rod size.
- Added `--disable-calf-floor` as a targeted mask variant. It cleared the floor/calf penetration on seed `20240611`
  while preserving high damage (`dealt=3.2472`, peak pen `0.0389`), but survival collapsed to `0.625`. On the blocker
  set (`20240611,20240641,20240681`) it stayed high-damage but only `0.667` survival and seed `20240681` still had
  zero damage. Rejected as a direct promotion path: the policy was using calf-ground contact as stance support.
- Added pair-specific `floor_calf_solref` support so only `floor` vs `*_calfg` contacts can be stiffened while
  striker/opponent contacts remain soft. Seed `20240611` sweep with the penalty checkpoint:
  `floor_calf=0.012` gave SPARC `+5.700`, pen `0.0171`, survival `0.88`;
  `floor_calf=0.010` gave SPARC `+7.857`, pen `0.0163`, survival `0.88`;
  `floor_calf=0.008` gave SPARC `+5.637`, dealt `1.4684`, margin `+1.4216`, pen `0.0153`, survival `1.00`.
  This is the first setting that fixes seed-`20240611` survival and peak penetration without globally stiffening all
  contacts.
- Full seven-seed replay with `floor_calf=0.008`, `rod_len=0.20`, `rod_radius=0.0135` was clean but below the
  promoted baseline: SPARC `+5.029`, dealt `1.1351`, taken `0.2003`, margin `+0.9348`, survival `1.00`, peak pen
  `0.0150`. Remaining blockers: seed `20240681` still has zero damage; seed `20240621` and `20240641` have slightly
  negative margin.
- Focused seed-`20240681` adaptation under `floor_calf=0.008` from the face-alignment checkpoint did not recover
  held-out damage. Training rollouts had contact, but benchmark seed `20240681` stayed clean/zero-damage at every
  eval: step `0` dealt `0.0`, step `327680` dealt `0.0`, step `655360` dealt `0.0`; survival `1.0`, peak pen
  `~0.0138`. No best checkpoint was saved. The current missing piece is reset-specific seed-81 contact acquisition,
  not general contact reward or global morphology.
- Added `--reset-bank-seed` / `--reset-bank-epis` to `train_adversarial.py` so training can sample the exact fixed
  benchmark reset keys. Seed-`20240681` reset-bank PPO under `floor_calf=0.008` still did not create deterministic
  benchmark damage: step `0/327680/655360` all had dealt `0.0`, survival `1.0`, peak pen `~0.014`. Stochastic replay
  of the final reset-bank checkpoint produced only tiny damage (`dealt=0.0072`). Conclusion: this is not just a
  deterministic-mean-vs-sampled-action issue.
- Patched `search_policy_bias.py` to support contact observations, striker geometry overrides, and floor/calf contact
  overrides. A direct action-bias CEM search on seed `20240681` under `floor_calf=0.008` immediately found a valid
  deterministic contact candidate (`dealt/margin=+0.281`, peak pen `0.0141`). This proves the seed-81 miss is
  reachable by a small deterministic correction, but a global action bias hurts other seeds.
- All-seven replay of the seed-81 action-bias candidate scored SPARC `+5.285`, dealt `1.1851`, taken `0.2206`,
  margin `+0.9646`, peak pen `0.0151`, survival `0.9821`. It fixed seed `20240681` (`dealt=0.2929`) but regressed
  seed `20240611` survival to `0.875` and left seed `20240641` negative margin. Rejected for promotion.
- A rank-1 output-kernel adapter search on blocker seeds `20240611,20240641,20240681` under
  `floor_calf=0.008`, `rod_len=0.20`, `rod_radius=0.0135` produced the strongest alternate branch so far:
  all-seven SPARC `+7.548`, dealt `1.6280`, taken `0.2873`, margin `+1.3408`, survival `1.0`, peak pen `0.0150`.
  It beats the current promoted baseline on the seven-seed aggregate, but not on the original strict/heldout split:
  strict-five average is about `+6.685` (below promoted `+7.196`), heldout `20240651,20240661` is `+1.533` (below
  promoted `+3.185`), and seed `20240681` is still zero-damage. Do not promote yet; keep as an alternate branch and
  use obs-conditioned adapters, not global action bias, for further work.
- Broader strict+heldout+seed81 rank-1 output search from that branch improved heldout but gave up aggregate. Candidate
  `sparc_strict_heldout_floorcalf_output_r1_0_best.pkl` replayed at all-seven SPARC `+7.163`, dealt `1.6249`,
  taken `0.3353`, margin `+1.2896`, survival `1.0`, peak pen `0.0148`; heldout `20240651,20240661` improved to
  SPARC `+2.254`, but still below the promoted heldout `+3.185`. Rejected for promotion.
- A weak-seed min-judge search on `20240621,20240641,20240661,20240681` from the broader candidate improved the
  weak-objective in search but replayed worse on all-seven: SPARC `+6.981`, dealt `1.5824`, taken `0.3531`,
  margin `+1.2293`, survival `1.0`, peak pen `0.0147`. Rejected.
- A heldout-only rank-1 output search from the broader candidate successfully pushed heldout above the promoted
  heldout score: `sparc_heldout_floorcalf_output_r1_0_best.pkl` replayed on `20240651,20240661` at SPARC `+3.256`,
  dealt `1.8629`, taken `0.8155`, margin `+1.0473`, survival `1.0`, peak pen `0.0145`. However, its all-seven replay
  fell to SPARC `+5.369`, dealt `1.1897`, taken `0.2262`, margin `+0.9635`, survival `1.0`, peak pen `0.0149`.
  Conclusion: the adapter space can trade between heldout and all-seven, but current single-adapter searches are not
  yet Pareto-superior to the promoted baseline. Next useful direction is multi-objective/router/gated adapter mixing
  rather than another single global adapter.
- Added `sweep_checkpoint_interpolation.py` to evaluate checkpoint interpolation across strict, heldout, and weak
  seeds in one compiled benchmark. Interpolating the all-seven branch
  (`sparc_blockers_floorcalf_output_r1_0_best.pkl`) with the heldout branch
  (`sparc_heldout_floorcalf_output_r1_0_best.pkl`) did not produce a useful Pareto blend. Alpha grid:
  `0.0` all-seven `+7.552`, strict `+6.691`, heldout `+1.526`;
  `0.45` all-seven `+4.761`, strict `+2.601`, heldout `+2.255`;
  `0.75` all-seven `+4.883`, strict `+2.698`, heldout `+2.670`;
  `1.0` all-seven `+5.258`, strict `+3.191`, heldout `+3.256`.
  Intermediate parameter interpolation collapses strict/all-seven, so a real action-level gate/router is needed if
  these skills are to be combined.
- Router mixing has now beaten single-checkpoint mixing on the strict+heldout+weak nine-seed suite. The current best
  router pointer is `sparc_router_time_actionboost_margin4161c0_policy_router.json`: all-nine SPARC `+6.599`, dealt
  `1.6233`, taken `0.3834`, margin `+1.2399`, survival/safe `1.0`, peak penetration `0.0153`, valid. The older
  promoted checkpoint remains stronger on the original strict-five and heldout-two aggregate, so the router is an
  active candidate rather than a replacement for `/root/proj/out/sparc_current_best.pkl`.
- The route to improvement is no longer "find one global adapter." Single adapters, checkpoint interpolation, and
  heldout-only tuning all trade off strict/heldout/weak seeds. The useful representation is a state/action-conditioned
  router over complementary branches: strict blocker branch A, seed-81 branch B, and seed-41/61 margin branch C.
- Time features helped the router slightly by giving it more handoff/timing context, but they did not fully solve
  seed `20240641` or seed `20240661`. The time-feature router preserved the seed-81 improvement while modestly
  improving seed-41/61 margins.
- Seed `20240661` and seed `20240681` are distinguishable in router features even when their mean distance is similar:
  seed-61 is more front-facing (`front ~0.525`, rod-close `~0.303`) while seed-81 is less front-facing (`front ~0.122`,
  rod-close `~0.165`). The seed-81 boost path is already mostly off for seed-61, so seed-61 regression is likely caused
  by constant per-action branch-B blending rather than the state boost itself.
- Added `--action-state-pull` to `search_policy_router.py` so the router can pull constant per-action gates back toward
  branch A in seed-61-like states while still allowing a state boost toward branch B in seed-81-like states. A
  `sparc_router_time_actionboost_pull_margin4161c0` run is currently testing whether that extra degree of freedom
  improves the all-nine/router candidate without losing the seed-81 contact fix.
- `sparc_router_time_actionboost_pull_margin4161c0` did not improve the router. It stayed valid but scored all-nine
  SPARC `+6.492` versus the current router's `+6.599`, with almost unchanged blocker behavior: seed `20240641` margin
  `-0.3008`, seed `20240661` margin `+0.0415`, seed `20240681` margin `+0.2790`. Conclusion: simply adding a
  state-conditioned pull-back parameter is not enough in this small CEM run; the current bottleneck is probably the
  underlying branch actions available to the router or the objective's weak pressure on seed-41/61, not just a missing
  gate degree of freedom.
- Found a loader-order bug in that pull-gate experiment: with `--action-state-pull --c-state-gate`, an old no-pull
  vector of length 40 matched the generic "add C gate" compatibility branch before the pull-padding branch. That put
  the old C-gate weights into the pull slot and disabled most of the actual C gate. The branch order is now fixed in
  `search_policy_router.py`; old no-pull vectors are padded as `action gate + boost + pull-off + existing C gate`.
- Replaying `sparc_current_best_router.json` under the current script gives a lower fresh all-nine result than the
  score stored in the JSON: no-pull replay SPARC `+6.082`, dealt `1.5466`, taken `0.3925`, margin `+1.1542`, valid.
  The largest change is high-damage seed `20240611` (`24.18 -> 19.65` SPARC) plus smaller drops on seed `20240681`.
  Conclusion: router promotion needs an explicit fresh replay artifact; stored search-time scores are too sensitive to
  tiny code-path / numeric differences to be treated as final proof.
- After fixing the loader order, `sparc_router_time_actionboost_pullfix_margin4161c0` found a valid fresh-run candidate
  at SPARC `+6.265`, dealt `1.5650`, taken `0.3711`, margin `+1.1938`. It improves over the fresh no-pull replay
  (`+6.082`) but not over the old stored current-router score (`+6.599`). The gain comes from stronger high/medium
  seeds and a better seed-`20240621` minimum dealt (`0.0482 -> 0.0922`); it worsens the actual blockers
  seed `20240641` (`-0.3009 -> -0.3248` margin) and seed `20240661` (`+0.0443 -> +0.0270`). Conclusion: mean-judge
  CEM will happily improve aggregate while moving away from the weak-seed fix, so the next router search needs
  explicit worst-seed/min-judge pressure or separate local branches for seed-41/61.
