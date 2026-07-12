# Warp Ladder — GPU Validation Results (A100 80GB, 2026-07-03)

> Historical benchmark record only. For current setup and launch validation,
> use [`notes/pre-gpu-test-entrypoint.md`](pre-gpu-test-entrypoint.md).

Closes the pragmatic-ladder goal from `notes/sim-engine-secret-sauce.md` §10.
Benchmarks via `sim/robot/bench_warp_vs_mjx.py` (1000 steps, 20-step warmup
excluded, CUDA graphs on). Engines: pinned MJX (mujoco 3.9/jax 0.6.2, the
training stack) vs mujoco_warp 3.10 (warp-lang 1.14, CUDA 12.9).

## Headline matrix (nenv=4096, steps=1000, A100)

| scene | MJX env-steps/s | warp env-steps/s | speedup | note |
|---|---|---|---|---|
| single (paramquad) | 58,772 | 122,802 | **2.1×** | contact-light |
| fight (2 robots + strikers) | 7,764 | 91,922 | **11.8×** | OUR training scene |
| mesh (real leg, quartic, dt=0.004) | 44,891 | 2,144,417 | ~48× ⚠ | see caveat |

⚠ mesh/warp reads implausibly high (wall 1.9 s for 4.1M env-steps) — likely a
short-circuit or async-timing artifact for this contact-sparse scene; treat as
"very fast, unquantified" until re-measured with a longer run. The MJX mesh
number (44.9k, finite throughout) independently validates the quartic-coupling
model on GPU at dt=0.004.

The 11.8× on the fight scene is the ladder's core claim confirmed on our own
workload: MJX's trace-frozen 778-contact/3,140-efc-row allocation (for ~20 real
contacts) vs warp's compacted dynamic pool.

## warp scaling, fight scene (steps=1000)

| nenv | env-steps/s |
|---|---|
| 1,024 | 97,779 |
| 4,096 | 91,922–101,997 |
| 8,192 | 100,792 |
| 16,384 | 99,105 |

Total throughput is FLAT from 1k→16k envs: the fight scene is per-step
serial-bound (solver iteration chain), not occupancy-bound. Implication for
training: ~1–4k envs is the sweet spot; 16k envs buys nothing but memory
pressure. (MJX at 4096 = 7.8k steps/s for comparison — warp at 1024 envs
already beats MJX at any size.)

Compile/warmup, fight scene: MJX ~57 s per fresh process (XLA, every time);
warp ~68 s cold cache once, then ~1.2 s warm (persistent kernel cache) — the
report's compile-time claim, confirmed.

## Our upstream fixes, validated on CUDA (integration branch `our-fixes-integration`)

Local editable install (`.venv-warp` → ~/Projects/mujoco_warp) and pod install
both run the merged PR branches (#1487 eq-anchor + #1488 pivot floor + the
health-check strengthening d79f0b4→rewritten).

- `block_cholesky_test.py` on A100: **4/4** — including the #1415 NaN
  regression and the finite-garbage-pivot test. **Answers the open cuSolverDx
  question: on indefinite input the cuSolverDx-backed tile factorization
  produces NaNs** (the in-test unguarded assertion passed on sm_80), the health
  check triggers on hardware, and the scalar repair branch's per-lane tile
  writes are sound on CUDA.
- `io_test.py -k "eq_data or slider_crank"` on A100: **3/3** — the connect
  anchor recomputation + the slider-crank tracking repro (our leg's geometry)
  at C-parity on GPU.
- Perf regression of the health check: none measurable — fixed build measured
  82.1k (cold) and 102.0k (warm) vs 91.9k unfixed single-sample; spread is
  cache/variance, not overhead (the check is O(nv) per matrix, hoisted).

## Thin bespoke layer (warplayer/) — ALL FOUR COMPONENTS BUILT (M3 GPU ratio pending)

M1 (analytic contacts: 0.004–0.05% of MuJoCo-C, kill bar 1%) and M2 (exact
loop-coordinate joint: dt=0.004 through TDC at the physical acceleration
ceiling 1.0e4 rad/s² vs 7e9 for the constraint model): 7/7 tests.
2026-07-03 (user overrode the earlier deferral): components (iii)+(iv) built —
`warplayer/lidar.py` (144-ray × nworld, reuses mujoco_warp's ray @wp.funcs;
parity vs C rangefinder ~1e-6 m), `warplayer/obsreward.py` (obs 44/182-D +
damage + every dense fight-reward term, line-cited, numpy oracle),
`warplayer/fused.py` + `bench_m3.py` (M3 harness: baseline=wrapper-style host
round-trip vs fused=one captured graph), `m4_train_demo.py` (M4: zero-copy
rollout→update cycles). 26/26 tests; appending our kernels leaves qpos
BIT-IDENTICAL to plain mujoco_warp.

GPU VERDICT (A100, 4096 worlds, 200 ctrl steps, both modes graph-captured):
**fused_over_baseline = 1.22× with lidar (102.3k vs 84.1k env-steps/s) and
0.92× without — the ≥2× kill criterion FAILS on both configs. The layer is
KILLED by its own charter.** The win that exists is lidar dedup (evaluate 144
rays once per control step instead of every substep), worth ~22% end-to-end —
available as a cheap wrapper-side optimization without any bespoke layer. The
obs/reward device↔host seam the layer was built to kill turned out to cost
almost nothing once the physics block is graph-captured: the solver dominates,
exactly as the flat scaling curve predicted. M1/M2's findings (exact loop
joint, analytic contacts) survive independently. M4 GPU demo blocked by a
hardcoded njmax in the demo script (moot given the kill; CPU-validated 26/26).

#868 GPU verdict (same pod, contact scene, nworld=8): reuse-only +1.9% and
rank-1 +3.2% SLOWER than baseline wall-clock at mean solver_niter 1.9 — with
~2 Newton iterations/step there is no refactorization to save and the caching
is pure overhead. The CPU kernel microbench's −55% reuse win only monetizes on
high-iteration scenes; the upstream story is the honest negative: measured,
does not transfer at low solver_niter, rank-1 loses to cooperative
refactorization everywhere we measured.

## Ladder verdict (all rungs)

1. **R1 warp backend** ✅ — 11.8× on the fight scene; harness + venvs in-tree.
2. **R2 quartic couplings** ✅ — dt=0.004 restored, 0.116 mm tracking, +31 mm
   loaded stomp (better than the connect model), 18/18 tests; GPU-validated.
3. **R3 contributions** ✅ — three upstream PRs open (mujoco_warp #1487, #1488,
   mujoco #3378), CUDA-validated locally, awaiting CLA + review.
4. **Thin layer** — ALL components built + tested (M1-M4, 26/26); GPU verdict
   1.22×/0.92× vs the ≥2× bar → **killed by its own criterion, as designed**.
   Salvage: lidar dedup (~22%) portable to the wrapper; M1/M2 stand.

## Cost

Benchmark + validation session 1: ~$2.7. Session 2 (meshwalk1 training 40.5M
steps + render + M3/M4/#868 GPU verdicts): ~$2.75. Balance $24.92 → $19.70.
All pods terminated at each session end (verified 0 remaining).
