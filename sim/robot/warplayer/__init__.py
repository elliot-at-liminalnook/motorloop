# SPDX-License-Identifier: MIT
"""warplayer — the thin bespoke simulation layer over NVIDIA Warp.

Implements notes/sim-engine-secret-sauce.md §10c, milestones M1-M4:
  M1  minimal rigid-body step: analytic capsule contacts + MuJoCo-style soft
      constraint math, (nworld, ...) data layout, cross-world atomic contact
      pool — validated against the MuJoCo C reference (kill: >1% off).
  M2  the exact loop-coordinate joint for the slider-crank blade foot —
      coordinate elimination instead of an equality constraint (§8 "fundamentally
      better" note): no constraint row, no dead-center singularity, exact
      toggle-press force profile via the projected Jacobian.
  M3  mujoco_warp-as-library on the two-robot fight scene with OUR kernels
      appended to the same launch sequence: lidar (lidar.py, component iii),
      obs/reward (obsreward.py, component iv), harness + baseline-vs-fused
      benchmark (fused.py, bench_m3.py — the >=2x kill criterion).
  M4  training-loop integration demo consuming the (nworld, ...) buffers
      zero-copy (m4_train_demo.py).

Reused (the 90%): Warp language/NVRTC/LLVM + typed arrays + kernel cache +
CUDA graph capture; mujoco_warp's step/types/raycasts AS A LIBRARY (M3/M4).
Bespoke (the 10%): curated pair table, analytic capsule kernels, soft-contact
solve, the loop joint, and OUR lidar/obs/reward fused into the step's launch
sequence — the §10c(iv) win no general engine provides. Data layout follows
mujoco_warp conventions (§4): Data arrays (nworld, thing); contact pool flat
with per-contact worldid. Everything is CUDA-ready: no host branches inside
the step, fixed launch dims, atomics for compaction; CPU (LLVM) runs the same
call sequence eagerly for local tests.

M1/M2 exports below are import-light (warp only). The M3/M4 modules
(warplayer.fused, .lidar, .obsreward, .bench_m3, .m4_train_demo) additionally
require mujoco_warp + the sim/robot scene builders and are imported explicitly.
"""
from .types import Model, Data, make_capsules_scene   # noqa: F401
from .step import step                                 # noqa: F401
from . import loopjoint                                # noqa: F401
