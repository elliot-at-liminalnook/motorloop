# Sim-Engine Secret Sauce: MuJoCo, MJX, Warp, mujoco_warp, Newton, PhysX 5, Genesis

> **Document status:** Historical · **Audience:** Simulation researchers · **Last reviewed:** 2026-07-12 · **Current backend guide:** [`pre-gpu-test-entrypoint.md`](pre-gpu-test-entrypoint.md)

This is a dated engine survey and migration record, not an active runbook.

Date: 2026-07-03. Method: 7 parallel code-reading threads over freshly cloned repos in
`~/Projects/` (line refs verified with `grep -n` against the checkouts), plus web checks for
governance/controversy. Commit pins: mujoco `14c0b0c9` (2026-07-02, MJX included), mujoco_warp
v3.10.0.1 (HEAD 2026-07-03), warp (6,971 commits since 2021-03-23), newton v1.4.0.dev0,
Genesis v1.2.1 (`0053ddc`), PhysX 5.x + IsaacLab main. Written for THIS project's five pain
points; project-local citations refer to `sim/robot/`.

## 0. The five problems, mapped

> ✅ **STATUS (2026-07-03): §10 has been executed end-to-end.** Rungs 1–3 done and GPU-validated; contribution targets 1–5 ALL drafted (1–3 as Google PRs #1487/#1488/#3378, 4 as a local draft with an honest mixed finding, 5 published as newton issue #3346 + draft PR #3347); thin layer M1–M4 fully built, 26/26 tests — only the ≥2× GPU kill-criterion measurement remains in flight. Green checks throughout mark what happened; numbers live in `notes/warp-ladder-results.md`.

| Pain point (where it lives here) | Root cause | Fixed by |
|---|---|---|
| Contact cost ∝ POSSIBLE pairs, two robots (`gen_robot_mjcf.py:85-88` F-SPEED culling) | MJX-JAX static pair enumeration (§2) | mujoco_warp atomic compaction (§4) |
| `mjx.ray` no heightfield (lidar sites, `gen_robot_mjcf.py:239-256`) | `_RAY_FUNC` has 6 types, silently skips HFIELD (§2.3) | done in mujoco_warp `ray.py:452` (§4.4); MJX-JAX port = contribution #2 (§10b) |
| Slider-crank needs dt=0.002 (`gen_mesh_robot_mjcf.py:183-188`) | soft `<connect>` = explicit stiff spring across steps at a singular Jacobian (§8) | quartic joint-equality coupling (§8, zero code); TGS/XPBD/Kamino for context |
| jit compile minutes | whole-pipeline XLA trace with baked shapes (§2.5) | Warp per-module NVRTC + disk cache, ~ms warm (§3.6) |
| brax wrapper bugs | brax sits on MJX-JAX | direct mjwarp / mjlab / Newton path (§4, §5) |

Everything below is from reading the engines' code, not their marketing.

### 0.1 Vocabulary primer (so this note stays readable in a year)

- **Baumgarte stabilization**: classic fix for constraint drift — add α·violation + β·violation-rate
  to the constraint RHS. Applied at the *acceleration or velocity* level with hand-tuned gains;
  overshoots when gains are stiff relative to dt. Nobody in this survey uses raw Baumgarte;
  everything below is a descendant that fixes its dt-sensitivity differently.
- **solref/solimp (MuJoCo)**: the constraint is a *virtual spring-damper whose impedance
  d ∈ [1e-4, 0.9999] varies with violation depth*. solref = (timeconst, dampratio) sets the
  spring; solimp = (dmin, dmax, width, midpoint, power) shapes d(r). Because d < 1 strictly,
  every constraint has finite stiffness → forces are bounded and the whole problem stays a
  strictly convex QP. Cost: constraints are never exactly satisfied (drift ∝ softness).
- **PGS vs Newton (MuJoCo solvers)**: PGS = per-constraint Gauss-Seidel sweeps on the dual
  (force) problem — cheap iterations, linear convergence. Newton = second-order descent on the
  primal (acceleration) problem with exact Hessian — few, expensive iterations. MuJoCo's Newton
  wins on CPU because of sparse factorization + rank-1 updates (§1).
- **TGS (Temporal Gauss-Seidel, PhysX)**: splits dt into posIters sub-steps *inside* the solver;
  every iteration integrates poses a little and re-measures constraint error, and springs are
  integrated backward-Euler per sub-step. Effect: stiffness that would explode explicit
  integration is unconditionally stable, and bias gains act against dt/posIters, not dt.
- **XPBD (extended position-based dynamics, Newton/others)**: constraints solved directly at the
  *position* level each substep with compliance α (inverse stiffness) and accumulated Lagrange
  multipliers. Stiffness-unconditionally-stable; physics accuracy is bought with substep count
  ("small steps" regime). Momentum conservation is approximate under heavy contact weighting.
- **NCP (nonlinear complementarity problem)**: the "exact" formulation of rigid contact
  (complementarity between gap and force). Hard to solve at scale; Disney's Kamino (§5) attacks
  it with proximal ADMM specifically because soft/relaxed formulations blur mechanism physics.
- **Featherstone / reduced coordinates**: parameterize by joint angles (nv DOFs), so joints are
  exact by construction — but the structure must be a *tree*; closed loops need either a cut
  joint + constraint (all engines here) or coordinate elimination (nobody, automatically; §8).
- **Dense vs sparse efc**: MuJoCo-family constraint solvers materialize a constraint Jacobian
  `efc_J` (rows = constraint equations). Dense (nefc × nv) is GPU-friendly but memory/FLOPs
  scale with *allocated* rows — which is why MJX's static worst-case allocation (§2) hurts and
  mujoco_warp's compaction (§4) doesn't.

## 1. MuJoCo core: what makes it fast on CPU, and why that resists GPUs

**Pipeline.** `mj_fwdPosition` (`src/engine/engine_forward.c:131`): kinematics → `mj_makeM`/
`mj_factorM` (143-144) → `mj_collision` (147) → `mj_makeConstraint` (159) →
`mj_projectConstraint` (164). The contact set and every constraint array are rebuilt from
scratch each step.

**Collision is dynamic, not static.** Broadphase is sweep-and-prune along a
covariance-adapted axis: `mj_broadphase` (`engine_collision_driver.c:1533`) eigen-decomposes
the geom-center covariance (`mju_eig3`, :1612) to pick the sweep direction, then `mj_SAP`
(:1400) sorts endpoints and prunes. contype/conaffinity is applied body-level inside SAP
(`add_pair`, :1315-1353); welded/parent-child pairs are dropped (`filterBodyPair`, :288).
Narrowphase dispatches through `mjCOLLISIONFUNC[type][type]` (:44-56) — analytic primitive
colliders (`engine_collision_primitive.c`: plane-sphere :53, capsule-capsule :521; dedicated
SAT box-box in `engine_collision_box.c:1351`) or the generic convex path, which since native
CCD became default is an in-house **GJK + EPA** (`engine_collision_gjk.c`: `gjk` :200, `epa`
:2402) with polygon-clipping multicontact (`multicontact`, :2122). Narrowphase is
multithreaded in chunks (`mj_narrowphase`, driver.c:1887-1941).

**Soft constraints (solref/solimp).** Per-row, `mj_makeImpedance`
(`engine_core_constraint.c:2152`) computes K = 1/(dmax²·tc²·dr²) (:2188), B = 2/(dmax·tc)
(:2197), and regularizer R = (1−d)/d · diagA (:2171) where impedance d(r) is a sigmoid of
position (`getimpedance`, :2100-2146) hard-clamped to [1e-4, 0.9999] (:2044-2047;
`mjMINIMP/mjMAXIMP` in `include/mujoco/mjmodel.h:27-28`). diagA ≈ diag(J M⁻¹ Jᵀ) comes from
**reference-configuration** inverse weights `body_invweight0`/`dof_invweight0`
(`mj_diagApprox`, :1720-1746) — precomputed at qpos0, not the current pose (matters in §8).
The reference acceleration is a spring-damper: aref = −B·vel − K·d(r)·(pos−margin)
(`mj_referenceConstraint`, :3130-3141). "refsafe" clamps solref[0] ≥ 2·dt (:2028-2030).

**The solver is a convex QP (Todorov 2014), and Newton is why it's fast.** Minimize over qacc:
½‖qacc − qacc_smooth‖²_M + s(J·qacc − aref), s convex (`mj_constraintUpdate_impl`,
:3155-3226). Default `mjSOL_NEWTON` (`engine_init.c:94`): builds H = M + JᵀDJ (`MakeHessian`,
`engine_solver.c:2010`), sparse Cholesky (:2135), then **rank-1 up/downdates instead of
refactorizing** as the active set changes (`HessianIncremental`, :2238-2276) plus exact
piecewise-quadratic linesearch (`PrimalSearch`, :1812). Because R > 0 always, the problem is
strictly convex — a handful of iterations converge. Equality constraints (connect: 3 rows via
`mj_jacDifPair`, weld: +3 quaternion rows; `mj_instantiateEquality`, :596-722) go through the
same machinery and are **always soft**: A + R ≻ 0 even when the mechanism Jacobian loses
rank, so per-step forces stay finite by construction (:3178-3182).

**implicitfast.** `mj_implicitSkip` (`engine_forward.c:1766`) solves
(M − h·∂f/∂v)·qacc = qfrc_smooth + qfrc_constraint. implicitfast calls
`mjd_smooth_vel(flg_bias=0)` (:1806) — actuator/joint/tendon damping and fluid drag
derivatives (`engine_derivative.c:1213,1880`), **skipping the RNE/Coriolis derivative** so the
matrix stays symmetric and factors with the cheap sparse LDL (:1809-1822). Damping becomes
unconditionally stable; note `qfrc_constraint` sits on the RHS — **constraint forces are
explicit across steps** (this is the toggle problem, §8).

**Why this design resists GPU-ification** (structural, from the code):
1. Dynamic shapes everywhere — ncon/nefc/nJ/nH re-counted and arena-allocated per step
   (`mj_makeConstraint`, constraint.c:2814-2838; mid-solve allocs, solver.c:2030-2072).
2. Branchy heterogeneous narrowphase behind a function-pointer table with data-dependent
   iteration counts (GJK/EPA polytopes) — hostile to SIMT warps.
3. Tree-ordered sparse algebra with loop-carried dependencies: CRB walks `dof_parentid`
   chains (`engine_core_smooth.c:1862`), `mj_factorI` is a fill-free reverse LDL over the
   tree (:1973-1997).
4. Stateful active-set logic: the rank-1 Cholesky-update trick is inherently sequential.
CPU-side it's the opposite: arena allocation, AVX kernels (`engine_util_sparse_avx.h`),
thread-chunked narrowphase and per-island solves (forward.c:910-935).

## 2. MJX: MuJoCo on JAX, and where the quadratic two-robot cost comes from

> ✅ **Measured on our fight scene (A100, 4096 envs):** the predicted static allocation is real — 778 contact slots / 3,140 efc rows for ~20 actual contacts — and costs 11.8× vs the warp path (7,764 vs 91,922 env-steps/s). The silent HFIELD ray miss is now fixed in [mujoco PR #3378](https://github.com/google-deepmind/mujoco/pull/3378) (with the unsupported-geom warning).

MJX (`mjx/mujoco/mjx/_src/`) is now a **three-backend front end** — `Impl.CPP / JAX / WARP`
(`types.py:29-35`) — with mujoco_warp vendored at `mjx/mujoco/mjx/third_party/mujoco_warp/`.
Everything below describes the JAX backend you're on via brax.

**2.1 Static contact allocation.** `geom_pairs()` (`collision_driver.py:128-196`) runs at
*trace time*: nested loops over bodies (:155,161) × `itertools.product` over their geoms
(:178), filtered only statically (explicit pairs :141-147, excludes :164-166, weld/parent
:168-174, contype/conaffinity :188-192). No positional filtering exists — the surviving list
is frozen into the compiled program. `_geom_groups()` (:199-250) buckets by function-key "to
guarantee static shapes for contacts and jacobians" (docstring :204-210); `make_condim()`
(:347-403) computes the static contact count from per-collider ncon (capsule-capsule 1,
plane-capsule 2, convex-convex 4). `_make_data_jax()` (`io.py:653-713`) bakes `ncon`, `nefc`,
and `efc_J: (nefc, nv)` into array shapes (:661,693). Every step, the collider **vmaps over
the whole pair list** (`collision_driver.py:406-461`, wrapper vmap at
`collision_convex.py:74`); non-touching pairs still pay full narrowphase, then get zeroed
downstream (`constraint.py:570,589`).

**2.2 Your numbers.** Two robots × ~17-21 geoms (`gen_robot_mjcf.py`: torso box + 4×
hip/thigh/calf capsules + foot spheres + optional rod/spear): cross-robot 17×17…21×21 =
**289-441 pairs** (different trees — no weld/parent filters apply), + 2×~20 floor-plane pairs.
All capsules → ~330-480 contact slots → at condim=3 pyramidal, `(condim−1)·2 = 4` efc rows
each (`constraint.py:660`) → **~1.3k-1.9k contact rows in a dense (nefc, nv) Jacobian**, per
step, per env, ×8192 envs — regardless of whether the robots are touching. Your F-SPEED
contype/conaffinity culling (`gen_robot_mjcf.py:368-394`) is exactly the right (only) lever on
this backend. The opt-in `max_geom_pairs`/`max_contact_points` numerics
(`collision_driver.py:414-456`, `lax.top_k` on sphere-bound distances) cut narrowphase but are
"experimental" (`doc/mjx.rst:584-591`) and skip PLANE/HFIELD groups (:120).

**2.3 Ray.** `_RAY_FUNC` (`ray.py:223-230`) = {PLANE, SPHERE, CAPSULE, ELLIPSOID, BOX, MESH}.
**No HFIELD, and no error** — the dispatch loop (:275) simply never tests unsupported geoms,
so a lidar ray at a heightfield returns −1 (miss) silently. Mesh rays are brute-force
vmap-over-all-triangles + argmin (:186-220), no BVH. hfield *collision* works
(`collision_convex.py:1048-1120`); only hfield *ray* is missing. Docs table: MJX-JAX ray
"Slow for meshes, hfield … unimplemented" vs MJX-Warp "All, BVH" (`doc/mjx.rst:510-513`).

**2.4 Architecture.** Pure-JAX pipeline (`forward.py:432-476`); levelwise `scan.body_tree`
kinematics (`scan.py:337-382`); dense mass matrix + `cho_factor` below nv=60
(`support.py:36-56`, `smooth.py:319-322` — your two-robot nv≈36-70 straddles this). Newton
solver builds a dense H and re-factors **every iteration** (`solver.py:396-412`) — no
incremental-update trick (that's mjx-side cost vs the C engine). `iterations=1` bypasses the
while_loop (:599-602); linesearch always pays `ls_iterations` via scan+cond (:239-253).

**2.5 Compile time.** Structural: the whole collision table, 9 concatenated efc builders
(`constraint.py:699-715`), solver, and integrator trace into one XLA graph with every shape a
compile-time constant; convex colliders specialize **per distinct mesh**
(`collision_driver.py:206-208`); numpy fields on Model are structural — touching them
recompiles (`doc/mjx.rst:360-363`). No shape polymorphism, no `donate_argnums` in MJX itself.
This is the price of vmap-everything; it cannot be patched away, only worked around.

**2.6 Dropped features** (trace-time `NotImplementedError`, `io.py:315-387`): flex, most
contact sensors, SDF pairs, several geom-type pairs, PGS/noslip, margin/gap on mesh/hfield.

**2.7 Tuning recipe for THIS env while still on MJX-JAX** (each item verified against the
code above; expected gains are educated estimates, not measurements):
- Keep/extend the F-SPEED contype/conaffinity mask — it is the only lever that shrinks the
  *trace-time* pair list, which is where the quadratic cost is baked (§2.1). Every masked pair
  removes narrowphase FLOPs *and* 1-4 dense efc rows.
- Add `<numeric name="max_geom_pairs" data="N"/>` + `max_contact_points` to the MJCF
  (`collision_driver.py:414-456`): caps narrowphase + contact slots by top-k sphere-bound
  distance. Experimental, skips plane groups — your floor pairs stay, which is fine (they're
  the support contacts).
- Solver: Newton, `iterations=1`, `ls_iterations=4-6`, dense Jacobian on A100
  (`doc/mjx.rst:553-582` makes exactly this recommendation for RL).
- Do NOT expect `eq_active=False` to save time — inactive equality rows are still allocated
  and multiplied, just zeroed (`constraint.py:138,157`).
- Compile time: nothing inside MJX helps; cache at the brax level (persistent compilation
  cache `jax.config jax_compilation_cache_dir`), and avoid touching any numpy field of
  `mjx.Model` between runs (each change = full retrace, `doc/mjx.rst:360-363`).
- The escape hatch shipped in this very repo: `mjx.put_model(m, impl='warp')` +
  `make_data(..., naconmax=, njmax=)` swaps the backend under the same API (`doc/mjx.rst:112-165`,
  `io.py:64-75` — JAX still default, env var `MJX_GPU_DEFAULT_WARP` flips it). Caveats: no
  autodiff through physics (fine for PPO), and the JAX interop path loses CUDA-graph
  conditionals, falling back to fixed-iteration solves (`mujoco_warp _src/solver.py:3499-3504`).

## 3. Warp: the "strictly typed language → simplest GPU ops" compiler already exists

> ✅ **Adopted:** `.venv-warp` (warp-lang 1.14) runs our benchmark harness and the `sim/robot/warplayer/` thin layer; the persistent-kernel-cache claim confirmed on the pod (~68 s cold once, ~1.2 s warm, vs MJX's ~57 s every fresh process).

This is precisely the user-vision artifact, 5 years mature, Apache-2.0, DCO-only.

**Codegen pipeline** (`warp/_src/codegen.py`): `@wp.kernel` (`context.py:1563`) inspects
source and `ast.parse`s it (codegen.py:1610). **Strict typing is enforced, not optional**:
missing arg annotations → `WarpCodegenError` (:1644-1645); variables cannot change type
(:4311,4400); only 24 whitelisted AST node types compile (`node_visitors`, :5176-5201) — any
other construct (lambda, comprehension, try, class) is a hard error (:5203-5212). The typed IR
(`Var`, :862) is emitted as C++/CUDA via literal f-string templates —
`cuda_kernel_template_forward` (:5973-5981) is the canonical
`extern "C" __global__ … _idx = blockIdx*blockDim+threadIdx; if (_idx >= dim.size) return;`.
CUDA compiles through **NVRTC** (`build.py:80` → `native/warp.cu:4767,4785`), CPU through an
embedded **Clang/LLVM JIT** (`native/clang/clang.cpp:7,135`). Modules are SHA-256
content-hashed and disk-cached (`context.py:2416,2605-2640`; cache skip at :3715-3721).

**Runtime primitives you'd otherwise rebuild:** typed device arrays (`types.py:3147`);
`wp.launch` with explicit forward/adjoint kernels (`context.py:10114-10155`); **CUDA graph
capture** (`utils.py:1829` → `cudaStreamBeginCapture`, warp.cu:3459) including **conditional
graph nodes** — data-dependent if/while *inside* a replayed graph (`wp.capture_if/while`,
context.py:11811,12046; warp.cu:3918-3938); tile ops with shared-memory GEMM / cuBLASDx LTO
(`native/tile_matmul.h`); source-transformed reverse-mode autodiff for every kernel
(`codegen_func_reverse`, codegen.py:6411; `wp.Tape`, tape.py:14-158); and ~70k lines of native
support — radix sort, hash grids, **BVH ray casting** (`wp.mesh_query_ray` →
`native/mesh.h:1764`, vendored cuBQL GPU BVH builder), NanoVDB volumes.

**Compile times:** ~168 ms cold per small module, ~5 ms warm from cache
(`docs/deep_dive/codegen.rst:25-40`); AOT prebaking exists (`wp.compile_aot_module`,
:1445-1459). Per-module C++ translation units, not whole-program XLA — this is the structural
reason mujoco_warp start-up is seconds while MJX is minutes.

**Effort so far:** first commit 2021-03-23, 6,971 commits, 100 contributors, ~8 sustained core
devs (Macklin 1,738 commits, Shi 2,133, …) — order **20-30 person-years**, and the hard 20%
(adjoint codegen, graphs, BVH, tiles, caching) is exactly what a bespoke build would need.

## 4. mujoco_warp: THE study — MuJoCo semantics as explicit Warp kernels

> ✅ **Validated + contributed to:** parity on our scenes at 1e-4–1e-6 (one 3.5e-2 divergence root-caused to OUR degenerate rod geometry, fixed in-tree); two upstream fixes drafted, CUDA-validated, and PR'd — [#1487](https://github.com/google-deepmind/mujoco_warp/pull/1487) (set_const never recomputed eq_data; slider-crank repro from our own leg) and [#1488](https://github.com/google-deepmind/mujoco_warp/pull/1488) (float32 Cholesky pivot floor + our floor-then-divide overflow finding). Local `our-fixes-integration` branch is editable-installed until upstream merges.

**Architecture.** `Data` = dataclass of `wp.array`s with explicit `nworld` leading dim
(`_src/types.py:2033,2196`); `Model` fields carry a *broadcastable* leading dim
(`body_mass: ("*","nbody")`, :1517) indexed `worldid % shape[0]` (`io.py:2580`) — per-world
domain randomization of 133 fields with **no recompile**. 308 `@wp.kernel`s implement the
pipeline explicitly (`forward.py:1315,1351`); the intended usage wraps the whole step in one
CUDA graph (`testspeed.py:281-286`) with conditional nodes on by default (`io.py:421`).

**Collision — the fix for problem #1.** Broadphase: NXN or SAP (tile-sort
`collision_driver.py:557` / segmented `:633`, sweep `:567`) with plane/sphere/AABB/OBB
prefilters (:99-282). Surviving pairs are **compacted with atomics** —
`pairid = wp.atomic_add(ncollision_out, 0, 1)` (:356-371); narrowphase launches fixed-dim
(graph-safe) but threads early-exit past the live count (`collision_primitive.py:1363-1365`).
Contacts go into a **single pool shared across all worlds** — `(naconmax,)` arrays with a
per-contact `worldid` (`types.py:1938-1975`), `wp.atomic_add(nacon_out,…)`
(`collision_core.py:268`), overflow flagged per world (`types.py:2181`), `naconmax =
nconmax × nworld` by default (`io.py:1385-1390`). **Runtime cost tracks actual
broadphase-surviving pairs**; clinching worlds borrow pool capacity from calm ones. Full GJK
(`collision_gjk.py:596`) + EPA (:2330) + manifold generation (:1993) as device functions.
Heightfield collision: HFIELD×{sphere,capsule,ellipsoid,cylinder,box,mesh}
(`collision_driver.py:54-59`; `collision_convex.py:164,382-432`).

**Solver.** CG + Newton (`solver.py:2809-2828`; no PGS/noslip). Newton: dense tiled Hessian +
blocked Cholesky (`block_cholesky.py`) for nv ≤ 60, sparse path auto above nv > 32
(`io.py:160-167`). **Per-world early exit inside the captured graph**: `_solve_done` flags
`ctx.done[worldid]` and decrements a global `nsolving` (`solver.py:3268-3300`); iteration
wrapped in `wp.capture_while(nsolving, …)` (:3498); converged worlds' kernels return
immediately (:845). Linesearch = one tile per world with `wp.tile_reduce` over efc rows
(:993-995). Islands + sleeping with DOF-compacted solves exist (`island.py`,
`solver.py:3715`).

**Ray + hfield ray: already done — not a contribution target.** `ray_geom` dispatch covers
plane/sphere/capsule/ellipsoid/cylinder/box (`ray.py:809-826`), mesh with **BVH**
(`ray_mesh_with_bvh`, :701, via `wp.mesh_query_ray`), flex (:763), and **`ray_hfield`**
(:452-620). Batched `rays()` API (nworld, nray) (:1224); rangefinder sensor implemented
(`sensor.py:179,568,817-827`). GitHub issues #94/#335/#943 (ray, hfield ray) all closed. The
gap survives only on the MJX-JAX side (§2.3).

**Equality constraints.** Full trio + tendon/flex: `_equality_connect` (`constraint.py:129`),
`_equality_joint` (:473), `_equality_weld` (:939); KBIP math identical to C (:85-107);
per-world `eq_solref` (:435) and runtime `eq_active` (`types.py:2204`). **Caveat: open issue
#1270** — "Connect constraint anchor computation does not account for joint reference
positions" (open since 2026-03-30). Test the slider-crank leg here before trusting it.

**Performance claims & how to measure.** The README publishes no headline multiplier (a
deliberate contrast with Genesis, §7); instead there's a nightly public benchmark dashboard
(google-deepmind.github.io/mujoco_warp/nightly — ns/step, memory, and **JIT time as a
first-class tracked metric**, RTX 6000 Ada) and a benchmark suite spanning humanoid,
unitree_g1 flat + hfield, aloha clutter/SDF/cloth, franka, myosim (`benchmarks/run.py`,
`sweep.py`). For your own numbers: `mjwarp-testspeed --event_trace` gives per-pipeline-stage
GPU-event timing (`_src/warp_util.py:100-117`) — run it on the two-robot MJCF before/after
any pair-culling change; that replaces guesswork about where step time goes.

**Consumption paths** (decision you'll actually face): (a) **via MJX** `impl='warp'` — keeps
brax/JAX training loop, loses graph-conditional early exit (§2.7); (b) **direct** — PyTorch
ecosystem via mjlab or Isaac Lab-Newton (README:111-114), full CUDA-graph benefits, but your
brax PPO stack would need replacing; (c) **via Newton SolverMuJoCo** (§5) — adds sensors,
importers, multi-solver A/B at the cost of a young API. All three share the same kernels.

**Maturity/effort.** First commit 2025-03-12; 2,003 commits/16 months (peak ~350/mo, steady
40-100/mo now); committers 1,198 google.com + 261 nvidia.com (Howell 1,025, Bayes 147,
Quaglino 119, Frey 119); package author "Newton Developers <mujoco@deepmind.com>"; versioned
in lockstep with MuJoCo (3.10.0.1); README: feature parity except PGS/noslip,
implicitfast-midpoint, plugins, flex-experimental; **no differentiability yet** (issue #500).
Estimate **10-15 person-years** to date at 5-8 sustained FTEs — for a *reimplementation* with
the semantics already specified by the C engine and its conformance tests. That is the honest
baseline for "build a GPU sim." Contribution posture is unusually welcoming: `uv sync
--all-extras`, pre-commit with a custom kernel_analyzer that runs on PRs, and an `AGENTS.md`
documenting AI-assisted-PR etiquette — they expect and merge external PRs.

## 5. Newton: the bespoke open engine is already being built

Newton (LF project, "initiated by Disney Research, Google DeepMind, and NVIDIA",
`README.md:13-15`) is a Warp-native, multi-solver engine: `Model`/`State`/`Control` +
`ModelBuilder` (`newton/_src/sim/model.py:37`, `builder.py:80` — 11k lines, holds *both*
maximal-coordinate `body_q` and generalized `joint_q` so both solver families share one
model), `SolverBase` plugin contract — "derive and override `step` and
`notify_model_changed`" (`_src/solvers/solver.py:177-184`). Eight solvers, capability matrix
maintained in `newton/solvers.py:37-115`:

| Solver | What it is | Status / relevance here |
|---|---|---|
| SolverMuJoCo (`solvers/mujoco/solver_mujoco.py:344`) | full conversion bridge to mujoco_warp: builds `MjSpec` programmatically (:4665, defaults NEWTON+implicitfast :4657-4660), `mujoco_warp.put_model` :6253, step :3491; can even inject Newton's own contacts (`use_mujoco_contacts=False`) | "primary backend"; the drop-in path for this project |
| SolverFeatherstone (`solvers/featherstone/…:54`) | reduced-coordinate CRBA, symplectic Euler | **silently ignores loop joints** — unusable for the mesh leg |
| SolverXPBD (`solvers/xpbd/solver_xpbd.py:33`) | Macklin/Müller XPBD, maximal coords | native loop joints w/ full drive/limit semantics (§8) |
| SolverVBD/AVBD (`solvers/vbd/…:76-92`) | vertex block descent; AVBD for rigids (ramped penalty + augmented-Lagrangian duals, γ=0.999 :207-228) | experimental |
| SolverSemiImplicit (`solvers/semi_implicit/…:30`) | legacy warp.sim symplectic Euler | loop joints native, low accuracy |
| SolverStyle3D | projective-dynamics cloth (Linctex contribution) | n/a |
| SolverImplicitMPM (`…:614-625`) | implicit MPM, granular | n/a |
| SolverKamino (`solvers/kamino/solver_kamino.py:52-71`) | Disney's proximal-ADMM **NCP** solver, built for **kinematic loops + hard contacts** (arXiv:2504.19771) | Beta — the most principled dead-center answer in this survey |

MJCF import fidelity is unusual: hfields, tendons, actuators, and `<equality>` either passed
through to SolverMuJoCo or **converted to native loop joints/mimic constraints** for other
solvers (`_src/utils/import_mjcf.py:162,299-302,2559-2564`). Loop closure is documented
per-solver (`docs/concepts/articulations.rst:750-853`): XPBD/SemiImplicit solve loop joints
natively with full drive/limit semantics; SolverMuJoCo synthesizes an equality;
**SolverFeatherstone silently ignores the loop** (trap). Sensors: contact, IMU, tiled-camera
raytracer (`_src/sensors/sensor_tiled_camera.py:20-32`), per-primitive analytic raycasts
(`_src/geometry/raycast.py:80-847`, `wp.mesh_query_ray` at :655) — **no turnkey lidar sensor**
(gap = contribution #5). Heightfield is a first-class geom (`_src/geometry/types.py:83`;
`ray_intersect_heightfield_local`, `_src/utils/heightfield.py:428`).

Maturity: v1.0.0 2026-03-10 → v1.3.0 2026-06-11, 2,119 commits since 2025-03-26, formal
deprecation policy (`docs/guide/compatibility.rst:131-158`). Committers: nvidia.com 1,083,
disney.com 15, zero @google.com (DeepMind's contribution flows via mujoco_warp). Isaac Lab
3.0 Beta ships the Newton backend (develop branch; MuJoCo-Warp solver only, flat-terrain
examples, PhysX↔Newton sim-to-sim policy transfer validated). Apache-2.0 + **CLA**
(`CONTRIBUTING.md:44-117`). Verdict: yes — this is the "bespoke engine, in the open," with
NVIDIA-centric committer concentration as the main governance caveat.

## 6. PhysX 5 (the Isaac secret sauce): temporal everything

**TGS.** Each "position iteration" is a real sub-timestep: `mStepDt = dt / posIters`
(`physx/source/lowleveldynamics/src/DyTGSDynamics.cpp:1898`). The island loop
(`iterativeSolveIsland`, :2515-2645) solves all constraints once, then **integrates bodies by
stepDt** (:2642) and accumulates `deltaLinDt/deltaAngDt` (:1470-1471); the contact kernel
re-measures separation from those accumulated deltas every iteration (`solveContact`,
`DyTGSContactPrep.cpp:1510-1557`), and joints re-project anchors through the accumulated
delta-rotation (`solve1DStep`, :2470-2514). Contrast PGS: bias baked once at prep
(`DyContactPrep.cpp:687-699`). Springs/drives are **backward-Euler implicit per substep**:
`a = stepDt(stepDt·k + c); x = 1/(1 + a·unitResponse)`
(`shared/DyCpuGpu1dConstraint.h:418-457`) — unconditionally stable for any stiffness. That
combination (small effective dt per iteration + implicit springs + re-measured error) is why
Isaac trains with stiffness 1e4 drives at dt=1/60.

**GPU pipeline: dynamic discovery, not static allocation.** Incremental 3-axis GPU SAP
emitting found/lost pair deltas (`gpubroadphase/src/CUDA/broadphase.cu:1504,1662-1669`), with
**env-ID filtering inside the broadphase kernel** (:62-81) — the hardware version of Isaac
Lab's per-env isolation. **Aggregates** = one broadphase entry per robot with a local mini-SAP
for self-pairs (`PxAggregate.h:74-92`; `aggregate.cu:557,928`) — why 30-link self-colliding
robots are cheap. **Persistent contact manifolds** cache ≤4 points in body-local space and
refresh by re-projection, regenerating only on drift/rotation thresholds
(`geomutils/src/pcm/GuPersistentContactManifold.h:44,723-754`). Actual contacts bump-allocate
into capacity-limited device streams (`atomicAdd` in `convexMeshOutput.cu:195-197`) that
**warn-and-drop on overflow** (`PxgNarrowphaseCore.cpp:1634`) — knobs =
`PxGpuDynamicsMemoryConfig` (`PxSceneDesc.h:457-486`). Articulations are reduced-coordinate
ABA trees (`DyFeatherstoneForwardDynamic.cpp:811,1014`) with drives/limits solved as implicit
internal constraints inside TGS substeps (`DyFeatherstoneArticulation.cpp:2610,3032`); loops
close via maximal-coordinate joints between links, or gear-ratio **mimic joints**
(`PxArticulationReducedCoordinate.h:1448`) — linear coupling only. GPU solver kernels are
open in-tree now (`gpusolver/src/CUDA/solverMultiBlockTGS.cu`).

**Isaac Lab on top:** manager-based MDP decomposition (`isaaclab/envs/manager_based_rl_env.py:25`,
`managers/*`), GridCloner with `enable_env_ids` collision filtering
(`scene/interactive_scene.py:244-246,303-308`), **RayCaster = Warp `mesh_query_ray`, static
single mesh only** (`sensors/ray_caster/ray_caster.py:40,51-53,164-167`;
`utils/warp/kernels.py:70`) — fine for terrain, useless for sensing the opponent robot —
PhysX contact views (`contact_sensor.py:286-385`), tiled zero-copy cameras. All BSD-3 Python
riding the **proprietary Isaac Sim binary** (`app/app_launcher.py:25-27`, README:113-121).

## 7. Genesis: honest assessment

Under the hood: a **Taichi program whose authors now maintain the compiler fork** — zero
`import taichi`; 1,032 `@qd.kernel/@qd.func` against `quadrants==1.0.2` (`pyproject.toml:13`),
the June-2025 gstaichi fork rebranded (README:238). The rigid solver is an **openly
acknowledged MuJoCo-lineage reimplementation**: solref/solimp 7-tuples
(`engine/solvers/rigid/rigid_solver.py:195-200`), CG/Newton + islands
(`options/solvers.py:442-533`), MPR/GJK narrowphase with EPA ported line-for-line from
MuJoCo (`collider/epa.py:53-54` cites `engine_collision_gjk.c#L1331`), an
`enable_mujoco_compatibility` flag, and step-for-step MuJoCo consistency tests
(`tests/test_rigid_physics.py:11-34`). Equality trio (connect/weld/joint) present
(`constraint/solver.py:938,1051,1395`). Real lidar (LBVH raycaster sensor,
`engine/sensors/raycaster.py:254`; `gs.sensors.Lidar` alias) and Isaac-Gym-derived terrain
(`genesis/ext/isaacgym/terrain_utils.py`).

The Dec-2024 claim ("430,000× realtime", "10-80× faster than Isaac/MJX") was real batched
throughput for a contact-light, mostly-static Franka scene (30k envs, 1 substep,
self-collision off); Stone Tao's corrected settings gave ~150× lower on the same 4090 and
3-10× *slower* than ManiSkill on contact-rich tasks
(stoneztao.substack.com/p/the-new-hyped-genesis-simulator-is, issue #181; Tassa:
"disingenuous", google-deepmind/mujoco discussion #2303). Authors published corrected
open-script benchmarks (Jan 2025) and deleted all speed claims from the README (commit
`ca92145`, 2026-05-27). Today (v1.2.1, 1,513 commits, professionalized by Genesis AI —
Duburcq 46% of commits): plausible for rigid-body RL, but its differentiator is
multi-physics/differentiable breadth, not rigid-body speed. Apache-2.0, **no CLA/DCO**. Risk:
a small team owning a whole compiler stack (Quadrants) under one product.

## 8. The toggle problem: the slider-crank dead center, engine by engine

> ✅ **Executed — see OUTCOME block below.** dt=0.004 restored (0.116 mm tracking, +31 mm loaded stomp, 18/18 tests); plus the thin layer's M2 proved the coordinate-elimination endgame: the exact loop joint runs TDC at the physical acceleration ceiling (1.0e4 rad/s² vs the connect model's 7e9 — the singularity simply doesn't exist in reduced coordinates).

The mechanism (`sim/robot/gen_mesh_robot_mjcf.py`): per leg, a knee crank (r=75 mm) + conrod
closes onto a pushrod slide via `<connect>` (:156-157); at TDC (φ=0) the loop Jacobian is
singular (:120-123); bodies are 50-80 g. Measured: dt=0.004 → 10 m slide error, |qacc| 7e9;
dt=0.002 → <1 mm (:183-188).

**Why MuJoCo blows up at dt=0.004 despite bounded forces.** Per-step, the convex solve
guarantees finite force (A + R ≻ 0, §1). But `qfrc_constraint` is applied **explicitly across
steps** (RHS of the implicitfast solve, `engine_forward.c:1766`), so the soft connect is a
stiff discrete-time oscillator: K = 1/(dmax²·tc²·dr²) with tc=0.02 nominally satisfies
refsafe at dt=0.004 (0.02 ≥ 2·0.004), but the *effective* frequency is √(K·d/m_eff) along the
constraint direction, and near TDC m_eff collapses (tiny bodies, near-singular J) while the
force direction flips sign as the piston reverses. Worse, R is scaled by `diagApprox` from
**qpos0 inverse weights** (`engine_core_constraint.c:1746`) — at the dead center the true
J M⁻¹ Jᵀ departs sharply from that reference estimate, so the regularization no longer
matches the configuration (inference flagged: the code facts are verified; the instability
mechanism is analysis). Halving dt restores the sampling of the flip — that's why 0.002 works.

**The cheap, exact-ish fix inside MuJoCo semantics (recommended first).** Replace the 3-row
`<connect>` with **1-row polynomial joint-equality couplings**: MuJoCo's `mjEQ_JOINT` couples
two joints via a quartic, q1 − ref1 = Σ c_k (q2−ref2)^k (`engine_core_constraint.c:727-751`;
MJX `constraint.py:248-265`; mujoco_warp `constraint.py:473`; Genesis
`constraint/solver.py:1051`). Concretely:
1. Fit c_0..c_4 of `<joint joint1="X_pushrod_slide" joint2="X_knee_blade"
   polycoef="c0 c1 c2 c3 c4"/>` against the closed form `slider_crank_s(phi)`
   (`gen_mesh_robot_mjcf.py:55-59`) over the working knee ROM (numpy `polyfit`, weight the
   TDC neighborhood); same for the toe: ψ(φ) via `loop_consistent_pose` (:63-70).
2. Delete the `<connect>` (:156-157) and the pushrod `<exclude>`s can stay.
3. The efc row is q_slide − poly(q_knee) = 0 with Jacobian [1, −poly′(φ)] — **full rank at
   TDC by construction** (the slide coefficient is the constant 1; poly′→0 is benign and
   correctly reproduces toggle-press force amplification ∝ 1/poly′; crank torque → slide
   force ratio diverges exactly as the real mechanism's mechanical advantage does).
4. Validation gates, in order: (a) quartic fit residual < 0.5 mm over ROM (else restrict ROM
   or keep connect + dt=0.002); (b) `test_mesh_robot_contract.py:115` loop-consistency check
   at dt=0.004; (c) the zero-g whip test that previously flipped the elbow branch
   (`gen_mesh_robot_mjcf.py:124-128` — the joint stops must still block the +200 mm branch,
   which the polynomial, being single-valued, can no longer reach at all — a side benefit);
   (d) toggle-press force profile vs the 1.2 kN connect response noted at :67.
This drops nefc by 1 row/leg (2 coupling rows replace 3 connect rows), removes the singular
direction entirely, and plausibly re-enables dt=0.004 — worth an afternoon before any
engine migration. If the quartic can't hold the tolerance, the same trick works piecewise:
MuJoCo also allows coupling via a **tendon** equality (`mjEQ_TENDON`,
`engine_core_constraint.c:728`) for more general shaping.

**OUTCOME (2026-07-03, implemented — `gen_mesh_robot_mjcf.py`, 18/18 tests at dt=0.004).**
The quartic couplings work as predicted: fit residual 0.116 mm / 0.031° over the ROM (c1(toe)
= −0.24997 ≈ R/L−1 and c2(slide) = −0.009373 match the analytic Taylor terms); an 8 s zero-g
TDC rest hold is bit-still; the full-ROM sweep tracks the closed form to sub-mm; the loaded
stomp improved from +12 mm to ~+30 mm of lift. Two additions the plan missed:
1. **The explicit test-servo damping was a co-culprit all along.** With the couplings, the
   knee's smooth reflected inertia is only ~2.6e-4 kg·m², so the tests' hand-rolled
   `-kd·qvel` (kd=0.3 via qfrc_applied / 0.2 via ctrl) has kd·dt/I ≈ 4.6 at dt=0.004 —
   past the discrete stability limit of 2 *regardless of loop formulation*. The connect
   masked it (exactly-zero residual at rest = no noise seed, plus reflected anchor inertia).
   Fix: P-only servos through the actuator path; damping belongs in `dof_damping`, which
   implicitfast integrates implicitly (§1) — this note's own lesson, applied to the harness.
2. **Kinematic couplings must also CARRY the load.** Soft rows (solref 0.02–0.04, default
   solimp) leaked ~21 mm under ~20 N/leg stance force — the slide blew through its +5 mm
   stop and the stomp lifted 1.6 mm. The full-rank [1, −poly′] Jacobian is precisely what
   lets the rows go near-hard at TDC where the connect could not: solref (0.008 1) = the
   2·dt refsafe floor + solimp dmax 0.9999 restores connect-grade holding (0.94 mm leak),
   still comfortably stable (ω·dt = 0.5, B·dt = 1.0). If dt rises, solref rises with it.
Net: dead-center singularity gone, flipped-elbow branch unreachable by construction,
dt=0.004 restored (halves mesh-robot rollout cost), and `test_timestep_is_fleet_standard`
pins the win against regression.

**Engine comparison for dead-center mechanisms:**
- **MuJoCo/MJX/mjwarp** (soft equality, acceleration-level): forces bounded, positions drift;
  explicit across steps → dt-limited near singularity (above). Same math in all three; mjwarp
  adds per-world eq_active and the #1270 anchor caveat (§4).
- **PhysX TGS**: loop joint = D6 between articulation links, solved with re-measured error +
  backward-Euler springs per substep (§6) — position-level and implicit, so the dead center
  does not force a global dt cut; but Isaac's own docs concede the loop-closing joint
  "accumulates the most error" (see `notes/sota-training-issues.md:310-312`), and mimic
  joints are linear-gear only, unable to express s(φ).
- **Newton XPBD**: native loop joints at position level with per-constraint compliance +
  Lagrange accumulators and relaxation 0.4-0.7 (`solver_xpbd.py:103-125,376-381`) —
  unconditionally stable in stiffness (accuracy bought with substeps), explicitly documented
  loop support (§5). **SolverKamino** is the most principled: proximal-ADMM NCP built by
  Disney *specifically* for closed loops + hard contacts (`solver_kamino.py:52-71`) — Beta,
  worth tracking. **SolverFeatherstone silently drops loops** — never use it for this leg.
- **Genesis**: MuJoCo trio, same behavior as MuJoCo.
- **Fundamentally better?** For a *known 1-DOF loop*, coordinate elimination (bake ψ(φ), s(φ)
  into kinematics — what a bespoke engine or Drake-style ScrewJoint thinking would do) beats
  every constraint formulation: no constraint, no singularity, exact toggle force profile via
  the projected Jacobian. No general-purpose engine here does that automatically; the quartic
  coupling is its 95% stand-in available today in your current stack.

## 9. Licenses and contribution mechanics

> ✅ **Exercised, with one scar:** three PRs opened under the Google CLA flow. Learned the hard way that mujoco_warp's AGENTS.md **forbids AI co-author trailers** (the CLA bot must match every commit author) — commits rewritten to single-author before signing. DCO/CLA table below held up exactly as written.

| Repo | License | Contribution gate |
|---|---|---|
| mujoco (+MJX) | Apache-2.0 (`LICENSE`) | Google CLA (`CONTRIBUTING.md:15-26`) |
| mujoco_warp | Apache-2.0 | Google CLA; friendly dev setup (uv/pytest/pre-commit), `AGENTS.md` even documents AI-assisted-PR etiquette |
| warp | Apache-2.0 (`LICENSE.md`) | **DCO only** (`CONTRIBUTING.md:4-10`) — lowest friction |
| newton | Apache-2.0 | CLA (LF newton-governance, `CONTRIBUTING.md:44-117`) |
| PhysX | BSD-3 (`LICENSE.md:1-31`) | DCO (`CONTRIBUTING.md:29-70`); GPU CUDA kernels now open in-tree |
| IsaacLab | BSD-3 (+Apache-2.0 mimic) | DCO 1.1 — but runtime needs proprietary Isaac Sim |
| Genesis | Apache-2.0 | none (plain PRs) |

**MIT preference, plainly:** for *using* these engines, Apache-2.0 and BSD-3 are practically
identical to MIT — permissive, no copyleft, commercial/closed use fine, your own code stays
MIT. Differences that matter: Apache-2.0 adds an explicit patent grant (protection you *want*
from NVIDIA/Google-owned code) plus keep-LICENSE/NOTICE obligations; BSD-3 adds only a
non-endorsement clause. You cannot relicense *their* files as MIT, but you can depend on,
vendor, or ship them inside an MIT project with attribution. For *contributing*: DCO repos
(warp, PhysX, IsaacLab) need only `git commit -s`; CLA repos (mujoco*, newton) need a
one-time signature — Google's CLA leaves your copyright with you.

## 10. Verdict

> ✅ **Ladder executed (2026-07-03):** (1) warp backend validated — **11.8×** on the fight scene, flat scaling 1k→16k envs (train at 1–4k); (2) quartic couplings landed — fleet dt=0.004 back, mechanism stronger than under the connect; (3) contribution targets 1–3 live as PRs #1487/#1488/#3378, CUDA-validated (cuSolverDx NaNs on indefinite input — question answered on the PR); thin layer **fully built M1–M4 (26/26 tests) and then KILLED by its own ≥2× criterion on the A100: fused-vs-baseline measured 1.22× with lidar / 0.92× without** — the device↔host seam costs ~nothing once physics is graph-captured; the honest salvage is lidar dedup (~22%, portable to the wrapper) and M1/M2's mechanism findings. Contribution targets 4 and 5 landed too: #868's rank-1 cholUpdate drafted with an honest mixed finding (factor REUSE −55% on 64% of events is the mergeable core; the rank-1 update itself is SLOWER than warp's cooperative refactorization on CPU), and the Newton lidar published as issue #3346 + draft PR #3347. #868's GPU verdict confirmed the negative: reuse +1.9%/rank-1 +3.2% slower at solver_niter≈2. Full numbers: `notes/warp-ladder-results.md`. §10 is now CLOSED — every claim measured or published.

**(a) Cost of a truly bespoke typed-language→GPU sim.** The language half is a solved,
reusable problem: Warp *is* the strictly-typed Python-subset→NVRTC/LLVM compiler, ~20-30
person-years in (§3). The engine half, measured on mujoco_warp — a team *reimplementing known
semantics with a conformance oracle* — is **10-15 person-years for MuJoCo parity and still
missing autodiff** (§4); Newton adds ~15 more NVIDIA-side person-years for the multi-solver
scaffold (§5). A from-scratch engine without a reference implementation buys back none of
that. A *scoped* sim for this project (below) is 2-4 person-months — but only its last 10% is
actually bespoke.

**(b) The pragmatic ladder — contribute, don't rebuild.** Concrete targets, in order of
payoff for this project:
1. **mujoco_warp #1270** (connect anchor vs joint reference positions) — you have a
   ready-made repro (the leg). Files: `mujoco_warp/_src/constraint.py:129`
   (`_equality_connect`) + a regression test against `slider_crank_s`. Directly de-risks
   migrating the mesh leg to the GPU path.
2. **MJX-JAX heightfield ray** — port `ray_hfield`
   (`mujoco_warp/_src/ray.py:452-620`, or C `engine_ray.c`) into
   `mjx/mujoco/mjx/_src/ray.py` `_RAY_FUNC` (:223-230). Unblocks lidar-over-terrain in your
   *current* brax stack without migration; the silent-miss behavior also deserves a warning
   upstream.
3. **mujoco_warp #1415** (block_cholesky lacks pivot floor → NaNs) — Newton-solver stability
   exactly in near-singular configurations like your TDC. File:
   `mujoco_warp/_src/block_cholesky.py` (+ `solver.py:26-27` call sites).
4. ✅ *(drafted 2026-07-03, local worktree; mixed finding — see §10 banner)* **mujoco_warp #868** (incremental Hessian) — port the C engine's rank-1
   Cholesky-update trick (`engine_solver.c:2238-2276`) to `_src/solver.py`; large solver
   speedup for contact-rich two-robot worlds.
5. ✅ *(published 2026-07-03: issue #3346 + draft PR #3347)* **Newton lidar sensor** — compose the existing analytic raycasts
   (`newton/_src/geometry/raycast.py:80-847`) into a `SensorLidar` beside
   `sensor_tiled_camera.py`; fills a documented gap and is DCO-adjacent visibility in the
   engine most likely to matter in 2027.

**(c) The thin bespoke layer over Warp, scoped to this project** (two robots + floor,
primitive geoms, one loop per leg, ~144 lidar rays): 90% reuse, by component —
- *Reused*: Warp language, NVRTC + kernel cache, CUDA graph capture w/ conditional nodes,
  `wp.mesh_query_ray`/BVH, tile reductions (§3); mujoco_warp's kernels **as a library** —
  types, GJK/EPA, solver, sensors (§4); Newton's `ModelBuilder` + MJCF importer if you want
  scene authoring (§5). PyTorch/JAX interop via dlpack is free in Warp.
- *Written (the actual 10%)*:
  (i) an exact loop-coordinate joint — one `@wp.func` evaluating ψ(φ), s(φ) and derivatives
  inside kinematics, eliminating the equality constraint entirely (the one thing no engine
  gives you; ~200 lines + tests against `test_mesh_robot_contract.py`);
  (ii) a curated static pair table — at ~40 geoms total, hand-culled static allocation is
  *optimal*: MJX's sin was uncurated enumeration + dense efc, not staticness per se
  (~60-120 pairs, analytic capsule-capsule/sphere/plane kernels, ~500 lines — note
  capsule-capsule contact is ~30 lines of segment-segment distance, no GJK needed);
  (iii) a 144-ray × nworld lidar kernel over primitives + hfield (~200 lines; one thread per
  (world, ray), analytic intersections from §3's primitives or cribbed from
  `mujoco_warp/_src/ray.py:809-826`);
  (iv) obs/reward kernels fused into the captured step graph — the real bespoke win, since no
  general engine will fuse *your* reward into *its* graph; this removes the
  device→host→device round-trip per control step that currently bounds brax wrapper
  latency.
- *Data layout if built*: mujoco_warp's conventions are the proven template — Data arrays
  `(nworld, thing)`, Model arrays broadcastable `(*, thing)` for per-world DR (§4), contacts
  in one cross-world atomic pool, whole step inside a single CUDA graph with
  `wp.capture_while` for the solver. Do not invent a different layout; interop with their
  kernels is the whole point.
- *Milestones with kill-criteria* (est. 2-4 person-months part-time total): M1 two-capsule
  drop test vs MuJoCo C reference (1-2 wk; kill if contact match > 1% off); M2 one leg with
  loop-coordinate joint vs `slider_crank_s` closed form (1-2 wk); M3 full two-robot step
  parity vs mjwarp at matched settings (2-4 wk; kill if < 2× faster end-to-end than the
  mjwarp path — the fused obs/reward is the only justification); M4 training-loop
  integration (2 wk).
- *Order of operations*: (1) move contacts+lidar to `mjx.put_model(m, impl='warp')` /
  direct mjwarp — all three MJX blockers disappear by construction (§4); (2) swap
  `<connect>` → quartic couplings and retry dt=0.004 (§8); (3) contribute targets 1-3 as you
  hit them; (4) only build layer (i)-(iv) if profiling (`mjwarp-testspeed --event_trace`)
  shows the step-graph seams (obs/reward round-trips) dominate — and build it *on*
  mujoco_warp, not beside it.

The one-line verdict: the bespoke compiler exists (Warp), the bespoke engine is being built in
the open by ~20 funded engineers (mujoco_warp/Newton), and this project's genuinely novel
needs — an exact toggle joint, curated pair tables, fused reward kernels, lidar — total a few
hundred lines *on top* of that stack, not a simulator *instead* of it.

## 11. Key-file index (for re-verifying or extending this note)

Line numbers drift; these anchors + a `grep -n` for the named symbol will re-locate everything.

| Question | Go to |
|---|---|
| How MuJoCo prices a contact | `mujoco/src/engine/engine_core_constraint.c` — `mj_makeImpedance`, `getimpedance`, `mj_diagApprox` |
| Why MuJoCo Newton is fast on CPU | `engine_solver.c` — `MakeHessian`, `HessianIncremental`, `PrimalSearch` |
| What implicitfast integrates implicitly | `engine_forward.c` `mj_implicitSkip`; `engine_derivative.c` `mjd_smooth_vel` |
| Where MJX bakes the pair list | `mjx/mujoco/mjx/_src/collision_driver.py` — `geom_pairs`, `_geom_groups`, `make_condim` |
| Where MJX sizes ncon/nefc | `mjx/mujoco/mjx/_src/io.py` `_make_data_jax`; `constraint.py` `make_efc_type`/`counts` |
| MJX ray coverage | `mjx/mujoco/mjx/_src/ray.py` `_RAY_FUNC` (HFIELD absent, silent miss) |
| Warp's AST→CUDA path | `warp/_src/codegen.py` (`ast.parse` in `Adjoint`, `node_visitors`, kernel templates); `warp/native/warp.cu` (NVRTC) |
| CUDA graph conditionals | `warp/_src/context.py` `capture_if`/`capture_while`; used at `mujoco_warp/_src/solver.py` `wp.capture_while(nsolving, …)` |
| mjwarp contact compaction | `mujoco_warp/_src/collision_driver.py` (atomic pairid), `collision_core.py` (atomic nacon), `types.py` (naconmax pool + per-contact worldid) |
| mjwarp hfield ray / rangefinder | `mujoco_warp/_src/ray.py` `ray_hfield`, `ray_mesh_with_bvh`; `sensor.py` rangefinder |
| Newton loop-joint semantics per solver | `newton/docs/concepts/articulations.rst` (§ on loop joints); `newton/_src/solvers/kamino/solver_kamino.py` header |
| Newton↔mujoco_warp bridge | `newton/_src/solvers/mujoco/solver_mujoco.py` (`MjSpec` build, `put_model`, step) |
| PhysX TGS substep loop | `PhysX/physx/source/lowleveldynamics/src/DyTGSDynamics.cpp` `iterativeSolveIsland`; implicit spring math in `shared/DyCpuGpu1dConstraint.h` |
| PhysX GPU contact streams / env filtering | `gpubroadphase/src/CUDA/broadphase.cu` (`filtering`, `performIncrementalSAP`); `PxgNarrowphaseCore.cpp` overflow warnings; `PxSceneDesc.h` `PxGpuDynamicsMemoryConfig` |
| Isaac Lab ray caster limits | `IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/ray_caster.py` (static single mesh); `utils/warp/kernels.py` |
| Genesis rigid-solver lineage | `Genesis/genesis/engine/solvers/rigid/` (`rigid_solver.py` solref unpack, `collider/epa.py` MuJoCo cite, `constraint/solver.py` equality trio) |
| This project's toggle repro | `sim/robot/gen_mesh_robot_mjcf.py:55-70,120-128,152-157,183-188`; `test_mesh_robot_contract.py:115` |
| This project's pair culling | `sim/robot/gen_robot_mjcf.py:85-88,368-394` (F-SPEED masks, `<pair>` solref overrides) |

Open items to re-check on next visit: mujoco_warp issues #1270 (connect anchors), #1415
(Cholesky pivot floor), #868 (incremental Hessian), #500 (differentiability); whether
`MJX_GPU_DEFAULT_WARP` has flipped to default; Newton SolverKamino graduation from Beta;
Isaac Lab 3.0 Newton backend leaving `develop`.
