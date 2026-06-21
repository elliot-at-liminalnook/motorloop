<!-- SPDX-License-Identifier: MIT -->
# `arena` framework — incremental build checklist

> **STATUS (2026-06-21): BUILT — all 8 phases `verified` in `arena/BUILD_STATE.json` with
> snapshots in `sim/build/fw-snapshots/`.** Validate with `make arena-prove` (every layer's
> selftest) or `make fw-status` (the ledger). 11 modules under `sim/robot/arena/`. The only
> remaining item is Phase 8's *live*-GPU exercise, which is the self-play transition
> (`arena.cli pipeline --runner pod`) on the existing A100 — it advances the overarching goal,
> so no second paid pod. Run it with `make gpu-arena` (or the `arena.cli` invocation).

Wrap the sprawling training harness (`train_adversarial.py` kernel + two near-duplicate drivers +
six `/tmp/rp_*.sh` pod scripts) in **one elegant framework** with a **universal trace/log spine**.

**Architecture (settled):** a *kernel/scheduler/runner* split over a *trace spine*.
- **Kernel** = `train_adversarial.py` — trains ONE stage; owns reward, the held-out benchmark,
  keep-best, checkpoint, resume. **Immutable** — we adapt to it, never rewrite it.
- **Layer 0 — trace spine** (`arena/trace.py`): one structured `Event` model, trace context that
  flows across process/machine boundaries, errors captured + classified at boundaries.
- **Layer 1 — `Stage`** (`arena/stage.py`): a declarative unit of work → a kernel CLI invocation.
- **Layer 2 — `Schedule` + engine** (`arena/schedule.py`, `arena/engine.py`): `Curriculum`,
  `League`, `Pipeline`; the ONE `drive()` loop (warm-from-best → gate → keep-best → resume).
- **Layer 3 — `Runner`** (`arena/runner.py`): `LocalRunner` / `PodRunner` (the pod lifecycle,
  hard-won fixes baked in), emitting pod + error events.
- **Layer 4 — `Run` + sugar** (`arena/run.py`): `Run(name, schedule, runner).go()` +
  `errors()/timeline()/metrics()/tail()/figure()`.

> **Opinionated ground rules (do not deviate without cause):**
> - Package lives at **`sim/robot/arena/`**; public API via `arena/__init__.py`; runnable via
>   `python -m arena.cli`. Every file: SPDX header, `py_compile`-clean, REUSE-clean.
> - **Validate every phase on the local CPU venv `$HOME/mjx-venv`** (free, pod-matched). GPU is
>   touched ONLY in Phase 8.
> - **The live GPU run is never touched** by this build — old drivers/scripts keep working until
>   Phase 7 migration; the framework becomes how we launch the *next* run.
> - **Verify-then-snapshot before advancing** a phase (the same checkpoint discipline we use for
>   training). A phase isn't "done" until its `PROVEN:` line is green AND a snapshot is taken.

## Build resumability + backup (set up FIRST, used throughout)
The implementation is itself a multi-stage run, so it gets the same resume/backup property:
- **Ledger** `sim/robot/arena/BUILD_STATE.json` — one record per phase:
  `{phase, title, status: todo|doing|done|verified, verify_cmd, snapshot, ts, notes}`.
- **Snapshots** `sim/build/fw-snapshots/<phaseN>-<ts>.tgz` — a tar of `arena/` + the ledger taken
  at each verified phase (the restore points; repo isn't git so snapshots ARE the backups —
  cf. the "Elliot commits himself" rule, so we never `git`).
- **Make targets** (Phase 0 adds them):
  - `make fw-status`  → render the ledger (what's done / next unverified phase + its verify cmd).
  - `make fw-snapshot PHASE=N` → verify-gated tar snapshot + mark the phase `verified` in the ledger.
  - `make fw-restore SNAP=<file>` → restore `arena/` + ledger from a snapshot.
- **Resume rule:** to continue, run `make fw-status`, do the named next phase, verify, snapshot. The
  ledger is the single source of truth for "where we are."

---

## Phase 0 — Scaffold + build-resumability tooling (do this first; it's the safety net)
- [ ] `mkdir sim/robot/arena`; add `arena/__init__.py` (empty public API stub), `arena/VERSION`.
- [ ] Write `arena/_ledger.py` (tiny): read/update `BUILD_STATE.json`; seed it with all 8 phases as
      `todo` (titles + verify commands from this file).
- [ ] Add `scripts/fw_snapshot.sh` (tar `sim/robot/arena` + ledger → `sim/build/fw-snapshots/`,
      only if the phase's verify command exits 0; then flip the phase to `verified`).
- [ ] Add Makefile targets `fw-status`, `fw-snapshot`, `fw-restore` (call the above).
- **Verify:** `make fw-status` prints all 8 phases as `todo` and names Phase 1 as next;
      `make fw-snapshot PHASE=0` (verify cmd = `python -c "import json,sys; json.load(open('sim/robot/arena/BUILD_STATE.json'))"`)
      produces a `.tgz` and flips Phase 0 → `verified`. → `PROVEN: build ledger + snapshot/restore round-trips`.
- [ ] **Checkpoint:** snapshot taken; ledger shows Phase 0 verified.

## Phase 1 — Layer 0: the trace spine `arena/trace.py` (everything emits into this)
- [ ] `Event` dataclass: `ts, run_id, stage, attempt, component, level, kind, msg, ctx, payload`.
- [ ] `Tracer(run_id, sink)`: append-only JSONL sink + a human console renderer (both, always).
- [ ] `span(stage, attempt, ctx)` context manager: stamps every event inside with the context;
      **on exception, auto-emits ONE `error` Event** (traceback + the span ctx).
- [ ] `classify(text|exit_code) -> cause`: rule table built from THIS session's real failures —
      `RESOURCE_EXHAUSTED/oom→gpu_oom`, `cuSolver→gpu_contention`, `no RECORD/externally-managed→
      pep668_pip`, `INTERNAL_SERVER_ERROR→runpod_graphql`, `not found on the registry→stale_image`,
      `exit 144→signal_kill/self_pkill`, empty-stdout-after-ship→`ship_stdin_conflict`, `FAILED rc=→
      stage_subprocess_fail`.
- [ ] `merge(*jsonl) -> events` by `ts` (for local⊕pod reconstruction).
- **Verify:** `python -m arena.trace --selftest` (on mjx-venv): emits a few events; a `with span():`
      that raises produces a classified `error` Event with the traceback + ctx; JSONL round-trips;
      `merge()` interleaves two files by ts. → `PROVEN: trace spine — structured events + span
      auto-error-capture + classifier + jsonl round-trip + merge`.
- [ ] **Checkpoint:** `make fw-snapshot PHASE=1`.

## Phase 2 — Emit from the CURRENT kernel + pod scripts (immediate value, before the refactor)
- [ ] Add a thin `arena.trace` shim the kernel uses: `train_adversarial`'s `METRIC()`/`print` for
      eval points → `span.metric("benchmark", …)` / `span.event("stage.start")`, reading
      `TRACE_RUN/TRACE_STAGE/TRACE_ATTEMPT` from env (absent ⇒ standalone, still works).
- [ ] Wrap the existing `/tmp/rp_watch.sh` check to emit classified `error` events into the run's
      `events.jsonl` (so the LIVE run's failures get classified now).
- **Verify:** a tiny `train_adversarial --tiny` (mjx-venv) writes `events.jsonl` with trace-stamped
      `stage.start` + `metric` events; feed a synthetic `RESOURCE_EXHAUSTED` line through the
      classifier → `gpu_oom`. → `PROVEN: kernel + watcher emit into the universal stream`.
- [ ] **Checkpoint:** `make fw-snapshot PHASE=2`.

## Phase 3 — Layer 1: `arena/stage.py` (declarative Stage → kernel invocation)
- [ ] `Stage` dataclass: `tag, task{striker, opponent: passive|frozen(ckpt), sep_lo/hi, azimuth},
      reward{clean,trade,disengage,fire,approach,shaping}, budget{steps,envs,batch}, bench{...},
      gate`. Method `.flags(warm, cum_base) -> [argv]` for `train_adversarial`.
- [ ] Helpers `Stage.curriculum_phase(name, …)` and `Stage.league_round(round, opp, bench_opp, …)`.
- **Verify:** `arena/stage_selftest` asserts `Stage.curriculum_phase("c2").flags(...)` is byte-equal
      to the argv `curriculum_drive` builds today, and a league Stage matches `selfplay_drive`. →
      `PROVEN: Stage round-trips both drivers' exact kernel CLIs`.
- [ ] **Checkpoint:** `make fw-snapshot PHASE=3`.

## Phase 4 — Layer 2: the unified engine + schedules (collapse the two drivers)
- [ ] `arena/engine.py`: `RunState` (best_ckpt, best_bench, cum_step, hof, idx, completed) with
      `save()/load()` (resume-safe) + `observe(res)` (keep-best + gate + cum_step) + `snapshot()`.
- [ ] `drive(schedule, runner, state)`: the ONE loop (`while stage := schedule.next(state)` →
      `runner.train` → `state.observe` → `schedule.on_done` → `state.save`), all inside `span`s.
- [ ] `arena/schedule.py`: `Curriculum` (sep-range phases, opponent=passive, gate+rollback),
      `League` (round → first-quarter HoF sample, fixed-ref benchmark, archive snapshot),
      `Pipeline([...])` (sequential, shares `best_ckpt`).
- **Verify (mjx-venv, LocalRunner stub):** (a) `Curriculum().next()` yields the SAME phases as
      `curriculum_drive.PHASES`; (b) a tiny `drive(Curriculum(), LocalRunner())` reproduces
      `curriculum_drive`'s gate/keep-best (compare `*_state.json`); (c) tiny `League` reproduces
      `selfplay_drive`; (d) **RESUME TEST** — interrupt mid-`Pipeline`, re-run, assert it continues
      from the saved stage (not from scratch) and the benchmark best is preserved. →
      `PROVEN: one engine = curriculum ⊕ self-play; resume continues mid-pipeline`.
- [ ] **Checkpoint:** `make fw-snapshot PHASE=4`.

## Phase 5 — Layer 3: `arena/runner.py` (Local + Pod, pod lifecycle unified)
- [ ] `LocalRunner.train(stage, warm) -> res` — subprocess the kernel on `sys.executable`, parse
      `{tag}_state.json`; emits `kernel`/`error` events.
- [ ] `Pod` lifecycle class (folds the six `/tmp/rp_*.sh`): `provision` (RunPod **REST** API, current
      image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, GPU sweep), `bringup` (REST ssh poll),
      `ship`, `setup` (venv/PEP-668), `launch` (nohup+sentinel, NO stdin+`&` conflict), `pull`,
      `watch` (10-min classified error checks), `terminate(at_budget)`. **Never** `pkill -f <self-matching>`.
- [ ] `PodRunner.train` = same interface as Local, over ssh; budget guard.
- **Verify:** LocalRunner runs a tiny stage end-to-end (mjx-venv); **mock-pod unit tests** for the
      REST request bodies + the sentinel-poll + the self-pkill guard (no real GPU); the classifier
      catches an injected pod failure (`INTERNAL_SERVER_ERROR→runpod_graphql`). →
      `PROVEN: runner abstraction — local trains; pod lifecycle composes + classifies failures` (real
      GPU smoke deferred to Phase 8).
- [ ] **Checkpoint:** `make fw-snapshot PHASE=5`.

## Phase 6 — Layer 4: `arena/run.py` (`Run`/`Pipeline` sugar + observability)
- [ ] `Run(name, schedule, runner)`: owns the run dir `runs/<run_id>/`, the `Tracer`, `state`,
      `.go()` (provision→drive→pull→terminate), `.resume()`.
- [ ] Views over the stream: `errors()`, `timeline()`, `metrics(kind)`, `tail()`, `figure()`
      (reuses `make_benchmark_figure`). Watcher = a consumer querying `kind=error`.
- **Verify (mjx-venv):** ONE `Run("smoke", Pipeline([Curriculum(tiny), League(tiny, seed=…)]),
      LocalRunner()).go()` runs the whole curriculum→league end-to-end; `run.errors()` returns an
      injected error; `run.figure()` writes the benchmark PNG; `run.resume()` after a kill continues.
      → `PROVEN: declarative Run reproduces the full pipeline + observability + resume`.
- [ ] **Checkpoint:** `make fw-snapshot PHASE=6`.

## Phase 7 — Migrate + retire the old surface
- [ ] `curriculum_drive.py` / `selfplay_drive.py` → thin shims calling `arena` (keep CLIs working);
      delete the `/tmp/rp_*.sh` reliance (PodRunner owns it). Update Makefile (`gpu-win-exchanges`,
      `gpu-win-exchanges-medium`, `win-exchanges-prove`, a new `gpu-arena`).
- [ ] Update `notes/gpu-pod-setup.md` + `notes/gpu-runbook.md` to point at `arena`; SPDX/REUSE pass.
- **Verify:** tiny runs via the shims are behavior-identical to the old scripts (diff `*_state.json`);
      `make win-exchanges-prove` green on mjx-venv; `python -m py_compile` over `arena/` + shims;
      `reuse lint` (or the project's checker) clean. → `PROVEN: old entrypoints delegate; nothing regressed`.
- [ ] **Checkpoint:** `make fw-snapshot PHASE=7`.

## Phase 8 — Real-scale GPU validation (the only paid phase)
- [ ] On a fresh pod via `PodRunner`, run `Run("striker-arena", Pipeline([Curriculum(), League(
      seed="curriculum_best")]), PodRunner(gpu="A100 80GB PCIe", budget=…))` at real scale.
- [ ] Confirm the trace spine captures local⊕pod events (merge by ts), errors are classified, the
      benchmark stays monotone, and `Run.resume()` works after a deliberate mid-run pod kill.
- **Verify:** `run.metrics("benchmark")` shows the rising curve across curriculum→league;
      `run.errors()` is empty or fully classified; a kill→`resume()` continues from the last stage;
      pod terminated at budget. → `PROVEN: the framework runs the real pipeline, observable +
      resumable, across machines`.
- [ ] **Checkpoint:** `make fw-snapshot PHASE=8`; record results in `notes/codesign-fighter-report.md`.

## Phase 9 — the adaptive Coach (DONE; `arena/coach.py`)
Verified competency controller: sparse-verdict judgment (win/survival/safe) + keep-best-on-win; reward
levers (clean/trade/fire/approach) + balance/energy competency levers + a curriculum lever (sep_hi) +
adaptive opponent difficulty. `make fw-snapshot PHASE=9` taken. (Roadmap for it: win-exchanges §2·C.)

---

## ROADMAP — the motorloop-specific spine (DONE 2026-06-21; Phases 10–12 verified + snapshotted)
The generic top-of-list items (metrics/batch/DR/coaching) were already built. The highest *new* lift
was the unique motorloop move: **connect the robot policy down to the verified component IP** (FOC/ADC/
encoder RTL), which had floated free of the robot sim. All three carried out (`make arena-prove`):
`arena/backend.py` (Actuator contract), `arena/rtl_gate.py` (FOC-envelope gate + RTL cosim hook),
`arena/manifest.py` (reproducibility + episode recorder + regression). Sequence (all `[x]`):

### Phase 10 — Backend/actuator CONTRACT (the architectural enabler, #2)
- [ ] Define a small `Backend` protocol — `reset / step / observe / act / metrics` (+ `render`) — so
      the actuator/physics layer is SWAPPABLE behind the kernel. Refactor `AdversarialEnv`'s MJX core
      behind it (small, ~no behavior change; the RL inner loop keeps the fast MJX-ideal actuator at
      8192 envs).
- **Verify:** the MJX backend passes byte-identical parity (`test_parity`) behind the contract; a toy
      stub backend plugs into the same `train_adversarial`/`arena` harness. `PROVEN: backend contract —
      MJX behind it unchanged; a second backend plugs in`.

### Phase 11 — RTL-cosim VALIDATION backend (THE differentiator, #6)
- [ ] A **multi-fidelity** backend: train on MJX-ideal (fast), then run a trained policy's actuator-
      command trajectory through the real **FOC/ADC RTL cosim** (cocotb/Verilator) as a GATE — does the
      controller actually deliver the demanded torque under current-limit + back-EMF + ADC latency +
      encoder quantization + PWM saturation? Reuse `multifidelity.py` + `reality_gap` + `motors.py` +
      `joint_torque_limit`; this grounds the Coach's `actuator-safety`/`energy` competency and the
      verdict's `safe_rate` in VERIFIED SILICON, not a toy clamp. (Same "fast for learning, real for
      judgment" split we used for rewards — now for the actuator.)
- **Verify:** a policy that respects the idealized limit but VIOLATES the real controller's envelope is
      caught by the RTL gate (fails `safe`); one that respects both passes. `PROVEN: co-design realized —
      robot behavior traced to the verified FOC/ADC IP`.

### Phase 12 — Episode/manifest CONTRACT (cheap + foundational, #1; do alongside 10–11)
- [ ] Extend the trace spine: optionally record full **per-step obs/action/reward/safety-events** for a
      run (replayable episodes), and write a **reproducibility manifest** per run — git commit, seed,
      config (Stage), robot model (`robot.toml` hash), policy checkpoint, backend, machine profile.
- **Verify:** a recorded run REPLAYS bit-identically from its manifest; `run.regression(other)` diffs two
      runs' metrics. `PROVEN: runs are regression evidence, not demos`.

> Explicitly DEFERRED (agree with the generic ranking's tail): high-fidelity rendering, URDF import
> adapters, Drake-style optimization/co-design — later, once the actuator-to-component spine is real.

---

## Done-when
All 8 phases `verified` in `BUILD_STATE.json` with snapshots; `Run(Pipeline([Curriculum, League]),
PodRunner).go()` is the single command that provisions, trains the skill curriculum then self-play,
keeps the benchmark monotone, captures every event/error across local+pod in one trace-stamped
stream, terminates at budget, and **resumes from any stage**. The old drivers/scripts are thin shims.

## What NOT to do
- Don't rewrite the kernel — `train_adversarial.py` is the immutable kernel; adapt to it.
- Don't touch the live GPU run while building (Phases 0–7 are local/free).
- Don't advance a phase before its `PROVEN:` verify is green AND a snapshot is taken (the build's
  own resume/backup property — mirror the training-run discipline).
- Don't build two logging paths — every print/metric/error is ONE `Event` into ONE stream.
- Don't `pkill -f <pattern that matches the caller>` (the exit-144 self-kill); kill by PID.
- Don't ship a script via stdin pipe AND background it in one ssh call (the empty-`run_long.sh` bug);
  ship, verify, THEN launch.
- Don't use the RunPod GraphQL deploy mutation (broken) — REST `rest.runpod.io/v1/pods`.
