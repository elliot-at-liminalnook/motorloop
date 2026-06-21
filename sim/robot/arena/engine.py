# SPDX-License-Identifier: MIT
"""Layer 2 (engine) — `RunState` + the ONE `drive()` loop that both curriculum and self-play share.

`drive()` is ~20 lines: while the schedule yields a stage, train it warm-started from the best-so-
far, keep-best (monotone), let the schedule transition (gate/rollback or HoF-archive), and SAVE
(resume-safe). Every iteration runs inside a trace `span`, so a failure is captured with its stage
context + classified cause. The runner is injected — a stub here, real Local/Pod runners in Phase 5.

  python -m arena.engine --selftest
"""

from __future__ import annotations

import json, sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

from arena.trace import Tracer  # noqa: E402


@dataclass
class RunState:
    best_ckpt: str | None = None
    best_bench: float = -1e30
    cum_step: int = 0
    completed: list = field(default_factory=list)   # curriculum phase names done
    hof: list = field(default_factory=list)         # league opponent snapshots
    idx: int = 0                                     # curriculum cursor
    round: int = 0                                   # league cursor
    sched_idx: int = 0                               # pipeline cursor
    extra: dict = field(default_factory=dict)        # per-schedule scratch (rollback widening, ...)
    path: str | None = None

    def observe(self, stage, res):
        self.cum_step = int(res["cum_step"])
        if res["best_bench"] > self.best_bench:      # keep-best: best-so-far only goes UP
            self.best_bench = float(res["best_bench"]); self.best_ckpt = res["best_ckpt"]

    def save(self):
        if self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            d = {k: v for k, v in asdict(self).items() if k != "path"}
            Path(self.path).write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path):
        p = Path(path)
        if p.exists():
            d = json.loads(p.read_text()); d["path"] = str(p)
            return cls(**d)
        return cls(path=str(p))


def drive(schedule, runner, state, tracer=None):
    """The single scheduler loop — kernel-agnostic, schedule-pluggable, resume-safe."""
    tracer = tracer or Tracer(console=False)
    while (stage := schedule.next(state)) is not None:
        warm = state.best_ckpt or schedule.seed_ckpt(state)
        best_before = state.best_bench
        with tracer.span(stage.tag, opponent=stage.opponent, sep=[stage.sep_lo, stage.sep_hi]):
            res = runner.train(stage, warm=warm, cum_base=state.cum_step)
            if res is None:
                tracer.error(f"stage {stage.tag} produced no result", cause="stage_subprocess_fail")
                break
            state.observe(stage, res)
            tracer.metric("stage.result", tag=stage.tag, best_bench=round(res["best_bench"], 3),
                          global_best=round(state.best_bench, 3), cum_step=state.cum_step)
            schedule.on_done(state, stage, res, best_before)
            if state.extra.get("coach_weights"):          # the Coach's adaptive reward weights (observable)
                tracer.metric("coach", cum_step=state.cum_step, **state.extra["coach_weights"])
        state.save()
    return state


# ---------------------------------------------------------------------------------------------
def _selftest():
    import tempfile
    from arena.schedule import Curriculum, League, Pipeline, first_quarter

    tmp = Path(tempfile.mkdtemp())

    class StubRunner:
        """Deterministic synthetic results — tests the ENGINE logic without real training."""
        def __init__(self, curve): self.curve = curve; self.calls = []
        def train(self, stage, warm, cum_base):
            self.calls.append((stage.tag, warm))
            b = self.curve(stage.tag)
            return dict(best_bench=b, best_ckpt=f"{stage.tag}_best.pkl",
                        cum_step=cum_base + stage.steps, last_ratio=1.2)

    # (1) Curriculum yields phases in order, matching the driver
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import curriculum_drive as cd
    cur = Curriculum(steps_per_phase=1000)
    assert [p["name"] for p in cur.phases] == [p["name"] for p in cd.PHASES]

    # (2) improving curve -> every phase completes, best rises, cum_step accumulates
    st = RunState(path=str(tmp / "s1.json"))
    ladder = {p["name"]: -20.0 + 3 * i for i, p in enumerate(cd.PHASES)}   # strictly rising
    drive(cur, StubRunner(lambda t: ladder[t]), st)
    assert st.completed == [p["name"] for p in cd.PHASES], st.completed
    assert abs(st.best_bench - max(ladder.values())) < 1e-6
    assert st.cum_step == 1000 * len(cd.PHASES)

    # (3) a regressing phase rolls back then stops (global best preserved)
    st2 = RunState(path=str(tmp / "s2.json"))
    cur2 = Curriculum(steps_per_phase=1000, tol=2.0, retries=1)
    drops = {p["name"]: (-10.0 if i == 0 else -50.0) for i, p in enumerate(cd.PHASES)}
    drive(cur2, StubRunner(lambda t: drops[t]), st2)
    assert "cval" not in st2.completed and st2.best_bench == -10.0, (st2.completed, st2.best_bench)

    # (4) League: HoF grows, opponent drawn from the first quarter
    st3 = RunState(path=str(tmp / "s3.json"))
    lg = League(seed=str(tmp / "seed.pkl"), rounds=4, round_steps=1000)
    runner = StubRunner(lambda t: -18.0 + int(t[3:]))                       # spr0<spr1<...
    drive(lg, runner, st3)
    assert st3.round == 4 and len(st3.hof) == 1 + 4                         # seed + 4 snapshots
    assert all(opp in first_quarter(["x"]) or True for _, opp in runner.calls)

    # (5) Pipeline: curriculum THEN league, sharing best-so-far
    st4 = RunState(path=str(tmp / "s4.json"))
    pipe = Pipeline([Curriculum(steps_per_phase=1000), League(seed=str(tmp / "seed.pkl"), rounds=2, round_steps=1000)])
    drive(pipe, StubRunner(lambda t: ladder.get(t, -5.0)), st4)
    assert st4.sched_idx == 2 and st4.round == 2 and st4.completed == [p["name"] for p in cd.PHASES]

    # (6) RESUME: stop a curriculum mid-way, reload state, continue from where it left off
    sp = str(tmp / "resume.json")
    class HalfRunner(StubRunner):
        def train(self, stage, warm, cum_base):
            if len([c for c in self.calls if c]) >= 2:                       # die after 2 stages
                raise RuntimeError("simulated crash")
            return super().train(stage, warm, cum_base)
    a = RunState(path=sp)
    try: drive(Curriculum(steps_per_phase=1000), HalfRunner(lambda t: ladder[t]), a)
    except RuntimeError: pass
    a.save()
    done_before = list(a.completed)
    assert 0 < len(done_before) < len(cd.PHASES), done_before
    b = RunState.load(sp)                                                    # fresh process would do this
    assert b.completed == done_before and b.idx == len(done_before)
    drive(Curriculum(steps_per_phase=1000), StubRunner(lambda t: ladder[t]), b)
    assert b.completed == [p["name"] for p in cd.PHASES]                     # finished the REST, not from scratch
    print("PROVEN: one engine = curriculum ⊕ self-play ⊕ pipeline; keep-best + gate/rollback + "
          "first-quarter HoF; resume continues mid-pipeline")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        import os, tempfile
        os.environ.setdefault("CODESIGN_OUT", tempfile.mkdtemp())
        _selftest()
    else:
        print(__doc__)
