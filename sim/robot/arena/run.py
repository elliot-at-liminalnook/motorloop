# SPDX-License-Identifier: MIT
"""Layer 4 — `Run`: the one declarative object that ties schedule + runner + the trace spine into a
resumable, observable experiment.

    Run("striker-arena",
        schedule=Pipeline([Curriculum(), League(seed="curriculum_best.pkl")]),
        runner=PodRunner(...)).go()

`go()` (provision→drive→pull→terminate for a PodRunner; just drive for Local) runs the whole
pipeline; `resume()` picks up from the saved `RunState`. Observability is pure views over the ONE
trace-stamped stream: `errors()`, `timeline()`, `metrics(kind)`, `tail()`, `figure()`.

  python -m arena.run --selftest
"""

from __future__ import annotations

import json, sys
from pathlib import Path

from arena.trace import Tracer, merge          # noqa: E402
from arena.engine import RunState, drive       # noqa: E402


class Run:
    def __init__(self, name, schedule, runner, rundir=None, console=True):
        self.name = name
        self.schedule = schedule
        self.runner = runner
        self.rundir = Path(rundir or f"runs/{name}")
        self.rundir.mkdir(parents=True, exist_ok=True)
        self.sink = self.rundir / "events.jsonl"
        self.tracer = Tracer(run_id=name, sink=str(self.sink), component="orchestrator", console=console)
        self.state = RunState.load(self.rundir / "state.json")
        # share the ONE stream + run_id with the runner so kernel/pod events join it
        self.runner.tracer = self.tracer
        self.runner.run_id = name
        self.runner.sink = str(self.sink)

    def go(self):
        self.tracer.event("run.start", self.name)
        if hasattr(self.runner, "provision"):
            self.runner.provision()                    # PodRunner: provision→ship→setup
        drive(self.schedule, self.runner, self.state, self.tracer)
        if hasattr(self.runner, "teardown"):
            self.runner.teardown()                     # PodRunner: pull→terminate(at budget)
        self.tracer.event("run.done", self.name,
                          best_bench=round(self.state.best_bench, 3), best_ckpt=self.state.best_ckpt or "")
        return self.state

    def resume(self):
        return self.go()                               # drive() continues from the saved RunState

    # --- observability: pure views over the trace-stamped stream (local ⊕ pulled-pod) ----------
    def _events(self, extra_jsonl=()):
        return merge(self.sink, *extra_jsonl)

    def errors(self):
        return [e for e in self._events() if e["kind"] == "error"]

    def timeline(self):
        return self._events()

    def metrics(self, kind=None):
        return [e for e in self._events() if e["kind"] == "metric" and (kind is None or e["msg"] == kind)]

    def tail(self, n=20):
        return self._events()[-n:]

    def figure(self, out=None):
        pts = []
        for e in self.metrics("benchmark"):
            p = e.get("payload", {})
            x = p.get("cum_step", p.get("step"))
            y = p.get("best", p.get("sparc"))
            if x is not None and y is not None:
                pts.append((x, y))
        if not pts:
            self.tracer.event("figure.skip", "no benchmark metrics in stream", "WARN"); return None
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        except Exception:
            self.tracer.event("figure.skip", "matplotlib unavailable", "WARN"); return None
        pts.sort()
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        out = Path(out or self.rundir / "benchmark_curve.png")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs, ys, ".-", color="tab:red"); ax.set_xlabel("cumulative env-steps")
        ax.set_ylabel("benchmark best (monotone)"); ax.set_title(f"{self.name} — keeps improving")
        ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
        return str(out)


# ---------------------------------------------------------------------------------------------
_STUB = (
    "import sys,json,os,time\n"
    "a=sys.argv\n"
    "tag=a[a.index('--tag')+1]; cum=int(a[a.index('--cum-base')+1]); steps=int(a[a.index('--steps')+1])\n"
    "o=os.environ['CODESIGN_OUT']; sink=os.environ.get('ARENA_SINK')\n"
    "fail=os.environ.get('STUB_FAIL_TAG')\n"
    "if tag==fail:\n"
    "    sys.stderr.write('RESOURCE_EXHAUSTED: out of memory'); sys.exit(1)\n"
    "best=-20.0+0.001*cum\n"
    "if sink:\n"
    "    open(sink,'a').write(json.dumps({'ts':time.time(),'run_id':os.environ.get('TRACE_RUN','r'),"
    "'component':'kernel','level':'INFO','kind':'metric','msg':'benchmark','stage':tag,'attempt':0,"
    "'ctx':{},'payload':{'cum_step':cum+steps,'best':best,'ratio':1.3}})+'\\n')\n"
    "json.dump({'tag':tag,'cum_step':cum+steps,'best_bench':best,'best_step':cum,'last_ratio':1.3},"
    " open(os.path.join(o,tag+'_state.json'),'w'))\n"
    "open(os.path.join(o,tag+'_best.pkl'),'wb').write(b'x')\n")


def _selftest():
    import os, tempfile
    from arena.runner import LocalRunner
    from arena.schedule import Curriculum, League, Pipeline
    tmp = Path(tempfile.mkdtemp())
    stub = tmp / "stub.py"; stub.write_text(_STUB)
    os.environ["CODESIGN_OUT"] = str(tmp / "out")

    # (1) ONE declarative Run reproduces the whole curriculum -> league pipeline end-to-end
    runner = LocalRunner(kernel=str(stub), out=tmp / "out", lean=True)
    seed = tmp / "out" / "seed.pkl"; seed.parent.mkdir(parents=True, exist_ok=True); seed.write_bytes(b"s")
    run = Run("smoke", Pipeline([Curriculum(steps_per_phase=1000), League(seed=str(seed), rounds=2, round_steps=1000)]),
              runner, rundir=tmp / "rundir", console=False)
    run.go()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import curriculum_drive as cd
    assert run.state.completed == [p["name"] for p in cd.PHASES] and run.state.round == 2, run.state.completed
    # observability: benchmark metrics present, figure writes a PNG, no errors
    assert len(run.metrics("benchmark")) >= len(cd.PHASES) + 2
    png = run.figure(); assert png and Path(png).exists(), png
    assert run.errors() == []

    # (2) an injected failure surfaces as a classified error in errors()
    os.environ["STUB_FAIL_TAG"] = "c1"
    run2 = Run("boom", Curriculum(steps_per_phase=1000), LocalRunner(kernel=str(stub), out=tmp / "out2", lean=True),
               rundir=tmp / "rundir2", console=False)
    run2.go()
    del os.environ["STUB_FAIL_TAG"]
    errs = run2.errors()
    assert any(e["payload"]["cause"] == "gpu_oom" for e in errs), errs   # kernel OOM classified

    # (3) RESUME: a Run that died mid-pipeline continues from saved state in a fresh Run object
    os.environ["STUB_FAIL_TAG"] = "c2"
    rd = tmp / "rundir3"
    Run("res", Curriculum(steps_per_phase=1000), LocalRunner(kernel=str(stub), out=tmp / "out3", lean=True),
        rundir=rd, console=False).go()
    partial = json.loads((rd / "state.json").read_text())["completed"]
    assert 0 < len(partial) < len(cd.PHASES), partial
    del os.environ["STUB_FAIL_TAG"]
    Run("res", Curriculum(steps_per_phase=1000), LocalRunner(kernel=str(stub), out=tmp / "out3", lean=True),
        rundir=rd, console=False).resume()
    finished = json.loads((rd / "state.json").read_text())["completed"]
    assert finished == [p["name"] for p in cd.PHASES], finished
    print("PROVEN: declarative Run reproduces curriculum→league + observability (metrics/errors/figure) "
          "+ resume continues a dead run from saved state")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        import os, tempfile
        os.environ.setdefault("CODESIGN_OUT", tempfile.mkdtemp())
        _selftest()
    else:
        print(__doc__)
