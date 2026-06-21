# SPDX-License-Identifier: MIT
"""Layer 0 — the universal trace spine.

ONE structured `Event` model for every log line, metric, and error across the whole system —
orchestrator, kernel subprocess, ssh, GPU pod, background watchers. Trace context (run_id / stage /
attempt) flows ACROSS process & machine boundaries via `TRACE_*` env vars (set by `inject()` before
a subprocess/ssh call; read here at construction), so an event logged inside PPO on the A100 carries
the same run_id/stage as the orchestrator's `stage.start` on the laptop. Errors are captured at the
boundary where they surface (a `span` auto-emits a classified `error` Event on any exception) with
cause + symptom + the full stage context. Everything is append-only JSONL (machine-queryable +
greppable + mergeable local⊕pod by ts) AND rendered to a console line — never two logging systems.

  python -m arena.trace --selftest
"""

from __future__ import annotations

import json, os, re, sys, time, traceback
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path

LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}


@dataclass
class Event:
    ts: float
    run_id: str
    component: str
    level: str
    kind: str
    msg: str = ""
    stage: str = ""
    attempt: int = 0
    ctx: dict = field(default_factory=dict)
    payload: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, line: str) -> "Event":
        return cls(**json.loads(line))


# --- error classifier: rules distilled from THIS project's real failures -------------------------
_RULES = [
    (r"RESOURCE_EXHAUSTED|out of memory|\bOOM\b", "gpu_oom"),
    (r"cuSolver|CUBLAS|CUDA_ERROR|cudaError", "gpu_contention"),
    (r"no RECORD|externally[- ]managed", "pep668_pip"),
    (r"INTERNAL_SERVER_ERROR", "runpod_graphql"),
    (r"not found on the (registry|machine)", "stale_image"),
    (r"no longer any instances|does not have the resources", "gpu_capacity"),
    (r"FAILED rc=|PHASE .* FAILED|ROUND .* FAILED", "stage_subprocess_fail"),
    (r"scan.*pytree|carry.*mismatch", "brax_pytree_mismatch"),
    (r"Traceback|Error:", "exception"),
]


def classify(text: str = "", exit_code=None) -> str:
    """Map a symptom (stderr text and/or an exit code) to a named cause."""
    if exit_code is not None:
        if exit_code in (144, 128 + 16):
            return "signal_kill/self_pkill"          # the exit-144 self-matching pkill trap
        if exit_code == 137:
            return "oom_kill(sigkill)"
        if exit_code == 255:
            return "ssh_failure"
    for pat, cause in _RULES:
        if re.search(pat, text or "", re.I):
            return cause
    return "unknown" if (text or exit_code is not None) else "none"


class Tracer:
    """Per-run tracer: append-only JSONL sink + console renderer, with a context stack threaded by
    `span()`. Inherits trace context from `TRACE_*` env (set by a parent across a boundary)."""

    def __init__(self, run_id=None, sink=None, component="orchestrator", console=True, min_level="INFO"):
        self.run_id = run_id or os.environ.get("TRACE_RUN") or "run"
        self.sink = Path(sink) if sink else None
        self.component = component
        self.console = console
        self.min_level = LEVELS.get(min_level, 20)
        self._stage = os.environ.get("TRACE_STAGE", "")
        self._attempt = int(os.environ.get("TRACE_ATTEMPT", "0") or 0)
        self._ctx: dict = {}
        if self.sink:
            self.sink.parent.mkdir(parents=True, exist_ok=True)

    def _emit(self, kind, msg="", level="INFO", ctx=None, payload=None, component=None) -> Event:
        ev = Event(ts=time.time(), run_id=self.run_id, component=component or self.component,
                   level=level, kind=kind, msg=msg, stage=self._stage, attempt=self._attempt,
                   ctx={**self._ctx, **(ctx or {})}, payload=payload or {})
        if self.sink:
            with open(self.sink, "a") as f:
                f.write(ev.to_json() + "\n")
        if self.console and LEVELS.get(level, 20) >= self.min_level:
            print(self._render(ev), flush=True)
        return ev

    # public emit helpers ------------------------------------------------------
    def event(self, kind, msg="", level="INFO", **ctx):
        return self._emit(kind, msg, level, ctx=ctx)

    def metric(self, name, **vals):
        return self._emit("metric", msg=name, level="INFO", payload=vals)

    def error(self, msg, cause=None, exit_code=None, **payload):
        return self._emit("error", msg, "ERROR",
                          payload={"cause": cause or classify(msg, exit_code), **payload})

    @contextmanager
    def span(self, stage, attempt=0, **ctx):
        prev = (self._stage, self._attempt, self._ctx)
        self._stage, self._attempt, self._ctx = str(stage), int(attempt), {**self._ctx, **ctx}
        self._emit("stage.start", f"stage {stage}")
        t0 = time.time()
        try:
            yield self
        except BaseException as e:                    # capture the error AT the boundary, with ctx
            tb = traceback.format_exc()
            self._emit("error", f"{type(e).__name__}: {e}", "ERROR",
                       payload={"cause": classify(f"{e}\n{tb}"), "traceback": tb[-2000:]})
            raise
        finally:
            self._emit("stage.done", f"stage {stage} ({time.time()-t0:.1f}s)")
            self._stage, self._attempt, self._ctx = prev

    def _render(self, ev: Event) -> str:
        c = {"DEBUG": "·", "INFO": " ", "WARN": "!", "ERROR": "✗"}.get(ev.level, " ")
        loc = ev.stage + (f"/{ev.attempt}" if ev.attempt else "")
        if ev.kind == "metric":
            extra = " " + " ".join(f"{k}={v}" for k, v in ev.payload.items())
        elif ev.kind == "error":
            extra = f"  cause={ev.payload.get('cause')}"
        else:
            extra = ""
        return f"{c} [{ev.component}:{loc}] {ev.kind} {ev.msg}{extra}"


def inject(env: dict, stage, attempt=0, run_id=None) -> dict:
    """Return a child env with TRACE_* set, so a subprocess/ssh kernel stamps the SAME trace
    context (this is what makes the trace flow across process & machine boundaries)."""
    e = dict(env)
    if run_id:
        e["TRACE_RUN"] = str(run_id)
    e["TRACE_STAGE"] = str(stage)
    e["TRACE_ATTEMPT"] = str(attempt)
    return e


def merge(*paths) -> list:
    """Merge multiple JSONL event files (local ⊕ pod) into one ts-ordered list of dicts."""
    evs = []
    for p in paths:
        p = Path(p)
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        evs.append(json.loads(line))
                    except Exception:
                        pass
    return sorted(evs, key=lambda e: e.get("ts", 0.0))


def _selftest():
    import tempfile
    d = Path(tempfile.mkdtemp())
    sink = d / "events.jsonl"
    t = Tracer(run_id="selftest", sink=sink, console=False)
    t.event("run.start", "begin")
    t.metric("benchmark", best=-15.1, ratio=1.3, step=5_000_000)
    try:
        with t.span("c2", attempt=1, sep="0.4-1.0", opponent="passive"):
            raise RuntimeError("RESOURCE_EXHAUSTED: out of memory while allocating")
    except RuntimeError:
        pass
    # classifier
    assert classify("RESOURCE_EXHAUSTED") == "gpu_oom"
    assert classify(exit_code=144) == "signal_kill/self_pkill"
    assert classify("INTERNAL_SERVER_ERROR") == "runpod_graphql"
    assert classify("no RECORD file for blinker") == "pep668_pip"
    # jsonl round-trip + span auto-error-capture carries cause + ctx
    evs = [Event.from_json(l) for l in sink.read_text().splitlines()]
    kinds = [e.kind for e in evs]
    assert {"run.start", "metric", "stage.start", "error", "stage.done"} <= set(kinds), kinds
    err = next(e for e in evs if e.kind == "error")
    assert err.payload["cause"] == "gpu_oom", err.payload
    assert err.stage == "c2" and err.attempt == 1 and err.ctx.get("opponent") == "passive"
    # cross-boundary context inheritance via env
    env = inject(os.environ.copy(), "spr3", attempt=0, run_id="selftest")
    assert env["TRACE_STAGE"] == "spr3"
    # merge local ⊕ "pod"
    sink2 = d / "pod.jsonl"
    Tracer(run_id="selftest", sink=sink2, console=False).metric("throughput", steps_s=5500)
    m = merge(sink, sink2)
    assert len(m) == len(evs) + 1
    assert all(m[i]["ts"] <= m[i + 1]["ts"] for i in range(len(m) - 1))
    print("PROVEN: trace spine — structured events + span auto-error-capture + classifier + "
          "cross-boundary ctx + jsonl round-trip + merge")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
