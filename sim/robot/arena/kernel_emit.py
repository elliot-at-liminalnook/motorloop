# SPDX-License-Identifier: MIT
"""Phase 2 — the thin shim the KERNEL (train_adversarial) and the WATCHER use to emit into the
universal trace stream, BEFORE the full framework refactor (immediate value).

The kernel calls `emit_metric/emit_event/emit_error`; they lazily build a `Tracer` from the env a
parent set across the boundary — `ARENA_SINK` (where to write) + `TRACE_RUN/STAGE/ATTEMPT` (the
context). Absent `ARENA_SINK` ⇒ no-op, so the kernel runs standalone unchanged. The watcher uses
`classify()` so the LIVE run's failures get named causes now.

  python -m arena.kernel_emit --selftest
"""

from __future__ import annotations

import os, sys
from pathlib import Path

from arena.trace import Tracer, classify  # noqa: E402

_TRACER = None
_BUILT = False


def _tracer():
    global _TRACER, _BUILT
    if not _BUILT:
        _BUILT = True
        sink = os.environ.get("ARENA_SINK")
        _TRACER = Tracer(sink=sink, component="kernel", console=False) if sink else None
    return _TRACER


def emit_metric(name, **vals):
    t = _tracer()
    if t:
        t.metric(name, **vals)


def emit_event(kind, msg="", level="INFO", **ctx):
    t = _tracer()
    if t:
        t.event(kind, msg, level, **ctx)


def emit_error(msg, cause=None, exit_code=None, **payload):
    t = _tracer()
    if t:
        t.error(msg, cause=cause, exit_code=exit_code, **payload)


def _selftest():
    import json, tempfile
    d = Path(tempfile.mkdtemp())
    os.environ["ARENA_SINK"] = str(d / "events.jsonl")
    os.environ["CODESIGN_OUT"] = str(d / "out")
    os.environ["TRACE_RUN"] = "selftest"; os.environ["TRACE_STAGE"] = "c2"; os.environ["TRACE_ATTEMPT"] = "0"
    # the shim emits trace-stamped events
    emit_event("stage.start", "kernel begin")
    emit_metric("benchmark", best=-15.1, dealt=0.08, taken=0.06, ratio=1.33)
    emit_error("RESOURCE_EXHAUSTED: out of memory")
    # the REAL kernel is wired: importing it + calling its METRIC emits into the same stream
    import importlib
    ta = importlib.import_module("train_adversarial")
    ta.METRIC(stage="warm_start", ok=1, resume="cval_ckpt.pkl")
    evs = [json.loads(l) for l in Path(os.environ["ARENA_SINK"]).read_text().splitlines()]
    assert all(e["run_id"] == "selftest" and e["component"] == "kernel" for e in evs), evs
    assert all(e["stage"] == "c2" for e in evs), [e["stage"] for e in evs]
    bench = next(e for e in evs if e["kind"] == "metric" and e["msg"] == "benchmark")
    assert bench["payload"]["ratio"] == 1.33
    err = next(e for e in evs if e["kind"] == "error")
    assert err["payload"]["cause"] == "gpu_oom", err["payload"]
    assert any(e["msg"] == "warm_start" or e.get("payload", {}).get("ok") == 1 for e in evs), \
        "kernel METRIC did not emit"
    print("PROVEN: kernel + watcher emit into the universal stream (trace-stamped; classifier live)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # sim/robot for train_adversarial
        _selftest()
    else:
        print(__doc__)
