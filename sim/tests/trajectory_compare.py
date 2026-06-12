"""Tolerance-based trajectory comparison shared by parity tests.

Trajectories are dicts of equal-length lists keyed by column name, always
including "time_s". Two comparison modes:

- compare_aligned: same time grid expected; strict per-sample bands.
- compare_interpolated: resamples b onto a's time grid first (for oracle
  runs whose solver picks its own output points).
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field


@dataclass
class Mismatch:
    column: str
    time_s: float
    a: float
    b: float
    error: float
    bound: float


@dataclass
class CompareResult:
    ok: bool
    worst: dict[str, float] = field(default_factory=dict)
    mismatches: list[Mismatch] = field(default_factory=list)

    def report(self, limit: int = 10) -> str:
        lines = [f"trajectory compare: {'OK' if self.ok else 'FAIL'}"]
        for col, err in sorted(self.worst.items()):
            lines.append(f"  worst |err| {col}: {err:.3e}")
        for m in self.mismatches[:limit]:
            lines.append(
                f"  MISMATCH {m.column} @ t={m.time_s:.6f}: "
                f"a={m.a:.6g} b={m.b:.6g} |err|={m.error:.3e} > {m.bound:.3e}"
            )
        if len(self.mismatches) > limit:
            lines.append(f"  ... {len(self.mismatches) - limit} more")
        return "\n".join(lines)


def _bound(ref: float, abs_tol: float, rel_tol: float) -> float:
    return abs_tol + rel_tol * abs(ref)


def _interp(times: list[float], values: list[float], t: float) -> float:
    i = bisect_left(times, t)
    if i <= 0:
        return values[0]
    if i >= len(times):
        return values[-1]
    t0, t1 = times[i - 1], times[i]
    v0, v1 = values[i - 1], values[i]
    if t1 == t0:
        return v1
    frac = (t - t0) / (t1 - t0)
    return v0 + frac * (v1 - v0)


def compare_aligned(
    a: dict[str, list[float]],
    b: dict[str, list[float]],
    columns: list[str],
    abs_tol: float = 1e-9,
    rel_tol: float = 1e-9,
) -> CompareResult:
    result = CompareResult(ok=True)
    assert len(a["time_s"]) == len(b["time_s"]), (
        f"length mismatch: {len(a['time_s'])} vs {len(b['time_s'])}"
    )
    for col in columns:
        worst = 0.0
        for t, va, vb in zip(a["time_s"], a[col], b[col]):
            err = abs(va - vb)
            worst = max(worst, err)
            bound = _bound(va, abs_tol, rel_tol)
            if err > bound:
                result.ok = False
                result.mismatches.append(Mismatch(col, t, va, vb, err, bound))
        result.worst[col] = worst
    return result


def compare_interpolated(
    a: dict[str, list[float]],
    b: dict[str, list[float]],
    columns: list[str],
    abs_tol: float,
    rel_tol: float,
    skip_initial_s: float = 0.0,
) -> CompareResult:
    """Compare b (own time grid) against a, linearly interpolating b.

    skip_initial_s ignores the start-up interval where solvers with event
    handling and fixed-step integrators legitimately differ the most.
    """
    result = CompareResult(ok=True)
    b_times = b["time_s"]
    for col in columns:
        worst = 0.0
        for t, va in zip(a["time_s"], a[col]):
            if t < skip_initial_s:
                continue
            vb = _interp(b_times, b[col], t)
            err = abs(va - vb)
            worst = max(worst, err)
            bound = _bound(va, abs_tol, rel_tol)
            if err > bound:
                result.ok = False
                result.mismatches.append(Mismatch(col, t, va, vb, err, bound))
        result.worst[col] = worst
    return result


def rows_to_columns(rows: list[dict[str, float]]) -> dict[str, list[float]]:
    """Convert the Python runner's list-of-dicts rows to column dict form."""
    if not rows:
        return {}
    return {key: [row[key] for row in rows] for key in rows[0]}
