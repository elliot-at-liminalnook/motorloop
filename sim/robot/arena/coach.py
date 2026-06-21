# SPDX-License-Identifier: MIT
"""The `Coach` — an automatic competency controller that replaces hand-tuned reward weights.

Instead of fixing `clean_weight`/`fire_shaping`/... by hand, the Coach measures each competency from
the held-out benchmark decomposition and moves its reward weight by a CONTROL LOOP toward a target:
the laggard's weight RISES, a satisfied competency's weight DECAYS (so it can't be over-optimized
into a degenerate optimum), and a STUCK competency (lagging but not improving despite a heavy weight
— currently unlearnable) BACKS OFF (don't pour reward into an unlearnable hole). Updates are
normalized per gauge (scale-free), so a target is an interpretable setpoint ("clean ≥ 0.1"), not a
magic weight. The Coach WRAPS a base `Schedule` (Curriculum/League/Pipeline): it injects the current
weights into each Stage and updates them from the previous stage's result.

This is the continuous-weight form of SAC-X / GradNorm / learning-progress curricula, driven by our
held-out signals. The brittle constants become a transient, self-annealing scaffold over the one
true objective (winning the exchange).

  python -m arena.coach --selftest
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable

from arena.schedule import Schedule  # noqa: E402


@dataclass
class Lever:
    field: str                 # the Stage reward field this controls (clean/trade/disengage/fire/approach)
    gauge: Callable            # signals(dict) -> competency value (HIGHER = better)
    target: float              # setpoint for the gauge
    lo: float = 0.0
    hi: float = 6.0
    lr: float = 0.4            # unitless gain (deficit normalized by gauge scale)
    w: float | None = None     # current weight (init = midpoint)
    _prev: float | None = None
    _tract: float = 0.5        # tractability — EMA of |progress|, starts OPTIMISTIC (give laggards a try)

    def init(self):
        if self.w is None:
            self.w = 0.5 * (self.lo + self.hi)
        return self


class Coach(Schedule):
    name = "coach"

    def __init__(self, base, levers, gate_eps=1e-3):
        self.base = base
        self.levers = [l.init() for l in levers]
        self.gate_eps = gate_eps
        self.last_weights = {l.field: l.w for l in self.levers}

    # --- the controller (pure; directly unit-testable) ---------------------------------------
    # For each competency: deficit = (target - gauge)/scale (scale-free). RAISE the weight ∝ deficit
    # but GATED by tractability (is it actually improving when attended?) so we don't pour reward into
    # a currently-unlearnable hole; DECAY whenever satisfied (gauge ≥ target) so it can't be
    # over-optimized into a degenerate optimum; gently RELEASE a heavy-but-stuck weight to free budget.
    def update(self, signals: dict) -> dict:
        for l in self.levers:
            g = float(l.gauge(signals))
            scale = max(abs(g), abs(l.target), 1e-3)
            prog = (g - l._prev) / scale if l._prev is not None else 0.0
            l._tract = 0.6 * l._tract + 0.4 * min(1.0, abs(prog) / 0.05)        # learning-progress, EMA
            deficit = (l.target - g) / scale
            span = l.hi - l.lo
            if deficit <= 0:                                                    # SATISFIED -> decay
                delta = l.lr * deficit * span
            else:                                                              # LAGGING -> raise, gated by progress
                delta = l.lr * deficit * span * l._tract
                if l.w > 0.6 * l.hi and l._tract < 0.1:                         # heavy + stuck -> release
                    delta = -0.05 * span
            l.w = min(l.hi, max(l.lo, l.w + delta))
            l._prev = g
        self.last_weights = {l.field: round(l.w, 3) for l in self.levers}
        return self.last_weights

    # --- Schedule wrapper --------------------------------------------------------------------
    def next(self, state):
        st = self.base.next(state)
        if st is None:
            return None
        for l in self.levers:                       # inject the controller's current weights
            setattr(st, l.field, round(l.w, 3))
        return st

    def on_done(self, state, stage, res, best_before):
        sig = (res or {}).get("signals") or {}
        if sig:
            state.extra["coach_weights"] = self.update(sig)    # engine emits this as a 'coach' metric
        self.base.on_done(state, stage, res, best_before)

    def seed_ckpt(self, state):
        return self.base.seed_ckpt(state)

    # default competency set over the held-out decomposition (targets are interpretable setpoints)
    @classmethod
    def default(cls, base):
        levers = [
            Lever("clean", lambda s: s.get("clean", 0.0), target=0.12, lo=2, hi=8, lr=0.4),       # land un-traded hits
            Lever("trade", lambda s: -s.get("trade", 0.0), target=-0.03, lo=1, hi=8, lr=0.4),     # punish trading when high
            Lever("approach", lambda s: s.get("closing", 0.0) - s.get("fleeing", 0.0),
                  target=0.0, lo=0, hi=3, lr=0.35),                                                # the anti-retreat lever
            Lever("fire", lambda s: s.get("fire", 0.0), target=1.5, lo=0, hi=3, lr=0.35),         # use the rod
        ]
        return cls(base, levers)


# ---------------------------------------------------------------------------------------------
def _selftest():
    # three synthetic competencies: LAGGING-then-RECOVERING, already-SATISFIED, permanently-STUCK
    lag = Lever("clean", lambda s: s["lag"], target=1.0, lo=0, hi=4, lr=0.4)
    sat = Lever("trade", lambda s: s["sat"], target=1.0, lo=0, hi=4, lr=0.4)
    stk = Lever("fire", lambda s: s["stk"], target=1.0, lo=0, hi=4, lr=0.4)
    coach = Coach(base=None, levers=[lag, sat, stk])
    w0 = {l.field: l.w for l in coach.levers}
    traj = {l.field: [l.w] for l in coach.levers}
    lagvals = [0.0, 0.0, 0.0, 0.0, 1.5, 1.5, 1.5]            # lags, then recovers above target
    for lv in lagvals:
        coach.update({"lag": lv, "sat": 2.0, "stk": 0.0})    # sat always satisfied; stk never moves
        for l in coach.levers:
            traj[l.field].append(l.w)
    # 1. the lagging competency's weight ROSE while it lagged
    assert max(traj["clean"][:5]) > w0["clean"] + 0.1, traj["clean"]
    # 2. once it RECOVERED (above target), its weight DECAYED
    assert traj["clean"][-1] < max(traj["clean"][:5]) - 0.1, traj["clean"]
    # 3. the already-SATISFIED competency's weight DECAYED (focus went to the laggard)
    assert traj["trade"][-1] < w0["trade"], traj["trade"]
    # 4. the STUCK competency BACKED OFF (progress gate) — never pinned to the ceiling
    assert max(traj["fire"]) < stk.hi and traj["fire"][-1] < max(traj["fire"]), traj["fire"]
    # bounded throughout
    assert all(l.lo <= w <= l.hi for l in coach.levers for w in traj[l.field])
    # FOCUS: during the lag phase the laggard outweighed the satisfied competency
    assert traj["clean"][4] > traj["trade"][4], (traj["clean"][4], traj["trade"][4])

    # integration: Coach wraps a base schedule -> injects weights into the Stage + updates from signals
    from arena.stage import Stage
    class StubBase(Schedule):
        def __init__(self): self.n = 0
        def next(self, state):
            self.n += 1
            return Stage(tag=f"s{self.n}") if self.n <= 2 else None
        def on_done(self, state, stage, res, best_before): pass
        def seed_ckpt(self, state): return "seed.pkl"
    from arena.engine import RunState
    c2 = Coach.default(StubBase()); stt = RunState()
    s1 = c2.next(stt)
    assert s1.clean == round(c2.levers[0].w, 3) and s1.fire == round(c2.levers[3].w, 3)  # weights injected
    c2.on_done(stt, s1, {"signals": {"clean": 0.0, "trade": 0.0, "fire": 0.0, "closing": 0.0, "fleeing": 5.0}}, -1e30)
    assert stt.extra.get("coach_weights"), "coach did not record weights"
    s2 = c2.next(stt)
    assert s2.fire >= s1.fire and s2.approach >= s1.approach   # under-fire + net-retreat -> those weights rose
    print("PROVEN: Coach — raises lagging weight, decays it on recovery, backs off the stuck, focuses "
          "off satisfied, stays bounded; wraps a schedule (injects weights, updates from signals)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
