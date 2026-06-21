# SPDX-License-Identifier: MIT
"""Layer 2 (schedules) — a `Schedule` yields the next `Stage` given the run state, and folds in its
transition policy. The two drivers collapse into TWO tiny schedules over ONE engine:
  * `Curriculum` — the sep-range phases (opponent=passive); gate + rollback (gentler widening).
  * `League`     — self-play rounds; training opponent sampled from the FIRST QUARTER of the HoF
                   (anti-cycling), benchmark vs a FIXED seed; archives each round's best.
  * `Pipeline`   — run schedules in sequence sharing the best-so-far (skill curriculum THEN self-play).

Phase configs are read from the EXISTING drivers (curriculum_drive.PHASES / selfplay_drive.RW) so
there is still one source of truth until the Phase-7 migration.

  python -m arena.schedule --selftest
"""

from __future__ import annotations

import sys
from pathlib import Path

from arena.stage import Stage  # noqa: E402


def _curriculum_phases():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import curriculum_drive as cd
    return cd.PHASES, cd.BENCH


def _league_rw():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import selfplay_drive as sd
    return sd.RW


def first_quarter(hof: list) -> list:
    return hof[:max(1, (len(hof) + 3) // 4)]


class Schedule:
    name = "schedule"
    def next(self, state): raise NotImplementedError
    def on_done(self, state, stage, res, best_before): pass
    def seed_ckpt(self, state): return None


class Curriculum(Schedule):
    name = "curriculum"

    def __init__(self, steps_per_phase=4_000_000, tol=0.05, retries=1, phases=None):  # tol = win-rate scale
        ph, _ = _curriculum_phases() if phases is None else (phases, None)
        self.phases = ph
        self.steps = steps_per_phase
        self.tol = tol
        self.retries = retries

    def next(self, state):
        if state.idx >= len(self.phases):
            return None
        ph = self.phases[state.idx]
        sep_hi = state.extra.get("sep_hi", ph["sep_hi"])      # may be narrowed by a rollback
        st = Stage.from_curriculum({**ph, "sep_hi": sep_hi}, self.steps)
        st.gate_tol = self.tol
        return st

    def on_done(self, state, stage, res, best_before):
        ph = self.phases[state.idx]
        if res["best_bench"] >= best_before - self.tol:        # GATE pass -> advance
            state.completed.append(ph["name"])
            state.idx += 1
            state.extra.pop("sep_hi", None); state.extra["attempt"] = 0
        else:                                                  # ROLLBACK -> retry with gentler widening
            a = state.extra.get("attempt", 0) + 1
            if a > self.retries:
                state.idx = len(self.phases)                   # give up; global best is preserved
            else:
                state.extra["attempt"] = a
                state.extra["sep_hi"] = ph["sep_lo"] + 0.5 * (stage.sep_hi - ph["sep_lo"])


class League(Schedule):
    name = "league"

    def __init__(self, seed, rounds=12, round_steps=10_000_000, tol=0.05, rw=None):  # tol = win-rate scale
        self.seed = str(seed)
        self.rounds = rounds
        self.steps = round_steps
        self.tol = tol
        self.rw = rw or _league_rw()

    def seed_ckpt(self, state):
        return self.seed

    def next(self, state):
        if state.round >= self.rounds:
            return None
        if not state.hof:
            state.hof = [self.seed]                             # seed the Hall of Fame
        # EXTENSION 1: the Coach's opp_difficulty grows the sampling window from the first quarter
        # (oldest/easiest — the anti-cycling floor) toward the whole HoF (recent/toughest snapshots).
        d = float(state.extra.get("opp_difficulty", 0.25))
        window = max(1, int((0.25 + 0.75 * d) * len(state.hof)))
        pool = state.hof[:window]
        opp = pool[state.round % len(pool)]                     # deterministic (resume-safe) pick
        return Stage.from_league(state.round, opp, self.seed, self.steps, self.rw)

    def on_done(self, state, stage, res, best_before):
        state.hof.append(res["best_ckpt"])                     # archive this round's best as an opponent
        state.round += 1


class Pipeline(Schedule):
    name = "pipeline"

    def __init__(self, schedules):
        self.schedules = list(schedules)

    def _cur(self, state):
        return self.schedules[state.sched_idx] if state.sched_idx < len(self.schedules) else None

    def next(self, state):
        while state.sched_idx < len(self.schedules):
            st = self.schedules[state.sched_idx].next(state)
            if st is not None:
                return st
            state.sched_idx += 1                               # sub-schedule exhausted -> advance
        return None

    def on_done(self, state, stage, res, best_before):
        self._cur(state).on_done(state, stage, res, best_before)

    def seed_ckpt(self, state):
        s = self._cur(state)
        return s.seed_ckpt(state) if s else None
