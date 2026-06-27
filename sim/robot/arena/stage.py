# SPDX-License-Identifier: MIT
"""Layer 1 — `Stage`: a declarative unit of work that compiles to a `train_adversarial` (kernel)
invocation. Curriculum phases and self-play rounds are both just Stages with different fields, so
the schedulers above produce Stages and the runners below execute them — the kernel CLI lives in
exactly ONE place.

  python -m arena.stage --selftest
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BENCH = dict(sep_lo=0.4, sep_hi=1.2, az=3.14159, epis=16, steps=200)


@dataclass
class Stage:
    tag: str
    sep_lo: float = 0.4
    sep_hi: float = 1.2
    approach: float = 0.0
    azimuth: float = 0.0
    shaping: float = 1.0
    clean: float = 0.0
    trade: float = 0.0
    disengage: float = 0.0
    fire: float = 0.0
    upright: float = 2.0       # BALANCE anchor (rich xmat uprightness; raised 0.6->2.0 so it isn't overwhelmed
                               # by approach=4 and lunge-sprawls; the QDD body has the torque to honor it.
                               # Coach raises further when survival lags.
    energy_penalty: float = 0.0  # ENERGY/actuator-safety (Coach competency lever — raise when slamming)
    steps: int = 4_000_000
    opponent: str = "passive"          # passive | frozen
    opp_ckpt: str | None = None
    bench_opp_ckpt: str | None = None
    bench: dict = field(default_factory=lambda: dict(DEFAULT_BENCH))
    gate_tol: float = 2.0              # accept if bench >= global_best - gate_tol

    def flags(self, warm=None, cum_base=0, envs=0, batch=0, lean=False, tiny=False) -> list:
        """The `train_adversarial` argv (after the script path) — the ONE place the kernel CLI is built."""
        f = ["--tag", self.tag, "--steps", str(self.steps), "--cum-base", str(cum_base)]
        if envs:
            f += ["--envs", str(envs)]
        if batch:
            f += ["--batch", str(batch)]
        f += ["--sep-lo", str(self.sep_lo), "--sep-hi", str(self.sep_hi),
              "--approach-weight", str(self.approach), "--azimuth", str(self.azimuth),
              "--shaping", str(self.shaping), "--clean-weight", str(self.clean),
              "--trade-weight", str(self.trade), "--disengage-weight", str(self.disengage),
              "--fire-shaping", str(self.fire), "--upright-weight", str(self.upright),
              "--energy-penalty", str(self.energy_penalty),
              "--bench-sep-lo", str(self.bench["sep_lo"]), "--bench-sep-hi", str(self.bench["sep_hi"]),
              "--bench-az", str(self.bench["az"]), "--bench-epis", str(self.bench["epis"]),
              "--bench-steps", str(self.bench["steps"])]
        if self.opponent == "frozen":
            f += ["--opponent", "frozen", "--opp-ckpt", str(self.opp_ckpt),
                  "--bench-opp-ckpt", str(self.bench_opp_ckpt or self.opp_ckpt)]
        if warm:
            f += ["--resume", str(warm)]
        if lean:
            f += ["--lean-contacts"]
        if tiny:
            f += ["--tiny"]
        return f

    # --- constructors that consume the existing drivers' configs (faithful round-trip) ----------
    @classmethod
    def from_curriculum(cls, ph: dict, steps: int) -> "Stage":
        return cls(tag=ph["name"], sep_lo=ph["sep_lo"], sep_hi=ph["sep_hi"], approach=ph["approach"],
                   azimuth=ph["azimuth"], shaping=ph["shaping"], clean=ph["clean"], trade=ph["trade"],
                   disengage=ph["disengage"], fire=ph.get("fire", 0.0), steps=steps)

    @classmethod
    def from_league(cls, rd: int, opp: str, bench_opp: str, steps: int, rw: dict) -> "Stage":
        return cls(tag=f"spr{rd}", opponent="frozen", opp_ckpt=opp, bench_opp_ckpt=bench_opp, steps=steps,
                   sep_lo=rw["sep_lo"], sep_hi=rw["sep_hi"], approach=rw["approach"], azimuth=rw["azimuth"],
                   shaping=rw["shaping"], clean=rw["clean"], trade=rw["trade"], disengage=rw["disengage"],
                   fire=rw["fire"])


def flags_to_dict(flags: list) -> dict:
    d, i = {}, 0
    while i < len(flags):
        f = flags[i]
        if f.startswith("--"):
            if i + 1 < len(flags) and not flags[i + 1].startswith("--"):
                d[f] = flags[i + 1]; i += 2
            else:
                d[f] = True; i += 1
        else:
            i += 1
    return d


def _selftest():
    import os, tempfile
    os.environ.setdefault("CODESIGN_OUT", tempfile.mkdtemp())     # kernel mkdir's OUT on import
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # sim/robot
    import curriculum_drive as cd
    import selfplay_drive as sd
    # (1) a curriculum phase -> Stage -> flags round-trips the driver's phase config + BENCH
    ph = cd.PHASES[2]                                            # "c1"
    st = Stage.from_curriculum(ph, steps=10_000_000)
    d = flags_to_dict(st.flags(warm="cval_best.pkl", cum_base=20_000_000, envs=8192, lean=True))
    assert d["--tag"] == ph["name"] and d["--steps"] == "10000000" and d["--cum-base"] == "20000000"
    assert d["--sep-lo"] == str(ph["sep_lo"]) and d["--sep-hi"] == str(ph["sep_hi"])
    assert d["--clean-weight"] == str(ph["clean"]) and d["--fire-shaping"] == str(ph["fire"])
    assert d["--bench-sep-lo"] == str(cd.BENCH["sep_lo"]) and d["--bench-epis"] == str(cd.BENCH["epis"])
    assert d["--envs"] == "8192" and d["--resume"] == "cval_best.pkl" and d["--lean-contacts"] is True
    assert "--opponent" not in d                                 # curriculum = passive
    # (2) a league round -> Stage -> flags carries the frozen-opponent self-play config
    sr = Stage.from_league(3, opp="spr1_best.pkl", bench_opp="selfplay_seed.pkl", steps=10_000_000, rw=sd.RW)
    d2 = flags_to_dict(sr.flags(warm="selfplay_best.pkl", envs=8192, lean=True))
    assert d2["--opponent"] == "frozen" and d2["--opp-ckpt"] == "spr1_best.pkl"
    assert d2["--bench-opp-ckpt"] == "selfplay_seed.pkl" and d2["--tag"] == "spr3"
    assert d2["--clean-weight"] == str(sd.RW["clean"]) and d2["--sep-hi"] == str(sd.RW["sep_hi"])
    print("PROVEN: Stage round-trips both drivers' exact kernel CLIs (curriculum + self-play)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
