# SPDX-License-Identifier: MIT
"""Phase 10 — the ACTUATOR backend contract (the architectural enabler for the motorloop spine).

The one thing we want swappable between the fast RL inner loop and the real-component validation is
the ACTUATOR: how a commanded action becomes *deliverable* torque. An `Actuator` maps joint speed →
the per-joint fraction of the static torque limit the drive can actually produce at that speed. Two
backends today:
  * `IdealActuator` — delivers full commanded torque at any speed (the idealized baseline).
  * `FocActuator`   — the real motorloop FOC/motor envelope (back-EMF + current limit + voltage sag +
                      thermal derate + gear efficiency), parameterized from a `motors.py` datasheet.
                      This is the component-IP-grounded actuator the RTL gate (Phase 11) validates against.

The env's `_ctrl_scale` already multiplies the action by exactly this fraction when `reality_gap` is
on — so `FocActuator` is that model made first-class + connectable to the verified FOC RTL.

  python -m arena.backend --selftest
"""

from __future__ import annotations

import dataclasses as dc
import sys
from pathlib import Path
import numpy as np


class Actuator:
    """Contract: `scale(joint_vel) -> per-joint deliverable-torque fraction in [0, ~1]`."""
    name = "actuator"
    def scale(self, joint_vel):
        raise NotImplementedError
    def params(self) -> dict:
        return {"model": self.name}


class IdealActuator(Actuator):
    name = "ideal"
    def scale(self, joint_vel):
        jv = np.asarray(joint_vel, dtype=float)
        return np.ones_like(jv)               # full commanded torque, any speed (the inner-loop assumption)


class FocActuator(Actuator):
    """The real FOC/motor torque-speed envelope (reality_gap.actuator_scale) — the bridge to silicon."""
    name = "foc"

    def __init__(self, dp: dict):
        self.dp = dp

    def scale(self, joint_vel):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from reality_gap import actuator_scale
        return np.asarray(actuator_scale(np.asarray(joint_vel, dtype=float), self.dp), dtype=float)

    def params(self) -> dict:
        return dict(self.dp)

    @classmethod
    def from_motor(cls, motor: str = "db42s03", gear: float = 6.0) -> "FocActuator":
        """Build the NOMINAL envelope from a `motors.py` datasheet (the center of each calibrated range)."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from reality_gap import default_uncertainty
        unc = default_uncertainty(motor, gear)
        dp = {"motor": motor, "gear": gear}
        for f in dc.fields(unc["act"]):
            dp[f.name] = float(getattr(unc["act"], f.name)[0])    # range = (center, lo, hi) -> center
        return cls(dp)


def _selftest():
    ideal = IdealActuator()
    foc = FocActuator.from_motor("db42s03", gear=6.0)
    dp = foc.params()
    no_load = dp["vbus"] / (dp["ke_phase"] * dp["gear"])         # joint speed where back-EMF kills torque
    speeds = no_load * np.array([0.0, 0.5, 0.85, 0.97])          # span standstill -> near no-load
    si = ideal.scale(speeds)
    sf = foc.scale(speeds)
    assert np.allclose(si, 1.0), si                              # ideal delivers full torque always
    # both satisfy the contract
    for a in (ideal, foc):
        assert hasattr(a, "scale") and isinstance(a.params(), dict) and a.name
    # the FOC envelope DROOPS with speed (back-EMF) and is bounded — the real-component behavior
    assert sf[0] > sf[-1] + 1e-3, sf                             # high speed delivers LESS than standstill
    assert 0.0 <= sf.min() and sf.max() <= 1.05, sf
    # it's grounded in the datasheet (params carry the real motor constants)
    p = foc.params()
    assert "ke_phase" in p and "i_limit" in p and p["gear"] == 6.0, p
    print(f"FOC envelope (db42s03, gear6) deliverable fraction vs speed {speeds.tolist()}: "
          f"{[round(float(x), 3) for x in sf]}")
    print("PROVEN: actuator backend contract — IdealActuator (full torque) + FocActuator (real "
          "back-EMF/current-limit envelope, datasheet-grounded) plug into one interface")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
