# SPDX-License-Identifier: MIT
"""Motor + FOC inner-loop actuator model for the RL env
(notes/rl-environment-checklist.md §1).

Pure-numpy (no gym/torch deps) so it imports in BOTH the rl-venv (the env) and
the system Python (the parity test against the cycle-accurate bench). Reuses the
motor profiles in sim/tests/motors.py: the policy's high-level command is rolled
off by the chosen motor's real **torque-speed envelope** (back-EMF current droop
+ current limit), so the agent learns under realistic motor limits - and the
motor choice (GM2804 / DB42S03 / EC 45) becomes part of the environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from motors import MOTORS, Motor  # noqa: E402


class MotorActuator:
    """One joint's BLDC + FOC inner loop, reduced to its torque-speed envelope.

    available motor torque at motor-shaft speed w_m:
        I = clip((Vbus - Ke*w_m)/R_ll, 0, I_limit);  tau_m = Kt * I
    joint torque = tau_m * gear * efficiency; the policy action is scaled by
    `torque_scale(w_joint)` = available/stall, in [0, 1].
    """

    def __init__(self, motor: Motor, gear_ratio: float = 1.0, vbus: float = 12.0,
                 efficiency: float = 0.9, current_limit_a: float | None = None,
                 peak_factor: float = 4.0):
        self.motor = motor
        self.gear = gear_ratio
        self.vbus = vbus
        self.eff = efficiency
        # Robots use PEAK (transient) torque, not the continuous rating - so the
        # default current limit is a peak multiple of rated, capped by what the
        # bus can drive (Vbus/R). Override with current_limit_a.
        self.i_limit = (current_limit_a if current_limit_a else
                        min(peak_factor * motor.rated_current_a,
                            vbus / motor.r_line_line))

    def _motor_torque(self, w_motor):
        w_motor = np.asarray(w_motor, dtype=float)
        i = np.clip((self.vbus - self.motor.ke_phase * np.abs(w_motor))
                    / self.motor.r_line_line, 0.0, self.i_limit)
        return self.motor.kt * i

    def stall_joint_torque(self) -> float:
        return float(self._motor_torque(0.0) * self.gear * self.eff)

    def joint_torque_cap(self, w_joint):
        """Max deliverable joint torque at joint speed w_joint (N*m)."""
        return self._motor_torque(np.asarray(w_joint) * self.gear) * self.gear * self.eff

    def torque_scale(self, w_joint):
        """Fraction of stall joint torque available at this joint speed, [0,1] -
        the speed-dependent rolloff the policy must respect."""
        stall = self.stall_joint_torque()
        return np.clip(self.joint_torque_cap(w_joint) / max(stall, 1e-9), 0.0, 1.0)

    def base_joint_speed_rad_s(self) -> float:
        """Joint speed at which back-EMF starts dropping torque below the current
        limit (the envelope 'knee')."""
        m = self.motor
        w_motor_knee = max((self.vbus - self.i_limit * m.r_line_line)
                           / m.ke_phase, 0.0)
        return w_motor_knee / self.gear


def make_actuator(name: str, **kw) -> MotorActuator:
    return MotorActuator(MOTORS[name], **kw)


if __name__ == "__main__":
    # quick envelope sanity per motor
    for n in ("gm2804", "db42s03", "maxon_ec45"):
        a = make_actuator(n, gear_ratio=6.0)
        ws = np.array([0, 5, 10, 20, 40])
        print(f"{n:11} stall={a.stall_joint_torque():.3f} Nm  knee={a.base_joint_speed_rad_s():.1f} rad/s  "
              f"scale@[0,5,10,20,40]={np.round(a.torque_scale(ws),2)}")
