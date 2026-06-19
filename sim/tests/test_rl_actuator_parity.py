# SPDX-License-Identifier: MIT
"""RL inner-loop parity (notes/rl-environment-checklist.md §6): the fast
actuator model used by the RL env must agree with the cycle-accurate co-sim
bench for a single joint - the trust anchor that what the policy learns will
transfer.

Runs in the system-python suite (the actuator is pure numpy; the bench is the
co-sim). Uses the DB42S03 motor because it is 4 pole pairs = the current
POLE_PAIRS build, so no re-Verilate is needed. The actuator is an *idealized*
envelope (ideal current control up to the back-EMF limit); it matches the real
FOC in the operating regime and is optimistic at extreme speed where the real
loop saturates - which is exactly why the verification tier exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from bench_factory import expected_init_time, foc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rl"))
from actuator import MotorActuator  # noqa: E402
from motors import MOTORS  # noqa: E402

VBUS = 12.0
ILIM = 8.0   # ~ Vbus/R for the DB42 (the bench's achievable stall current)


def _bench_peak_current(bldcsim, params, omega):
    """Median peak phase-current amplitude (A) at a clamped speed under a
    max-torque command, with the DB42 plant."""
    m = MOTORS["db42s03"]
    cfg = foc(params)
    cfg["motor"].update(m.cfg_motor(cfg["motor"]))
    b = bldcsim.Bench(cfg)
    b.run_for(expected_init_time(params))
    b.set_align_offset(int(params.value("foc.align_offset")))
    b.set_id_target(0)
    b.set_iq_target(2000)              # demand max torque -> back-EMF-limited
    b.set_speed_clamp(True, float(omega))
    b.set_mode(3)
    b.run_for(0.06)
    samp = []
    for _ in range(200):
        b.run_for(2e-5)
        samp.append(float(np.abs(b.currents).max()))
    return float(np.median(samp))


def test_actuator_envelope_matches_bench(bldcsim, params):
    act = MotorActuator(MOTORS["db42s03"], gear_ratio=1.0, vbus=VBUS,
                        current_limit_a=ILIM)

    def i_avail(w):  # actuator's predicted achievable current (A)
        return float(act._motor_torque(w) / act.motor.kt)

    # Operating regime (<= ~half the no-load speed): the reduced model and the
    # cycle-accurate FOC agree within tolerance.
    for omega, tol in [(20.0, 0.30), (60.0, 0.30)]:
        ib = _bench_peak_current(bldcsim, params, omega)
        ia = i_avail(omega)
        assert abs(ib - ia) <= tol * max(ia, 1e-6), (
            f"omega={omega}: bench {ib:.2f} A vs actuator {ia:.2f} A")

    # Both exhibit the back-EMF rolloff: current at 120 < current at 20.
    assert _bench_peak_current(bldcsim, params, 120.0) \
        < 0.9 * _bench_peak_current(bldcsim, params, 20.0)
    assert i_avail(120.0) < i_avail(20.0)


def test_motor_profiles_give_distinct_envelopes():
    """The three motors give materially different torque-speed envelopes (the
    RL<->motor coupling): the gimbal's knee is far lower than the servo's."""
    knees = {n: MotorActuator(MOTORS[n], gear_ratio=6.0, vbus=VBUS)
             .base_joint_speed_rad_s() for n in ("gm2804", "db42s03", "maxon_ec45")}
    assert knees["gm2804"] < knees["db42s03"] < knees["maxon_ec45"]
