# SPDX-License-Identifier: MIT
"""FOC stage 7: AS5600 angle-latency compensation (Q22).

The AS5600's frame + filter latency makes the reported angle lag the true
rotor angle; for FOC that rotates the dq frame off-true, and the error grows
with speed (cos(lag) torque loss + cross-coupling). The RTL optionally
advances the angle by omega_e * t_latency to undo it.

The study clamps the rotor at a series of speeds (dyno), commands a fixed
torque current, and measures the TRUE developed torque (from the plant
currents and the true rotor angle) with extrapolation off vs on. Without
compensation the torque falls as speed rises (the frame lags); with it, the
torque is recovered. Placeholder motor params - this quantifies the
mechanism, not a hardware number; it feeds Q18/Q22.
"""

from __future__ import annotations

import math
import statistics

import pytest

from bench_factory import foc, expected_init_time


def _developed_torque(b, ke, pp):
    te = pp * b.theta
    ia, ib, ic = b.currents
    return ke * (math.sin(te) * ia
                 + math.sin(te - 2 * math.pi / 3) * ib
                 + math.sin(te - 4 * math.pi / 3) * ic)


def _torque_at_speed(bldcsim, params, omega, extrap, iq_cmd=80):
    ke = params.value("motor.Ke")
    pp = int(params.value("motor.pole_pairs"))
    b = bldcsim.Bench(foc(params, "sensor"))   # AS5600 latency/imperfection on
    b.run_for(expected_init_time(params))
    b.set_speed_clamp(True, omega)
    b.set_align_offset(int(params.value("foc.align_offset")))
    b.set_foc_extrap(extrap)
    b.set_id_target(0)
    b.set_iq_target(iq_cmd)
    b.set_mode(3)
    b.run_for(0.1)
    samples = [(_developed_torque(b, ke, pp))
               for _ in range(200) if not b.run_for(2e-4)]
    return statistics.mean(samples), statistics.pstdev(samples)


SPEEDS = [30, 70, 110]


def test_latency_costs_torque_without_compensation(bldcsim, params):
    """Without extrapolation the developed torque falls as speed rises (the
    frame lag grows); the loss is small at low speed and material at high."""
    t_lo, _ = _torque_at_speed(bldcsim, params, SPEEDS[0], extrap=False)
    t_hi, _ = _torque_at_speed(bldcsim, params, SPEEDS[-1], extrap=False)
    # Same commanded current; torque should be markedly lower at high speed.
    assert t_hi < 0.92 * t_lo, (
        f"expected torque loss with speed: {t_lo:.4f} -> {t_hi:.4f}")


def test_extrapolation_recovers_high_speed_torque(bldcsim, params):
    """At high speed, enabling extrapolation recovers torque lost to the
    frame lag - the Q22 result."""
    omega = SPEEDS[-1]
    t_off, _ = _torque_at_speed(bldcsim, params, omega, extrap=False)
    t_on, _ = _torque_at_speed(bldcsim, params, omega, extrap=True)
    assert t_on > t_off, (
        f"extrapolation should recover torque: off {t_off:.4f} "
        f"on {t_on:.4f}")
    # The recovery is a few percent or more at this speed/latency.
    assert t_on > 1.03 * t_off, (
        f"recovery too small to matter: off {t_off:.4f} on {t_on:.4f}")


def test_extrapolation_neutral_at_low_speed(bldcsim, params):
    """At low speed the lag is small, so extrapolation neither helps nor hurts
    much (it must not destabilize the well-aligned low-speed case)."""
    omega = SPEEDS[0]
    t_off, _ = _torque_at_speed(bldcsim, params, omega, extrap=False)
    t_on, _ = _torque_at_speed(bldcsim, params, omega, extrap=True)
    assert t_on == pytest.approx(t_off, rel=0.05), (
        f"low-speed torque should be ~unchanged: off {t_off:.4f} "
        f"on {t_on:.4f}")
