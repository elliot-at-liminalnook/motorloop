"""FOC stage 5: the current loop (torque control at a fixed angle).

With the rotor clamped (dyno) the FOC datapath regulates the dq currents to
their commands - the first true closed-loop FOC milestone and the cleanest
place to shake out fixed-point bugs. Then a torque-linearity sweep confirms
iq is genuinely the torque-producing axis (T = 1.5*Ke*iq) once the dq frame is
aligned (foc.align_offset). Placeholder motor params: this checks the control
law, not a hardware number.
"""

from __future__ import annotations

import math
import statistics

import pytest

from bench_factory import foc, expected_init_time


def _foc_align(params):
    return int(params.value("foc.align_offset"))


def _developed_torque(b, params):
    """Plant torque from the phase currents and rotor angle (blend=0):
    T = Ke * sum_k sin(theta_e - k*120deg) * i_k."""
    ke = params.value("motor.Ke")
    pp = int(params.value("motor.pole_pairs"))
    te = pp * b.theta
    ia, ib, ic = b.currents
    return ke * (math.sin(te) * ia
                 + math.sin(te - 2 * math.pi / 3) * ib
                 + math.sin(te - 4 * math.pi / 3) * ic)


def _settle_fixed_angle(bldcsim, params, id_t, iq_t, settle_s=0.2):
    b = bldcsim.Bench(foc(params))
    b.run_for(expected_init_time(params))
    assert b.angle_valid, "sensor angle not valid"
    b.set_speed_clamp(True, 0.0)            # hold the rotor
    b.set_align_offset(_foc_align(params))
    b.set_id_target(id_t)
    b.set_iq_target(iq_t)
    b.set_mode(3)
    b.run_for(settle_s)
    return b


@pytest.mark.parametrize("id_t,iq_t", [(0, 60), (0, 30), (0, -50), (20, 60)])
def test_current_loop_regulates_dq(bldcsim, params, id_t, iq_t):
    """At a fixed angle the loop drives measured id->id* and iq->iq* with zero
    shoot-through - FOC current regulation closes."""
    b = _settle_fixed_angle(bldcsim, params, id_t, iq_t)
    assert b.shoot_through_violations == 0
    assert b.foc_id == pytest.approx(id_t, abs=3), (
        f"id {b.foc_id} != target {id_t}")
    assert b.foc_iq == pytest.approx(iq_t, abs=3), (
        f"iq {b.foc_iq} != target {iq_t}")


def test_current_loop_zero_command_is_quiet(bldcsim, params):
    """id*=iq*=0 holds essentially zero current (the loop is not chattering)."""
    b = _settle_fixed_angle(bldcsim, params, 0, 0)
    assert b.shoot_through_violations == 0
    assert max(abs(x) for x in b.currents) < 0.3, "idle current too large"


def test_torque_linear_in_iq(bldcsim, params):
    """Sweep iq* (id*=0) at a clamped spin and confirm the developed torque is
    linear in iq* with the expected surface-PMSM slope (T = 1.5*Ke*iq), id~0.
    The torque-producing-axis check."""
    ke = params.value("motor.Ke")
    lpa = (params.value("drv8301.amp_gain")
           * params.value("feedback.current.shunt")
           * 4096.0 / params.value("adc.vref"))

    iq_cmds = [20, 40, 60, 80]
    torques = []
    for iq_t in iq_cmds:
        b = bldcsim.Bench(foc(params))
        b.run_for(expected_init_time(params))
        b.set_speed_clamp(True, 30.0)       # spin through all angles
        b.set_align_offset(_foc_align(params))
        b.set_id_target(0)
        b.set_iq_target(iq_t)
        b.set_mode(3)
        b.run_for(0.1)
        samples = []
        ids = []
        for _ in range(150):
            b.run_for(2e-4)
            samples.append(_developed_torque(b, params))
            ids.append(b.foc_id)
        assert b.shoot_through_violations == 0
        assert abs(statistics.mean(ids)) < 6, "id not held near 0"
        torques.append(statistics.mean(samples))

    # Linearity: each torque close to 1.5*Ke*(iq*/lpa) within 10%.
    for iq_t, T in zip(iq_cmds, torques):
        ideal = 1.5 * ke * (iq_t / lpa)
        assert T == pytest.approx(ideal, rel=0.1), (
            f"iq*={iq_t}: T={T:.5f} vs ideal {ideal:.5f}")
    # Monotonic and roughly constant slope.
    slopes = [torques[i + 1] - torques[i] for i in range(len(torques) - 1)]
    assert all(s > 0 for s in slopes), "torque not monotincreasing in iq*"
    assert max(slopes) < 2.0 * min(slopes), "slope not roughly constant"
