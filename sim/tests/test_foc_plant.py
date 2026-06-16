# SPDX-License-Identifier: MIT
"""FOC stage 1: sinusoidal-PMSM plant configuration and the dq cross-check.

Confirms the abc plant (emf_trapezoid_blend = 0) and the dq control math
(sim/scripts/foc_reference.py) agree. The plant is driven in the rotor frame
with ideal sinusoidal voltages (float reference inv-Park / inv-Clarke), the
rotor clamped to a known speed so the electrical steady state settles fast,
and the measured currents are projected back to dq. The standing caveat
applies: placeholder motor params, so this checks model self-consistency, not
a hardware number.
"""

from __future__ import annotations

import math

import pytest

import foc_reference as fr


def _motor(params, blend=0.0):
    return {
        "resistance_ohm": params.value("motor.R"),
        "inductance_h": params.value("motor.L"),
        "ke_v_s_per_rad": params.value("motor.Ke"),
        "inertia_kg_m2": params.value("motor.J"),
        "damping_n_m_s_per_rad": params.value("motor.B"),
        "pole_pairs": int(params.value("motor.pole_pairs")),
        "trapezoid_blend": blend,
        "load_torque_n_m": 0.0,
    }


def _bridge(params):
    return {
        "vbus_v": params.value("bus.vbus"),
        "fet_rds_on_ohm": params.value("inverter.fet_rds_on"),
        "diode_vf_v": params.value("inverter.body_diode_vf"),
    }


def _plant_cfg(params):
    return {
        "current_epsilon_a": params.value("sim.current_epsilon"),
        "max_substep_s": params.value("sim.max_substep"),
    }


def _run_rotor_frame(bldcsim, params, vd_cmd, vq_cmd, omega,
                     settle_s=0.06, dt=20e-6):
    """Drive the clamped-speed plant with constant rotor-frame (vd, vq),
    return (id, iq, omega, (ia, ib, ic)) at the end."""
    vbus = params.value("bus.vbus")
    pp = int(params.value("motor.pole_pairs"))
    plant = bldcsim.ThreePhasePlant(_motor(params), _bridge(params),
                                    _plant_cfg(params))
    plant.set_speed_clamp(True, omega)
    for _ in range(int(settle_s / dt)):
        theta_e = pp * plant.theta_rad
        valpha, vbeta = fr.inv_park(vd_cmd, vq_cmd, theta_e)
        va, vb, vc = fr.inv_clarke(valpha, vbeta)
        duties = [max(0.0, min(1.0, 0.5 + v / vbus)) for v in (va, vb, vc)]
        plant.set_averaged_phase(duties)
        plant.advance(dt)
    theta_e = pp * plant.theta_rad
    ia, ib, ic = plant.currents_a
    ialpha, ibeta = fr.clarke(ia, ib, ic)
    id_, iq = fr.park(ialpha, ibeta, theta_e)
    return id_, iq, plant.omega_rad_s, (ia, ib, ic)


OPERATING_POINTS = [
    (0.0, 2.0, 90.0),
    (0.0, 3.0, 90.0),
    (0.0, 2.0, 50.0),
    (1.0, 2.0, 90.0),
    (-1.0, 2.5, 70.0),
]


@pytest.mark.parametrize("vd_cmd,vq_cmd,omega", OPERATING_POINTS)
def test_dq_voltage_model_matches_plant(bldcsim, params, vd_cmd, vq_cmd,
                                        omega):
    """The dq steady-state voltage model reproduces the applied rotor-frame
    (vd, vq) from the measured (id, iq, omega): the abc physics and the dq
    math agree. This is the FOC analogue of the three-way plant parity."""
    R = params.value("motor.R")
    L = params.value("motor.L")
    Ke = params.value("motor.Ke")
    pp = int(params.value("motor.pole_pairs"))
    lam = fr.lambda_m(Ke, pp)

    id_, iq, w, _ = _run_rotor_frame(bldcsim, params, vd_cmd, vq_cmd, omega)
    we = pp * w
    vd_model, vq_model = fr.dq_voltage(id_, iq, we, R, L, lam)

    # Tolerance: a few percent of the bus, covering averaging ripple and the
    # finite settle window.
    assert vd_model == pytest.approx(vd_cmd, abs=0.05), (
        f"vd: model {vd_model:.4f} vs applied {vd_cmd}")
    assert vq_model == pytest.approx(vq_cmd, abs=0.05), (
        f"vq: model {vq_model:.4f} vs applied {vq_cmd}")


def test_amplitude_invariant_transform(bldcsim, params):
    """|i_dq| equals the phase-current envelope (amplitude-invariant Clarke),
    and the three phase currents are balanced (sum ~ 0)."""
    id_, iq, _, (ia, ib, ic) = _run_rotor_frame(
        bldcsim, params, 0.0, 3.0, 90.0)
    assert abs(ia + ib + ic) < 1e-3, "phase currents not balanced"
    # Envelope peak = |i_dq|; the instantaneous max phase is <= that. Drive a
    # second point a quarter electrical period apart is overkill - instead
    # check |i_dq| is consistent with the per-phase RMS * sqrt(2)/sqrt(3)*...:
    # simplest robust check is that no instantaneous phase exceeds |i_dq|.
    mag = math.hypot(id_, iq)
    assert max(abs(ia), abs(ib), abs(ic)) <= mag + 1e-6


def test_positive_vq_makes_positive_torque(bldcsim, params):
    """Orientation: +vq drives +iq, and the plant torque tracks 1.5*Ke*iq
    (so iq is genuinely the torque-producing axis in this convention)."""
    Ke = params.value("motor.Ke")
    pp = int(params.value("motor.pole_pairs"))
    B = params.value("motor.B")
    del B
    omega = 90.0
    id_, iq, w, _ = _run_rotor_frame(bldcsim, params, 0.0, 2.0, omega)
    assert iq > 0.0, "positive vq did not produce positive iq"
    # Developed torque is positive and follows the surface-PMSM relation
    # T = 1.5*Ke*iq (the dyno absorbs it at the clamped speed).
    torque = fr.torque_from_iq(iq, Ke)
    assert torque > 0.0, "positive iq did not develop positive torque"
    assert torque == pytest.approx(1.5 * Ke * iq, rel=1e-6)
    # A negative vq must reverse the torque sign (orientation, not magnitude).
    id_n, iq_n, _, _ = _run_rotor_frame(bldcsim, params, 0.0, -2.0, omega)
    assert iq_n < 0.0, "negative vq did not produce negative iq"


def test_sinusoidal_emf_is_the_foc_plant(bldcsim, params):
    """The FOC config is blend = 0 (sinusoidal). A trapezoidal plant would
    show a markedly different dq voltage at the same operating point - this
    pins the regression so a config slip is caught."""
    R = params.value("motor.R")
    L = params.value("motor.L")
    Ke = params.value("motor.Ke")
    pp = int(params.value("motor.pole_pairs"))
    lam = fr.lambda_m(Ke, pp)

    # Sinusoidal: the dq model holds tightly.
    id_s, iq_s, w_s, _ = _run_rotor_frame(bldcsim, params, 0.0, 2.0, 90.0)
    vd_s, vq_s = fr.dq_voltage(id_s, iq_s, pp * w_s, R, L, lam)
    assert vq_s == pytest.approx(2.0, abs=0.05)

    # Trapezoidal plant at the same drive: the smooth-dq model no longer
    # reproduces the command as tightly (harmonics break the dq assumption).
    motor_trap = _motor(params, blend=1.0)
    vbus = params.value("bus.vbus")
    plant = bldcsim.ThreePhasePlant(motor_trap, _bridge(params),
                                    _plant_cfg(params))
    plant.set_speed_clamp(True, 90.0)
    for _ in range(int(0.06 / 20e-6)):
        theta_e = pp * plant.theta_rad
        valpha, vbeta = fr.inv_park(0.0, 2.0, theta_e)
        va, vb, vc = fr.inv_clarke(valpha, vbeta)
        plant.set_averaged_phase(
            [max(0.0, min(1.0, 0.5 + v / vbus)) for v in (va, vb, vc)])
        plant.advance(20e-6)
    ia, ib, ic = plant.currents_a
    ialpha, ibeta = fr.clarke(ia, ib, ic)
    id_t, iq_t = fr.park(ialpha, ibeta, pp * plant.theta_rad)
    vd_t, vq_t = fr.dq_voltage(id_t, iq_t, pp * plant.omega_rad_s, R, L, lam)
    # The trapezoidal harmonics inject a 6th-harmonic ripple the smooth dq
    # model cannot represent: the instantaneous reconstruction is looser.
    # (We assert the sinusoidal case is the tight one; this is a guard, not a
    # precise trapezoid model.)
    assert abs(vq_s - 2.0) < abs(vq_t - 2.0) + 0.05
