# SPDX-License-Identifier: MIT
"""FOC stage 8.1: the Python FOC controller reference closed around the C++
plant, and its agreement with the RTL FOC loop at the loop level.

The reference (foc_reference.FocController) is the executable spec of the
control law (foc_core.v + speed_iq_pi.v) in fixed-point, driving the averaged
3-phase plant. The math primitives are already bit-verified (stage 3); this
checks the assembled loop reaches the same steady state as the RTL - extending
the three-way-parity story up into the controller. Placeholder motor params.
"""

from __future__ import annotations

import math
import statistics

import pytest

import foc_reference as fr
from bench_factory import foc, expected_init_time


def _gains(params):
    half = int(round(params.value("rtl.clock_frequency")
                     / (2 * params.value("pwm.frequency"))))
    return dict(
        half=half,
        vlim=int(round(half * params.value("foc.v_circle_limit"))),
        cur_kp=int(params.value("foc.cur_pi_kp")),
        cur_kish=int(params.value("foc.cur_pi_ki_shift")),
        speed_kp=int(params.value("foc.speed_kp")),
        speed_kish=int(params.value("foc.speed_ki_shift")),
        iq_max=int(params.value("foc.iq_max")),
        sincos_bits=int(params.value("foc.sincos_table_bits")),
    )


def _lpa(params):
    return (params.value("drv8301.amp_gain")
            * params.value("feedback.current.shunt")
            * 4096.0 / params.value("adc.vref"))


def _run_reference(bldcsim, params, target, steps=8000):
    """Python FOC controller driving the averaged plant; ideal angle/speed
    (the executable spec does not model sensor imperfections)."""
    pp = int(params.value("motor.pole_pairs"))
    align = int(params.value("foc.align_offset"))
    lpa = _lpa(params)
    dt = 1.0 / params.value("pwm.frequency")

    motor = {
        "resistance_ohm": params.value("motor.R"),
        "inductance_h": params.value("motor.L"),
        "ke_v_s_per_rad": params.value("motor.Ke"),
        "inertia_kg_m2": params.value("motor.J"),
        "damping_n_m_s_per_rad": params.value("motor.B"),
        "pole_pairs": pp, "trapezoid_blend": 0.0, "load_torque_n_m": 0.0,
    }
    bridge = {
        "vbus_v": params.value("bus.vbus"),
        "fet_rds_on_ohm": params.value("inverter.fet_rds_on"),
        "diode_vf_v": params.value("inverter.body_diode_vf"),
    }
    cfg = {"current_epsilon_a": params.value("sim.current_epsilon"),
           "max_substep_s": params.value("sim.max_substep")}
    plant = bldcsim.ThreePhasePlant(motor, bridge, cfg)
    ctl = fr.FocController(**_gains(params))

    speeds = []
    for step in range(steps):
        ia, ib, _ = plant.currents_a
        theta16 = (((int(round(plant.theta_rad * pp * 4096 / (2 * math.pi)))
                     + align) & 0xFFF) << 4) & 0xFFFF
        (da, db, dc), _, _ = ctl.step(
            int(ia * lpa), int(ib * lpa), theta16,
            int(round(plant.omega_rad_s)), target)
        plant.set_averaged_phase([da / ctl.half, db / ctl.half,
                                  dc / ctl.half])
        plant.advance(dt)
        if step > steps - 1000:
            speeds.append(plant.omega_rad_s)
    return statistics.mean(speeds)


@pytest.mark.parametrize("target", [60, 80, 100])
def test_reference_loop_reaches_target(bldcsim, params, target):
    """The Python FOC reference spins the averaged plant to the target."""
    omega = _run_reference(bldcsim, params, target)
    assert omega == pytest.approx(target, rel=0.06), (
        f"reference settled at {omega:.1f}, target {target}")


def test_reference_agrees_with_rtl_loop(bldcsim, params):
    """Loop-level parity: the Python reference and the RTL FOC loop settle to
    the same speed for the same command (different plant detail - averaged vs
    switched + ADC + sensor - so behavioral, not bit-exact)."""
    target = 80
    ref_omega = _run_reference(bldcsim, params, target)

    b = bldcsim.Bench(foc(params))
    b.run_for(expected_init_time(params))
    b.set_align_offset(int(params.value("foc.align_offset")))
    b.set_id_target(0)
    b.set_foc_speed_loop(True)
    b.set_target_speed(target)
    b.set_mode(3)
    b.run_for(1.2)
    rtl_speeds = []
    for _ in range(1500):
        b.run_for(2e-4)
        rtl_speeds.append(b.omega)
    rtl_omega = statistics.mean(rtl_speeds)

    assert ref_omega == pytest.approx(rtl_omega, rel=0.06), (
        f"reference {ref_omega:.1f} vs RTL {rtl_omega:.1f}")
    assert b.shoot_through_violations == 0
