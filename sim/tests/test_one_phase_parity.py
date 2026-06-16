# SPDX-License-Identifier: MIT
"""Stage 1.4/1.5: C++ vs Python one-phase parity and analytic sanity."""

from __future__ import annotations

import math

from trajectory_compare import compare_aligned, rows_to_columns

COLUMNS = [
    "duty",
    "phase_voltage_v",
    "back_emf_v",
    "current_a",
    "torque_n_m",
    "omega_rad_s",
    "theta_rad",
]


def scenario_inputs(params):
    return {
        "stop_s": params.value("scenario.one_phase_reference.stop"),
        "dt_s": params.value("scenario.one_phase_reference.dt"),
        "schedule": [
            tuple(seg)
            for seg in params.value("scenario.one_phase_reference.duty_schedule")
        ],
    }


def python_params(params, one_phase_reference):
    return one_phase_reference.params_from_config(params)


def cpp_params(params):
    return {
        "resistance_ohm": params.value("motor.R"),
        "inductance_h": params.value("motor.L"),
        "back_emf_v_s_per_rad": params.value("motor.Ke"),
        "torque_n_m_per_a": params.value("motor.Kt"),
        "inertia_kg_m2": params.value("motor.J"),
        "damping_n_m_s_per_rad": params.value("motor.B"),
        "vbus_v": params.value("bus.vbus"),
        "load_torque_n_m": params.value(
            "scenario.one_phase_reference.load_torque"
        ),
    }


def test_cpp_matches_python_reference(params, bldcsim, one_phase_reference):
    inputs = scenario_inputs(params)

    schedule_fn = one_phase_reference.make_duty_schedule(
        [list(seg) for seg in inputs["schedule"]]
    )
    rows = one_phase_reference.simulate(
        inputs["stop_s"],
        inputs["dt_s"],
        python_params(params, one_phase_reference),
        schedule_fn,
    )
    py_trace = rows_to_columns(rows)

    cpp_trace = bldcsim.simulate_one_phase(
        cpp_params(params), inputs["schedule"], inputs["stop_s"], inputs["dt_s"]
    )

    result = compare_aligned(py_trace, cpp_trace, COLUMNS, abs_tol=1e-9, rel_tol=1e-9)
    assert result.ok, result.report()


def test_locked_rotor_current_step(params, bldcsim):
    """Huge inertia pins omega ~ 0: current follows V/R(1 - exp(-tR/L))."""
    p = cpp_params(params)
    p["inertia_kg_m2"] = 1e9
    R, L, vbus = p["resistance_ohm"], p["inductance_h"], p["vbus_v"]
    duty = 0.5
    tau = L / R
    stop, dt = 8 * tau, tau / 200

    trace = bldcsim.simulate_one_phase(p, [(0.0, duty)], stop, dt)

    for t, i in zip(trace["time_s"], trace["current_a"]):
        expected = duty * vbus / R * (1.0 - math.exp(-t / tau))
        assert abs(i - expected) < 1e-3 * (duty * vbus / R), (
            f"t={t}: {i} vs {expected}"
        )


def test_no_load_steady_speed(params, bldcsim):
    """Steady state: duty*Vbus = Ke*w + R*B*w/Kt."""
    p = cpp_params(params)
    p["inertia_kg_m2"] = 1e-5  # settle fast
    duty = 0.6
    Ke, Kt = p["back_emf_v_s_per_rad"], p["torque_n_m_per_a"]
    R, B = p["resistance_ohm"], p["damping_n_m_s_per_rad"]
    omega_expected = duty * p["vbus_v"] / (Ke + R * B / Kt)

    trace = bldcsim.simulate_one_phase(p, [(0.0, duty)], 0.5, 1e-6)

    omega_final = trace["omega_rad_s"][-1]
    assert abs(omega_final - omega_expected) < 5e-3 * omega_expected, (
        f"{omega_final} vs {omega_expected}"
    )
