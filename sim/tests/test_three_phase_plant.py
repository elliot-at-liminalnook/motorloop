"""Stage 2: three-phase plant — parity, analytic sanity, switched-bridge
behavior, and integrator convergence."""

from __future__ import annotations

import math

from trajectory_compare import compare_aligned, rows_to_columns

TWO_PI = 2.0 * math.pi

PARITY_COLUMNS = [
    "ia_a", "ib_a", "ic_a",
    "va_v", "vb_v", "vc_v",
    "ea_v", "eb_v", "ec_v",
    "torque_n_m", "omega_rad_s", "theta_rad", "sector",
]


def motor_dict(params, **overrides):
    d = {
        "resistance_ohm": params.value("motor.R"),
        "inductance_h": params.value("motor.L"),
        "ke_v_s_per_rad": params.value("motor.Ke"),
        "inertia_kg_m2": params.value("motor.J"),
        "damping_n_m_s_per_rad": params.value("motor.B"),
        "pole_pairs": int(params.value("motor.pole_pairs")),
        "trapezoid_blend": params.value("motor.emf_trapezoid_blend"),
        "load_torque_n_m": params.value(
            "scenario.three_phase_open_loop.load_torque"),
    }
    d.update(overrides)
    return d


def bridge_dict(params, **overrides):
    d = {
        "vbus_v": params.value("bus.vbus"),
        "fet_rds_on_ohm": params.value("inverter.fet_rds_on"),
        "diode_vf_v": params.value("inverter.body_diode_vf"),
    }
    d.update(overrides)
    return d


def config_dict(params, **overrides):
    d = {
        "current_epsilon_a": params.value("sim.current_epsilon"),
        "max_substep_s": params.value("sim.max_substep"),
    }
    d.update(overrides)
    return d


def scenario_dict(params):
    prefix = "scenario.three_phase_open_loop"
    return {
        "duty": params.value(f"{prefix}.duty"),
        "f_elec_final_hz": params.value(f"{prefix}.f_elec_final"),
        "ramp_time_s": params.value(f"{prefix}.ramp_time"),
        "stop_s": params.value(f"{prefix}.stop"),
        "dt_s": params.value(f"{prefix}.dt"),
        "max_substep_s": params.value(f"{prefix}.max_substep"),
    }


# ---------------------------------------------------------------------------
# EMF shape
# ---------------------------------------------------------------------------

def test_emf_shape_sinusoidal_limit(bldcsim):
    for theta in [0.0, 0.3, 1.0, 2.5, 4.0, 6.0]:
        assert abs(bldcsim.emf_shape(theta, 0.0) - math.sin(theta)) < 1e-12


def test_emf_shape_trapezoid_flat_top(bldcsim):
    # |sin| >= 0.5 (30..150 deg) saturates at +/-1 for blend = 1.
    for deg in [35.0, 60.0, 90.0, 120.0, 145.0]:
        theta = math.radians(deg)
        assert abs(bldcsim.emf_shape(theta, 1.0) - 1.0) < 1e-12
        assert abs(bldcsim.emf_shape(-theta, 1.0) + 1.0) < 1e-12
    assert abs(bldcsim.emf_shape(0.0, 1.0)) < 1e-12


# ---------------------------------------------------------------------------
# Parity: C++ vs Python reference (averaged open-loop scenario)
# ---------------------------------------------------------------------------

def test_cpp_matches_python_three_phase(params, bldcsim):
    import run_three_phase_reference as ref

    scenario = scenario_dict(params)
    rows = ref.simulate(
        ref.motor_from_config(params),
        ref.BridgeParams(vbus_v=params.value("bus.vbus")),
        duty=scenario["duty"],
        f_elec_final_hz=scenario["f_elec_final_hz"],
        ramp_time_s=scenario["ramp_time_s"],
        stop_s=scenario["stop_s"],
        dt_s=scenario["dt_s"],
        current_epsilon_a=params.value("sim.current_epsilon"),
        max_substep_s=scenario["max_substep_s"],
    )
    py_trace = rows_to_columns(rows)

    cpp_trace = bldcsim.simulate_three_phase_open_loop(
        motor_dict(params),
        bridge_dict(params),
        config_dict(params, max_substep_s=scenario["max_substep_s"]),
        scenario["duty"],
        scenario["f_elec_final_hz"],
        scenario["ramp_time_s"],
        scenario["stop_s"],
        scenario["dt_s"],
    )

    result = compare_aligned(py_trace, cpp_trace, PARITY_COLUMNS,
                             abs_tol=1e-8, rel_tol=1e-8)
    assert result.ok, result.report()


# ---------------------------------------------------------------------------
# Analytic sanity
# ---------------------------------------------------------------------------

def test_locked_rotor_two_phase_current_step(params, bldcsim):
    """Sector 0 (A high, B low), rotor pinned by huge inertia at theta = 0:
    series R-L pair, i_a -> duty*vbus/(2R) with tau = L/R."""
    motor = motor_dict(params, inertia_kg_m2=1e9)
    R = motor["resistance_ohm"]
    L = motor["inductance_h"]
    vbus = params.value("bus.vbus")
    duty = 0.5
    tau = L / R
    dt = tau / 500

    plant = bldcsim.ThreePhasePlant(motor, bridge_dict(params),
                                    config_dict(params, max_substep_s=dt))
    plant.set_averaged(0, duty)

    i_final = duty * vbus / (2.0 * R)
    for step in range(1, 2001):
        plant.advance(dt)
        t = step * dt
        expected = i_final * (1.0 - math.exp(-t / tau))
        ia = plant.currents_a[0]
        assert abs(ia - expected) < 2e-3 * i_final, f"t={t}: {ia} vs {expected}"
    assert abs(plant.currents_a[1] + plant.currents_a[0]) < 1e-9
    assert abs(plant.currents_a[2]) < 1e-9


def test_open_loop_reaches_sync_speed(params, bldcsim):
    """Long gentle ramp: mean speed over the tail hunts around the commanded
    synchronous speed 2*pi*f/pp."""
    f_elec = params.value("scenario.three_phase_open_loop.f_elec_final")
    pp = int(params.value("motor.pole_pairs"))
    omega_sync = TWO_PI * f_elec / pp

    trace = bldcsim.simulate_three_phase_open_loop(
        motor_dict(params), bridge_dict(params),
        config_dict(params, max_substep_s=1e-5),
        0.6, f_elec, 0.4, 2.0, 1e-5,
    )
    tail = [w for t, w in zip(trace["time_s"], trace["omega_rad_s"]) if t > 1.2]
    mean_tail = sum(tail) / len(tail)
    assert abs(mean_tail - omega_sync) < 0.05 * omega_sync, (
        f"mean tail speed {mean_tail} vs sync {omega_sync}"
    )


# ---------------------------------------------------------------------------
# Switched-bridge behavior
# ---------------------------------------------------------------------------

def test_currents_sum_to_zero_switched(params, bldcsim):
    plant = bldcsim.ThreePhasePlant(motor_dict(params), bridge_dict(params),
                                    config_dict(params))
    # Hard-switch through a few sector patterns with all-off gaps.
    patterns = [
        ([True, False, False], [False, True, False]),
        ([False, False, False], [False, False, False]),
        ([False, True, False], [False, False, True]),
        ([False, False, True], [True, False, False]),
    ]
    eps = params.value("sim.current_epsilon")
    for gh, gl in patterns:
        plant.set_gates(gh, gl)
        for _ in range(50):
            plant.advance(1e-5)
            ia, ib, ic = plant.currents_a
            assert abs(ia + ib + ic) < 3 * eps + 1e-9


def test_dead_time_diode_freewheel(params, bldcsim):
    """Conduct A->B, then open all gates: ia freewheels through the low diode
    of leg A (va = -Vf) and returns through the high diode of leg B
    (vb = vbus + Vf), decays to zero, then all legs float."""
    vbus = params.value("bus.vbus")
    vf = params.value("inverter.body_diode_vf")
    motor = motor_dict(params, inertia_kg_m2=1e9)  # rotor pinned: no EMF
    plant = bldcsim.ThreePhasePlant(motor, bridge_dict(params),
                                    config_dict(params))

    plant.set_gates([True, False, False], [False, True, False])
    plant.advance(2e-3)  # ~ tau: build current
    ia_before = plant.currents_a[0]
    assert ia_before > 1.0

    plant.set_gates([False, False, False], [False, False, False])
    plant.advance(2e-6)  # still conducting through diodes
    out = plant.outputs()
    assert out["mode"][0] == bldcsim.LegMode.DIODE_LOW
    assert out["mode"][1] == bldcsim.LegMode.DIODE_HIGH
    assert abs(out["terminal_v"][0] - (-vf)) < 1e-9
    assert abs(out["terminal_v"][1] - (vbus + vf)) < 1e-9
    # Reverse voltage across the loop decays the current.
    assert plant.currents_a[0] < ia_before

    plant.advance(5e-3)  # long after decay
    out = plant.outputs()
    assert all(m == bldcsim.LegMode.FLOAT for m in out["mode"])
    assert all(abs(i) < 1e-9 for i in plant.currents_a)


def test_shoot_through_request_counted_and_blocked(params, bldcsim):
    plant = bldcsim.ThreePhasePlant(motor_dict(params), bridge_dict(params),
                                    config_dict(params))
    plant.set_gates([True, False, False], [True, False, False])
    assert plant.shoot_through_requests == 1
    plant.advance(1e-4)
    # Treated as both-off: no current path was created.
    assert all(abs(i) < 1e-6 for i in plant.currents_a)


def test_energy_consistent_torque_coupling(params, bldcsim):
    """Power balance of the EMF/torque formulation: at any state,
    sum(e_k * i_k) == torque * omega (electromechanical conversion is
    lossless in the coupling itself)."""
    plant = bldcsim.ThreePhasePlant(motor_dict(params), bridge_dict(params),
                                    config_dict(params))
    plant.set_averaged(0, 0.6)
    checked = 0
    for step in range(60):
        plant.advance(2e-3)
        if step % 6 == 0:
            plant.set_averaged((step // 6) % 6, 0.6)
        out = plant.outputs()
        omega = plant.omega_rad_s
        if abs(omega) < 1.0:
            continue
        p_emf = sum(e * i for e, i in zip(out["emf_v"], plant.currents_a))
        p_mech = out["torque_n_m"] * omega
        assert abs(p_emf - p_mech) < 1e-9 + 1e-9 * abs(p_mech)
        checked += 1
    assert checked > 10, "rotor never spun; test exercised nothing"


# ---------------------------------------------------------------------------
# Integrator convergence (stage 2.5)
# ---------------------------------------------------------------------------

def test_substep_convergence(params, bldcsim):
    scenario = scenario_dict(params)

    def run(substep):
        return bldcsim.simulate_three_phase_open_loop(
            motor_dict(params), bridge_dict(params),
            config_dict(params, max_substep_s=substep),
            scenario["duty"], scenario["f_elec_final_hz"],
            scenario["ramp_time_s"], scenario["stop_s"], scenario["dt_s"],
        )

    coarse = run(1e-5)
    fine = run(1e-6)
    w_coarse = coarse["omega_rad_s"][-1]
    w_fine = fine["omega_rad_s"][-1]
    assert abs(w_coarse - w_fine) < 2e-3 * abs(w_fine), (
        f"{w_coarse} vs {w_fine}"
    )
