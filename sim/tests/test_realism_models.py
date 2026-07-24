# SPDX-License-Identifier: MIT
"""Realism-checklist unit tier: supply/bus physics, mechanical terms,
correlated disturbances, ADC nonidealities, thermal lumps, sensor
imperfection — each validated in isolation against hand calcs."""

from __future__ import annotations

import math

from bench_factory import bench_config


def motor_dict(params, **overrides):
    d = bench_config(params)["motor"]
    d.update(overrides)
    return d


def bridge_dict(params, **overrides):
    d = bench_config(params)["bridge"]
    d.update(overrides)
    return d


def plant_cfg(params, **overrides):
    d = bench_config(params)["plant"]
    d.update(overrides)
    return d


def supply_dict(params, **overrides):
    d = bench_config(params)["supply"]
    d["enabled"] = True
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# Stage 1: supply/bus dynamics (plant level)
# ---------------------------------------------------------------------------

def test_supply_cc_fold_collapses_bus_under_overload(params, bldcsim):
    """Static stall draws far beyond i_limit: the bus must sag toward
    i_limit * R_load — below the DRV UVLO threshold (the brownout cause)."""
    plant = bldcsim.ThreePhasePlant(
        motor_dict(params, inertia_kg_m2=1e9),  # locked rotor, no EMF
        bridge_dict(params), plant_cfg(params),
        supply=supply_dict(params, i_limit_a=1.0))
    plant.set_gates([True, False, False], [False, True, False])
    plant.advance(50e-3)
    # Equilibrium: i = i_limit through ~2R+2rds -> v_bus ~ 1A * 1.01 Ohm.
    assert plant.supply_in_cc
    assert plant.bus_v < params.value("drv8301_uvlo.pvdd_falling"), (
        f"bus {plant.bus_v}: overload must brown out the DRV"
    )
    assert abs(plant.currents_a[0] - 1.0) < 0.1  # current-limited


def test_supply_regen_pumps_bus_with_no_sink(params, bldcsim):
    """Spin fast (averaged mode, stiff rails), then open all gates: BEMF
    rectifies through the body diodes into the bus cap, which the supply
    cannot discharge - the bus voltage must RISE."""
    plant = bldcsim.ThreePhasePlant(
        motor_dict(params, inertia_kg_m2=2e-5),
        bridge_dict(params), plant_cfg(params),
        supply=supply_dict(params))
    plant.set_gates([False] * 3, [False] * 3)
    # External mechanical drive (e.g. another machine spinning the shaft):
    # negative load torque accelerates the rotor past the rectify threshold.
    plant.set_load_torque(-0.06)
    plant.advance(1.5)
    omega = plant.omega_rad_s
    ke_ll = params.value("motor_spec.ke_line_line_peak")
    assert omega * ke_ll > params.value("bus.vbus") + 2.0, (
        f"need BEMF above bus to rectify: omega={omega:.0f}"
    )
    # Voltage-triggered diode rectification pumps the bus cap; the supply
    # cannot sink the current.
    assert plant.bus_v > params.value("bus.vbus") + 1.0, (
        f"regen did not pump the bus: {plant.bus_v:.2f}"
    )
    assert plant.supply_in_cc  # diode-blocked: supply cannot sink


def test_supply_disabled_keeps_bus_stiff(params, bldcsim):
    plant = bldcsim.ThreePhasePlant(motor_dict(params), bridge_dict(params),
                                    plant_cfg(params))
    plant.set_gates([True, False, False], [False, True, False])
    plant.advance(5e-3)
    assert plant.bus_v == params.value("bus.vbus")


# ---------------------------------------------------------------------------
# Stage 2: mechanical reality
# ---------------------------------------------------------------------------

def test_coulomb_breakaway_and_bounded_creep(params, bldcsim):
    tau_c = params.value("motor.coulomb_friction")
    eps = params.value("motor.stiction_omega_eps")
    plant = bldcsim.ThreePhasePlant(
        motor_dict(params, coulomb_friction_nm=tau_c,
                   stiction_omega_eps=eps),
        bridge_dict(params), plant_cfg(params))
    # Sub-breakaway drive: regularization permits only bounded creep
    # omega ~ eps * tau/tau_c (documented artifact).
    plant.set_load_torque(-0.5 * tau_c)  # negative load = driving torque
    plant.advance(1.0)
    creep = eps * 0.5 * tau_c / tau_c
    assert abs(plant.omega_rad_s) < 2.5 * creep + 0.05

    # Above breakaway: accelerates well past the creep band.
    plant.set_load_torque(-3.0 * tau_c)
    plant.advance(1.0)
    assert plant.omega_rad_s > 5.0


def test_cogging_detents_are_equilibria(params, bldcsim):
    amp = params.value("motor.cogging_torque")
    order = int(params.value("motor.cogging_order"))
    plant = bldcsim.ThreePhasePlant(
        motor_dict(params, cogging_torque_nm=amp, cogging_order=order,
                   coulomb_friction_nm=params.value(
                       "motor.coulomb_friction")),
        bridge_dict(params), plant_cfg(params))
    # Small kick: the rotor must stay trapped within one detent pitch.
    plant.set_load_torque(-0.3 * amp)
    plant.advance(0.2)
    plant.set_load_torque(0.0)
    plant.advance(1.0)
    assert abs(plant.theta_rad) < 2.0 * math.pi / order
    assert abs(plant.omega_rad_s) < 0.2


def test_spin_down_matches_coulomb_viscous_analytic(params, bldcsim):
    """From omega0 with no drive: J dw/dt = -(B w + tau_c). Time to reach
    w1: t = (J/B) ln((B w0 + tau_c)/(B w1 + tau_c))."""
    J = params.value("motor.J")
    B = params.value("motor.B")
    tau_c = params.value("motor.coulomb_friction")
    plant = bldcsim.ThreePhasePlant(
        motor_dict(params, coulomb_friction_nm=tau_c),
        bridge_dict(params), plant_cfg(params))
    plant.set_gates([False] * 3, [False] * 3)  # truly open: no braking
    # Directly reach speed via a driving load torque, then release.
    plant.set_load_torque(-0.05)
    plant.advance(0.4)
    w0 = plant.omega_rad_s
    plant.set_load_torque(0.0)
    w1 = 5.0
    t_pred = (J / B) * math.log((B * w0 + tau_c) / (B * w1 + tau_c))
    t = 0.0
    while plant.omega_rad_s > w1 and t < 5.0:
        plant.advance(1e-3)
        t += 1e-3
    assert abs(t - t_pred) < 0.08 * t_pred, f"spin-down {t} vs {t_pred}"


# ---------------------------------------------------------------------------
# Stage 3: correlated disturbances + ADC nonidealities (model level)
# ---------------------------------------------------------------------------

def chain_cfg(params, **overrides):
    d = bench_config(params)["chain"]
    d.update(overrides)
    return d


def test_ground_shift_is_common_mode(params, bldcsim):
    r = params.value("disturbance.gnd_shift_r")
    chain = bldcsim.FeedbackChain(chain_cfg(params, gnd_shift_r_ohm=r))
    plant = bldcsim.ThreePhasePlant(
        bench_config(params)["motor"], bench_config(params)["bridge"],
        bench_config(params)["plant"])
    plant.advance(1e-3)
    chain.set_bus_return_current(10.0, 0.0)  # 10 A return
    chain.update_from_plant(1e-3, plant, params.value("bus.vbus"))
    baseline = bldcsim.FeedbackChain(chain_cfg(params))
    baseline.update_from_plant(1e-3, plant, params.value("bus.vbus"))
    for ch in range(7):
        delta = chain.channel(ch) - baseline.channel(ch)
        assert abs(delta - r * 10.0) < 1e-9, f"ch{ch}: {delta}"


def test_edge_spike_decays_and_aperture_placement_pays_off(params, bldcsim):
    """The 3.8 payoff check at model level: right after a gate edge the
    spike corrupts the channels by many LSB; by the off-window center
    (>= 10 us away at duty <= 0.9) it is gone."""
    spike = params.value("disturbance.gate_edge_spike")
    tau = params.value("disturbance.gate_edge_tau")
    chain = bldcsim.FeedbackChain(chain_cfg(
        params, edge_spike_v=spike, edge_spike_tau_s=tau))
    plant = bldcsim.ThreePhasePlant(
        bench_config(params)["motor"], bench_config(params)["bridge"],
        bench_config(params)["plant"])
    plant.advance(1e-3)
    lsb = params.value("adc.vref") / 4096

    chain.notify_gate_edge(1.0)
    chain.update_from_plant(200e-9, plant, params.value("bus.vbus"))
    early = chain.channel(0) - params.value("feedback.current.offset")
    assert abs(early) > 4 * lsb, "spike must be visible near the edge"

    chain.update_from_plant(10e-6, plant, params.value("bus.vbus"))
    late = chain.channel(0) - params.value("feedback.current.offset")
    assert abs(late) < 0.5 * lsb, (
        f"spike must be gone at the off-window center: {late / lsb:.2f} LSB"
    )


def test_adc_transfer_nonidealities(params, bldcsim):
    cfg = {
        "vref_v": 3.3, "max_sclk_hz": 1e6, "min_cs_high_s": 500e-9,
        "offset_lsb": 2.0, "gain_error": 0.0, "inl_bow_lsb": 0.0,
    }
    from test_adc_and_sensor_models import AdcHarness
    a = bldcsim.Mcp3208(cfg)
    a.set_channels([1.65] * 8)
    code, _ = AdcHarness(a).convert(0)
    assert code == 2048 + 2  # pure offset

    cfg2 = dict(cfg, offset_lsb=0.0, inl_bow_lsb=1.5)
    b = bldcsim.Mcp3208(cfg2)
    b.set_channels([1.65] * 8)
    code2, _ = AdcHarness(b).convert(0)
    assert code2 == 2048 + 1  # bow peaks at midscale: +1.5 floored

    cfg3 = dict(cfg, offset_lsb=0.0, gain_error=2.4e-4)
    c = bldcsim.Mcp3208(cfg3)
    c.set_channels([3.2] * 8)
    code3, _ = AdcHarness(c).convert(0)
    ideal = 3.2 * 4096 / 3.3
    assert code3 == int(ideal * (1 + 2.4e-4))


def test_vref_ripple_ratiometric_error(params, bldcsim):
    from test_adc_and_sensor_models import AdcHarness
    cfg = {"vref_v": 3.3, "max_sclk_hz": 1e6, "min_cs_high_s": 500e-9}
    a = bldcsim.Mcp3208(cfg)
    a.set_channels([1.65] * 8)
    h = AdcHarness(a)
    code_nominal, _ = h.convert(0)
    a.set_live_vref(3.3 + 0.01)
    h.idle(1e-6)
    code_rippled, _ = h.convert(0)
    assert code_rippled < code_nominal  # higher vref -> smaller code


# ---------------------------------------------------------------------------
# Stage 4: thermal lumps
# ---------------------------------------------------------------------------

def thermal_cfg(params, **overrides):
    d = bench_config(params)["thermal"]
    d["enabled"] = True
    d.update(overrides)
    return d


def test_thermal_step_follows_rc_analytic(params, bldcsim):
    t = bldcsim.ThermalModel(thermal_cfg(params))
    rth = (params.value("thermal.fet_rth_jc")
           + params.value("thermal.fet_rth_ca"))
    cth = params.value("thermal.fet_cth")
    p = 2.0
    dt = 1e-3
    sim_t = 0.0
    while sim_t < 5.0:
        t.update(dt, [p, 0.0, 0.0], 0.0)
        sim_t += dt
    tau = rth * cth
    expected = params.value("thermal.ambient_c") + p * rth * (
        1.0 - math.exp(-sim_t / tau))
    assert abs(t.fet_tj_max_c - expected) < 0.05 * (expected - 25.0)


def test_drift_laws_follow_winding_temperature(params, bldcsim):
    t = bldcsim.ThermalModel(thermal_cfg(params))
    p_w = 20.0  # winding watts
    sim_t = 0.0
    for _ in range(4000):
        t.update(5e-3, [0.0] * 3, p_w)
        sim_t += 5e-3
    rth = params.value("thermal.motor_rth_wa")
    tau = rth * params.value("thermal.motor_cth")
    expected_dT = p_w * rth * (1.0 - math.exp(-sim_t / tau))
    dT = t.motor_t_c - params.value("thermal.ambient_c")
    assert abs(dT - expected_dT) < 0.05 * expected_dT
    assert abs(t.r_scale - (1 + params.value("thermal.alpha_cu") * dT)) < 1e-6
    assert abs(t.ke_scale - (1 - params.value("thermal.ke_derate") * dT)) < 1e-6


def test_emergent_otw_otsd_from_die_temp(params, bldcsim):
    """DRV model OTW/OTSD driven by the die-temperature input - no
    injection calls anywhere."""
    drv = bldcsim.Drv8301(bench_config(params)["drv"])
    pins = {"en_gate": True, "inh": [False] * 3, "inl": [False] * 3}
    drv.update(1e-3, pins, [0.0] * 3, 12.0, 25.0)
    assert drv.nfault and drv.noctw

    drv.update(2e-3, pins, [0.0] * 3, 12.0, 135.0)  # above OTW
    assert not drv.noctw, "OTW must pulse the warning pin"
    assert drv.nfault, "OTW alone is not a fault"

    drv.update(3e-3, pins, [0.0] * 3, 12.0, 155.0)  # above OTSD
    assert not drv.nfault, "OTSD must latch the fault"

    # Cooling does not clear the latch; GATE_RESET does (via quick pulse).
    drv.update(4e-3, pins, [0.0] * 3, 12.0, 25.0)
    assert not drv.nfault
    pins_off = dict(pins, en_gate=False)
    drv.update(4e-3 + 1e-6, pins_off, [0.0] * 3, 12.0, 25.0)
    drv.update(4e-3 + 3e-6, pins, [0.0] * 3, 12.0, 25.0)  # quick reset
    drv.update(4e-3 + 5e-6, pins, [0.0] * 3, 12.0, 25.0)
    assert drv.nfault


def test_pvdd_uvlo_resets_registers_emergently(params, bldcsim):
    """PVDD below the UVLO threshold silently resets the registers - the
    slva552 cause, no injection."""
    drv = bldcsim.Drv8301(bench_config(params)["drv"])
    pins = {"en_gate": True, "inh": [False] * 3, "inl": [False] * 3}
    drv.update(1e-3, pins, [0.0] * 3, 12.0, 25.0)
    assert not drv.pvdd_uv_active
    drv.update(2e-3, pins, [0.0] * 3, 5.0, 25.0)  # below 5.9 V
    assert drv.pvdd_uv_active
    assert not drv.nfault
    sr1_pvdd_uv = 1 << 8
    # Recover above threshold + hysteresis.
    drv.update(3e-3, pins, [0.0] * 3, 6.5, 25.0)
    assert not drv.pvdd_uv_active
    assert drv.nfault
    assert drv.reg(0) & sr1_pvdd_uv


# ---------------------------------------------------------------------------
# Stage 5: sensor imperfection
# ---------------------------------------------------------------------------

def test_eccentricity_error_profile(params, bldcsim):
    e1 = 0.05
    phi1 = 0.3
    cfg = bench_config(params)["encoder"]
    cfg.update({"eccentricity_e1_rad": e1, "eccentricity_phi1_rad": phi1,
                "eccentricity_e2_rad": 0.0, "eccentricity_phi2_rad": 0.0,
                "angle_noise_lsb": 0.0})
    sensor = bldcsim.As5600(cfg)
    for theta in (0.5, 1.5, 3.0, 5.0):
        t = 0.0
        while t < 20e-3:  # full settle at static angle
            t += 50e-6
            sensor.update(t, theta)
        err = sensor.filtered_angle_rad - theta
        # wrap to [-pi, pi]
        err = (err + math.pi) % (2 * math.pi) - math.pi
        expected = e1 * math.sin(theta + phi1)
        assert abs(err - expected) < 0.004, f"theta={theta}: {err} vs {expected}"
        # fresh sensor per angle to avoid filter history
        sensor = bldcsim.As5600(cfg)


def test_angle_noise_seed_determinism(params, bldcsim):
    cfg = bench_config(params)["encoder"]
    cfg.update({"angle_noise_lsb": 2.0, "noise_seed": 42})
    readings = []
    for _ in range(2):
        s = bldcsim.As5600(cfg)
        t = 0.0
        vals = []
        while t < 5e-3:
            t += 50e-6
            s.update(t, 1.0)
            vals.append(s.filtered_angle_rad)
        readings.append(vals)
    assert readings[0] == readings[1]
