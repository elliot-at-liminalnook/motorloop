"""Builds Bench configuration dicts from sim/config/params.toml."""

from __future__ import annotations


def bench_config(params, **overrides):
    cfg = {
        "clk_hz": params.value("rtl.clock_frequency"),
        "vbus_v": params.value("bus.vbus"),
        "trace_interval_s": 10e-6,
        "motor": {
            "resistance_ohm": params.value("motor.R"),
            "inductance_h": params.value("motor.L"),
            "ke_v_s_per_rad": params.value("motor.Ke"),
            "inertia_kg_m2": params.value("motor.J"),
            "damping_n_m_s_per_rad": params.value("motor.B"),
            "pole_pairs": int(params.value("motor.pole_pairs")),
            "trapezoid_blend": params.value("motor.emf_trapezoid_blend"),
            "load_torque_n_m": 0.0,
        },
        "bridge": {
            "vbus_v": params.value("bus.vbus"),
            "fet_rds_on_ohm": params.value("inverter.fet_rds_on"),
            "diode_vf_v": params.value("inverter.body_diode_vf"),
        },
        "plant": {
            "current_epsilon_a": params.value("sim.current_epsilon"),
            "max_substep_s": params.value("sim.max_substep"),
        },
        "drv": {
            "en_gate_ready_time_s": params.value(
                "drv8301.en_gate_ready_time"),
            "quick_reset_pulse_max_s": params.value(
                "drv8301.quick_reset_pulse_max"),
            "dead_time_floor_s": params.value("inverter.dtc_dead_time_floor"),
            "noctw_pulse_s": 64e-6,
            "rds_on_ohm": params.value("inverter.fet_rds_on"),
            "oc_adj_set_default": int(params.value("drv8301.oc_adj_set")),
        },
        "adc": {
            "vref_v": params.value("adc.vref"),
            "max_sclk_hz": params.value("adc.sclk"),
            "min_cs_high_s": 500e-9,
            # ch0-2 IOUT (buffered), ch3-5 EMF (cap charge share),
            # ch6 VBUS (unbuffered settling), ch7 unused.
            "sample_residual": [
                params.value("adc.sample_residual_iout"),
                params.value("adc.sample_residual_iout"),
                params.value("adc.sample_residual_iout"),
                params.value("adc.sample_residual_emf"),
                params.value("adc.sample_residual_emf"),
                params.value("adc.sample_residual_emf"),
                params.value("adc.sample_residual_bus"),
                0.0,
            ],
        },
        "encoder": {
            "sample_period_s": params.value("angle_sensor.sample_period"),
            "filter_settling_s": params.value("angle_sensor.filter_settling"),
            "pwm_carrier_hz": params.value("angle_sensor.pwm_carrier"),
        },
        "chain": {
            "shunt_ohm": params.value("feedback.current.shunt"),
            "amp_gain": params.value("drv8301.amp_gain"),
            "amp_offset_v": params.value("feedback.current.offset"),
            "emf_divider": params.value("feedback.emf.divider_ratio"),
            "emf_rc_cutoff_hz": params.value("feedback.emf.rc_cutoff"),
            "bus_divider": params.value(
                "feedback.bus_voltage.divider_ratio"),
            "rail_v": params.value("adc.vref"),
        },
        # Realism layers: present but DISABLED/zero by default; enable named
        # effect groups per scenario via realism().
        "supply": {
            "enabled": False,
            "v_set_v": params.value("bus.vbus"),
            "i_limit_a": params.value("bus.supply_current_limit"),
            "r_source_ohm": params.value("supply.r_source"),
            "capacitance_f": params.value("bus.capacitance"),
        },
        "thermal": {
            "enabled": False,
            "fet_rth_jc_k_w": params.value("thermal.fet_rth_jc"),
            "fet_rth_ca_k_w": params.value("thermal.fet_rth_ca"),
            "fet_cth_j_k": params.value("thermal.fet_cth"),
            "drv_rth_ja_k_w": params.value("thermal.drv_rth_ja"),
            "drv_cth_j_k": params.value("thermal.drv_cth"),
            "motor_rth_wa_k_w": params.value("thermal.motor_rth_wa"),
            "motor_cth_j_k": params.value("thermal.motor_cth"),
            "sw_loss_k_j_va": params.value("thermal.sw_loss_k"),
            "gate_drive_e_per_edge_j": params.value(
                "thermal.gate_drive_e_per_edge"),
            "drv_quiescent_w": params.value("thermal.drv_quiescent_w"),
            "alpha_cu_1_k": params.value("thermal.alpha_cu"),
            "ke_derate_1_k": params.value("thermal.ke_derate"),
            "rds_tempco_1_k": params.value("thermal.rds_tempco"),
            "ambient_c": params.value("thermal.ambient_c"),
        },
        "vref_ripple_v": 0.0,
        "pwm_frequency_hz": params.value("pwm.frequency"),
        "uart_baud": params.value("rtl.uart_baud"),
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


def realism(params, *groups, **extra_overrides):
    """bench_config with named realism effect groups enabled:
    'supply', 'mechanical', 'disturbance', 'thermal', 'sensor'."""
    overrides = {}
    if "supply" in groups:
        overrides["supply"] = {"enabled": True}
    if "mechanical" in groups:
        overrides["motor"] = {
            "cogging_torque_nm": params.value("motor.cogging_torque"),
            "cogging_order": int(params.value("motor.cogging_order")),
            "coulomb_friction_nm": params.value("motor.coulomb_friction"),
            "stiction_omega_eps": params.value("motor.stiction_omega_eps"),
        }
    if "disturbance" in groups:
        overrides["chain"] = {
            "gnd_shift_r_ohm": params.value("disturbance.gnd_shift_r"),
            "gnd_shift_l_h": params.value("disturbance.gnd_shift_l"),
            "edge_spike_v": params.value("disturbance.gate_edge_spike"),
            "edge_spike_tau_s": params.value("disturbance.gate_edge_tau"),
        }
        overrides["vref_ripple_v"] = params.value("disturbance.vref_ripple")
    if "thermal" in groups:
        overrides["thermal"] = {"enabled": True}
    if "sensor" in groups:
        overrides["encoder"] = {
            "eccentricity_e1_rad": params.value(
                "sensor_imperfection.eccentricity_e1"),
            "eccentricity_phi1_rad": params.value(
                "sensor_imperfection.eccentricity_phi1"),
            "eccentricity_e2_rad": params.value(
                "sensor_imperfection.eccentricity_e2"),
            "eccentricity_phi2_rad": params.value(
                "sensor_imperfection.eccentricity_phi2"),
            "angle_noise_lsb": params.value(
                "sensor_imperfection.angle_noise_lsb"),
        }
    # Deep-merge caller overrides on top.
    for key, value in extra_overrides.items():
        if isinstance(value, dict) and isinstance(overrides.get(key), dict):
            overrides[key].update(value)
        else:
            overrides[key] = value
    return bench_config(params, **overrides)


def uart_write_frame(addr, value):
    return [0x80 | (addr & 0x7F), (value >> 8) & 0xFF, value & 0xFF]


def uart_read_frame(addr):
    return [addr & 0x7F]


def freq_word(f_elec_hz, clk_hz):
    """Open-loop phase-accumulator word for an electrical frequency."""
    return int(f_elec_hz * (2**32) / clk_hz)


def expected_init_time(params):
    """Bench time by which the DRV manager should reach RUN."""
    return (1e-3                                           # power-up settle
            + params.value("drv8301.en_gate_ready_time") + 2e-3  # EN wait
            + 200e-6                                       # DC_CAL window
            + 2e-3)                                        # SPI config margin
