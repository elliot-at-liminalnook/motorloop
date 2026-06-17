# SPDX-License-Identifier: MIT
"""Builds Bench configuration dicts from sim/config/params.toml."""

from __future__ import annotations

# Platform profiles (platform-abstraction stage 3): each names the concrete
# peripheral model the C++ factory builds for every role. The current ZONRI
# board is the default; a second BOM (the TI reference) registers here when its
# models land (Phase B). Keep all models in the tree - selection is per
# scenario, so old parts double as the "cheap-out" regression.
PLATFORMS = {
    "zonri_drv8301": {
        "driver_name": "drv8301",
        "adc_name": "mcp3208",
        "angle_name": "as5600",
        "drv_hw_mode": False,
    },
    # The ZONRI board's DRV8302 assembly variant (datasheet-backed, Phase B):
    # same family, but hardware-configured (no SPI) - the controller skips the
    # SPI register sequence (drv_manager hw_mode). Demonstrates a genuine model
    # + RTL swap end to end.
    "zonri_drv8302": {
        "driver_name": "drv8302",
        "adc_name": "mcp3208",
        "angle_name": "as5600",
        "drv_hw_mode": True,
    },
    # Phase C, stage 14: the assembled reference BOMs.
    # ti_reference (clean): DRV8316R integrated FET+CSA + AS5047P SPI angle, a
    # modest MCP3208 reading the integrated CSA. Fewest passives (retires Q7);
    # runs on the driver's power-on defaults.
    "ti_reference": {
        "driver_name": "drv8316r",
        "adc_name": "mcp3208",
        "angle_name": "as5047p",
        "drv_hw_mode": True,
        "angle_spi_mode": True,
        "cur_norm_shift": 3,
        "chain": {"current_sense_source": 1, "csa_gain_v_per_a": 0.15},
    },
    # ti_reference_hp (external-FET): DRV8323RS + ADS9224R 16-bit simultaneous
    # current ADC (retires Q21) + AS5047P SPI angle. Higher power envelope.
    "ti_reference_hp": {
        "driver_name": "drv8323rs",
        "adc_name": "mcp3208",
        "angle_name": "as5047p",
        "drv_hw_mode": False,
        "angle_spi_mode": True,
        "adc_dual_mode": True,
        "cur_norm_shift": 3,
        "foc": {"current_sample_scheme": 1},
    },
    # Phase C, stage 11: the external-FET DRV8323RS driver. Same SPI frame
    # family as the DRV8301, so the controller configures it with the existing
    # write/verify handshake (no RTL change); the DRV8323 runs on its datasheet
    # power-on defaults (6x PWM, auto-retry OCP at VDS_LVL=0.75 V).
    "zonri_drv8323rs": {
        "driver_name": "drv8323rs",
        "adc_name": "mcp3208",
        "angle_name": "as5600",
        "drv_hw_mode": False,
    },
    # Phase C, stage 12: the DRV8316R integrated-FET driver with INTEGRATED
    # current-sense amplifiers (the clean reference BOM - no external shunts,
    # retires Q7). The chain routes current sensing through the integrated CSA
    # (kIntegratedDriverCsa, ~0.15 V/A); cur_norm_shift renormalizes its much
    # larger codes/A back to the canonical FOC fixed-point scale. The part is
    # operational on power-on defaults (6x PWM), so the controller runs it via
    # the hardware path (drv_hw_mode, no SPI reconfiguration).
    "zonri_drv8316r": {
        "driver_name": "drv8316r",
        "adc_name": "mcp3208",
        "angle_name": "as5600",
        "drv_hw_mode": True,
        "cur_norm_shift": 3,
        "chain": {
            "current_sense_source": 1,   # kIntegratedDriverCsa
            "csa_gain_v_per_a": 0.15,
        },
    },
    # Phase C, stage 13: the ADS9224R 16-bit dual-simultaneous current ADC for
    # the FOC current path (retires Q21 in hardware - one CONVST samples both
    # phase currents at the same instant). adc_dual_mode routes the FOC currents
    # from the ADS9224R master; cur_norm_shift renormalizes the 16-bit codes/A;
    # sample_scheme=1 leaves the chain live so the ADS9224R itself provides the
    # simultaneity. The EMF/bus path still uses the MCP3208.
    "zonri_ads9224r": {
        "driver_name": "drv8301",
        "adc_name": "mcp3208",
        "angle_name": "as5600",
        "drv_hw_mode": False,
        "adc_dual_mode": True,
        "cur_norm_shift": 3,
        "foc": {"current_sample_scheme": 1},
    },
    # Phase C, stage 10: swap only the angle sensor to the AS5047P (SPI, 14-bit,
    # DAEC). The controller uses the SPI angle master instead of the AS5600 PWM
    # capture (ctrl_angle_spi_mode). Isolates the angle-role swap for testing;
    # the assembled reference BOMs (stage 14) combine it with the TI parts.
    "zonri_as5047p": {
        "driver_name": "drv8301",
        "adc_name": "mcp3208",
        "angle_name": "as5047p",
        "drv_hw_mode": False,
        "angle_spi_mode": True,
    },
}
DEFAULT_PLATFORM = "zonri_drv8301"

# Part-comparison study (notes/part-comparison-checklist.md): one source of
# truth for each comparison and the single part it isolates. Verified clean:
#   * angle sensor - zonri_drv8301 vs zonri_as5047p differ ONLY in angle_name
#     (driver, ADC, drv_hw_mode all equal), so the pair isolates the sensor.
#   * current sampling - the whole-BOM adc_dual_mode swap (zonri_ads9224r /
#     ti_reference_hp) also shifts the operating point of the shared placeholder
#     loop (a confound), so Q21 skew is isolated by the foc current_sample_scheme
#     toggle on a fixed stable platform instead (0 = simultaneous / ADS9224R,
#     1 = sequential skew / MCP3208). See part_compare.py.
COMPARISONS = {
    "angle_sensor": {"kind": "platform", "a": "zonri_drv8301",
                     "b": "zonri_as5047p",
                     "labels": ("AS5600", "AS5047P"), "varies": "angle_name"},
    "current_sampling": {"kind": "scheme", "base": "zonri_drv8301",
                         "a": 1, "b": 0,
                         "labels": ("MCP3208 (sequential skew)",
                                    "ADS9224R (simultaneous)"),
                         "varies": "foc.current_sample_scheme"},
    "whole_bom": {"kind": "platform", "a": "zonri_drv8301",
                  "b": "ti_reference_hp",
                  "labels": ("default BOM", "TI HP BOM"),
                  "varies": "driver+adc+angle (system snapshot only)"},
}


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
        # FOC runtime config (C++/bench-side; the RTL gains are build-time
        # constants in rtl_params.vh). Inert until ctrl_mode = 3 and the
        # stage-4/7 plumbing consume current_sample_scheme / angle_latency.
        "foc": {
            "current_sample_scheme": int(
                params.value("foc.current_sample_scheme")),
            "angle_extrap_enable": int(
                params.value("foc.angle_extrap_enable")),
            "angle_latency_s": params.value("foc.angle_latency_s"),
        },
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


def foc(params, *groups, **extra_overrides):
    """bench_config for FOC scenarios: confirms the sinusoidal-PMSM plant
    (emf_trapezoid_blend = 0) and selects FOC runtime options. Accepts the
    same realism group names as realism() so FOC can be exercised with
    supply/mechanical/thermal/sensor effects, plus FOC-specific overrides
    (sample_scheme=, angle_extrap=, angle_latency_s=)."""
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
    # Ensure the sinusoidal-PMSM plant for FOC regardless of caller.
    motor = overrides.setdefault("motor", {})
    motor["trapezoid_blend"] = 0.0
    # FOC-specific convenience overrides map onto the foc sub-dict.
    foc_over = {}
    if "sample_scheme" in extra_overrides:
        foc_over["current_sample_scheme"] = int(
            extra_overrides.pop("sample_scheme"))
    if "angle_extrap" in extra_overrides:
        foc_over["angle_extrap_enable"] = int(
            extra_overrides.pop("angle_extrap"))
    if "angle_latency_s" in extra_overrides:
        foc_over["angle_latency_s"] = float(
            extra_overrides.pop("angle_latency_s"))
    if foc_over:
        overrides["foc"] = foc_over
    for key, value in extra_overrides.items():
        if isinstance(value, dict) and isinstance(overrides.get(key), dict):
            overrides[key].update(value)
        else:
            overrides[key] = value
    return bench_config(params, **overrides)


def platform(params, name=DEFAULT_PLATFORM, **overrides):
    """bench_config selecting a named platform's peripheral model set. Unknown
    name raises (so a typo fails loudly, not silently to the default). Extra
    overrides deep-merge on top, so realism()/foc()-style sub-dicts still work:
        platform(params, "zonri_drv8301", motor={"trapezoid_blend": 0.0})"""
    if name not in PLATFORMS:
        raise KeyError(f"unknown platform {name!r}; known: {list(PLATFORMS)}")
    over = dict(PLATFORMS[name])
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(over.get(key), dict):
            over[key].update(value)
        else:
            over[key] = value
    cfg = bench_config(params, **over)
    cfg["platform"] = name      # for telemetry / the active-platform banner
    return cfg


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
