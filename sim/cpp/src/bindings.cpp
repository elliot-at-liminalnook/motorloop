#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "as5600.hpp"
#include "bench.hpp"
#include "drv8301.hpp"
#include "duty_schedule.hpp"
#include "feedback_chain.hpp"
#include "mcp3208.hpp"
#include "one_phase_plant.hpp"
#include "three_phase_plant.hpp"
#include "three_phase_scenario.hpp"

namespace py = pybind11;

namespace {

// Python-facing MCP3208 wrapper holding settable channel voltages.
class PyMcp3208 {
 public:
  explicit PyMcp3208(const bldcsim::Mcp3208Config& config)
      : adc_(config, [this](int ch) { return channels_[ch & 0x7]; }) {}

  void set_channels(const std::array<double, 8>& v) { channels_ = v; }
  bldcsim::Mcp3208& adc() { return adc_; }

 private:
  std::array<double, 8> channels_{};
  bldcsim::Mcp3208 adc_;
};

bldcsim::OnePhaseParams one_phase_params_from_dict(const py::dict& d) {
  bldcsim::OnePhaseParams p{};
  p.resistance_ohm = d["resistance_ohm"].cast<double>();
  p.inductance_h = d["inductance_h"].cast<double>();
  p.back_emf_v_s_per_rad = d["back_emf_v_s_per_rad"].cast<double>();
  p.torque_n_m_per_a = d["torque_n_m_per_a"].cast<double>();
  p.inertia_kg_m2 = d["inertia_kg_m2"].cast<double>();
  p.damping_n_m_s_per_rad = d["damping_n_m_s_per_rad"].cast<double>();
  p.vbus_v = d["vbus_v"].cast<double>();
  p.load_torque_n_m = d["load_torque_n_m"].cast<double>();
  return p;
}

py::dict trace_to_dict(const bldcsim::OnePhaseTrace& t) {
  py::dict out;
  out["time_s"] = t.time_s;
  out["duty"] = t.duty;
  out["duty_limited"] = t.duty_limited;
  out["phase_voltage_v"] = t.phase_voltage_v;
  out["back_emf_v"] = t.back_emf_v;
  out["current_a"] = t.current_a;
  out["torque_n_m"] = t.torque_n_m;
  out["omega_rad_s"] = t.omega_rad_s;
  out["theta_rad"] = t.theta_rad;
  return out;
}

bldcsim::ThreePhaseMotorParams motor_params_from_dict(const py::dict& d) {
  bldcsim::ThreePhaseMotorParams p{};
  p.resistance_ohm = d["resistance_ohm"].cast<double>();
  p.inductance_h = d["inductance_h"].cast<double>();
  p.ke_v_s_per_rad = d["ke_v_s_per_rad"].cast<double>();
  p.inertia_kg_m2 = d["inertia_kg_m2"].cast<double>();
  p.damping_n_m_s_per_rad = d["damping_n_m_s_per_rad"].cast<double>();
  p.pole_pairs = d["pole_pairs"].cast<int>();
  p.trapezoid_blend = d["trapezoid_blend"].cast<double>();
  p.load_torque_n_m = d["load_torque_n_m"].cast<double>();
  if (d.contains("cogging_torque_nm")) {
    p.cogging_torque_nm = d["cogging_torque_nm"].cast<double>();
    p.cogging_order = d["cogging_order"].cast<int>();
  }
  if (d.contains("coulomb_friction_nm")) {
    p.coulomb_friction_nm = d["coulomb_friction_nm"].cast<double>();
  }
  if (d.contains("stiction_omega_eps")) {
    p.stiction_omega_eps = d["stiction_omega_eps"].cast<double>();
  }
  return p;
}

bldcsim::BridgeParams bridge_params_from_dict(const py::dict& d) {
  bldcsim::BridgeParams p{};
  p.vbus_v = d["vbus_v"].cast<double>();
  p.fet_rds_on_ohm = d["fet_rds_on_ohm"].cast<double>();
  p.diode_vf_v = d["diode_vf_v"].cast<double>();
  return p;
}

bldcsim::PlantConfig plant_config_from_dict(const py::dict& d) {
  bldcsim::PlantConfig c{};
  c.current_epsilon_a = d["current_epsilon_a"].cast<double>();
  c.max_substep_s = d["max_substep_s"].cast<double>();
  return c;
}

bldcsim::Drv8301Config drv_config_from_dict(const py::dict& d) {
  bldcsim::Drv8301Config c;
  c.en_gate_ready_time_s = d["en_gate_ready_time_s"].cast<double>();
  c.quick_reset_pulse_max_s = d["quick_reset_pulse_max_s"].cast<double>();
  c.dead_time_floor_s = d["dead_time_floor_s"].cast<double>();
  c.noctw_pulse_s = d["noctw_pulse_s"].cast<double>();
  c.rds_on_ohm = d["rds_on_ohm"].cast<double>();
  c.oc_adj_set_default = d["oc_adj_set_default"].cast<int>();
  return c;
}

bldcsim::Mcp3208Config adc_config_from_dict(const py::dict& d) {
  bldcsim::Mcp3208Config c;
  c.vref_v = d["vref_v"].cast<double>();
  c.max_sclk_hz = d["max_sclk_hz"].cast<double>();
  c.min_cs_high_s = d["min_cs_high_s"].cast<double>();
  if (d.contains("sample_residual")) {
    c.sample_residual = d["sample_residual"].cast<std::array<double, 8>>();
  }
  if (d.contains("offset_lsb")) c.offset_lsb = d["offset_lsb"].cast<double>();
  if (d.contains("gain_error")) c.gain_error = d["gain_error"].cast<double>();
  if (d.contains("inl_bow_lsb")) {
    c.inl_bow_lsb = d["inl_bow_lsb"].cast<double>();
  }
  return c;
}

bldcsim::SupplyParams supply_params_from_dict(const py::dict& d) {
  bldcsim::SupplyParams s;
  s.enabled = d["enabled"].cast<bool>();
  s.v_set_v = d["v_set_v"].cast<double>();
  s.i_limit_a = d["i_limit_a"].cast<double>();
  s.r_source_ohm = d["r_source_ohm"].cast<double>();
  s.capacitance_f = d["capacitance_f"].cast<double>();
  if (d.contains("ramp_time_s")) {
    s.ramp_time_s = d["ramp_time_s"].cast<double>();
  }
  return s;
}

bldcsim::ThermalConfig thermal_config_from_dict(const py::dict& d) {
  bldcsim::ThermalConfig t;
  t.enabled = d["enabled"].cast<bool>();
  t.fet_rth_jc_k_w = d["fet_rth_jc_k_w"].cast<double>();
  t.fet_rth_ca_k_w = d["fet_rth_ca_k_w"].cast<double>();
  t.fet_cth_j_k = d["fet_cth_j_k"].cast<double>();
  t.drv_rth_ja_k_w = d["drv_rth_ja_k_w"].cast<double>();
  t.drv_cth_j_k = d["drv_cth_j_k"].cast<double>();
  t.motor_rth_wa_k_w = d["motor_rth_wa_k_w"].cast<double>();
  t.motor_cth_j_k = d["motor_cth_j_k"].cast<double>();
  t.sw_loss_k_j_va = d["sw_loss_k_j_va"].cast<double>();
  t.gate_drive_e_per_edge_j = d["gate_drive_e_per_edge_j"].cast<double>();
  t.drv_quiescent_w = d["drv_quiescent_w"].cast<double>();
  t.alpha_cu_1_k = d["alpha_cu_1_k"].cast<double>();
  t.ke_derate_1_k = d["ke_derate_1_k"].cast<double>();
  t.rds_tempco_1_k = d["rds_tempco_1_k"].cast<double>();
  t.ambient_c = d["ambient_c"].cast<double>();
  return t;
}

bldcsim::As5600Config encoder_config_from_dict(const py::dict& d) {
  bldcsim::As5600Config c;
  c.sample_period_s = d["sample_period_s"].cast<double>();
  c.filter_settling_s = d["filter_settling_s"].cast<double>();
  c.pwm_carrier_hz = d["pwm_carrier_hz"].cast<double>();
  return c;
}

bldcsim::FeedbackChainConfig chain_config_from_dict(const py::dict& d) {
  bldcsim::FeedbackChainConfig c;
  c.shunt_ohm = d["shunt_ohm"].cast<double>();
  c.amp_gain = d["amp_gain"].cast<double>();
  c.amp_offset_v = d["amp_offset_v"].cast<double>();
  c.emf_divider = d["emf_divider"].cast<double>();
  c.emf_rc_cutoff_hz = d["emf_rc_cutoff_hz"].cast<double>();
  c.bus_divider = d["bus_divider"].cast<double>();
  c.rail_v = d["rail_v"].cast<double>();
  if (d.contains("noise_rms_v")) {
    c.noise_rms_v = d["noise_rms_v"].cast<double>();
  }
  if (d.contains("noise_seed")) {
    c.noise_seed = d["noise_seed"].cast<unsigned>();
  }
  if (d.contains("gnd_shift_r_ohm")) {
    c.gnd_shift_r_ohm = d["gnd_shift_r_ohm"].cast<double>();
  }
  if (d.contains("gnd_shift_l_h")) {
    c.gnd_shift_l_h = d["gnd_shift_l_h"].cast<double>();
  }
  if (d.contains("edge_spike_v")) {
    c.edge_spike_v = d["edge_spike_v"].cast<double>();
  }
  if (d.contains("edge_spike_tau_s")) {
    c.edge_spike_tau_s = d["edge_spike_tau_s"].cast<double>();
  }
  return c;
}

bldcsim::As5600Config encoder_config_full_from_dict(const py::dict& d) {
  bldcsim::As5600Config c;
  c.sample_period_s = d["sample_period_s"].cast<double>();
  c.filter_settling_s = d["filter_settling_s"].cast<double>();
  c.pwm_carrier_hz = d["pwm_carrier_hz"].cast<double>();
  if (d.contains("eccentricity_e1_rad")) {
    c.eccentricity_e1_rad = d["eccentricity_e1_rad"].cast<double>();
    c.eccentricity_phi1_rad = d["eccentricity_phi1_rad"].cast<double>();
    c.eccentricity_e2_rad = d["eccentricity_e2_rad"].cast<double>();
    c.eccentricity_phi2_rad = d["eccentricity_phi2_rad"].cast<double>();
  }
  if (d.contains("angle_noise_lsb")) {
    c.angle_noise_lsb = d["angle_noise_lsb"].cast<double>();
  }
  if (d.contains("mounting_offset_rad")) {
    c.mounting_offset_rad = d["mounting_offset_rad"].cast<double>();
  }
  if (d.contains("noise_seed")) {
    c.noise_seed = d["noise_seed"].cast<unsigned>();
  }
  return c;
}

bldcsim::BenchConfig bench_config_from_dict(const py::dict& d) {
  bldcsim::BenchConfig c;
  c.clk_hz = d["clk_hz"].cast<double>();
  c.vbus_v = d["vbus_v"].cast<double>();
  c.trace_interval_s = d["trace_interval_s"].cast<double>();
  c.motor = motor_params_from_dict(d["motor"].cast<py::dict>());
  c.bridge = bridge_params_from_dict(d["bridge"].cast<py::dict>());
  c.plant = plant_config_from_dict(d["plant"].cast<py::dict>());
  c.drv = drv_config_from_dict(d["drv"].cast<py::dict>());
  c.adc = adc_config_from_dict(d["adc"].cast<py::dict>());
  c.encoder = encoder_config_full_from_dict(d["encoder"].cast<py::dict>());
  c.chain = chain_config_from_dict(d["chain"].cast<py::dict>());
  if (d.contains("supply")) {
    c.supply = supply_params_from_dict(d["supply"].cast<py::dict>());
  }
  if (d.contains("thermal")) {
    c.thermal = thermal_config_from_dict(d["thermal"].cast<py::dict>());
  }
  if (d.contains("vref_ripple_v")) {
    c.vref_ripple_v = d["vref_ripple_v"].cast<double>();
  }
  if (d.contains("pwm_frequency_hz")) {
    c.pwm_frequency_hz = d["pwm_frequency_hz"].cast<double>();
  }
  if (d.contains("uart_baud")) {
    c.uart_baud = d["uart_baud"].cast<double>();
  }
  return c;
}

py::dict trace_to_dict(const bldcsim::ThreePhaseTrace& t) {
  py::dict out;
  out["time_s"] = t.time_s;
  out["ia_a"] = t.ia_a;
  out["ib_a"] = t.ib_a;
  out["ic_a"] = t.ic_a;
  out["va_v"] = t.va_v;
  out["vb_v"] = t.vb_v;
  out["vc_v"] = t.vc_v;
  out["ea_v"] = t.ea_v;
  out["eb_v"] = t.eb_v;
  out["ec_v"] = t.ec_v;
  out["neutral_v"] = t.neutral_v;
  out["torque_n_m"] = t.torque_n_m;
  out["omega_rad_s"] = t.omega_rad_s;
  out["theta_rad"] = t.theta_rad;
  out["sector"] = t.sector;
  return out;
}

}  // namespace

PYBIND11_MODULE(bldcsim, m) {
  m.doc() = "BLDC co-simulation testbench: C++ plant and lockstep bench";

  m.def(
      "simulate_one_phase",
      [](const py::dict& params,
         const std::vector<std::pair<double, double>>& schedule, double stop_s,
         double dt_s) {
        const bldcsim::DutySchedule duty(schedule);
        return trace_to_dict(bldcsim::simulate_one_phase(
            one_phase_params_from_dict(params), duty, stop_s, dt_s));
      },
      py::arg("params"), py::arg("duty_schedule"), py::arg("stop_s"),
      py::arg("dt_s"),
      "Run the averaged one-phase reference scenario (mirrors "
      "run_one_phase_reference.py).");

  py::enum_<bldcsim::LegMode>(m, "LegMode")
      .value("DRIVEN_HIGH", bldcsim::LegMode::kDrivenHigh)
      .value("DRIVEN_LOW", bldcsim::LegMode::kDrivenLow)
      .value("DIODE_LOW", bldcsim::LegMode::kDiodeLow)
      .value("DIODE_HIGH", bldcsim::LegMode::kDiodeHigh)
      .value("FLOAT", bldcsim::LegMode::kFloat);

  py::class_<bldcsim::ThreePhasePlant>(m, "ThreePhasePlant")
      .def(py::init([](const py::dict& motor, const py::dict& bridge,
                       const py::dict& config, const py::object& supply) {
             bldcsim::SupplyParams s;
             if (!supply.is_none()) {
               s = supply_params_from_dict(supply.cast<py::dict>());
             }
             return bldcsim::ThreePhasePlant(motor_params_from_dict(motor),
                                             bridge_params_from_dict(bridge),
                                             plant_config_from_dict(config),
                                             s);
           }),
           py::arg("motor"), py::arg("bridge"), py::arg("config"),
           py::arg("supply") = py::none())
      .def_property_readonly("bus_v", &bldcsim::ThreePhasePlant::bus_v)
      .def_property_readonly("supply_in_cc",
                             &bldcsim::ThreePhasePlant::supply_in_cc)
      .def_property_readonly("bus_current_a",
                             &bldcsim::ThreePhasePlant::bus_current_a)
      .def("set_gates",
           [](bldcsim::ThreePhasePlant& self, const std::array<bool, 3>& gh,
              const std::array<bool, 3>& gl) { self.set_gates(gh, gl); })
      .def("set_averaged", &bldcsim::ThreePhasePlant::set_averaged)
      .def("advance", &bldcsim::ThreePhasePlant::advance)
      .def("set_load_torque", &bldcsim::ThreePhasePlant::set_load_torque)
      .def_property_readonly("time_s", &bldcsim::ThreePhasePlant::time_s)
      .def_property_readonly("shoot_through_requests",
                             &bldcsim::ThreePhasePlant::shoot_through_requests)
      .def_property_readonly(
          "currents_a",
          [](const bldcsim::ThreePhasePlant& self) {
            return self.state().current_a;
          })
      .def_property_readonly(
          "omega_rad_s",
          [](const bldcsim::ThreePhasePlant& self) {
            return self.state().omega_rad_s;
          })
      .def_property_readonly(
          "theta_rad",
          [](const bldcsim::ThreePhasePlant& self) {
            return self.state().theta_rad;
          })
      .def("outputs", [](const bldcsim::ThreePhasePlant& self) {
        const bldcsim::ThreePhaseOutputs out = self.outputs();
        py::dict d;
        d["terminal_v"] = out.terminal_v;
        d["emf_v"] = out.emf_v;
        d["neutral_v"] = out.neutral_v;
        d["torque_n_m"] = out.torque_n_m;
        d["mode"] = out.mode;
        return d;
      });

  m.def("emf_shape", &bldcsim::emf_shape, py::arg("theta_e_rad"),
        py::arg("trapezoid_blend"));
  m.def("open_loop_sector", &bldcsim::open_loop_sector, py::arg("t_s"),
        py::arg("f_elec_final_hz"), py::arg("ramp_time_s"));

  m.def(
      "simulate_three_phase_open_loop",
      [](const py::dict& motor, const py::dict& bridge, const py::dict& config,
         double duty, double f_elec_final_hz, double ramp_time_s,
         double stop_s, double dt_s) {
        return trace_to_dict(bldcsim::simulate_three_phase_open_loop(
            motor_params_from_dict(motor), bridge_params_from_dict(bridge),
            plant_config_from_dict(config), duty, f_elec_final_hz, ramp_time_s,
            stop_s, dt_s));
      },
      py::arg("motor"), py::arg("bridge"), py::arg("config"), py::arg("duty"),
      py::arg("f_elec_final_hz"), py::arg("ramp_time_s"), py::arg("stop_s"),
      py::arg("dt_s"),
      "Averaged-mode open-loop six-step ramp (mirrors "
      "run_three_phase_reference.py and the Modelica oracle).");

  // ---- DRV8301 ------------------------------------------------------------
  py::class_<bldcsim::Drv8301>(m, "Drv8301")
      .def(py::init([](const py::dict& d) {
             bldcsim::Drv8301Config c;
             c.en_gate_ready_time_s = d["en_gate_ready_time_s"].cast<double>();
             c.quick_reset_pulse_max_s =
                 d["quick_reset_pulse_max_s"].cast<double>();
             c.dead_time_floor_s = d["dead_time_floor_s"].cast<double>();
             c.noctw_pulse_s = d["noctw_pulse_s"].cast<double>();
             c.rds_on_ohm = d["rds_on_ohm"].cast<double>();
             c.oc_adj_set_default = d["oc_adj_set_default"].cast<int>();
             return bldcsim::Drv8301(c);
           }),
           py::arg("config"))
      .def(
          "update",
          [](bldcsim::Drv8301& self, double t_s, const py::dict& pins,
             const std::array<double, 3>& currents, double pvdd_v,
             double die_temp_c) {
            bldcsim::Drv8301Inputs in;
            in.en_gate = pins["en_gate"].cast<bool>();
            in.dc_cal = pins.contains("dc_cal") ? pins["dc_cal"].cast<bool>()
                                                : false;
            in.inh = pins["inh"].cast<std::array<bool, 3>>();
            in.inl = pins["inl"].cast<std::array<bool, 3>>();
            in.nscs = pins.contains("nscs") ? pins["nscs"].cast<bool>() : true;
            in.sclk = pins.contains("sclk") ? pins["sclk"].cast<bool>() : false;
            in.sdi = pins.contains("sdi") ? pins["sdi"].cast<bool>() : false;
            self.update(t_s, in, currents, pvdd_v, die_temp_c);
          },
          py::arg("t_s"), py::arg("pins"), py::arg("currents"),
          py::arg("pvdd_v") = 12.0, py::arg("die_temp_c") = 25.0)
      .def_property_readonly("pvdd_uv_active",
                             &bldcsim::Drv8301::pvdd_uv_active)
      .def_property_readonly("sdo", &bldcsim::Drv8301::sdo)
      .def_property_readonly("nfault", &bldcsim::Drv8301::nfault)
      .def_property_readonly("noctw", &bldcsim::Drv8301::noctw)
      .def_property_readonly("gate_high", &bldcsim::Drv8301::gate_high)
      .def_property_readonly("gate_low", &bldcsim::Drv8301::gate_low)
      .def_property_readonly("ready", &bldcsim::Drv8301::ready)
      .def_property_readonly("frame_errors", &bldcsim::Drv8301::frame_errors)
      .def_property_readonly("oc_events", &bldcsim::Drv8301::oc_events)
      .def_property_readonly("vds_threshold_v",
                             &bldcsim::Drv8301::vds_threshold_v)
      .def("reg", &bldcsim::Drv8301::reg)
      .def("dc_cal_active", &bldcsim::Drv8301::dc_cal_active)
      .def("inject_register_reset", &bldcsim::Drv8301::inject_register_reset)
      .def("inject_otw", &bldcsim::Drv8301::inject_otw);

  // ---- MCP3208 ------------------------------------------------------------
  py::class_<PyMcp3208>(m, "Mcp3208")
      .def(py::init([](const py::dict& d) {
             return std::make_unique<PyMcp3208>(adc_config_from_dict(d));
           }),
           py::arg("config"))
      .def_property_readonly(
          "last_sample_theft_v",
          [](PyMcp3208& self) { return self.adc().last_sample_theft_v(); })
      .def("set_channels", &PyMcp3208::set_channels)
      .def("set_live_vref",
           [](PyMcp3208& self, double v) { self.adc().set_live_vref(v); })
      .def("update",
           [](PyMcp3208& self, double t_s, bool cs_n, bool sclk, bool din) {
             self.adc().update(t_s, cs_n, sclk, din);
           })
      .def_property_readonly(
          "dout", [](PyMcp3208& self) { return self.adc().dout(); })
      .def_property_readonly("last_sample",
                             [](PyMcp3208& self) {
                               const auto& s = self.adc().last_sample();
                               py::dict d;
                               d["time_s"] = s.time_s;
                               d["channel"] = s.channel;
                               d["analog_v"] = s.analog_v;
                               d["code"] = s.code;
                               return d;
                             })
      .def_property_readonly(
          "conversions",
          [](PyMcp3208& self) { return self.adc().conversions(); })
      .def_property_readonly(
          "sclk_too_fast_count",
          [](PyMcp3208& self) { return self.adc().sclk_too_fast_count(); })
      .def_property_readonly(
          "cs_too_short_count",
          [](PyMcp3208& self) { return self.adc().cs_too_short_count(); })
      .def_property_readonly("differential_requests", [](PyMcp3208& self) {
        return self.adc().differential_requests();
      });

  // ---- AS5600 -------------------------------------------------------------
  py::class_<bldcsim::As5600>(m, "As5600")
      .def(py::init([](const py::dict& d) {
             return bldcsim::As5600(encoder_config_full_from_dict(d));
           }),
           py::arg("config"))
      .def("update", &bldcsim::As5600::update, py::arg("t_s"),
           py::arg("theta_mech_rad"))
      .def_property_readonly("out", &bldcsim::As5600::out)
      .def_property_readonly("angle12", &bldcsim::As5600::angle12)
      .def_property_readonly("filtered_angle_rad",
                             &bldcsim::As5600::filtered_angle_rad)
      .def("inject_magnet_loss", &bldcsim::As5600::inject_magnet_loss);

  // ---- Lockstep bench -------------------------------------------------------
  py::class_<bldcsim::Bench>(m, "Bench")
      .def(py::init([](const py::dict& d) {
             return std::make_unique<bldcsim::Bench>(
                 bench_config_from_dict(d));
           }),
           py::arg("config"))
      .def("set_mode", &bldcsim::Bench::set_mode)
      .def("set_duty", &bldcsim::Bench::set_duty)
      .def("set_target_speed", &bldcsim::Bench::set_target_speed)
      .def("set_open_loop", &bldcsim::Bench::set_open_loop)
      .def("set_align_offset", &bldcsim::Bench::set_align_offset)
      .def("run_for", &bldcsim::Bench::run_for)
      .def("run_cycles", &bldcsim::Bench::run_cycles)
      .def("inject_drv_register_reset",
           &bldcsim::Bench::inject_drv_register_reset)
      .def("inject_drv_latched_fault",
           &bldcsim::Bench::inject_drv_latched_fault)
      .def("inject_drv_otw", &bldcsim::Bench::inject_drv_otw)
      .def("inject_magnet_loss", &bldcsim::Bench::inject_magnet_loss)
      .def("set_line_glitches", &bldcsim::Bench::set_line_glitches,
           py::arg("mask"), py::arg("rate_hz"), py::arg("end_t_s"),
           py::arg("seed"))
      .def("run_config_window", &bldcsim::Bench::run_config_window,
           py::arg("seconds"), py::arg("en_gate_pulldown"), py::arg("seed"))
      .def("uart_send", &bldcsim::Bench::uart_send)
      .def("uart_take_received", &bldcsim::Bench::uart_take_received)
      .def_property_readonly("bus_v", &bldcsim::Bench::bus_v)
      .def_property_readonly("bus_v_min", &bldcsim::Bench::bus_v_min)
      .def_property_readonly("bus_v_max", &bldcsim::Bench::bus_v_max)
      .def_property_readonly("supply_in_cc", &bldcsim::Bench::supply_in_cc)
      .def_property_readonly("drv_pvdd_uv", &bldcsim::Bench::drv_pvdd_uv)
      .def_property_readonly("pvdd_uv_events",
                             &bldcsim::Bench::pvdd_uv_events)
      .def_property_readonly("fet_tj_max_c", &bldcsim::Bench::fet_tj_max_c)
      .def_property_readonly("drv_t_c", &bldcsim::Bench::drv_t_c)
      .def_property_readonly("motor_t_c", &bldcsim::Bench::motor_t_c)
      .def_property_readonly("config_window_gate_activity",
                             &bldcsim::Bench::config_window_gate_activity)
      .def_property_readonly("drv_dead", &bldcsim::Bench::drv_dead)
      .def_property_readonly("locked_out", &bldcsim::Bench::locked_out)
      .def_property_readonly("stalled", &bldcsim::Bench::stalled)
      .def_property_readonly("adc_stuck", &bldcsim::Bench::adc_stuck)
      .def_property_readonly("offset_fault", &bldcsim::Bench::offset_fault)
      .def_property_readonly("speed_reverse", &bldcsim::Bench::speed_reverse)
      .def("set_load_torque", &bldcsim::Bench::set_load_torque)
      .def("set_load_oscillation", &bldcsim::Bench::set_load_oscillation)
      .def("open_vcd", &bldcsim::Bench::open_vcd)
      .def("close_vcd", &bldcsim::Bench::close_vcd)
      .def("clear_trace", &bldcsim::Bench::clear_trace)
      .def_property_readonly("time_s", &bldcsim::Bench::time_s)
      .def_property_readonly("currents", &bldcsim::Bench::currents)
      .def_property_readonly("omega", &bldcsim::Bench::omega)
      .def_property_readonly("theta", &bldcsim::Bench::theta)
      .def_property_readonly("sector", &bldcsim::Bench::dbg_sector)
      .def_property_readonly("duty", &bldcsim::Bench::dbg_duty)
      .def_property_readonly("configured", &bldcsim::Bench::dbg_configured)
      .def_property_readonly("fault_count", &bldcsim::Bench::dbg_fault_count)
      .def_property_readonly("mismatch_count",
                             &bldcsim::Bench::dbg_mismatch_count)
      .def_property_readonly("speed", &bldcsim::Bench::dbg_speed)
      .def_property_readonly("angle", &bldcsim::Bench::dbg_angle)
      .def_property_readonly("angle_valid", &bldcsim::Bench::dbg_angle_valid)
      .def_property_readonly(
          "gates",
          [](const bldcsim::Bench& b) {
            py::list hi;
            py::list lo;
            for (int i = 0; i < 3; ++i) {
              hi.append(b.drv().gate_high()[i]);
              lo.append(b.drv().gate_low()[i]);
            }
            return py::make_tuple(hi, lo);
          })
      .def_property_readonly(
          "encoder_angle_rad",
          [](const bldcsim::Bench& b) {
            return b.encoder().filtered_angle_rad();
          })
      .def_property_readonly("offset_a", &bldcsim::Bench::dbg_offset_a)
      .def_property_readonly("offset_b", &bldcsim::Bench::dbg_offset_b)
      .def_property_readonly("noctw_count", &bldcsim::Bench::dbg_noctw_count)
      .def_property_readonly("shoot_through_violations",
                             &bldcsim::Bench::shoot_through_violations)
      .def_property_readonly("plant_shoot_through_requests",
                             &bldcsim::Bench::plant_shoot_through_requests)
      .def_property_readonly("min_dead_time_s",
                             &bldcsim::Bench::min_dead_time_s)
      .def_property_readonly("drv_ready",
                             [](const bldcsim::Bench& b) {
                               return b.drv().ready();
                             })
      .def_property_readonly("drv_frame_errors",
                             [](const bldcsim::Bench& b) {
                               return b.drv().frame_errors();
                             })
      .def("drv_reg",
           [](const bldcsim::Bench& b, int addr) { return b.drv().reg(addr); })
      .def_property_readonly(
          "trace",
          [](const bldcsim::Bench& b) {
            py::dict d;
            const auto& rows = b.trace();
            auto col = [&](auto getter) {
              py::list values;
              for (const auto& r : rows) values.append(getter(r));
              return values;
            };
            d["time_s"] = col([](const auto& r) { return r.time_s; });
            d["ia_a"] = col([](const auto& r) { return r.current_a[0]; });
            d["ib_a"] = col([](const auto& r) { return r.current_a[1]; });
            d["ic_a"] = col([](const auto& r) { return r.current_a[2]; });
            d["omega_rad_s"] = col([](const auto& r) { return r.omega_rad_s; });
            d["theta_rad"] = col([](const auto& r) { return r.theta_rad; });
            d["sector"] = col([](const auto& r) { return r.sector; });
            d["duty"] = col([](const auto& r) { return r.duty; });
            d["configured"] = col([](const auto& r) { return r.configured; });
            d["fault_count"] = col([](const auto& r) { return r.fault_count; });
            d["mismatch_count"] =
                col([](const auto& r) { return r.mismatch_count; });
            d["speed_rad_s"] = col([](const auto& r) { return r.speed_rad_s; });
            d["cur_code"] = col([](const auto& r) { return r.cur_code; });
            d["emf_code"] = col([](const auto& r) { return r.emf_code; });
            d["vbus_code"] = col([](const auto& r) { return r.vbus_code; });
            d["vbus_v"] = col([](const auto& r) { return r.vbus_v; });
            d["fet_tj_c"] = col([](const auto& r) { return r.fet_tj_c; });
            d["motor_t_c"] = col([](const auto& r) { return r.motor_t_c; });
            return d;
          })
      .def_property_readonly("adc_samples", [](const bldcsim::Bench& b) {
        py::list out;
        for (const auto& s : b.adc_samples()) {
          py::dict d;
          d["time_s"] = s.time_s;
          d["channel"] = s.channel;
          d["analog_v"] = s.analog_v;
          d["code"] = s.code;
          d["pwm_counter"] = s.pwm_counter;
          d["pwm_up"] = s.pwm_up;
          d["duty"] = s.duty;
          out.append(d);
        }
        return out;
      });

  // ---- Thermal model (standalone, for unit tests) ---------------------------
  py::class_<bldcsim::ThermalModel>(m, "ThermalModel")
      .def(py::init([](const py::dict& d) {
             return bldcsim::ThermalModel(thermal_config_from_dict(d));
           }),
           py::arg("config"))
      .def("add_switch_edge", &bldcsim::ThermalModel::add_switch_edge)
      .def("update", &bldcsim::ThermalModel::update, py::arg("dt_s"),
           py::arg("leg_conduction_w"), py::arg("winding_w"))
      .def_property_readonly("fet_tj_max_c",
                             &bldcsim::ThermalModel::fet_tj_max_c)
      .def_property_readonly("drv_t_c", &bldcsim::ThermalModel::drv_t_c)
      .def_property_readonly("motor_t_c", &bldcsim::ThermalModel::motor_t_c)
      .def_property_readonly("r_scale", &bldcsim::ThermalModel::r_scale)
      .def_property_readonly("ke_scale", &bldcsim::ThermalModel::ke_scale)
      .def_property_readonly("rds_scale", &bldcsim::ThermalModel::rds_scale);

  // ---- Feedback chain -----------------------------------------------------
  py::class_<bldcsim::FeedbackChain>(m, "FeedbackChain")
      .def(py::init([](const py::dict& d) {
             return bldcsim::FeedbackChain(chain_config_from_dict(d));
           }),
           py::arg("config"))
      .def("set_dc_cal", &bldcsim::FeedbackChain::set_dc_cal)
      .def("set_bus_return_current",
           &bldcsim::FeedbackChain::set_bus_return_current)
      .def("notify_gate_edge", &bldcsim::FeedbackChain::notify_gate_edge)
      .def_property_readonly("gnd_shift_v",
                             &bldcsim::FeedbackChain::gnd_shift_v)
      .def(
          "update_from_plant",
          [](bldcsim::FeedbackChain& self, double dt_s,
             const bldcsim::ThreePhasePlant& plant, double vbus_v) {
            self.update(dt_s, plant.state(), plant.outputs(), vbus_v);
          },
          py::arg("dt_s"), py::arg("plant"), py::arg("vbus_v"))
      .def_property_readonly("channels",
                             &bldcsim::FeedbackChain::channels)
      .def("channel", &bldcsim::FeedbackChain::channel);
}
