// SPDX-License-Identifier: MIT
#include "three_phase_scenario.hpp"

#include <cmath>

namespace bldcsim {

namespace {
constexpr double kPi = 3.1415926535897932384626433832795;
}

double open_loop_phase_e(double t_s, double f_elec_final_hz,
                         double ramp_time_s) {
  if (t_s < ramp_time_s) {
    return 2.0 * kPi * (0.5 * f_elec_final_hz * t_s * t_s / ramp_time_s);
  }
  return 2.0 * kPi * (0.5 * f_elec_final_hz * ramp_time_s +
                      f_elec_final_hz * (t_s - ramp_time_s));
}

int open_loop_sector(double t_s, double f_elec_final_hz, double ramp_time_s) {
  const double phase = open_loop_phase_e(t_s, f_elec_final_hz, ramp_time_s);
  const long idx = static_cast<long>(std::floor(phase / (kPi / 3.0)));
  return static_cast<int>(((idx % 6) + 6) % 6);
}

ThreePhaseTrace simulate_three_phase_open_loop(
    const ThreePhaseMotorParams& motor, const BridgeParams& bridge,
    const PlantConfig& config, double duty, double f_elec_final_hz,
    double ramp_time_s, double stop_s, double dt_s) {
  ThreePhasePlant plant(motor, bridge, config);
  ThreePhaseTrace trace;

  const long steps = std::lround(stop_s / dt_s);
  for (long step = 0; step <= steps; ++step) {
    const double t_s = static_cast<double>(step) * dt_s;
    const int sector = open_loop_sector(t_s, f_elec_final_hz, ramp_time_s);
    plant.set_averaged(sector, duty);

    const ThreePhaseState& s = plant.state();
    const ThreePhaseOutputs out = plant.outputs();
    trace.time_s.push_back(t_s);
    trace.ia_a.push_back(s.current_a[0]);
    trace.ib_a.push_back(s.current_a[1]);
    trace.ic_a.push_back(s.current_a[2]);
    trace.va_v.push_back(out.terminal_v[0]);
    trace.vb_v.push_back(out.terminal_v[1]);
    trace.vc_v.push_back(out.terminal_v[2]);
    trace.ea_v.push_back(out.emf_v[0]);
    trace.eb_v.push_back(out.emf_v[1]);
    trace.ec_v.push_back(out.emf_v[2]);
    trace.neutral_v.push_back(out.neutral_v);
    trace.torque_n_m.push_back(out.torque_n_m);
    trace.omega_rad_s.push_back(s.omega_rad_s);
    trace.theta_rad.push_back(s.theta_rad);
    trace.sector.push_back(static_cast<double>(sector));

    if (step < steps) {
      plant.advance(dt_s);
    }
  }
  return trace;
}

}  // namespace bldcsim
