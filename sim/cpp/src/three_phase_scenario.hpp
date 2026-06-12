#pragma once

#include <vector>

#include "three_phase_plant.hpp"

namespace bldcsim {

// Open-loop forced six-step commutation: electrical phase ramps quadratically
// to f_elec_final over ramp_time, then advances linearly. All implementations
// (C++, Python, Modelica) compute this identical closed form.
double open_loop_phase_e(double t_s, double f_elec_final_hz,
                         double ramp_time_s);
int open_loop_sector(double t_s, double f_elec_final_hz, double ramp_time_s);

struct ThreePhaseTrace {
  std::vector<double> time_s;
  std::vector<double> ia_a, ib_a, ic_a;
  std::vector<double> va_v, vb_v, vc_v;
  std::vector<double> ea_v, eb_v, ec_v;
  std::vector<double> neutral_v;
  std::vector<double> torque_n_m;
  std::vector<double> omega_rad_s;
  std::vector<double> theta_rad;
  std::vector<double> sector;
};

// Averaged-mode open-loop run. Sector and duty are held constant within each
// outer step of dt_s (evaluated at the step start), exactly like the Python
// reference implementation.
ThreePhaseTrace simulate_three_phase_open_loop(
    const ThreePhaseMotorParams& motor, const BridgeParams& bridge,
    const PlantConfig& config, double duty, double f_elec_final_hz,
    double ramp_time_s, double stop_s, double dt_s);

}  // namespace bldcsim
