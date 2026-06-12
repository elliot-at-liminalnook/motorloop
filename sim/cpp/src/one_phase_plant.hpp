#pragma once

#include <vector>

#include "duty_schedule.hpp"

namespace bldcsim {

// Mirrors sim/scripts/run_one_phase_reference.py and
// sim/modelica/BldcCosimTestbench package: averaged one-phase
// electromechanical plant, RK4 fixed step.

struct OnePhaseParams {
  double resistance_ohm;
  double inductance_h;
  double back_emf_v_s_per_rad;
  double torque_n_m_per_a;
  double inertia_kg_m2;
  double damping_n_m_s_per_rad;
  double vbus_v;
  double load_torque_n_m;
};

struct OnePhaseState {
  double current_a = 0.0;
  double omega_rad_s = 0.0;
  double theta_rad = 0.0;
};

struct OnePhaseTrace {
  std::vector<double> time_s;
  std::vector<double> duty;
  std::vector<double> duty_limited;
  std::vector<double> phase_voltage_v;
  std::vector<double> back_emf_v;
  std::vector<double> current_a;
  std::vector<double> torque_n_m;
  std::vector<double> omega_rad_s;
  std::vector<double> theta_rad;
};

class OnePhasePlant {
 public:
  OnePhasePlant(const OnePhaseParams& params, const OnePhaseState& initial)
      : params_(params), state_(initial) {}

  const OnePhaseState& state() const { return state_; }

  // One RK4 step; duty is evaluated through the schedule at substep times,
  // matching the Python reference integrator stage for stage.
  void rk4_step(double t_s, double dt_s, const DutySchedule& schedule);

  OnePhaseState derivatives(const OnePhaseState& s, double duty) const;

 private:
  OnePhaseParams params_;
  OnePhaseState state_;
};

OnePhaseTrace simulate_one_phase(const OnePhaseParams& params,
                                 const DutySchedule& schedule, double stop_s,
                                 double dt_s);

}  // namespace bldcsim
