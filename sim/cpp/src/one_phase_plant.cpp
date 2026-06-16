// SPDX-License-Identifier: MIT
#include "one_phase_plant.hpp"

#include <algorithm>
#include <cmath>

namespace bldcsim {

namespace {

double clamp_duty(double duty) { return std::max(-1.0, std::min(1.0, duty)); }

OnePhaseState add_scaled(const OnePhaseState& s, const OnePhaseState& d,
                         double scale) {
  return OnePhaseState{
      s.current_a + scale * d.current_a,
      s.omega_rad_s + scale * d.omega_rad_s,
      s.theta_rad + scale * d.theta_rad,
  };
}

}  // namespace

OnePhaseState OnePhasePlant::derivatives(const OnePhaseState& s,
                                         double duty) const {
  const double duty_limited = clamp_duty(duty);
  const double phase_voltage_v = duty_limited * params_.vbus_v;
  const double back_emf_v = params_.back_emf_v_s_per_rad * s.omega_rad_s;

  const double d_current =
      (phase_voltage_v - params_.resistance_ohm * s.current_a - back_emf_v) /
      params_.inductance_h;

  const double torque = params_.torque_n_m_per_a * s.current_a;
  const double d_omega =
      (torque - params_.damping_n_m_s_per_rad * s.omega_rad_s -
       params_.load_torque_n_m) /
      params_.inertia_kg_m2;

  return OnePhaseState{d_current, d_omega, s.omega_rad_s};
}

void OnePhasePlant::rk4_step(double t_s, double dt_s,
                             const DutySchedule& schedule) {
  const OnePhaseState k1 = derivatives(state_, schedule.at(t_s));
  const OnePhaseState k2 = derivatives(add_scaled(state_, k1, dt_s / 2.0),
                                       schedule.at(t_s + dt_s / 2.0));
  const OnePhaseState k3 = derivatives(add_scaled(state_, k2, dt_s / 2.0),
                                       schedule.at(t_s + dt_s / 2.0));
  const OnePhaseState k4 =
      derivatives(add_scaled(state_, k3, dt_s), schedule.at(t_s + dt_s));

  state_ = OnePhaseState{
      state_.current_a + dt_s *
                             (k1.current_a + 2.0 * k2.current_a +
                              2.0 * k3.current_a + k4.current_a) /
                             6.0,
      state_.omega_rad_s + dt_s *
                               (k1.omega_rad_s + 2.0 * k2.omega_rad_s +
                                2.0 * k3.omega_rad_s + k4.omega_rad_s) /
                               6.0,
      state_.theta_rad + dt_s *
                             (k1.theta_rad + 2.0 * k2.theta_rad +
                              2.0 * k3.theta_rad + k4.theta_rad) /
                             6.0,
  };
}

OnePhaseTrace simulate_one_phase(const OnePhaseParams& params,
                                 const DutySchedule& schedule, double stop_s,
                                 double dt_s) {
  OnePhasePlant plant(params, OnePhaseState{});
  OnePhaseTrace trace;

  const long steps = std::lround(stop_s / dt_s);
  for (long step = 0; step <= steps; ++step) {
    const double t_s = static_cast<double>(step) * dt_s;
    const double duty = schedule.at(t_s);
    const double duty_limited = clamp_duty(duty);
    const OnePhaseState& s = plant.state();

    trace.time_s.push_back(t_s);
    trace.duty.push_back(duty);
    trace.duty_limited.push_back(duty_limited);
    trace.phase_voltage_v.push_back(duty_limited * params.vbus_v);
    trace.back_emf_v.push_back(params.back_emf_v_s_per_rad * s.omega_rad_s);
    trace.current_a.push_back(s.current_a);
    trace.torque_n_m.push_back(params.torque_n_m_per_a * s.current_a);
    trace.omega_rad_s.push_back(s.omega_rad_s);
    trace.theta_rad.push_back(s.theta_rad);

    if (step < steps) {
      plant.rk4_step(t_s, dt_s, schedule);
    }
  }
  return trace;
}

}  // namespace bldcsim
