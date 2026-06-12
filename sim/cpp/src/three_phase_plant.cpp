#include "three_phase_plant.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bldcsim {

namespace {
constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kPhaseShift = kTwoPi / 3.0;
}  // namespace

double emf_shape(double theta_e_rad, double trapezoid_blend) {
  const double s = std::sin(theta_e_rad);
  // clamp(2 sin, -1, 1) has a 120-degree flat top: |sin| >= 0.5 saturates.
  const double trap = std::max(-1.0, std::min(1.0, 2.0 * s));
  return (1.0 - trapezoid_blend) * s + trapezoid_blend * trap;
}

std::array<int, 2> six_step_phases(int sector) {
  // {high, low}: 0:AB 1:AC 2:BC 3:BA 4:CA 5:CB
  static constexpr std::array<std::array<int, 2>, 6> kTable{{
      {0, 1}, {0, 2}, {1, 2}, {1, 0}, {2, 0}, {2, 1},
  }};
  return kTable[static_cast<std::size_t>(sector % 6)];
}

ThreePhasePlant::ThreePhasePlant(const ThreePhaseMotorParams& motor,
                                 const BridgeParams& bridge,
                                 const PlantConfig& config,
                                 const SupplyParams& supply)
    : motor_(motor), bridge_(bridge), config_(config), supply_(supply) {
  // E27: RK4 stability guard - the electrical time constant must be well
  // resolved by the substep or the integrator silently blows up. Refuse
  // parameter sets outside the valid envelope instead.
  const double tau_e = motor_.inductance_h / motor_.resistance_ohm;
  if (config_.max_substep_s > 0.5 * tau_e) {
    throw std::invalid_argument(
        "max_substep too large for L/R = " + std::to_string(tau_e) +
        " s; reduce sim.max_substep below 0.5*L/R");
  }
  state_.vbus_v = supply_.enabled
                      ? (supply_.ramp_time_s > 0.0 ? 0.0 : supply_.v_set_v)
                      : bridge_.vbus_v;
}

double ThreePhasePlant::bus_current_for(
    const ThreePhaseState& s, const std::array<LegMode, 3>& modes) const {
  // Current drawn from the positive rail: legs conducting through the high
  // side (FET or diode). DiodeHigh carries negative phase current back INTO
  // the rail - that is regeneration.
  double i_bus = 0.0;
  for (int k = 0; k < 3; ++k) {
    if (modes[k] == LegMode::kDrivenHigh || modes[k] == LegMode::kDiodeHigh) {
      i_bus += s.current_a[k];
    }
  }
  return i_bus;
}

double ThreePhasePlant::bus_current_a() const {
  return bus_current_for(state_, evaluate(state_).mode);
}

bool ThreePhasePlant::supply_in_cc() const {
  if (!supply_.enabled) return false;
  double v_set = supply_.v_set_v;
  if (supply_.ramp_time_s > 0.0 && time_s_ < supply_.ramp_time_s) {
    v_set *= time_s_ / supply_.ramp_time_s;
  }
  const double demand = (v_set - state_.vbus_v) / supply_.r_source_ohm;
  return demand >= supply_.i_limit_a || demand <= 0.0;
}

std::array<double, 3> ThreePhasePlant::leg_conduction_w() const {
  const ThreePhaseOutputs out = evaluate(state_);
  std::array<double, 3> w{};
  for (int k = 0; k < 3; ++k) {
    const double i = std::abs(state_.current_a[k]);
    switch (out.mode[k]) {
      case LegMode::kDrivenHigh:
      case LegMode::kDrivenLow:
        w[k] = i * i * bridge_.fet_rds_on_ohm * rds_scale_;
        break;
      case LegMode::kDiodeLow:
      case LegMode::kDiodeHigh:
        w[k] = i * bridge_.diode_vf_v;
        break;
      case LegMode::kFloat:
        break;
    }
  }
  return w;
}

void ThreePhasePlant::set_gates(const std::array<bool, 3>& gate_high,
                                const std::array<bool, 3>& gate_low) {
  averaged_mode_ = false;
  for (int k = 0; k < 3; ++k) {
    if (gate_high[k] && gate_low[k]) {
      ++shoot_through_requests_;
      gate_high_[k] = false;
      gate_low_[k] = false;
    } else {
      gate_high_[k] = gate_high[k];
      gate_low_[k] = gate_low[k];
    }
  }
}

void ThreePhasePlant::set_averaged(int sector, double duty) {
  averaged_mode_ = true;
  sector_ = ((sector % 6) + 6) % 6;
  duty_ = std::max(0.0, std::min(1.0, duty));
}

ThreePhaseOutputs ThreePhasePlant::evaluate(const ThreePhaseState& s) const {
  const double eps = config_.current_epsilon_a;
  std::array<LegMode, 3> modes{};

  for (int k = 0; k < 3; ++k) {
    const double i = s.current_a[k];
    if (averaged_mode_) {
      const auto [hi, lo] = six_step_phases(sector_);
      if (k == hi) {
        modes[k] = LegMode::kDrivenHigh;
      } else if (k == lo) {
        modes[k] = LegMode::kDrivenLow;
      } else if (i > eps) {
        modes[k] = LegMode::kDiodeLow;
      } else if (i < -eps) {
        modes[k] = LegMode::kDiodeHigh;
      } else {
        modes[k] = LegMode::kFloat;
      }
    } else {
      if (gate_high_[k]) {
        modes[k] = LegMode::kDrivenHigh;
      } else if (gate_low_[k]) {
        modes[k] = LegMode::kDrivenLow;
      } else if (i > eps) {
        modes[k] = LegMode::kDiodeLow;
      } else if (i < -eps) {
        modes[k] = LegMode::kDiodeHigh;
      } else {
        modes[k] = LegMode::kFloat;
      }
    }
  }

  // Voltage-triggered diode turn-ON (switched mode): a floating leg whose
  // open-circuit terminal voltage exceeds the rails starts conducting even
  // at zero current (e.g. external spin-up rectifying BEMF into the bus).
  // Promote violating legs and re-evaluate; converges in <= 3 passes.
  if (!averaged_mode_) {
    for (int pass = 0; pass < 3; ++pass) {
      const ThreePhaseOutputs probe = evaluate_with_modes(s, modes);
      const double rail = s.vbus_v;
      bool changed = false;
      for (int k = 0; k < 3; ++k) {
        if (modes[k] != LegMode::kFloat) continue;
        if (probe.terminal_v[k] > rail + bridge_.diode_vf_v + 1e-9) {
          modes[k] = LegMode::kDiodeHigh;
          changed = true;
        } else if (probe.terminal_v[k] < -bridge_.diode_vf_v - 1e-9) {
          modes[k] = LegMode::kDiodeLow;
          changed = true;
        }
      }
      if (!changed) break;
    }
  }
  return evaluate_with_modes(s, modes);
}

ThreePhaseOutputs ThreePhasePlant::evaluate_with_modes(
    const ThreePhaseState& s, const std::array<LegMode, 3>& modes) const {
  ThreePhaseOutputs out;
  out.mode = modes;
  const double theta_e = motor_.pole_pairs * s.theta_rad;

  const double ke = motor_.ke_v_s_per_rad * ke_scale_;
  std::array<double, 3> shape{};
  for (int k = 0; k < 3; ++k) {
    shape[k] = emf_shape(theta_e - k * kPhaseShift, motor_.trapezoid_blend);
    out.emf_v[k] = ke * shape[k] * s.omega_rad_s;
  }

  // Averaged (parity) mode always sees the stiff source; switched mode sees
  // the live bus-cap voltage (== the stiff value when the supply model is
  // disabled, since dvbus is then zero).
  const double rail = averaged_mode_ ? bridge_.vbus_v : s.vbus_v;
  const double rds = bridge_.fet_rds_on_ohm * rds_scale_;

  std::array<bool, 3> connected{};
  for (int k = 0; k < 3; ++k) {
    const double i = s.current_a[k];
    switch (modes[k]) {
      case LegMode::kDrivenHigh:
        out.terminal_v[k] =
            averaged_mode_ ? duty_ * rail : rail - i * rds;
        break;
      case LegMode::kDrivenLow:
        out.terminal_v[k] = averaged_mode_ ? 0.0 : -i * rds;
        break;
      case LegMode::kDiodeLow:
        out.terminal_v[k] = averaged_mode_ ? 0.0 : -bridge_.diode_vf_v;
        break;
      case LegMode::kDiodeHigh:
        out.terminal_v[k] =
            averaged_mode_ ? rail : rail + bridge_.diode_vf_v;
        break;
      case LegMode::kFloat:
        break;  // observed below from the neutral voltage
    }
    connected[k] = modes[k] != LegMode::kFloat;
  }

  // Isolated neutral: over connected legs the currents sum to ~0, so
  // v_n = mean(v_t - e). Fewer than two connected legs cannot carry current.
  int n = 0;
  double acc = 0.0;
  for (int k = 0; k < 3; ++k) {
    if (connected[k]) {
      ++n;
      acc += out.terminal_v[k] - out.emf_v[k];
    }
  }
  if (n >= 2) {
    out.neutral_v = acc / n;
  } else {
    // Open circuit: neutral floats with the EMFs; no current can flow.
    out.neutral_v = -(out.emf_v[0] + out.emf_v[1] + out.emf_v[2]) / 3.0;
    for (int k = 0; k < 3; ++k) {
      if (connected[k]) {
        connected[k] = false;
        out.mode[k] = LegMode::kFloat;
      }
    }
  }

  // Floating legs: terminal voltage is observed, not imposed.
  for (int k = 0; k < 3; ++k) {
    if (out.mode[k] == LegMode::kFloat) {
      out.terminal_v[k] = out.neutral_v + out.emf_v[k];
    }
  }

  double torque = 0.0;
  for (int k = 0; k < 3; ++k) {
    torque += ke * shape[k] * s.current_a[k];
  }
  out.torque_n_m = torque;
  return out;
}

ThreePhasePlant::Derivatives ThreePhasePlant::derivatives(
    const ThreePhaseState& s, const std::array<LegMode, 3>& modes) const {
  const ThreePhaseOutputs out = evaluate_with_modes(s, modes);
  const double r = motor_.resistance_ohm * r_scale_;
  Derivatives d;
  for (int k = 0; k < 3; ++k) {
    if (out.mode[k] == LegMode::kFloat) {
      d.di[k] = 0.0;
    } else {
      d.di[k] = (out.terminal_v[k] - out.neutral_v - out.emf_v[k] -
                 r * s.current_a[k]) /
                motor_.inductance_h;
    }
  }
  // Mechanical realism terms (zero-parameter == ideal viscous model):
  // regularized Coulomb friction and a single cogging harmonic.
  double tau_fric = 0.0;
  if (motor_.coulomb_friction_nm != 0.0) {
    tau_fric = motor_.coulomb_friction_nm *
               std::tanh(s.omega_rad_s / motor_.stiction_omega_eps);
  }
  double tau_cog = 0.0;
  if (motor_.cogging_torque_nm != 0.0) {
    tau_cog = motor_.cogging_torque_nm *
              std::sin(motor_.cogging_order * s.theta_rad);
  }
  d.domega = (out.torque_n_m - motor_.damping_n_m_s_per_rad * s.omega_rad_s -
              motor_.load_torque_n_m - tau_fric - tau_cog) /
             motor_.inertia_kg_m2;
  d.dtheta = s.omega_rad_s;

  if (supply_.enabled && !averaged_mode_) {
    double v_set = supply_.v_set_v;
    if (supply_.ramp_time_s > 0.0 && time_s_ < supply_.ramp_time_s) {
      v_set *= time_s_ / supply_.ramp_time_s;  // E18 power-on ramp
    }
    const double demand = (v_set - s.vbus_v) / supply_.r_source_ohm;
    const double i_supply =
        std::max(0.0, std::min(supply_.i_limit_a, demand));
    d.dvbus = (i_supply - bus_current_for(s, modes)) / supply_.capacitance_f;
  }
  return d;
}

void ThreePhasePlant::substep(double dt_s) {
  // Leg modes are resolved ONCE at the substep start and held fixed for all
  // RK4 stages: stage-level re-resolution lets diode modes flip-flop around
  // a zero crossing, which averages the stage derivatives into a phantom
  // equilibrium and currents never decay. The post-step clamps below handle
  // the actual zero crossing (ideal diode blocking).
  const ThreePhaseState s0 = state_;
  const std::array<LegMode, 3> modes = evaluate(s0).mode;
  const auto stage = [&](const Derivatives& d, double scale) {
    ThreePhaseState s = s0;
    for (int k = 0; k < 3; ++k) s.current_a[k] += scale * d.di[k];
    s.omega_rad_s += scale * d.domega;
    s.theta_rad += scale * d.dtheta;
    s.vbus_v += scale * d.dvbus;
    return s;
  };

  const Derivatives k1 = derivatives(s0, modes);
  const Derivatives k2 = derivatives(stage(k1, dt_s / 2.0), modes);
  const Derivatives k3 = derivatives(stage(k2, dt_s / 2.0), modes);
  const Derivatives k4 = derivatives(stage(k3, dt_s), modes);

  for (int k = 0; k < 3; ++k) {
    state_.current_a[k] +=
        dt_s * (k1.di[k] + 2.0 * k2.di[k] + 2.0 * k3.di[k] + k4.di[k]) / 6.0;
  }
  state_.omega_rad_s += dt_s *
                        (k1.domega + 2.0 * k2.domega + 2.0 * k3.domega +
                         k4.domega) /
                        6.0;
  state_.theta_rad += dt_s *
                      (k1.dtheta + 2.0 * k2.dtheta + 2.0 * k3.dtheta +
                       k4.dtheta) /
                      6.0;
  state_.vbus_v += dt_s *
                   (k1.dvbus + 2.0 * k2.dvbus + 2.0 * k3.dvbus + k4.dvbus) /
                   6.0;

  // Diode-blocking clamp: a leg that was freewheeling cannot have its current
  // cross zero within the substep; the residual is zeroed (ideal diode).
  for (int k = 0; k < 3; ++k) {
    if (modes[k] == LegMode::kDiodeLow && state_.current_a[k] < 0.0) {
      state_.current_a[k] = 0.0;
    } else if (modes[k] == LegMode::kDiodeHigh && state_.current_a[k] > 0.0) {
      state_.current_a[k] = 0.0;
    } else if (modes[k] == LegMode::kFloat) {
      state_.current_a[k] = 0.0;
    }
  }
}

void ThreePhasePlant::advance(double dt_s) {
  double remaining = dt_s;
  while (remaining > 1e-15) {
    const double h = std::min(remaining, config_.max_substep_s);
    substep(h);
    remaining -= h;
  }
  time_s_ += dt_s;
}

ThreePhaseOutputs ThreePhasePlant::outputs() const { return evaluate(state_); }

}  // namespace bldcsim
