#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace bldcsim {

// Three-phase BLDC plant: per-phase R/L with position-dependent back-EMF
// (sinusoid-to-trapezoid blend), isolated neutral, switched bridge with body
// diodes, plus an averaged six-step mode for fast runs and oracle parity.
//
// Conventions:
//  - Phase current i[k] is positive flowing OUT of the bridge leg INTO the
//    motor winding (toward the isolated neutral).
//  - theta is the mechanical angle; electrical angle = pole_pairs * theta.
//  - EMF shape f_k(theta_e) has unit peak; e_k = Ke * f_k * omega and
//    torque = Ke * sum(f_k * i_k), so electrical power balances exactly.

struct ThreePhaseMotorParams {
  double resistance_ohm;          // per phase
  double inductance_h;            // per phase
  double ke_v_s_per_rad;          // peak back-EMF constant (mechanical)
  double inertia_kg_m2;
  double damping_n_m_s_per_rad;
  int pole_pairs;
  double trapezoid_blend;         // 0 = sinusoidal, 1 = 120-deg flat trapezoid
  double load_torque_n_m;
  // Mechanical realism (stage 2); zero = ideal viscous-only model.
  double cogging_torque_nm = 0.0;     // peak detent torque
  int cogging_order = 24;             // detents per mechanical revolution
  double coulomb_friction_nm = 0.0;   // kinetic friction magnitude
  double stiction_omega_eps = 0.5;    // tanh regularization width [rad/s]
};

struct BridgeParams {
  double vbus_v;
  double fet_rds_on_ohm;   // switched mode only
  double diode_vf_v;       // switched mode only
};

// Bench-supply model (realism stage 1): CV source with series impedance and
// a CC fold at i_limit, feeding the bus capacitance. The supply cannot sink
// current, so regenerative deceleration pumps the bus voltage up. Disabled
// (stiff source at BridgeParams::vbus_v) for parity scenarios.
struct SupplyParams {
  bool enabled = false;
  double v_set_v = 12.0;
  double i_limit_a = 1.0;
  double r_source_ohm = 0.05;
  double capacitance_f = 1.66e-3;
  // E18: power-on ramp - the effective setpoint rises linearly over
  // ramp_time_s (0 = instant), sweeping the bus through the UVLO region.
  double ramp_time_s = 0.0;
};

struct PlantConfig {
  double current_epsilon_a = 1e-6;  // float-mode threshold
  double max_substep_s = 1e-6;      // integration substep cap
};

enum class LegMode : std::uint8_t {
  kDrivenHigh,
  kDrivenLow,
  kDiodeLow,    // both FETs off, current freewheeling through low diode
  kDiodeHigh,   // both FETs off, current returning through high diode
  kFloat,       // disconnected, current held at ~0
};

struct ThreePhaseState {
  std::array<double, 3> current_a{0.0, 0.0, 0.0};
  double omega_rad_s = 0.0;
  double theta_rad = 0.0;
  double vbus_v = 0.0;  // bus-cap voltage; constant when supply disabled
};

struct ThreePhaseOutputs {
  std::array<double, 3> terminal_v{};
  std::array<double, 3> emf_v{};
  std::array<LegMode, 3> mode{};
  double neutral_v = 0.0;
  double torque_n_m = 0.0;
};

// Per-phase normalized EMF shape, unit peak.
double emf_shape(double theta_e_rad, double trapezoid_blend);

class ThreePhasePlant {
 public:
  ThreePhasePlant(const ThreePhaseMotorParams& motor, const BridgeParams& bridge,
                  const PlantConfig& config,
                  const SupplyParams& supply = SupplyParams{});

  // Switched mode: real gate signals (already dead-time-processed by the
  // DRV8301 model). gate_high[k] && gate_low[k] is counted as a shoot-through
  // request and treated as both-off.
  void set_gates(const std::array<bool, 3>& gate_high,
                 const std::array<bool, 3>& gate_low);

  // Averaged mode: ideal six-step drive. sector in [0, 6), duty in [0, 1].
  // Sector table (hi, lo): 0:AB 1:AC 2:BC 3:BA 4:CA 5:CB.
  void set_averaged(int sector, double duty);

  // Averaged 3-phase continuous mode: each leg driven-high with its own duty
  // fraction d[k] in [0, 1] (averaged terminal voltage = d[k]*vbus). The
  // continuous-modulation analogue of set_averaged's six-step drive; used for
  // the dq cross-check (foc-checklist stage 1) and the Python FOC reference
  // (stage 8). Physics solver is identical - only the leg drive differs.
  void set_averaged_phase(const std::array<double, 3>& duty);

  void advance(double dt_s);

  const ThreePhaseState& state() const { return state_; }
  ThreePhaseOutputs outputs() const;  // evaluated at the current state
  double time_s() const { return time_s_; }
  long shoot_through_requests() const { return shoot_through_requests_; }

  void set_load_torque(double n_m) { motor_.load_torque_n_m = n_m; }

  // Test/dyno affordance: clamp the rotor to a fixed speed (mechanical
  // integration overridden, theta advances at omega). Lets a test read the
  // electrical steady state (L/R ~ ms) at a known speed without waiting out
  // the mechanical time constant (J/B ~ s). Physics of the electrical solve
  // is unchanged.
  void set_speed_clamp(bool enabled, double omega_rad_s = 0.0);

  // Live thermal-drift multipliers (realism stage 4); default 1.0.
  void set_r_scale(double s) { r_scale_ = s; }
  void set_ke_scale(double s) { ke_scale_ = s; }
  void set_rds_scale(double s) { rds_scale_ = s; }

  // Bus/supply probes.
  double bus_v() const { return state_.vbus_v; }
  double bus_current_a() const;   // current drawn from the bus by the bridge
  bool supply_in_cc() const;      // CC fold or diode-blocked (regen)
  // Per-leg conduction loss + total mech-side losses for the thermal model.
  std::array<double, 3> leg_conduction_w() const;

 private:
  struct Derivatives {
    std::array<double, 3> di{};
    double domega = 0.0;
    double dtheta = 0.0;
    double dvbus = 0.0;
  };

  double bus_current_for(const ThreePhaseState& s,
                         const std::array<LegMode, 3>& modes) const;

  // Resolves leg modes from the state (diode conduction by current sign).
  ThreePhaseOutputs evaluate(const ThreePhaseState& s) const;
  // Same electrical solve but with externally fixed leg modes; used by the
  // RK4 stages so modes cannot flip-flop within a substep.
  ThreePhaseOutputs evaluate_with_modes(
      const ThreePhaseState& s, const std::array<LegMode, 3>& modes) const;
  Derivatives derivatives(const ThreePhaseState& s,
                          const std::array<LegMode, 3>& modes) const;
  void substep(double dt_s);

  ThreePhaseMotorParams motor_;
  BridgeParams bridge_;
  PlantConfig config_;
  SupplyParams supply_;
  ThreePhaseState state_;
  double time_s_ = 0.0;
  double r_scale_ = 1.0;
  double ke_scale_ = 1.0;
  double rds_scale_ = 1.0;

  bool averaged_mode_ = false;
  bool avg_three_phase_ = false;   // continuous per-leg averaged drive
  bool speed_clamp_ = false;       // dyno: hold omega fixed
  double clamp_omega_ = 0.0;
  std::array<bool, 3> gate_high_{false, false, false};
  std::array<bool, 3> gate_low_{false, false, false};
  int sector_ = 0;
  double duty_ = 0.0;
  std::array<double, 3> avg_duty_{0.0, 0.0, 0.0};
  long shoot_through_requests_ = 0;
};

// Six-step sector table: returns {high_phase, low_phase} indices (0=A,1=B,2=C).
std::array<int, 2> six_step_phases(int sector);

}  // namespace bldcsim
