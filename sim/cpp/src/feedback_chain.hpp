#pragma once

#include <array>
#include <cmath>
#include <random>

#include "three_phase_plant.hpp"

namespace bldcsim {

// Analog feedback chain between the power stage and the MCP3208 inputs,
// using TI-EVM-baseline values (status ti-evm-baseline, Q7/Q11 — verify on
// the physical ZONRI board):
//
//  ch0..2  IOUTA/B/C : shunt + amp, vo = offset + gain * Rsh * i_phase while
//                      the leg's LOW-side path conducts (low-side shunts see
//                      no current during high-side conduction!); offset
//                      otherwise. DC_CAL shorts amp inputs on ch0/ch1.
//                      Sign convention documented here is a Q7 verify item.
//  ch3..5  EMF-A/B/C : divider * terminal voltage through first-order RC.
//  ch6     VBUS      : divider * bus voltage.
//  ch7     unused    : 0 V.
//
// All outputs clamp to [0, rail].

struct FeedbackChainConfig {
  double shunt_ohm = 2.0e-3;
  double amp_gain = 10.0;
  double amp_offset_v = 1.65;
  double emf_divider = 0.095;
  double emf_rc_cutoff_hz = 1.6e3;
  double bus_divider = 0.157;
  double rail_v = 3.3;
  // Additive gaussian noise on every channel, deterministic via the seed
  // (the bench must stay reproducible run to run).
  double noise_rms_v = 0.0;
  unsigned noise_seed = 0x5eed;
  // Correlated disturbances (realism stage 3; zero = off):
  // ground shift: common-mode offset = r*i_return + l*di_return/dt.
  double gnd_shift_r_ohm = 0.0;
  double gnd_shift_l_h = 0.0;
  // gate-edge transient: decaying spike on current/EMF channels per edge.
  double edge_spike_v = 0.0;
  double edge_spike_tau_s = 1e-6;
};

class FeedbackChain {
 public:
  explicit FeedbackChain(const FeedbackChainConfig& config)
      : config_(config), rng_(config.noise_seed) {}

  void set_dc_cal(int channel, bool active) {
    if (channel == 0 || channel == 1) dc_cal_[channel] = active;
  }

  // ADC sampling charge-share feedback: the EMF channels' filter cap is the
  // reservoir the sample cap charges from, so each conversion shifts the
  // filtered node by the theft voltage; it then recovers through the RC.
  void apply_sample_theft(int channel, double delta_v) {
    if (channel >= 3 && channel <= 5) {
      emf_filtered_[channel - 3] += delta_v;
    }
  }

  // Correlated-disturbance inputs (bench-driven, realism stage 3).
  void set_bus_return_current(double i_a, double didt_a_s) {
    gnd_shift_v_ = config_.gnd_shift_r_ohm * i_a +
                   config_.gnd_shift_l_h * didt_a_s;
  }
  // Called on any gate switching edge; sign keys the spike polarity.
  void notify_gate_edge(double sign) {
    if (config_.edge_spike_v != 0.0) {
      std::uniform_real_distribution<double> jitter(0.7, 1.3);
      edge_spike_state_ += sign * config_.edge_spike_v * jitter(rng_);
    }
  }
  double gnd_shift_v() const { return gnd_shift_v_; }

  // Advance RC states by dt and recompute channel voltages from the plant.
  void update(double dt_s, const ThreePhaseState& state,
              const ThreePhaseOutputs& out, double vbus_v) {
    const double tau = 1.0 / (2.0 * M_PI * config_.emf_rc_cutoff_hz);
    const double alpha = 1.0 - std::exp(-dt_s / tau);
    edge_spike_state_ *= std::exp(-dt_s / config_.edge_spike_tau_s);

    // Ground shift is a common-mode term on every channel; edge spikes hit
    // the current/EMF measurement paths.
    const double cm = gnd_shift_v_;
    const double spike = edge_spike_state_;

    for (int k = 0; k < 3; ++k) {
      // Current channels: low-side shunt conducts in DRIVEN_LOW / DIODE_LOW.
      const bool low_conducting = out.mode[k] == LegMode::kDrivenLow ||
                                  out.mode[k] == LegMode::kDiodeLow;
      double vo = config_.amp_offset_v;
      const bool calibrating = (k < 2) && dc_cal_[k];
      if (low_conducting && !calibrating) {
        vo += config_.amp_gain * config_.shunt_ohm * state.current_a[k];
      }
      channels_[k] = clamp(vo + cm + spike);

      // EMF channels: divider + RC.
      const double target = config_.emf_divider * out.terminal_v[k];
      emf_filtered_[k] += alpha * (target - emf_filtered_[k]);
      channels_[3 + k] = clamp(emf_filtered_[k] + cm + spike);
    }
    channels_[6] = clamp(config_.bus_divider * vbus_v + cm);
    channels_[7] = 0.0;

    if (config_.noise_rms_v > 0.0) {
      std::normal_distribution<double> noise(0.0, config_.noise_rms_v);
      for (int k = 0; k < 7; ++k) {
        channels_[k] = clamp(channels_[k] + noise(rng_));
      }
    }
  }

  // FOC phase-current sample-and-hold (foc-checklist stage 4, Q21). In the
  // simultaneous scheme an external S/H (or dual ADC) freezes all three
  // low-side-shunt currents at the PWM-counter peak, so the sequencer's two
  // sequential reads both reflect the same instant; in the sequential scheme
  // this is off and each read samples live (the ~22 us inter-conversion skew
  // is real). The bench arms the latch at each counter peak.
  void set_simultaneous_currents(bool on) { simultaneous_currents_ = on; }
  void latch_currents() {
    for (int k = 0; k < 3; ++k) held_current_[k] = channels_[k];
    current_latched_ = true;
  }

  double channel(int idx) const {
    if (simultaneous_currents_ && current_latched_ && idx >= 0 && idx < 3) {
      return held_current_[idx];
    }
    return channels_[idx];
  }
  const std::array<double, 8>& channels() const { return channels_; }

 private:
  double clamp(double v) const {
    if (v < 0.0) return 0.0;
    if (v > config_.rail_v) return config_.rail_v;
    return v;
  }

  FeedbackChainConfig config_;
  std::mt19937 rng_;
  std::array<double, 8> channels_{};
  std::array<double, 3> emf_filtered_{};
  std::array<bool, 2> dc_cal_{false, false};
  double gnd_shift_v_ = 0.0;
  double edge_spike_state_ = 0.0;
  bool simultaneous_currents_ = false;
  bool current_latched_ = false;
  std::array<double, 3> held_current_{};
};

}  // namespace bldcsim
