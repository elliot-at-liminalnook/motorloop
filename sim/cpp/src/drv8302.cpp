// SPDX-License-Identifier: MIT
#include "drv8302.hpp"

#include <cmath>

namespace bldcsim {

void Drv8302::update(double t_s, const DriverInputs& in,
                     const std::array<double, 3>& phase_currents_a,
                     double pvdd_v, double die_temp_c) {
  dc_cal_ = in.dc_cal;

  // EN_GATE sequencing: ready after the settle window (no SPI config step).
  if (in.en_gate && !prev_en_gate_) en_rise_time_ = t_s;
  if (!in.en_gate) {
    en_rise_time_ = -1.0;
    ready_ = false;
    otsd_latched_ = false;  // EN_GATE reset clears latched faults
  } else if (en_rise_time_ >= 0.0 &&
             t_s - en_rise_time_ >= config_.en_gate_ready_time_s) {
    ready_ = true;
  }
  prev_en_gate_ = in.en_gate;

  // PVDD UVLO with hysteresis (the brownout cause).
  if (pvdd_v < config_.pvdd_uvlo_falling_v) {
    pvdd_uv_ = true;
  } else if (pvdd_v >
             config_.pvdd_uvlo_falling_v + config_.pvdd_uvlo_hysteresis_v) {
    pvdd_uv_ = false;
  }

  // Overtemperature: OTW (warning, nOCTW) and OTSD (latched shutdown, nFAULT).
  const bool otw = otw_injected_ || die_temp_c >= config_.otw_c;
  if (die_temp_c >= config_.otsd_c) otsd_latched_ = true;

  // nOCTW: pulse-stretched low while an OTW/OC report is active.
  if (otw) noctw_until_ = t_s + config_.noctw_pulse_s;

  const bool enabled = in.en_gate && ready_ && !pvdd_uv_ && !otsd_latched_;

  for (int k = 0; k < 3; ++k) {
    if (in.inh[k] && !prev_inh_[k]) cbc_block_high_[k] = false;
    if (in.inl[k] && !prev_inl_[k]) cbc_block_low_[k] = false;
    prev_inh_[k] = in.inh[k];
    prev_inl_[k] = in.inl[k];

    // 6-PWM truth table (Table 1) with shoot-through guard.
    bool want_gh = enabled && in.inh[k] && !in.inl[k];
    bool want_gl = enabled && in.inl[k] && !in.inh[k];
    if (leg_latched_[k]) { want_gh = false; want_gl = false; }
    if (cbc_block_high_[k]) want_gh = false;
    if (cbc_block_low_[k]) want_gl = false;

    // DTC dead-time floor + complementary hand-shake.
    if (want_gh && !gh_[k] &&
        t_s - gl_off_time_[k] < config_.dead_time_floor_s && !gl_[k]) {
      want_gh = false;
    }
    if (want_gh && gl_[k]) want_gh = false;
    if (want_gl && !gl_[k] &&
        t_s - gh_off_time_[k] < config_.dead_time_floor_s && !gh_[k]) {
      want_gl = false;
    }
    if (want_gl && gh_[k]) want_gl = false;

    if (gh_[k] && !want_gh) gh_off_time_[k] = t_s;
    if (gl_[k] && !want_gl) gl_off_time_[k] = t_s;
    gh_[k] = want_gh;
    gl_[k] = want_gl;

    // VDS overcurrent on the conducting FET.
    if (!config_.oc_disabled && enabled && (gh_[k] || gl_[k])) {
      const double vds = std::abs(phase_currents_a[k]) * config_.rds_on_ohm;
      if (vds > config_.oc_threshold_v) {
        ++oc_events_;
        noctw_until_ = t_s + config_.noctw_pulse_s;  // report OC on nOCTW
        if (config_.oc_latch_mode) {
          leg_latched_[k] = true;
          gh_[k] = false; gl_[k] = false;
          gh_off_time_[k] = t_s; gl_off_time_[k] = t_s;
        } else {
          // Current-limit: shut the offending FET off until its input cycles.
          if (gh_[k]) { cbc_block_high_[k] = true; gh_[k] = false;
                        gh_off_time_[k] = t_s; }
          if (gl_[k]) { cbc_block_low_[k] = true; gl_[k] = false;
                        gl_off_time_[k] = t_s; }
        }
      }
    }
  }

  // nFAULT: low (asserted) on UVLO or any latched shutdown; high once ready.
  const bool any_latched = otsd_latched_ ||
      leg_latched_[0] || leg_latched_[1] || leg_latched_[2];
  nfault_ = !(pvdd_uv_ || any_latched) && ready_;
  noctw_ = !(t_s < noctw_until_);
}

}  // namespace bldcsim
