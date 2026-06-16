// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstdint>

#include "i_gate_driver.hpp"

namespace bldcsim {

// Behavioral DRV8302 gate driver (datasheet SLES267C). Same family as the
// DRV8301 but HARDWARE-controlled - there is no SPI register file; dead time,
// OC threshold/mode, PWM mode, and amp gain are set by strapping pins
// (modeled as config). Models, at lockstep-bench granularity:
//
//  - EN_GATE sequencing: not-ready window after enable; nFAULT releases (high)
//    once the driver is ready (internal EN_GATE high). No SPI config step.
//  - 6-PWM truth table (Table 1) with shoot-through hand-shake and the DTC
//    dead-time floor between complementary transitions.
//  - VDS overcurrent (VDS = |i|*Rds_on vs the OC_ADJ threshold) with the two
//    M_OC modes: current-limit (cycle-by-cycle, nOCTW 64 us pulse) and latch
//    shutdown (leg FETs off, nFAULT latched until an EN_GATE reset).
//  - PVDD UVLO (gates low + nFAULT) and OTW/OTSD via the die temperature.
//
// The current-shunt amplifiers (Vo = Vref/2 - G*(SN-SP), G in {10,40}) are
// modeled by the FeedbackChain like the DRV8301 path; this model owns the gate
// drive and protection only.
struct Drv8302Config {
  double en_gate_ready_time_s = 1e-3;
  double dead_time_floor_s = 50e-9;     // DTC shorted = 50 ns minimum
  double noctw_pulse_s = 64e-6;
  double rds_on_ohm = 4.6e-3;
  double oc_threshold_v = 0.403;        // OC_ADJ VDS trip (resistor-set)
  bool oc_latch_mode = false;           // M_OC: false=current-limit, true=latch
  bool oc_disabled = false;             // OC_ADJ tied to DVDD disables OCP
  double pvdd_uvlo_falling_v = 8.0;     // PVDD UVLO (8-60 V operating range)
  double pvdd_uvlo_hysteresis_v = 0.3;
  double otw_c = 130.0;
  double otsd_c = 150.0;
};

class Drv8302 : public IGateDriver {
 public:
  explicit Drv8302(const Drv8302Config& config) : config_(config) {}

  void update(double t_s, const DriverInputs& in,
              const std::array<double, 3>& phase_currents_a,
              double pvdd_v, double die_temp_c) override;

  const std::array<bool, 3>& gate_high() const override { return gh_; }
  const std::array<bool, 3>& gate_low() const override { return gl_; }
  bool nfault() const override { return nfault_; }
  bool noctw() const override { return noctw_; }
  bool sdo() const override { return false; }          // no SPI
  bool pvdd_uv_active() const override { return pvdd_uv_; }
  bool dc_cal_active(int channel) const override {
    return dc_cal_ && (channel == 0 || channel == 1);
  }
  std::uint16_t reg(int /*addr*/) const override { return 0; }  // no registers
  bool ready() const override { return ready_; }
  long frame_errors() const override { return 0; }     // no SPI frames

  void inject_otw(bool active) override { otw_injected_ = active; }
  void inject_latched_fault() override { otsd_latched_ = true; }
  // The slva-style PVDD brownout: a transient drops the driver out of ready.
  void inject_register_reset() override { ready_ = false; en_rise_time_ = -1.0; }

  long oc_events() const { return oc_events_; }

 private:
  Drv8302Config config_;

  bool ready_ = false;
  bool prev_en_gate_ = false;
  double en_rise_time_ = -1.0;
  bool dc_cal_ = false;

  std::array<bool, 3> gh_{false, false, false};
  std::array<bool, 3> gl_{false, false, false};
  std::array<double, 3> gh_off_time_{-1e9, -1e9, -1e9};
  std::array<double, 3> gl_off_time_{-1e9, -1e9, -1e9};
  std::array<bool, 3> leg_latched_{};
  std::array<bool, 3> cbc_block_high_{};
  std::array<bool, 3> cbc_block_low_{};
  std::array<bool, 3> prev_inh_{};
  std::array<bool, 3> prev_inl_{};

  bool nfault_ = true;   // open-drain, true = released (high)
  bool noctw_ = true;
  double noctw_until_ = -1.0;
  bool pvdd_uv_ = false;
  bool otsd_latched_ = false;
  bool otw_injected_ = false;
  long oc_events_ = 0;
};

}  // namespace bldcsim
