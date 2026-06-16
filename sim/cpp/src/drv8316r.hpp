// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstdint>

#include "i_gate_driver.hpp"

namespace bldcsim {

// Behavioral DRV8316R integrated-FET motor driver (datasheet SLVSF16B). The
// "clean reference BOM" driver: it integrates the three half-bridges, the power
// FETs, and the three current-sense amplifiers - so there are no external
// shunts or discrete amps (retires the Q7 clone-passive uncertainty).
//
// What this model owns: gate drive (6-PWM truth table + dead-time floor),
// integrated-FET overcurrent (a fixed 16 A / 24 A current limit, not a VDS
// sense), VM UVLO, OTW/OTSD, EN/nSLEEP sequencing. The DRV8316R is operational
// on its power-on defaults (6x PWM), so the reference platform runs it via the
// controller's hardware path (no SPI reconfiguration) - hence the SPI surface
// is represented at datasheet defaults (sdo/reg/frame_errors are inert here).
//
// The integrated current-sense amplifier (Vo = VREF/2 +/- GCSA*I, GCSA in V/A,
// bidirectional, low-side) is modeled by the FeedbackChain's
// kIntegratedDriverCsa source, NOT here: this model owns gate drive and
// protection, the chain owns the analog sense node the host ADC digitizes (the
// integrated CSA replaces the external shunt + discrete amp, not the ADC).
struct Drv8316rConfig {
  double en_gate_ready_time_s = 1e-3;   // nSLEEP wake / ready window
  double dead_time_floor_s = 50e-9;
  double noctw_pulse_s = 64e-6;
  double rds_on_ohm = 95e-3;            // integrated leg Rds(on) (envelope)
  double ocp_level_a = 16.0;            // OCP_LVL = 0 -> 16 A integrated limit
  double pvdd_uvlo_falling_v = 4.2;     // VM UVLO falling (rising 4.4)
  double pvdd_uvlo_hysteresis_v = 0.2;
  double otw_c = 170.0;                 // OTW typ (datasheet)
  double otsd_c = 185.0;                // OTS typ (datasheet)
  bool ocp_latch = true;               // OCP_MODE default = latched
};

class Drv8316r : public IGateDriver {
 public:
  explicit Drv8316r(const Drv8316rConfig& config) : config_(config) {}

  void update(double t_s, const DriverInputs& in,
              const std::array<double, 3>& phase_currents_a,
              double pvdd_v, double die_temp_c) override;

  const std::array<bool, 3>& gate_high() const override { return gh_; }
  const std::array<bool, 3>& gate_low() const override { return gl_; }
  bool nfault() const override { return nfault_; }
  bool noctw() const override { return noctw_; }
  bool sdo() const override { return false; }       // SPI inert (defaults)
  bool pvdd_uv_active() const override { return pvdd_uv_; }
  // The integrated CSA offset calibration tracks the DC_CAL window (the chain
  // shorts the CSA to capture the VREF/2 midpoint offset).
  bool dc_cal_active(int /*channel*/) const override { return dc_cal_; }
  std::uint16_t reg(int /*addr*/) const override { return 0; }
  bool ready() const override { return ready_; }
  long frame_errors() const override { return 0; }

  void inject_register_reset() override { ready_ = false; en_rise_time_ = -1.0; }
  void inject_otw(bool active) override { otw_injected_ = active; }
  void inject_latched_fault() override { otsd_latched_ = true; }

  long oc_events() const { return oc_events_; }

 private:
  Drv8316rConfig config_;

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

  bool nfault_ = true;
  bool noctw_ = true;
  double noctw_until_ = -1.0;
  bool pvdd_uv_ = false;
  bool otsd_latched_ = false;
  bool otw_injected_ = false;
  long oc_events_ = 0;
};

}  // namespace bldcsim
