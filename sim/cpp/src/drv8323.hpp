// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstdint>

#include "i_gate_driver.hpp"

namespace bldcsim {

// Behavioral DRV8323RS smart gate driver (datasheet SLVSDJ3D). External-FET
// driver of the same SPI family as the DRV8301 - identical 16-bit frame
// (R/W[15] | addr[14:11] | data[10:0], mode 1), so it reuses the existing SPI
// master and the drv_manager write/verify handshake unchanged. The differences
// the model captures:
//
//  - DRV8323 register map (0x00-0x07): Fault Status 1/2, Driver Control,
//    Gate Drive HS/LS, OCP Control, CSA Control. Writes to the R/W registers
//    (0x02-0x06) are stored verbatim and echoed on read (so the controller's
//    write-addr2/addr3 + readback-verify completes); the OCP and CSA registers
//    keep their datasheet power-on defaults (the controller does not rewrite
//    them, so the part runs on those valid defaults: 6x PWM, auto-retry OCP at
//    VDS_LVL = 0.75 V, CSA gain 20 V/V).
//  - VDS overcurrent against the 16-step VDS_LVL threshold (OCP Control 0x05),
//    with the four OCP modes (latched / retry / report-only / disabled).
//  - External FETs: the OC trip senses the *external* FET's VDS = |i|*Rds_on,
//    so Rds_on is the power-stage MOSFET's, a platform param.
//  - 6-PWM truth table + DTC dead-time floor (identical to the DRV8301 path).
//  - VM UVLO (6-60 V operating range), OTW/OTSD via the die temperature.
//
// The current-sense amplifiers (gain {5,10,20,40} V/V) are modeled by the
// FeedbackChain's external-shunt-driver-amp path, like the DRV8301.
struct Drv8323Config {
  double en_gate_ready_time_s = 1e-3;     // tREADY: SPI ready <= 1 ms
  double quick_reset_pulse_max_s = 40e-6;  // EN reset pulse 8-40 us
  double dead_time_floor_s = 100e-9;       // DEAD_TIME default 100 ns
  double noctw_pulse_s = 64e-6;
  double rds_on_ohm = 4.6e-3;              // EXTERNAL power FET Rds_on
  int vds_lvl_code = 9;                    // OCP Control default 1001b = 0.75 V
  int ocp_mode_default = 1;                // 0 latch,1 retry,2 report,3 disabled
  double pvdd_uvlo_falling_v = 5.5;        // VM UVLO (6 V min operating)
  double pvdd_uvlo_hysteresis_v = 0.3;
  double otw_c = 130.0;                    // DRV830x-family typical
  double otsd_c = 150.0;
};

class Drv8323 : public IGateDriver {
 public:
  explicit Drv8323(const Drv8323Config& config);

  void update(double t_s, const DriverInputs& in,
              const std::array<double, 3>& phase_currents_a,
              double pvdd_v, double die_temp_c) override;

  const std::array<bool, 3>& gate_high() const override { return gh_; }
  const std::array<bool, 3>& gate_low() const override { return gl_; }
  bool nfault() const override { return nfault_; }
  bool noctw() const override { return noctw_; }
  bool sdo() const override { return sdo_; }
  bool pvdd_uv_active() const override { return pvdd_uv_; }
  bool dc_cal_active(int channel) const override {
    // CSA offset calibration: CSA_CAL_A/B/C are bits 4/3/2 of CSA Control 0x06.
    if (channel == 0) return (regs_[6] >> 4) & 0x1;
    if (channel == 1) return (regs_[6] >> 3) & 0x1;
    return (regs_[6] >> 2) & 0x1;
  }
  std::uint16_t reg(int addr) const override {
    return regs_[addr & 0x7];
  }
  bool ready() const override { return ready_; }
  long frame_errors() const override { return frame_errors_; }

  void inject_register_reset() override { reset_registers(); }
  void inject_otw(bool active) override { otw_injected_ = active; }
  void inject_latched_fault() override { latched_shutdown_ = true; }

  long oc_events() const { return oc_events_; }

 private:
  // OCP Control (0x05) fields.
  int ocp_mode() const { return (regs_[5] >> 6) & 0x3; }
  int vds_lvl() const { return regs_[5] & 0xF; }
  double vds_threshold_v() const;
  // Driver Control (0x02): PWM_MODE bits[6:5] (00 = 6x).
  int pwm_mode() const { return (regs_[2] >> 5) & 0x3; }

  void reset_registers();
  void latch_status1(int bit);
  void process_spi(double t_s, const DriverInputs& in);
  void latch_frame();
  void process_gates(double t_s, const DriverInputs& in,
                     const std::array<double, 3>& phase_currents_a);
  void pulse_noctw(double t_s);

  Drv8323Config config_;

  // Register file 0x00..0x07 (11-bit payloads).
  std::array<std::uint16_t, 8> regs_{};

  bool ready_ = false;
  bool prev_en_gate_ = false;
  double en_rise_time_ = -1.0;
  double en_fall_time_ = -1.0;

  // SPI shift state (mode 1, 16-bit; mirrors the DRV8301 slave).
  bool prev_sclk_ = false;
  bool prev_nscs_ = true;
  std::uint16_t shift_in_ = 0;
  std::uint16_t shift_out_ = 0;
  std::uint16_t pending_response_ = 0;
  int bit_count_ = 0;
  bool frame_valid_ = true;
  bool sdo_ = false;
  long frame_errors_ = 0;

  std::array<bool, 3> gh_{false, false, false};
  std::array<bool, 3> gl_{false, false, false};
  std::array<double, 3> gh_off_time_{-1e9, -1e9, -1e9};
  std::array<double, 3> gl_off_time_{-1e9, -1e9, -1e9};

  bool nfault_ = true;
  bool noctw_ = true;
  double noctw_until_ = -1.0;
  bool latched_shutdown_ = false;
  std::array<bool, 3> leg_latched_{};
  std::array<bool, 3> cbc_block_high_{};
  std::array<bool, 3> cbc_block_low_{};
  std::array<bool, 3> prev_inh_{};
  std::array<bool, 3> prev_inl_{};
  bool otw_injected_ = false;
  bool pvdd_uv_ = false;
  bool otsd_latched_ = false;
  long oc_events_ = 0;
};

}  // namespace bldcsim
