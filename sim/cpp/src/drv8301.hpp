#pragma once

#include <array>
#include <cstdint>

namespace bldcsim {

// Behavioral DRV8301 gate driver model (datasheet SLOS719F; verified facts in
// notes/docs-digest.md). Models, at lockstep-bench granularity:
//
//  - EN_GATE sequencing: not-ready window after enable, quick-reset pulse
//    (faults + status cleared, no full restart) vs full shutdown (registers
//    reset to power-on defaults).
//  - 6-PWM input truth table with shoot-through guard (INH=INL=1 -> both off)
//    and the DTC dead-time floor between complementary transitions.
//  - SPI slave, mode 1 (SDO shifts on SCLK rising, SDI sampled on falling),
//    16-bit frames, N+1 pipelined responses, frame-fault flag, write->SR1
//    response, SR1 latched bits cleared on read.
//  - VDS overcurrent comparison (|i| * rds_on vs OC_ADJ_SET table) with all
//    four OCP modes; nOCTW 64 us pulse stretching; nFAULT latching.
//  - Fault-injection hooks: silent register reset (slva552 brownout), OTW.
//
// Not modeled: buck converter, GVDD/AVDD rails (beyond injection), 3-PWM
// mode (RTL uses 6-PWM), SPI gate-current setting effects, analog amp
// internals (the feedback chain models those).

struct Drv8301Config {
  double en_gate_ready_time_s = 10e-3;
  double quick_reset_pulse_max_s = 10e-6;
  double dead_time_floor_s = 50e-9;
  double noctw_pulse_s = 64e-6;
  double rds_on_ohm = 4.6e-3;
  int oc_adj_set_default = 0;  // power-on default unconfirmed (Q10)
  // PVDD UVLO (the slva552 brownout cause): below falling threshold the
  // device drops gates, asserts nFAULT/PVDD_UV, and silently resets every
  // register to power-on defaults.
  double pvdd_uvlo_falling_v = 5.9;
  double pvdd_uvlo_hysteresis_v = 0.3;
  // OTW/OTSD become emergent when a thermal model supplies the die
  // temperature (realism stage 4); injection API retained.
  double otw_c = 130.0;
  double otsd_c = 150.0;
};

struct Drv8301Inputs {
  bool en_gate = false;
  bool dc_cal = false;
  std::array<bool, 3> inh{false, false, false};
  std::array<bool, 3> inl{false, false, false};
  bool nscs = true;
  bool sclk = false;
  bool sdi = false;
};

class Drv8301 {
 public:
  explicit Drv8301(const Drv8301Config& config);

  // Advance to absolute time t_s with the given pin inputs and phase currents
  // (positive out of the bridge into the motor). Call on every RTL clock
  // edge; edge detection is internal. pvdd_v defaults to a healthy rail;
  // die_temp_c drives emergent OTW/OTSD when a thermal model supplies it.
  void update(double t_s, const Drv8301Inputs& in,
              const std::array<double, 3>& phase_currents_a,
              double pvdd_v = 12.0, double die_temp_c = 25.0);

  bool pvdd_uv_active() const { return pvdd_uv_; }

  // Pin outputs.
  bool sdo() const { return sdo_; }
  bool nfault() const { return nfault_; }
  bool noctw() const { return noctw_; }
  const std::array<bool, 3>& gate_high() const { return gh_; }
  const std::array<bool, 3>& gate_low() const { return gl_; }

  // Amp calibration state for the feedback chain (channel 0 = SO1/phase A,
  // channel 1 = SO2/phase B). True when inputs are shorted.
  bool dc_cal_active(int channel) const;

  // Register access for assertions (addr 0..3).
  std::uint16_t reg(int addr) const { return regs_[addr]; }
  bool ready() const { return ready_; }
  double vds_threshold_v() const;

  // Fault injection.
  void inject_register_reset();  // slva552 silent brownout reset
  void inject_otw(bool active) { otw_injected_ = active; }
  // Force a latched shutdown (as if OTSD/OC-latch tripped): asserts nFAULT
  // until a quick-reset EN_GATE pulse or GATE_RESET clears it.
  void inject_latched_fault() { latched_shutdown_ = true; }

  // Diagnostics.
  long frame_errors() const { return frame_errors_; }
  long oc_events() const { return oc_events_; }

 private:
  // Control register 1 fields (verified bit map).
  int ocp_mode() const { return (regs_[2] >> 4) & 0x3; }
  int oc_adj_set() const { return (regs_[2] >> 6) & 0x1F; }
  bool pwm_mode_3() const { return (regs_[2] >> 3) & 0x1; }
  // Control register 2 fields.
  int octw_mode() const { return regs_[3] & 0x3; }

  void reset_registers();
  void latch_status(int bit);
  void process_spi(double t_s, const Drv8301Inputs& in);
  void latch_frame(double t_s);
  void process_gates(double t_s, const Drv8301Inputs& in,
                     const std::array<double, 3>& phase_currents_a);
  void pulse_noctw(double t_s);

  Drv8301Config config_;

  // Register file: 0=SR1, 1=SR2, 2=CR1, 3=CR2 (11-bit payloads).
  std::array<std::uint16_t, 4> regs_{};

  // EN_GATE sequencing.
  bool ready_ = false;
  bool prev_en_gate_ = false;
  double en_rise_time_ = -1.0;
  double en_fall_time_ = -1.0;

  // SPI shift state.
  bool prev_sclk_ = false;
  bool prev_nscs_ = true;
  std::uint16_t shift_in_ = 0;
  std::uint16_t shift_out_ = 0;
  std::uint16_t pending_response_ = 0;
  int bit_count_ = 0;
  bool frame_valid_ = true;
  bool sdo_ = false;
  long frame_errors_ = 0;

  // Gate outputs and dead-time floor tracking.
  std::array<bool, 3> gh_{false, false, false};
  std::array<bool, 3> gl_{false, false, false};
  std::array<double, 3> gh_off_time_{-1e9, -1e9, -1e9};
  std::array<double, 3> gl_off_time_{-1e9, -1e9, -1e9};

  // Protection state.
  bool nfault_ = true;  // open-drain, true = released (high)
  bool noctw_ = true;
  double noctw_until_ = -1.0;
  bool latched_shutdown_ = false;       // OTSD / OC-latch style shutdown
  std::array<bool, 3> leg_latched_{};   // per-leg OC latch (latch mode)
  // Cycle-by-cycle current-limit: FET held off until its input deasserts.
  std::array<bool, 3> cbc_block_high_{};
  std::array<bool, 3> cbc_block_low_{};
  std::array<bool, 3> prev_inh_{};
  std::array<bool, 3> prev_inl_{};
  bool otw_injected_ = false;
  bool pvdd_uv_ = false;
  bool otsd_latched_ = false;
  long oc_events_ = 0;
  bool last_dc_cal_pin_ = false;

  double time_s_ = 0.0;
};

}  // namespace bldcsim
