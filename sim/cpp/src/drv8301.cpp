// SPDX-License-Identifier: MIT
#include "drv8301.hpp"

#include <cmath>

namespace bldcsim {

namespace {

// OC_ADJ_SET code -> VDS trip threshold (SLOS719F Table 13).
constexpr std::array<double, 32> kVdsTable{
    0.060, 0.068, 0.076, 0.086, 0.097, 0.109, 0.123, 0.138,
    0.155, 0.175, 0.197, 0.222, 0.250, 0.282, 0.317, 0.358,
    0.403, 0.454, 0.511, 0.576, 0.648, 0.730, 0.822, 0.926,
    1.043, 1.175, 1.324, 1.491, 1.679, 1.892, 2.131, 2.400,
};

// Status register 1 bit positions (verified map).
constexpr int kBitFault = 10;
constexpr int kBitPvddUv = 8;
constexpr int kBitOtsd = 7;
constexpr int kBitOtw = 6;
constexpr std::array<int, 3> kBitFetHighOc{5, 3, 1};  // FETHA, FETHB, FETHC
constexpr std::array<int, 3> kBitFetLowOc{4, 2, 0};   // FETLA, FETLB, FETLC

constexpr std::uint16_t kDeviceId = 0x1;  // SR2 D[3:0]; verify on hardware

constexpr int kOcpCurrentLimit = 0;
constexpr int kOcpLatchShutdown = 1;
constexpr int kOcpReportOnly = 2;
constexpr int kOcpDisabled = 3;

}  // namespace

Drv8301::Drv8301(const Drv8301Config& config) : config_(config) {
  reset_registers();
}

void Drv8301::reset_registers() {
  regs_ = {0, kDeviceId, 0, 0};
  // OC_ADJ_SET power-on default is configurable because the datasheet table
  // does not clearly mark it (Q10); everything else defaults to zero.
  regs_[2] = static_cast<std::uint16_t>((config_.oc_adj_set_default & 0x1F)
                                        << 6);
}

double Drv8301::vds_threshold_v() const {
  return kVdsTable[static_cast<std::size_t>(oc_adj_set())];
}

bool Drv8301::dc_cal_active(int channel) const {
  // Pin applies to both channels; SPI bits are per channel (CR2 D4/D5).
  const bool spi_bit = (channel == 0) ? ((regs_[3] >> 4) & 0x1)
                                      : ((regs_[3] >> 5) & 0x1);
  return last_dc_cal_pin_ || spi_bit;
}

void Drv8301::latch_status(int bit) {
  regs_[0] = static_cast<std::uint16_t>(regs_[0] | (1u << bit) |
                                        (1u << kBitFault));
}

void Drv8301::pulse_noctw(double t_s) {
  noctw_until_ = t_s + config_.noctw_pulse_s;
}

void Drv8301::update(double t_s, const Drv8301Inputs& in,
                     const std::array<double, 3>& phase_currents_a,
                     double pvdd_v, double die_temp_c) {
  time_s_ = t_s;
  last_dc_cal_pin_ = in.dc_cal;

  // ---- PVDD UVLO (slva552 brownout cause) ---------------------------------
  if (!pvdd_uv_ && pvdd_v < config_.pvdd_uvlo_falling_v) {
    pvdd_uv_ = true;
    reset_registers();  // silent register reset while the device keeps running
    latch_status(kBitPvddUv);
  } else if (pvdd_uv_ &&
             pvdd_v > config_.pvdd_uvlo_falling_v +
                          config_.pvdd_uvlo_hysteresis_v) {
    pvdd_uv_ = false;
  }

  // ---- Emergent thermal warnings/shutdown (die temperature) ---------------
  if (die_temp_c >= config_.otsd_c && !otsd_latched_) {
    otsd_latched_ = true;
    latch_status(kBitOtsd);
    pulse_noctw(t_s);
  }
  if (die_temp_c >= config_.otw_c) {
    latch_status(kBitOtw);
    if (octw_mode() == 0 || octw_mode() == 1) {
      pulse_noctw(t_s);
    }
  }

  // ---- EN_GATE sequencing -------------------------------------------------
  if (in.en_gate && !prev_en_gate_) {
    const double low_time = (en_fall_time_ >= 0.0) ? t_s - en_fall_time_ : 1e9;
    if (low_time <= config_.quick_reset_pulse_max_s && en_rise_time_ >= 0.0) {
      // Quick reset: clear faults and status registers, stay configured.
      regs_[0] = 0;
      latched_shutdown_ = false;
      otsd_latched_ = false;
      leg_latched_ = {false, false, false};
      // ready_ keeps its previous value: no full restart.
    } else {
      en_rise_time_ = t_s;
      ready_ = false;
    }
  }
  if (!in.en_gate && prev_en_gate_) {
    en_fall_time_ = t_s;
  }
  if (!in.en_gate) {
    // Holding low beyond the quick-reset window is a full shutdown.
    if (en_fall_time_ >= 0.0 &&
        t_s - en_fall_time_ > config_.quick_reset_pulse_max_s) {
      if (ready_ || regs_[2] != 0 || regs_[3] != 0) {
        reset_registers();
      }
      ready_ = false;
      latched_shutdown_ = false;
      otsd_latched_ = false;
      leg_latched_ = {false, false, false};
    }
  } else if (!ready_ && en_rise_time_ >= 0.0 &&
             t_s - en_rise_time_ >= config_.en_gate_ready_time_s) {
    ready_ = true;
  }
  prev_en_gate_ = in.en_gate;

  // ---- SPI ----------------------------------------------------------------
  process_spi(t_s, in);

  // ---- GATE_RESET self-clearing bit (CR1 D2) ------------------------------
  if ((regs_[2] >> 2) & 0x1) {
    latched_shutdown_ = false;
    otsd_latched_ = false;
    leg_latched_ = {false, false, false};
    regs_[2] = static_cast<std::uint16_t>(regs_[2] & ~(1u << 2));
  }

  // ---- OTW injection ------------------------------------------------------
  if (otw_injected_) {
    latch_status(kBitOtw);
    if (octw_mode() == 0 || octw_mode() == 1) {  // report OT
      pulse_noctw(t_s);
    }
  }

  // ---- Gates + overcurrent ------------------------------------------------
  process_gates(t_s, in, phase_currents_a);

  // ---- Fault pins ---------------------------------------------------------
  const bool any_latched =
      latched_shutdown_ || otsd_latched_ || leg_latched_[0] ||
      leg_latched_[1] || leg_latched_[2];
  nfault_ = !any_latched && !pvdd_uv_;
  noctw_ = !(t_s < noctw_until_);
}

void Drv8301::process_spi(double t_s, const Drv8301Inputs& in) {
  if (!prev_nscs_ && in.nscs) {
    // Frame end.
    if (frame_valid_ && bit_count_ == 16 && !in.sclk) {
      latch_frame(t_s);
    } else {
      ++frame_errors_;
      pending_response_ = static_cast<std::uint16_t>(1u << 15);
    }
    sdo_ = false;  // high impedance, modeled as low
  }
  if (prev_nscs_ && !in.nscs) {
    // Frame start: clock must be low when nSCS falls.
    frame_valid_ = !in.sclk;
    bit_count_ = 0;
    shift_in_ = 0;
    shift_out_ = pending_response_;
  }
  if (!in.nscs) {
    if (in.sclk && !prev_sclk_) {
      // Rising edge: shift response out, MSB first.
      sdo_ = (shift_out_ & 0x8000u) != 0;
      shift_out_ = static_cast<std::uint16_t>(shift_out_ << 1);
    }
    if (!in.sclk && prev_sclk_) {
      // Falling edge: sample SDI.
      shift_in_ = static_cast<std::uint16_t>((shift_in_ << 1) |
                                             (in.sdi ? 1u : 0u));
      ++bit_count_;
      if (bit_count_ > 16) {
        frame_valid_ = false;
      }
    }
  }
  prev_sclk_ = in.sclk;
  prev_nscs_ = in.nscs;
}

void Drv8301::latch_frame(double /*t_s*/) {
  const bool read = (shift_in_ >> 15) & 0x1;
  const int addr = (shift_in_ >> 11) & 0xF;
  const std::uint16_t data = shift_in_ & 0x7FF;

  if (read) {
    if (addr <= 3) {
      pending_response_ = static_cast<std::uint16_t>(
          ((addr & 0xF) << 11) | (regs_[addr] & 0x7FF));
      if (addr == 0) {
        regs_[0] = 0;  // SR1 latched bits clear on read
      }
    } else {
      pending_response_ = static_cast<std::uint16_t>((addr & 0xF) << 11);
    }
  } else {
    // Writes only land once the device is up (SPI ready follows EN_GATE).
    if (ready_ && (addr == 2 || addr == 3)) {
      regs_[addr] = data;
    }
    // Write response carries Status Register 1.
    pending_response_ =
        static_cast<std::uint16_t>((0x0 << 11) | (regs_[0] & 0x7FF));
  }
}

void Drv8301::process_gates(double t_s, const Drv8301Inputs& in,
                            const std::array<double, 3>& phase_currents_a) {
  const bool enabled = in.en_gate && ready_ && !latched_shutdown_ &&
                       !pvdd_uv_ && !otsd_latched_;

  for (int k = 0; k < 3; ++k) {
    // Input edges release cycle-by-cycle blocks (next PWM cycle).
    if (in.inh[k] && !prev_inh_[k]) cbc_block_high_[k] = false;
    if (in.inl[k] && !prev_inl_[k]) cbc_block_low_[k] = false;
    prev_inh_[k] = in.inh[k];
    prev_inl_[k] = in.inl[k];

    // 6-PWM truth table with shoot-through guard.
    bool want_gh = enabled && in.inh[k] && !in.inl[k];
    bool want_gl = enabled && in.inl[k] && !in.inh[k];

    if (leg_latched_[k]) {
      want_gh = false;
      want_gl = false;
    }
    if (cbc_block_high_[k]) want_gh = false;
    if (cbc_block_low_[k]) want_gl = false;

    // DTC dead-time floor: a gate may not turn on until the complementary
    // gate has been off for at least the floor time.
    if (want_gh && !gh_[k] &&
        t_s - gl_off_time_[k] < config_.dead_time_floor_s && !gl_[k]) {
      want_gh = false;
    }
    if (want_gh && gl_[k]) {
      want_gh = false;  // complementary still on (handshake)
    }
    if (want_gl && !gl_[k] &&
        t_s - gh_off_time_[k] < config_.dead_time_floor_s && !gh_[k]) {
      want_gl = false;
    }
    if (want_gl && gh_[k]) {
      want_gl = false;
    }

    if (gh_[k] && !want_gh) gh_off_time_[k] = t_s;
    if (gl_[k] && !want_gl) gl_off_time_[k] = t_s;
    gh_[k] = want_gh;
    gl_[k] = want_gl;

    // VDS overcurrent on the conducting FET.
    if (ocp_mode() != kOcpDisabled && enabled) {
      const double vds = std::abs(phase_currents_a[k]) * config_.rds_on_ohm;
      const bool high_conducting = gh_[k];
      const bool low_conducting = gl_[k];
      if ((high_conducting || low_conducting) && vds > vds_threshold_v()) {
        ++oc_events_;
        latch_status(high_conducting ? kBitFetHighOc[k] : kBitFetLowOc[k]);
        if (octw_mode() == 0 || octw_mode() == 2) {  // report OC
          pulse_noctw(t_s);
        }
        switch (ocp_mode()) {
          case kOcpCurrentLimit:
            if (high_conducting) {
              cbc_block_high_[k] = true;
              gh_[k] = false;
              gh_off_time_[k] = t_s;
            }
            if (low_conducting) {
              cbc_block_low_[k] = true;
              gl_[k] = false;
              gl_off_time_[k] = t_s;
            }
            break;
          case kOcpLatchShutdown:
            leg_latched_[k] = true;
            gh_[k] = false;
            gl_[k] = false;
            gh_off_time_[k] = t_s;
            gl_off_time_[k] = t_s;
            break;
          default:
            break;  // report only
        }
      }
    }
  }
}

void Drv8301::inject_register_reset() {
  // slva552: supply brownout silently resets every register to power-on
  // defaults while the device keeps running.
  reset_registers();
}

}  // namespace bldcsim
