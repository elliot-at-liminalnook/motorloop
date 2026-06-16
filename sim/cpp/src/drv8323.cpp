// SPDX-License-Identifier: MIT
#include "drv8323.hpp"

#include <cmath>

namespace bldcsim {

namespace {

// OCP Control VDS_LVL code -> VDS overcurrent threshold (SLVSDJ3D Table 8-18).
constexpr std::array<double, 16> kVdsLvlTable{
    0.060, 0.130, 0.200, 0.260, 0.310, 0.450, 0.530, 0.600,
    0.680, 0.750, 0.940, 1.130, 1.300, 1.500, 1.700, 1.880,
};

// Fault Status 1 (0x00) bit positions (SLVSDJ3D Table 8-12).
constexpr int kBitFault = 10;     // FAULT (mirrors nFAULT)
constexpr int kBitVdsOcp = 9;
constexpr int kBitUvlo = 7;
constexpr int kBitOtsd = 6;
// Per-FET VDS_OCP bits: HA5 LA4 HB3 LB2 HC1 LC0.
constexpr std::array<int, 3> kBitFetHighOc{5, 3, 1};
constexpr std::array<int, 3> kBitFetLowOc{4, 2, 0};

constexpr int kOcpLatched = 0;
constexpr int kOcpRetry = 1;
constexpr int kOcpReport = 2;
constexpr int kOcpDisabled = 3;

}  // namespace

Drv8323::Drv8323(const Drv8323Config& config) : config_(config) {
  reset_registers();
}

void Drv8323::reset_registers() {
  regs_ = {0, 0, 0, 0, 0, 0, 0, 0};
  // Datasheet power-on defaults for the registers the controller does NOT
  // rewrite, so the part runs on valid defaults.
  // Driver Control (0x02): PWM_MODE = 6x (0).
  regs_[2] = 0;
  // Gate Drive HS/LS (0x03/0x04): LOCK = 011b unlock, IDRIVE = max (1111).
  regs_[3] = static_cast<std::uint16_t>((0x3 << 8) | (0xF << 4) | 0xF);
  regs_[4] = static_cast<std::uint16_t>((0x1 << 8) | (0xF << 4) | 0xF);
  // OCP Control (0x05): OCP_MODE, DEAD_TIME=100ns(01), OCP_DEG=4us(01), VDS_LVL.
  regs_[5] = static_cast<std::uint16_t>(
      ((config_.ocp_mode_default & 0x3) << 6) | (0x1 << 8) | (0x1 << 4) |
      (config_.vds_lvl_code & 0xF));
  // CSA Control (0x06): CSA_GAIN=20V/V(10b), VREF_DIV bidirectional, SEN_LVL=1V.
  regs_[6] = static_cast<std::uint16_t>((0x1 << 9) | (0x2 << 6) | 0x3);
}

double Drv8323::vds_threshold_v() const {
  return kVdsLvlTable[static_cast<std::size_t>(vds_lvl())];
}

void Drv8323::latch_status1(int bit) {
  regs_[0] = static_cast<std::uint16_t>(regs_[0] | (1u << bit) |
                                        (1u << kBitFault));
}

void Drv8323::pulse_noctw(double t_s) {
  noctw_until_ = t_s + config_.noctw_pulse_s;
}

void Drv8323::update(double t_s, const DriverInputs& in,
                     const std::array<double, 3>& phase_currents_a,
                     double pvdd_v, double die_temp_c) {
  // ---- VM UVLO ------------------------------------------------------------
  if (!pvdd_uv_ && pvdd_v < config_.pvdd_uvlo_falling_v) {
    pvdd_uv_ = true;
    reset_registers();
    latch_status1(kBitUvlo);
  } else if (pvdd_uv_ && pvdd_v > config_.pvdd_uvlo_falling_v +
                                      config_.pvdd_uvlo_hysteresis_v) {
    pvdd_uv_ = false;
  }

  // ---- Overtemperature ----------------------------------------------------
  if (die_temp_c >= config_.otsd_c && !otsd_latched_) {
    otsd_latched_ = true;
    latch_status1(kBitOtsd);
    pulse_noctw(t_s);
  }
  if (otw_injected_ || die_temp_c >= config_.otw_c) pulse_noctw(t_s);

  // ---- EN_GATE sequencing -------------------------------------------------
  if (in.en_gate && !prev_en_gate_) {
    const double low_time = (en_fall_time_ >= 0.0) ? t_s - en_fall_time_ : 1e9;
    if (low_time <= config_.quick_reset_pulse_max_s && en_rise_time_ >= 0.0) {
      // Short EN reset pulse clears faults without a full restart.
      regs_[0] = 0;
      latched_shutdown_ = false;
      otsd_latched_ = false;
      leg_latched_ = {false, false, false};
    } else {
      en_rise_time_ = t_s;
      ready_ = false;
    }
  }
  if (!in.en_gate && prev_en_gate_) en_fall_time_ = t_s;
  if (!in.en_gate) {
    if (en_fall_time_ >= 0.0 &&
        t_s - en_fall_time_ > config_.quick_reset_pulse_max_s) {
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

  // ---- SPI + gates --------------------------------------------------------
  process_spi(t_s, in);
  process_gates(t_s, in, phase_currents_a);

  // ---- Fault pins ---------------------------------------------------------
  const bool any_latched = latched_shutdown_ || otsd_latched_ ||
                           leg_latched_[0] || leg_latched_[1] || leg_latched_[2];
  nfault_ = !any_latched && !pvdd_uv_;
  noctw_ = !(t_s < noctw_until_);
}

void Drv8323::process_spi(double t_s, const DriverInputs& in) {
  (void)t_s;
  if (!prev_nscs_ && in.nscs) {
    if (frame_valid_ && bit_count_ == 16 && !in.sclk) {
      latch_frame();
    } else {
      ++frame_errors_;
      pending_response_ = static_cast<std::uint16_t>(1u << 15);
    }
    sdo_ = false;
  }
  if (prev_nscs_ && !in.nscs) {
    frame_valid_ = !in.sclk;
    bit_count_ = 0;
    shift_in_ = 0;
    shift_out_ = pending_response_;
  }
  if (!in.nscs) {
    if (in.sclk && !prev_sclk_) {
      sdo_ = (shift_out_ & 0x8000u) != 0;
      shift_out_ = static_cast<std::uint16_t>(shift_out_ << 1);
    }
    if (!in.sclk && prev_sclk_) {
      shift_in_ = static_cast<std::uint16_t>((shift_in_ << 1) |
                                             (in.sdi ? 1u : 0u));
      ++bit_count_;
      if (bit_count_ > 16) frame_valid_ = false;
    }
  }
  prev_sclk_ = in.sclk;
  prev_nscs_ = in.nscs;
}

void Drv8323::latch_frame() {
  const bool read = (shift_in_ >> 15) & 0x1;
  const int addr = (shift_in_ >> 11) & 0xF;
  const std::uint16_t data = shift_in_ & 0x7FF;

  if (read) {
    if (addr <= 7) {
      pending_response_ = static_cast<std::uint16_t>(
          ((addr & 0xF) << 11) | (regs_[addr] & 0x7FF));
      if (addr == 0) regs_[0] = 0;  // Fault Status 1 clears on read
    } else {
      pending_response_ = static_cast<std::uint16_t>((addr & 0xF) << 11);
    }
  } else {
    // Writes land on the R/W registers (0x02-0x06) once the device is ready;
    // stored verbatim so the controller's readback-verify matches.
    if (ready_ && addr >= 2 && addr <= 6) regs_[addr] = data;
    pending_response_ =
        static_cast<std::uint16_t>((0x0 << 11) | (regs_[0] & 0x7FF));
  }
}

void Drv8323::process_gates(double t_s, const DriverInputs& in,
                            const std::array<double, 3>& phase_currents_a) {
  const bool enabled = in.en_gate && ready_ && !latched_shutdown_ &&
                       !pvdd_uv_ && !otsd_latched_;

  for (int k = 0; k < 3; ++k) {
    if (in.inh[k] && !prev_inh_[k]) cbc_block_high_[k] = false;
    if (in.inl[k] && !prev_inl_[k]) cbc_block_low_[k] = false;
    prev_inh_[k] = in.inh[k];
    prev_inl_[k] = in.inl[k];

    bool want_gh = enabled && in.inh[k] && !in.inl[k];
    bool want_gl = enabled && in.inl[k] && !in.inh[k];
    if (leg_latched_[k]) { want_gh = false; want_gl = false; }
    if (cbc_block_high_[k]) want_gh = false;
    if (cbc_block_low_[k]) want_gl = false;

    // DTC dead-time floor + complementary handshake.
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

    // VDS overcurrent on the conducting external FET.
    if (ocp_mode() != kOcpDisabled && enabled) {
      const double vds = std::abs(phase_currents_a[k]) * config_.rds_on_ohm;
      const bool high_conducting = gh_[k];
      const bool low_conducting = gl_[k];
      if ((high_conducting || low_conducting) && vds > vds_threshold_v()) {
        ++oc_events_;
        latch_status1(kBitVdsOcp);
        latch_status1(high_conducting ? kBitFetHighOc[k] : kBitFetLowOc[k]);
        pulse_noctw(t_s);
        switch (ocp_mode()) {
          case kOcpRetry:  // cycle-by-cycle current limit (retry on next edge)
            if (high_conducting) {
              cbc_block_high_[k] = true; gh_[k] = false; gh_off_time_[k] = t_s;
            }
            if (low_conducting) {
              cbc_block_low_[k] = true; gl_[k] = false; gl_off_time_[k] = t_s;
            }
            break;
          case kOcpLatched:
            leg_latched_[k] = true;
            gh_[k] = false; gl_[k] = false;
            gh_off_time_[k] = t_s; gl_off_time_[k] = t_s;
            break;
          default:
            break;  // report-only
        }
      }
    }
  }
}

}  // namespace bldcsim
