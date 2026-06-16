// SPDX-License-Identifier: MIT
#include "ads9224r.hpp"

#include <algorithm>
#include <cmath>

namespace bldcsim {

std::uint16_t Ads9224r::quantize(double v) const {
  // Differential input = (sense voltage - midpoint); 16-bit two's-complement,
  // zero = mid-scale. +/- fsr_half_v maps to +/- 32767.
  const double diff = v - config_.vmid_v;
  double code = std::round(diff / config_.fsr_half_v * 32767.0);
  code = std::clamp(code, -32768.0, 32767.0);
  return static_cast<std::uint16_t>(static_cast<int>(code) & 0xFFFF);
}

void Ads9224r::update(double t_s, bool convst, bool cs_n, bool sclk,
                      double v_cha, double v_chb) {
  // CONVST rising edge: latch BOTH channels at the same instant (simultaneous).
  if (convst && !prev_convst_) {
    const std::uint16_t a = quantize(v_cha);
    const std::uint16_t b = quantize(v_chb);
    shift_a_ = a;
    shift_b_ = b;
    // Sign-extend the 16-bit codes for the diagnostics.
    code_a_ = static_cast<int>(static_cast<std::int16_t>(a));
    code_b_ = static_cast<int>(static_cast<std::int16_t>(b));
    converting_ = true;
    ready_ = false;
    convert_done_t_ = t_s + config_.tdrdy_s;
    ++conversions_;
  }
  if (converting_ && t_s >= convert_done_t_) {
    converting_ = false;
    ready_ = true;
  }

  // Serial readout: CS low frames the transfer; on each SCLK rising (leading)
  // edge present the next MSB-first bit, the master samples at the trailing
  // edge - identical to the DRV8301 slave shifter (one-cycle bench feedback).
  if (!cs_n && sclk && !prev_sclk_) {
    sdo_a_ = (shift_a_ & 0x8000u) != 0;
    shift_a_ = static_cast<std::uint16_t>(shift_a_ << 1);
    sdo_b_ = (shift_b_ & 0x8000u) != 0;
    shift_b_ = static_cast<std::uint16_t>(shift_b_ << 1);
  }

  prev_convst_ = convst;
  prev_cs_n_ = cs_n;
  prev_sclk_ = sclk;
}

}  // namespace bldcsim
