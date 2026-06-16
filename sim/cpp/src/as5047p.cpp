// SPDX-License-Identifier: MIT
#include "as5047p.hpp"

#include <cmath>

namespace bldcsim {

namespace {
constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr int kDataUnits = 16384;  // 14-bit

double wrap_angle(double rad) {
  double w = std::fmod(rad, kTwoPi);
  if (w < 0.0) w += kTwoPi;
  return w;
}

double circular_delta(double from, double to) {
  double d = std::fmod(to - from, kTwoPi);
  if (d > kTwoPi / 2.0) d -= kTwoPi;
  if (d < -kTwoPi / 2.0) d += kTwoPi;
  return d;
}
}  // namespace

As5047p::As5047p(const As5047pConfig& config)
    : config_(config), rng_(config.noise_seed) {}

void As5047p::update(double t_s, double theta_mech_rad) {
  const double dt = (last_t_ >= 0.0) ? (t_s - last_t_) : 0.0;
  last_t_ = t_s;

  double measured = theta_mech_rad + config_.mounting_offset_rad;
  if (config_.eccentricity_e1_rad != 0.0 ||
      config_.eccentricity_e2_rad != 0.0) {
    measured += config_.eccentricity_e1_rad *
                    std::sin(theta_mech_rad + config_.eccentricity_phi1_rad) +
                config_.eccentricity_e2_rad *
                    std::sin(2.0 * theta_mech_rad +
                             config_.eccentricity_phi2_rad);
  }
  if (config_.angle_noise_lsb > 0.0) {
    std::normal_distribution<double> noise(
        0.0, config_.angle_noise_lsb * kTwoPi / kDataUnits);
    measured += noise(rng_);
  }
  const double sampled = wrap_angle(measured);

  // The reported angle lags the true angle by the effective read latency;
  // DAEC shrinks that lag from ~100 us to ~1.7 us (the Q22 quantity).
  const double tau =
      config_.daec_enable ? config_.daec_latency_s : config_.raw_latency_s;
  if (first_ || dt <= 0.0) {
    filtered_rad_ = sampled;
    first_ = false;
  } else {
    const double alpha = 1.0 - std::exp(-dt / tau);
    filtered_rad_ = wrap_angle(
        filtered_rad_ + alpha * circular_delta(filtered_rad_, sampled));
  }

  int a = static_cast<int>(std::floor(filtered_rad_ / kTwoPi * kDataUnits));
  if (a < 0) a = 0;
  if (a > kDataUnits - 1) a = kDataUnits - 1;
  angle14_ = a;
}

std::uint16_t As5047p::response_word() const {
  const std::uint16_t data = static_cast<std::uint16_t>(angle14_ & 0x3FFF);
  const std::uint16_t ef = magnet_lost_ ? 1u : 0u;  // EF: error flag
  std::uint16_t w = static_cast<std::uint16_t>((ef << 14) | data);
  // PARD (bit 15) = even parity over bits[14:0].
  int ones = 0;
  for (int i = 0; i < 15; ++i)
    if ((w >> i) & 1u) ++ones;
  if (ones & 1) w = static_cast<std::uint16_t>(w | (1u << 15));
  return w;
}

void As5047p::spi_io(bool cs_n, bool sclk, bool mosi) {
  if (prev_cs_n_ && !cs_n) {
    // Frame start: clock must idle low (CPHA=1, CPOL=0).
    bit_count_ = 0;
    shift_in_ = 0;
    shift_out_ = pending_response_;
  }
  if (!prev_cs_n_ && cs_n) {
    // Frame end: the response to the command just clocked in is presented on
    // the NEXT frame (pipelined read), and reflects the latest angle.
    pending_response_ = response_word();
    sdo_ = false;
  }
  if (!cs_n) {
    if (sclk && !prev_sclk_) {
      // Rising edge: shift the response out, MSB first.
      sdo_ = (shift_out_ & 0x8000u) != 0;
      shift_out_ = static_cast<std::uint16_t>(shift_out_ << 1);
    }
    if (!sclk && prev_sclk_) {
      // Falling edge: sample the command bit.
      shift_in_ = static_cast<std::uint16_t>((shift_in_ << 1) | (mosi ? 1u : 0u));
      ++bit_count_;
    }
  }
  prev_sclk_ = sclk;
  prev_cs_n_ = cs_n;
}

}  // namespace bldcsim
