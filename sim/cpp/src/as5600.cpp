#include "as5600.hpp"

#include <cmath>

namespace bldcsim {

namespace {
constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr int kInitUnits = 128;
constexpr int kDataUnits = 4096;
constexpr int kErrorUnits = 127;
constexpr int kFrameUnits = kInitUnits + kDataUnits + kErrorUnits;  // 4351

double wrap_angle(double rad) {
  double w = std::fmod(rad, kTwoPi);
  if (w < 0.0) w += kTwoPi;
  return w;
}

// Shortest signed circular distance from `from` to `to`.
double circular_delta(double from, double to) {
  double d = std::fmod(to - from, kTwoPi);
  if (d > kTwoPi / 2.0) d -= kTwoPi;
  if (d < -kTwoPi / 2.0) d += kTwoPi;
  return d;
}

}  // namespace

As5600::As5600(const As5600Config& config)
    : config_(config), rng_(config.noise_seed) {
  frame_period_s_ = 1.0 / config_.pwm_carrier_hz;
  unit_s_ = frame_period_s_ / kFrameUnits;
}

void As5600::update(double t_s, double theta_mech_rad) {
  // Internal sampling + slow filter (tau such that settling ~= 4 tau).
  const double tau = config_.filter_settling_s / 4.0;
  while (t_s >= next_sample_time_) {
    // Mounting eccentricity + offset + angle noise distort the reading.
    double measured = theta_mech_rad + config_.mounting_offset_rad;
    if (config_.eccentricity_e1_rad != 0.0 ||
        config_.eccentricity_e2_rad != 0.0) {
      measured += config_.eccentricity_e1_rad *
                      std::sin(theta_mech_rad + config_.eccentricity_phi1_rad)
                  + config_.eccentricity_e2_rad *
                        std::sin(2.0 * theta_mech_rad +
                                 config_.eccentricity_phi2_rad);
    }
    if (config_.angle_noise_lsb > 0.0) {
      std::normal_distribution<double> noise(
          0.0, config_.angle_noise_lsb * kTwoPi / 4096.0);
      measured += noise(rng_);
    }
    sampled_rad_ = wrap_angle(measured);
    if (first_sample_) {
      filtered_rad_ = sampled_rad_;
      first_sample_ = false;
    } else {
      const double alpha = 1.0 - std::exp(-config_.sample_period_s / tau);
      filtered_rad_ = wrap_angle(
          filtered_rad_ + alpha * circular_delta(filtered_rad_, sampled_rad_));
    }
    next_sample_time_ += config_.sample_period_s;
  }

  // PWM frame generation; the angle is latched at each frame start.
  while (t_s - frame_start_ >= frame_period_s_) {
    frame_start_ += frame_period_s_;
    frame_angle_ = static_cast<int>(
        std::floor(filtered_rad_ / kTwoPi * kDataUnits));
    if (frame_angle_ > kDataUnits - 1) frame_angle_ = kDataUnits - 1;
  }

  if (magnet_lost_) {
    out_ = false;
    return;
  }
  const double in_frame = t_s - frame_start_;
  const double high_time = (kInitUnits + frame_angle_) * unit_s_;
  out_ = in_frame < high_time;
}

}  // namespace bldcsim
