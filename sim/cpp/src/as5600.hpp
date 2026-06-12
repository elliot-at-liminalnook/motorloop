#pragma once

#include <random>

namespace bldcsim {

// Behavioral AS5600 magnetic angle sensor in PWM-output mode (DS000365;
// digest facts). Deliberately latency-faithful:
//
//  - the true mechanical angle is sampled every sample_period_s (150 us),
//  - a first-order slow filter (tau = settling/4) tracks the sampled angle
//    on the shortest circular path,
//  - the OUT pin emits PWM frames at pwm_carrier_hz: a frame of
//    128 (init high) + 4096 (data) + 127 (error/low) = 4351 units, with the
//    high time = 128 + angle12 units. The filtered angle is latched once per
//    frame at the frame start.
//
// Magnet-loss injection drives the output low (simplification; flagged
// status registers belong to a future I2C model).

struct As5600Config {
  double sample_period_s = 150e-6;
  double filter_settling_s = 2.2e-3;  // slow filter SF=00
  double pwm_carrier_hz = 920.0;
  // Mounting imperfection (realism stage 5, Q20): measured angle =
  // true + e1*sin(theta+phi1) + e2*sin(2*theta+phi2) + gaussian noise.
  double eccentricity_e1_rad = 0.0;
  double eccentricity_phi1_rad = 0.0;
  double eccentricity_e2_rad = 0.0;
  double eccentricity_phi2_rad = 0.0;
  double angle_noise_lsb = 0.0;
  // Static sensor-vs-rotor offset (pi = flipped magnet, E24).
  double mounting_offset_rad = 0.0;
  unsigned noise_seed = 0xa5c0;
};

class As5600 {
 public:
  explicit As5600(const As5600Config& config);

  // Advance to absolute time t_s with the true mechanical angle.
  void update(double t_s, double theta_mech_rad);

  bool out() const { return out_; }
  int angle12() const { return frame_angle_; }       // angle in current frame
  double filtered_angle_rad() const { return filtered_rad_; }

  void inject_magnet_loss(bool lost) { magnet_lost_ = lost; }

 private:
  As5600Config config_;
  double next_sample_time_ = 0.0;
  double sampled_rad_ = 0.0;
  double filtered_rad_ = 0.0;
  bool first_sample_ = true;

  double frame_period_s_;
  double unit_s_;
  double frame_start_ = 0.0;
  int frame_angle_ = 0;
  bool out_ = false;
  bool magnet_lost_ = false;
  std::mt19937 rng_{0xa5c0};
};

}  // namespace bldcsim
