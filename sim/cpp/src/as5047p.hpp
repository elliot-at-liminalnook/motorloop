// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <random>

#include "i_angle_sensor.hpp"

namespace bldcsim {

// Behavioral AS5047P 14-bit magnetic angle sensor, SPI mode-1 readout
// (datasheet DS000324). Unlike the AS5600 (PWM output, high latency) the
// AS5047P is read over SPI and applies DAEC (Dynamic Angle Error Compensation):
// the raw ~90-110 us read-pipeline latency is predicted forward to ~1.5-1.9 us
// residual at constant speed (datasheet pp.7-9). That is the hardware form of
// the RTL's omega*t_latency extrapolation (Q22), so this model lets the bench
// compare the two head to head.
//
// SPI: 16-bit frames, mode 1 (CPOL=0/CPHA=1), MSB first, *pipelined* read - the
// angle for a Read-ANGLECOM command returns on the NEXT CS frame. The FPGA
// master streams Read-0x3FFF and this slave returns PARD[15] | EF[14] |
// ANGLE[13:0] with even parity over bits[14:0]. Magnet loss sets EF, which the
// master treats as an invalid angle (the realism-scenario portable behaviour).
struct As5047pConfig {
  double daec_latency_s = 1.7e-6;   // residual read latency with DAEC (p.7)
  double raw_latency_s = 100e-6;    // uncompensated pipeline latency (p.7)
  bool daec_enable = true;          // SETTINGS1.DAECDIS = 0 -> DAEC on
  // Portable mounting / measurement nonidealities (shared with the AS5600
  // path so the realism scenarios stay platform-independent).
  double mounting_offset_rad = 0.0;
  double eccentricity_e1_rad = 0.0;
  double eccentricity_phi1_rad = 0.0;
  double eccentricity_e2_rad = 0.0;
  double eccentricity_phi2_rad = 0.0;
  double angle_noise_lsb = 0.0;     // 1-sigma, in 14-bit LSB
  unsigned noise_seed = 0x5047;
};

class As5047p : public IAngleSensor {
 public:
  explicit As5047p(const As5047pConfig& config);

  void update(double t_s, double theta_mech_rad) override;
  bool out() const override { return false; }  // not a PWM part
  double filtered_angle_rad() const override { return filtered_rad_; }
  void inject_magnet_loss(bool lost) override { magnet_lost_ = lost; }

  // SPI-slave bus driven by the RTL angle master (no-op default elsewhere).
  void spi_io(bool cs_n, bool sclk, bool mosi) override;
  bool miso() const override { return sdo_; }

  int angle14() const { return angle14_; }  // diagnostic

 private:
  std::uint16_t response_word() const;

  As5047pConfig config_;
  double filtered_rad_ = 0.0;
  bool first_ = true;
  double last_t_ = -1.0;
  int angle14_ = 0;
  bool magnet_lost_ = false;
  std::mt19937 rng_{0x5047};

  // SPI slave state (mode 1, 16-bit; mirrors the DRV8301 slave shifter).
  bool prev_cs_n_ = true;
  bool prev_sclk_ = false;
  int bit_count_ = 0;
  std::uint16_t shift_in_ = 0;
  std::uint16_t shift_out_ = 0;
  std::uint16_t pending_response_ = 0xC000;  // EF=1 until the first read lands
  bool sdo_ = false;
};

}  // namespace bldcsim
