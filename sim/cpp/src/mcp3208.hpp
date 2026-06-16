// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstdint>
#include <functional>

#include "i_current_adc.hpp"

namespace bldcsim {

// Behavioral MCP3208 12-bit SAR ADC, SPI mode 0,0 (DS21298E; digest facts).
//
// Frame (after CS falls): DIN carries start(1), SGL/DIFF, D2, D1, D0; the
// input is sampled during the 1.5-clock window that ends on the falling edge
// of the 5th clock after the start bit (the HOLD instant — recorded for the
// aperture-placement assertions); DOUT then shifts a null bit followed by 12
// data bits MSB-first on falling edges.
//
// Modeled guards (diagnostic counters, not hard errors):
//  - SCLK period shorter than the configured maximum frequency,
//  - CS high time shorter than 500 ns between conversions,
//  - differential mode requests (not supported by this model).

struct Mcp3208Config {
  double vref_v = 3.3;
  double max_sclk_hz = 1.0e6;
  double min_cs_high_s = 500e-9;
  // Shared-sample-cap residual per channel: fraction of (previous cap
  // voltage - target) left in the held sample. Derived from the codified
  // ADC front-end circuit (derive_params.py): charge share against a local
  // filter cap, or settling residual for unbuffered sources.
  std::array<double, 8> sample_residual{};
  // Transfer nonidealities (datasheet-grade; realism stage 3):
  // code = ideal*(1+gain_error) + offset_lsb + inl_bow*sin(pi*ideal/4096).
  double offset_lsb = 0.0;
  double gain_error = 0.0;
  double inl_bow_lsb = 0.0;
};

// The completed-conversion struct is the generic AdcSample (i_current_adc.hpp);
// the alias keeps existing call sites compiling.
using Mcp3208Sample = AdcSample;

class Mcp3208 : public ICurrentAdc {
 public:
  using AnalogSource = std::function<double(int channel)>;

  Mcp3208(const Mcp3208Config& config, AnalogSource source);

  // Advance to absolute time t_s with current pin inputs. Call on every RTL
  // clock edge; SPI edge detection is internal.
  void update(double t_s, bool cs_n, bool sclk, bool din);

  // Live reference voltage (the 3.3 V rail is the reference: PWM ripple on
  // it makes conversions ratiometrically wrong). Defaults to config vref.
  void set_live_vref(double v) { live_vref_v_ = v; }

  bool dout() const { return dout_; }

  const Mcp3208Sample& last_sample() const { return last_sample_; }
  // Voltage the sampling event removed from the source node (negative when
  // the cap arrived below the target). For channels with a local reservoir
  // cap, the bench feeds this back into the feedback chain so the droop
  // persists and recovers through the channel RC.
  double last_sample_theft_v() const { return last_theft_v_; }
  long conversions() const { return conversions_; }
  long sclk_too_fast_count() const { return sclk_too_fast_; }
  long cs_too_short_count() const { return cs_too_short_; }
  long differential_requests() const { return differential_requests_; }

 private:
  int quantize(double v) const;

  Mcp3208Config config_;
  AnalogSource source_;

  bool prev_cs_n_ = true;
  bool prev_sclk_ = false;
  double cs_rise_time_ = -1.0;
  double last_sclk_rise_ = -1.0;

  // Frame state.
  int falling_edges_ = 0;     // since CS fell
  bool started_ = false;
  int start_edge_ = -1;       // falling-edge index when start bit was seen
  std::uint8_t command_ = 0;  // SGL/DIFF + D2..D0 accumulator
  int command_bits_ = 0;
  bool sampled_ = false;
  std::uint16_t data_ = 0;
  int data_bits_out_ = 0;
  bool dout_ = false;

  Mcp3208Sample last_sample_;
  double cap_v_ = 0.0;        // sample-cap voltage held from last conversion
  double last_theft_v_ = 0.0;
  double live_vref_v_ = -1.0;  // < 0: use config vref
  long conversions_ = 0;
  long sclk_too_fast_ = 0;
  long cs_too_short_ = 0;
  long differential_requests_ = 0;
};

}  // namespace bldcsim
