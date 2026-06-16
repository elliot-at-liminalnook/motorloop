// SPDX-License-Identifier: MIT
#pragma once

namespace bldcsim {

// A single completed ADC conversion (generic; formerly Mcp3208Sample).
struct AdcSample {
  double time_s = -1.0;
  int channel = -1;
  double analog_v = 0.0;
  int code = 0;
};

// Role interface for the current/feedback ADC (MCP3208, ADS9224R, …). Captures
// the bench<->model SPI call surface; conversion timing, aperture, and
// nonidealities live inside each concrete model. The bench holds an
// ICurrentAdc; a factory picks the part from config.
class ICurrentAdc {
 public:
  virtual ~ICurrentAdc() = default;

  // Advance to absolute time with the current SPI pin inputs.
  virtual void update(double t_s, bool cs_n, bool sclk, bool din) = 0;

  // The live reference voltage (ratiometric conversions track rail ripple).
  virtual void set_live_vref(double v) = 0;

  virtual bool dout() const = 0;
  virtual const AdcSample& last_sample() const = 0;
  virtual double last_sample_theft_v() const = 0;
  virtual long conversions() const = 0;
};

}  // namespace bldcsim
