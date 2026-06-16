// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>

namespace bldcsim {

// Behavioral ADS9224R: 16-bit, dual *simultaneous*-sampling SAR ADC (datasheet
// SBAS876C). The reference platform's current ADC - it samples both phase
// currents on one CONVST edge (40 ps aperture mismatch => truly simultaneous),
// retiring Q21 (the MCP3208's ~22 us sequential inter-channel skew) in hardware.
//
// Bench surface: the RTL ads9224r_master drives CONVST/CS/SCLK; this slave
// presents two serial data lines SDO_A / SDO_B. On a CONVST rising edge both
// channels' sample-and-holds latch the current-sense voltages at the same
// instant; after tDRDY READY asserts and the two 16-bit two's-complement codes
// (zero = mid-scale = no current) shift out MSB-first. The signed format means
// no offset subtraction is needed downstream (unlike the offset-binary MCP3208).
//
// Read transport: 16-bit, CPOL=0/CPHA=1 class (shift out on the leading edge,
// master samples on the trailing edge) - matches the existing SPI masters.
struct Ads9224rConfig {
  double vref_v = 2.5;          // internal reference (datasheet)
  double fsr_half_v = 4.096;    // +/- full-scale = 3.2768 * VREF/2 ... ~4.096 V
  double vmid_v = 1.65;         // current-sense midpoint (AINM, = VREF/2 rail)
  double tdrdy_s = 315e-9;      // CONVST -> data ready (low latency)
};

class Ads9224r {
 public:
  explicit Ads9224r(const Ads9224rConfig& config) : config_(config) {}

  // Advance with the RTL master's pins and the two current-sense voltages
  // (FeedbackChain channels 0 and 1). Call every RTL clock edge.
  void update(double t_s, bool convst, bool cs_n, bool sclk,
              double v_cha, double v_chb);

  bool sdo_a() const { return sdo_a_; }
  bool sdo_b() const { return sdo_b_; }
  bool ready() const { return ready_; }

  int code_a() const { return code_a_; }   // signed 16-bit, diagnostic
  int code_b() const { return code_b_; }
  long conversions() const { return conversions_; }

 private:
  std::uint16_t quantize(double v) const;

  Ads9224rConfig config_;

  bool prev_convst_ = false;
  bool prev_cs_n_ = true;
  bool prev_sclk_ = false;

  // Conversion state.
  bool converting_ = false;
  double convert_done_t_ = -1.0;
  bool ready_ = false;
  int code_a_ = 0, code_b_ = 0;          // signed (sign-extended from 16-bit)

  // Serial readout shifters (one per SDO line).
  std::uint16_t shift_a_ = 0, shift_b_ = 0;
  bool sdo_a_ = false, sdo_b_ = false;
  long conversions_ = 0;
};

}  // namespace bldcsim
