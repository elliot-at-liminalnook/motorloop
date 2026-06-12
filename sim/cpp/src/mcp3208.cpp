#include "mcp3208.hpp"

#include <cmath>

namespace bldcsim {

Mcp3208::Mcp3208(const Mcp3208Config& config, AnalogSource source)
    : config_(config), source_(std::move(source)) {}

int Mcp3208::quantize(double v) const {
  const double vref = (live_vref_v_ > 0.0) ? live_vref_v_ : config_.vref_v;
  const double ideal = v * 4096.0 / vref;
  // Transfer nonidealities: gain error, offset, smooth INL bow.
  const double bowed = ideal * (1.0 + config_.gain_error) +
                       config_.offset_lsb +
                       config_.inl_bow_lsb * std::sin(M_PI * ideal / 4096.0);
  int code = static_cast<int>(std::floor(bowed));
  if (code < 0) code = 0;
  if (code > 4095) code = 4095;
  return code;
}

void Mcp3208::update(double t_s, bool cs_n, bool sclk, bool din) {
  if (cs_n && !prev_cs_n_) {
    cs_rise_time_ = t_s;
    dout_ = false;
  }
  if (!cs_n && prev_cs_n_) {
    if (cs_rise_time_ >= 0.0 &&
        t_s - cs_rise_time_ < config_.min_cs_high_s) {
      ++cs_too_short_;
    }
    falling_edges_ = 0;
    started_ = false;
    command_ = 0;
    command_bits_ = 0;
    sampled_ = false;
    data_ = 0;
    data_bits_out_ = 0;
    last_sclk_rise_ = -1.0;
  }

  if (!cs_n) {
    if (sclk && !prev_sclk_) {
      // Rising edge: device samples DIN.
      if (last_sclk_rise_ >= 0.0) {
        const double period = t_s - last_sclk_rise_;
        if (period < 0.999 / config_.max_sclk_hz) {
          ++sclk_too_fast_;
        }
      }
      last_sclk_rise_ = t_s;

      if (!started_) {
        if (din) {
          started_ = true;
          falling_edges_ = 0;
        }
      } else if (command_bits_ < 4) {
        command_ = static_cast<std::uint8_t>((command_ << 1) | (din ? 1 : 0));
        ++command_bits_;
      }
    }

    if (!sclk && prev_sclk_ && started_) {
      // falling_edges_ == k means falling edge F(k-1) counting from the
      // clock cycle that carried the start bit. The sample window ends on
      // the falling edge of the 5th clock FOLLOWING the start bit (F5).
      ++falling_edges_;
      if (falling_edges_ == 6 && command_bits_ >= 4) {
        // HOLD instant: end of the 1.5-clock sample window.
        const bool single_ended = (command_ >> 3) & 0x1;
        const int channel = command_ & 0x7;
        if (!single_ended) {
          ++differential_requests_;
        }
        const double v_target = source_ ? source_(channel) : 0.0;
        // Shared sample cap arrives holding the previous conversion's
        // voltage; the residual fraction of that difference survives.
        const double d = config_.sample_residual[channel & 0x7];
        const double v = v_target + d * (cap_v_ - v_target);
        last_theft_v_ = v - v_target;
        cap_v_ = v;
        last_sample_ = Mcp3208Sample{t_s, channel, v, quantize(v)};
        data_ = static_cast<std::uint16_t>(last_sample_.code);
        sampled_ = true;
        ++conversions_;
        dout_ = false;  // null bit
        data_bits_out_ = 0;
      } else if (sampled_ && data_bits_out_ < 12) {
        dout_ = (data_ >> (11 - data_bits_out_)) & 0x1;
        ++data_bits_out_;
      } else if (sampled_ && data_bits_out_ >= 12) {
        dout_ = false;
      }
    }
  }

  prev_cs_n_ = cs_n;
  prev_sclk_ = sclk;
}

}  // namespace bldcsim
