// SPDX-License-Identifier: MIT
#include "bench.hpp"

#include <algorithm>
#include <cmath>

#include <verilated.h>
#include <verilated_vcd_c.h>

#include "Vcontroller_top.h"
#include "peripheral_factory.hpp"

namespace bldcsim {

Bench::Bench(const BenchConfig& config)
    : config_(config),
      ctx_(std::make_unique<VerilatedContext>()),
      plant_(config.motor, config.bridge, config.plant, config.supply),
      chain_(config.chain),
      drv_(make_gate_driver(config.driver_name, config.drv)),
      adc_(make_current_adc(config.adc_name, config.adc,
                            [this](int ch) { return chain_.channel(ch); })),
      encoder_(make_angle_sensor(config.angle_name, config.encoder)),
      adc2_(Ads9224rConfig{config.adc.vref_v, 4.096,
                           config.chain.amp_offset_v, 315e-9}),
      thermal_(config.thermal) {
  ctx_->traceEverOn(true);
  top_ = std::make_unique<Vcontroller_top>(ctx_.get());
  period_s_ = 1.0 / config_.clk_hz;
  uart_bit_s_ = 1.0 / config_.uart_baud;

  top_->rst_n = 0;
  top_->clk = 0;
  top_->ctrl_mode = 0;
  top_->ctrl_duty = 0;
  top_->ctrl_target_speed = 0;
  top_->ctrl_ol_freq_word = 0;
  top_->ctrl_ol_ramp_inc = 0;
  top_->ctrl_align_offset = 0;
  top_->ctrl_foc_sample = 0;
  top_->ctrl_id_target = 0;
  top_->ctrl_iq_target = 0;
  top_->ctrl_foc_speed_loop = 0;
  top_->ctrl_foc_extrap = config_.foc.angle_extrap_enable ? 1 : 0;
  top_->ctrl_drv_hw_mode = config_.drv_hw_mode ? 1 : 0;
  top_->ctrl_angle_spi_mode = config_.angle_spi_mode ? 1 : 0;
  top_->ctrl_cur_norm_shift = config_.cur_norm_shift & 0x7;
  top_->ctrl_adc_dual_mode = config_.adc_dual_mode ? 1 : 0;
  top_->adc2_sdo_a = 0;
  top_->adc2_sdo_b = 0;
  top_->adc2_ready = 0;
  top_->nfault = 1;
  top_->noctw = 1;
  top_->drv_miso = 0;
  top_->adc_miso = 0;
  top_->angle_pwm = 0;
  top_->angle_miso = 0;
  top_->uart_rx_pin = 1;  // idle high
  // FOC phase-current sampling scheme (Q21): simultaneous (scheme 0) freezes
  // the low-side-shunt currents at the PWM peak via an external S/H.
  chain_.set_simultaneous_currents(config_.foc.current_sample_scheme == 0);
  // A few reset cycles.
  for (int n = 0; n < 8; ++n) {
    top_->clk = 1; top_->eval();
    top_->clk = 0; top_->eval();
  }
  top_->rst_n = 1;
}

Bench::~Bench() { close_vcd(); }

void Bench::open_vcd(const std::string& path) {
  vcd_ = std::make_unique<VerilatedVcdC>();
  top_->trace(vcd_.get(), 99);
  vcd_->open(path.c_str());
}

void Bench::close_vcd() {
  if (vcd_) {
    vcd_->close();
    vcd_.reset();
  }
}

void Bench::set_mode(int mode) { top_->ctrl_mode = mode & 0x3; }
void Bench::set_duty(int duty) { top_->ctrl_duty = duty & 0xFFFF; }
void Bench::set_target_speed(int rad_s) {
  top_->ctrl_target_speed = rad_s & 0xFFFF;
}
void Bench::set_open_loop(std::uint32_t freq_word, std::uint32_t ramp_inc) {
  top_->ctrl_ol_freq_word = freq_word;
  top_->ctrl_ol_ramp_inc = ramp_inc;
}
void Bench::set_align_offset(int offset12) {
  top_->ctrl_align_offset = offset12 & 0xFFF;
}
void Bench::set_foc_sample(bool on) { top_->ctrl_foc_sample = on ? 1 : 0; }

namespace {
// Sign-extend an 18-bit two's-complement value held in a uint32.
int sx18(std::uint32_t v) {
  v &= 0x3FFFF;
  return (v & 0x20000) ? static_cast<int>(v | 0xFFFC0000) : static_cast<int>(v);
}
}  // namespace

int Bench::foc_cur_a() const { return sx18(top_->dbg_foc_cur_a); }
int Bench::foc_cur_b() const { return sx18(top_->dbg_foc_cur_b); }
bool Bench::foc_valid() const { return top_->dbg_foc_valid; }
int Bench::foc_id() const { return sx18(top_->dbg_foc_id); }
int Bench::foc_iq() const { return sx18(top_->dbg_foc_iq); }
int Bench::foc_vd() const { return sx18(top_->dbg_foc_vd); }
int Bench::foc_vq() const { return sx18(top_->dbg_foc_vq); }

void Bench::set_id_target(int lsb) {
  top_->ctrl_id_target = static_cast<std::uint32_t>(lsb) & 0x3FFFF;
}
void Bench::set_iq_target(int lsb) {
  top_->ctrl_iq_target = static_cast<std::uint32_t>(lsb) & 0x3FFFF;
}
void Bench::set_speed_clamp(bool on, double omega_rad_s) {
  plant_.set_speed_clamp(on, omega_rad_s);
}
void Bench::set_foc_speed_loop(bool on) {
  top_->ctrl_foc_speed_loop = on ? 1 : 0;
}
void Bench::set_foc_extrap(bool on) {
  top_->ctrl_foc_extrap = on ? 1 : 0;
}

int Bench::dbg_sector() const { return top_->dbg_sector; }
int Bench::dbg_duty() const { return top_->dbg_duty; }
bool Bench::dbg_configured() const { return top_->dbg_configured; }
int Bench::dbg_fault_count() const { return top_->dbg_fault_count; }
int Bench::dbg_mismatch_count() const { return top_->dbg_mismatch_count; }
int Bench::dbg_speed() const { return top_->dbg_speed; }
int Bench::dbg_angle() const { return top_->dbg_angle; }
bool Bench::dbg_angle_valid() const { return top_->dbg_angle_valid; }
int Bench::dbg_offset_a() const { return top_->dbg_offset_a; }
int Bench::dbg_offset_b() const { return top_->dbg_offset_b; }
int Bench::dbg_noctw_count() const { return top_->dbg_noctw_count; }
bool Bench::drv_dead() const { return top_->dbg_drv_dead; }
bool Bench::locked_out() const { return top_->dbg_locked_out; }
bool Bench::stalled() const { return top_->dbg_stall; }
bool Bench::adc_stuck() const { return top_->dbg_adc_stuck; }
bool Bench::offset_fault() const { return top_->dbg_offset_fault; }
bool Bench::speed_reverse() const { return top_->dbg_reverse; }

void Bench::sync_plant() {
  const double lag = time_s_ - plant_time_s_;
  if (lag <= 0.0) return;
  if (load_osc_amp_ != 0.0) {
    plant_.set_load_torque(base_load_nm_ +
                           load_osc_amp_ * std::sin(2.0 * M_PI *
                                                    load_osc_hz_ * time_s_));
  }
  plant_.advance(lag);

  // Thermal lumps follow the plant; their drift multipliers feed back.
  if (thermal_.enabled()) {
    double winding_w = 0.0;
    const double r_hot = config_.motor.resistance_ohm * thermal_.r_scale();
    for (double i : plant_.state().current_a) winding_w += i * i * r_hot;
    thermal_.update(lag, plant_.leg_conduction_w(), winding_w);
    plant_.set_r_scale(thermal_.r_scale());
    plant_.set_ke_scale(thermal_.ke_scale());
    plant_.set_rds_scale(thermal_.rds_scale());
  }

  // Ground-shift disturbance follows the bus return current.
  const double i_bus = plant_.bus_current_a();
  double didt = 0.0;
  if (last_bus_i_t_ >= 0.0 && time_s_ > last_bus_i_t_) {
    didt = (i_bus - last_bus_i_) / (time_s_ - last_bus_i_t_);
  }
  last_bus_i_ = i_bus;
  last_bus_i_t_ = time_s_;
  chain_.set_bus_return_current(i_bus, didt);

  const double vbus = config_.supply.enabled ? plant_.bus_v() : config_.vbus_v;
  chain_.update(lag, plant_.state(), plant_.outputs(), vbus);
  bus_v_min_ = std::min(bus_v_min_, plant_.bus_v());
  bus_v_max_ = std::max(bus_v_max_, plant_.bus_v());
  plant_time_s_ = time_s_;
}

bool Bench::glitched(unsigned bit, bool value) {
  if (!(glitch_mask_ & (1u << bit)) || time_s_ > glitch_end_t_) return value;
  std::uniform_real_distribution<double> u(0.0, 1.0);
  if (u(glitch_rng_) < glitch_rate_hz_ * period_s_) {
    return !value;
  }
  return value;
}

void Bench::set_line_glitches(unsigned mask, double rate_hz, double end_t_s,
                              unsigned seed) {
  glitch_mask_ = mask;
  glitch_rate_hz_ = rate_hz;
  glitch_end_t_ = end_t_s;
  glitch_rng_.seed(seed);
}

void Bench::run_config_window(double seconds, bool en_gate_pulldown,
                              unsigned seed) {
  // Models the FPGA-configuration window: pins float, so the DRV inputs see
  // random levels. The external EN_GATE pull-down (Q13 mitigation) is the
  // only thing keeping the power stage dead. The RTL is not ticked.
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> coin(0, 1);
  const double step = 5e-6;
  double t = pre_time_s_;
  const double end = pre_time_s_ + seconds;
  // A floating EN_GATE drifts slowly (worst case: high long enough for the
  // DRV to become ready); the PWM-pin noise toggles fast.
  bool en_gate_float = false;
  long step_count = 0;
  while (t < end) {
    t += step;
    ++step_count;
    if (step_count % 4000 == 0) en_gate_float = coin(rng) != 0;  // ~20 ms
    Drv8301Inputs in;
    in.en_gate = en_gate_pulldown ? false : en_gate_float;
    in.dc_cal = coin(rng) != 0;
    for (int k = 0; k < 3; ++k) {
      in.inh[k] = coin(rng) != 0;
      in.inl[k] = coin(rng) != 0;
    }
    in.nscs = coin(rng) != 0;
    in.sclk = coin(rng) != 0;
    in.sdi = coin(rng) != 0;
    drv_->update(t, in, plant_.state().current_a,
                config_.supply.enabled ? plant_.bus_v() : config_.vbus_v,
                thermal_.drv_t_c());
    const auto& gh = drv_->gate_high();
    const auto& gl = drv_->gate_low();
    bool any = false;
    for (int k = 0; k < 3; ++k) any = any || gh[k] || gl[k];
    if (any) ++config_window_gate_activity_;
    plant_.set_gates(gh, gl);
    plant_.advance(step);
  }
  pre_time_s_ = end;
  plant_time_s_ = end;
}

void Bench::uart_send(const std::vector<int>& bytes) {
  for (int b : bytes) uart_tx_queue_.push_back(b & 0xFF);
}

std::vector<int> Bench::uart_take_received() {
  std::vector<int> out;
  out.swap(uart_rx_bytes_);
  return out;
}

void Bench::tick() {
  time_s_ = pre_time_s_ + static_cast<double>(cycle_) * period_s_;

  // Posedge.
  top_->clk = 1;
  top_->eval();
  if (vcd_) vcd_->dump(static_cast<std::uint64_t>(cycle_ * 2));

  // RTL outputs -> DRV8301 (with optional line corruption on the SPI).
  Drv8301Inputs drv_in;
  drv_in.en_gate = top_->en_gate;
  drv_in.dc_cal = top_->dc_cal;
  for (int k = 0; k < 3; ++k) {
    drv_in.inh[k] = (top_->inh >> k) & 1;
    drv_in.inl[k] = (top_->inl >> k) & 1;
  }
  drv_in.nscs = top_->drv_ncs;
  drv_in.sclk = glitched(4, top_->drv_sclk);
  drv_in.sdi = glitched(3, top_->drv_mosi);
  const double pvdd =
      config_.supply.enabled ? plant_.bus_v() : config_.vbus_v;
  drv_->update(time_s_, drv_in, plant_.state().current_a, pvdd,
              thermal_.drv_t_c());
  if (drv_->pvdd_uv_active() && !last_pvdd_uv_) ++pvdd_uv_events_;
  last_pvdd_uv_ = drv_->pvdd_uv_active();

  // DC_CAL state flows into the feedback chain.
  chain_.set_dc_cal(0, drv_->dc_cal_active(0));
  chain_.set_dc_cal(1, drv_->dc_cal_active(1));

  // Gate edges force a plant sync; otherwise cap the lag.
  const auto& gh = drv_->gate_high();
  const auto& gl = drv_->gate_low();
  bool gates_changed = false;
  for (int k = 0; k < 3; ++k) {
    if (gh[k] != last_gh_[k] || gl[k] != last_gl_[k]) gates_changed = true;
  }
  if (gates_changed || time_s_ - plant_time_s_ >= config_.plant.max_substep_s) {
    sync_plant();
    if (gates_changed) {
      plant_.set_gates(gh, gl);
    }
  }

  // Checkers: shoot-through + observed dead time on the model's gate pins.
  // Gate edges also feed the switching-loss accumulator and the correlated
  // edge-spike disturbance.
  for (int k = 0; k < 3; ++k) {
    if (gh[k] && gl[k]) ++shoot_through_;
    const bool edge = gh[k] != last_gh_[k] || gl[k] != last_gl_[k];
    if (edge) {
      thermal_.add_switch_edge(k, pvdd, plant_.state().current_a[k]);
      chain_.notify_gate_edge((gh[k] && !last_gh_[k]) ? 1.0 : -1.0);
    }
    if (last_gh_[k] && !gh[k]) gh_off_at_[k] = time_s_;
    if (last_gl_[k] && !gl[k]) gl_off_at_[k] = time_s_;
    if (gh[k] && !last_gh_[k] && gl_off_at_[k] >= 0.0) {
      const double dead = time_s_ - gl_off_at_[k];
      if (dead < min_dead_time_) min_dead_time_ = dead;
    }
    if (gl[k] && !last_gl_[k] && gh_off_at_[k] >= 0.0) {
      const double dead = time_s_ - gh_off_at_[k];
      if (dead < min_dead_time_) min_dead_time_ = dead;
    }
    last_gh_[k] = gh[k];
    last_gl_[k] = gl[k];
  }

  // FOC current sample-and-hold (Q21): freeze the low-side-shunt currents at
  // the PWM-counter peak (pwm_up 1->0), where every low side conducts. Only
  // active while the RTL is FOC-sampling (mode 3 or forced) - it must not
  // perturb six-step, which samples on the down-slope. Scheme 1 (sequential)
  // leaves the hold off and each ADC read samples live.
  const bool foc_active = top_->ctrl_foc_sample || (top_->ctrl_mode == 3);
  const bool pwm_up_now = top_->dbg_pwm_up;
  if (foc_active && last_pwm_up_ && !pwm_up_now &&
      config_.foc.current_sample_scheme == 0) {
    sync_plant();             // refresh the chain to the peak instant
    chain_.latch_currents();
  }
  last_pwm_up_ = pwm_up_now;

  // MCP3208 (analog source reads the feedback chain). The reference is the
  // live 3.3 V rail: PWM-synchronized ripple makes conversions ratiometric.
  if (config_.vref_ripple_v != 0.0) {
    const int half = static_cast<int>(
        std::lround(config_.clk_hz / (2.0 * config_.pwm_frequency_hz)));
    const double phase =
        top_->dbg_pwm_up
            ? static_cast<double>(top_->dbg_pwm_counter) / (2.0 * half)
            : 1.0 - static_cast<double>(top_->dbg_pwm_counter) / (2.0 * half);
    adc_->set_live_vref(config_.adc.vref_v +
                       config_.vref_ripple_v *
                           std::sin(2.0 * M_PI * phase));
  }
  adc_->update(time_s_, top_->adc_ncs, top_->adc_sclk, top_->adc_mosi);
  if (adc_->conversions() != last_adc_conversions_) {
    last_adc_conversions_ = adc_->conversions();
    const auto& s = adc_->last_sample();
    adc_log_.push_back(AdcSampleLog{s.time_s, s.channel, s.analog_v, s.code,
                                    static_cast<int>(top_->dbg_pwm_counter),
                                    static_cast<bool>(top_->dbg_pwm_up),
                                    static_cast<int>(top_->dbg_duty)});
    // Reservoir-cap channels see the sampling charge theft.
    chain_.apply_sample_theft(s.channel, adc_->last_sample_theft_v());
  }

  // ADS9224R dual-simultaneous current ADC (FOC path). On a CONVST edge it
  // latches both phase-current channels at the same instant; refresh the chain
  // first so the sampled currents are fresh at that instant.
  if (top_->adc2_convst && !last_adc2_convst_) sync_plant();
  last_adc2_convst_ = top_->adc2_convst;
  adc2_.update(time_s_, top_->adc2_convst, top_->adc2_ncs, top_->adc2_sclk,
               chain_.channel(0), chain_.channel(1));

  // Angle sensor: advance the physical angle, then service the SPI bus (the
  // AS5047P path; a no-op for the AS5600 PWM part).
  encoder_->update(time_s_, plant_.state().theta_rad);
  encoder_->spi_io(top_->angle_ncs, top_->angle_sclk, top_->angle_mosi);

  // Peripheral outputs -> RTL inputs (with optional line corruption).
  top_->drv_miso = glitched(0, drv_->sdo());
  top_->nfault = drv_->nfault();
  top_->noctw = drv_->noctw();
  top_->adc_miso = glitched(1, adc_->dout());
  top_->angle_pwm = glitched(2, encoder_->out());
  top_->angle_miso = encoder_->miso();
  top_->adc2_sdo_a = adc2_.sdo_a();
  top_->adc2_sdo_b = adc2_.sdo_b();
  top_->adc2_ready = adc2_.ready();

  // UART host model: serialize queued bytes onto the RTL's RX pin and
  // decode its TX pin (8N1).
  if (uart_tx_bit_idx_ < 0 && !uart_tx_queue_.empty()) {
    uart_tx_bit_idx_ = 0;
    uart_tx_next_edge_ = time_s_;
  }
  if (uart_tx_bit_idx_ >= 0 && time_s_ >= uart_tx_next_edge_) {
    const int byte = uart_tx_queue_.front();
    if (uart_tx_bit_idx_ == 0) {
      top_->uart_rx_pin = 0;  // start
    } else if (uart_tx_bit_idx_ <= 8) {
      top_->uart_rx_pin = (byte >> (uart_tx_bit_idx_ - 1)) & 1;
    } else if (uart_tx_bit_idx_ == 9) {
      top_->uart_rx_pin = 1;  // stop bit, held for a FULL bit time
    } else {
      // Stop bit complete: byte done, line stays idle-high.
      uart_tx_bit_idx_ = -1;
      uart_tx_queue_.pop_front();
    }
    if (uart_tx_bit_idx_ >= 0) {
      uart_tx_next_edge_ = time_s_ + uart_bit_s_;
      ++uart_tx_bit_idx_;
    }
  }
  const bool tx_line = top_->uart_tx_pin;
  if (uart_rx_bit_idx_ < 0) {
    if (uart_prev_line_ && !tx_line) {  // start edge
      uart_rx_bit_idx_ = 0;
      uart_rx_shift_ = 0;
      uart_rx_next_sample_ = time_s_ + 1.5 * uart_bit_s_;  // mid of bit 0
    }
  } else if (time_s_ >= uart_rx_next_sample_) {
    if (uart_rx_bit_idx_ < 8) {
      uart_rx_shift_ |= (tx_line ? 1 : 0) << uart_rx_bit_idx_;
      ++uart_rx_bit_idx_;
      uart_rx_next_sample_ += uart_bit_s_;
    } else {
      if (tx_line) uart_rx_bytes_.push_back(uart_rx_shift_);  // stop valid
      uart_rx_bit_idx_ = -1;
    }
  }
  uart_prev_line_ = tx_line;

  // Negedge.
  top_->clk = 0;
  top_->eval();
  if (vcd_) vcd_->dump(static_cast<std::uint64_t>(cycle_ * 2 + 1));

  // Trace.
  if (time_s_ >= next_trace_at_) {
    next_trace_at_ = time_s_ + config_.trace_interval_s;
    BenchTraceRow row;
    row.time_s = time_s_;
    row.current_a = plant_.state().current_a;
    row.omega_rad_s = plant_.state().omega_rad_s;
    row.theta_rad = plant_.state().theta_rad;
    row.sector = top_->dbg_sector;
    row.duty = top_->dbg_duty;
    row.configured = top_->dbg_configured;
    row.fault_count = top_->dbg_fault_count;
    row.mismatch_count = top_->dbg_mismatch_count;
    row.speed_rad_s = top_->dbg_speed;
    row.cur_code = top_->dbg_cur_code;
    row.emf_code = top_->dbg_emf_code;
    row.vbus_code = top_->dbg_vbus_code;
    row.vbus_v = config_.supply.enabled ? plant_.bus_v() : config_.vbus_v;
    row.fet_tj_c = thermal_.fet_tj_max_c();
    row.motor_t_c = thermal_.motor_t_c();
    trace_.push_back(row);
  }

  ++cycle_;
}

void Bench::run_cycles(std::uint64_t cycles) {
  for (std::uint64_t n = 0; n < cycles; ++n) {
    tick();
  }
  sync_plant();
}

void Bench::run_for(double seconds) {
  run_cycles(static_cast<std::uint64_t>(seconds * config_.clk_hz));
}

}  // namespace bldcsim
