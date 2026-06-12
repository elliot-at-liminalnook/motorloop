#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <random>

#include "as5600.hpp"
#include "drv8301.hpp"
#include "feedback_chain.hpp"
#include "mcp3208.hpp"
#include "thermal_model.hpp"
#include "three_phase_plant.hpp"

class Vcontroller_top;
class VerilatedContext;
class VerilatedVcdC;

namespace bldcsim {

// Lockstep bench: the Verilated controller_top, the behavioral peripherals,
// and the C++ plant advance together under a single time authority
// (see notes/architecture.md).
//
// Per RTL clock cycle:
//   1. posedge eval,
//   2. RTL outputs -> DRV8301 model (gates, SPI), MCP3208, AS5600,
//   3. plant catch-up: on any gate change immediately, otherwise capped at
//      sim.max_substep lag; feedback chain follows the plant,
//   4. peripheral outputs -> RTL inputs, negedge eval.
//
// Built-in checkers: shoot-through monitor on the post-DRV gate outputs and
// minimum observed dead time per leg.

struct BenchTraceRow {
  double time_s;
  std::array<double, 3> current_a;
  double omega_rad_s;
  double theta_rad;
  int sector;
  int duty;
  bool configured;
  int fault_count;
  int mismatch_count;
  int speed_rad_s;
  int cur_code;
  int emf_code;
  int vbus_code;
  double vbus_v;       // live bus voltage (== config value, stiff supply)
  double fet_tj_c;     // hottest FET junction (ambient when thermal off)
  double motor_t_c;
};

struct AdcSampleLog {
  double time_s;
  int channel;
  double analog_v;
  int code;
  int pwm_counter;   // RTL PWM counter at the hold instant
  bool pwm_up;
  int duty;          // active duty compare at the hold instant
};

struct BenchConfig {
  double clk_hz = 25e6;
  double vbus_v = 12.0;
  ThreePhaseMotorParams motor{};
  BridgeParams bridge{};
  PlantConfig plant{};
  SupplyParams supply{};      // realism stage 1 (disabled = stiff source)
  ThermalConfig thermal{};    // realism stage 4 (disabled = static params)
  Drv8301Config drv{};
  Mcp3208Config adc{};
  As5600Config encoder{};
  FeedbackChainConfig chain{};
  double vref_ripple_v = 0.0;     // PWM-synchronized ADC reference ripple
  double pwm_frequency_hz = 20e3;
  double uart_baud = 115200.0;
  double trace_interval_s = 10e-6;
};

class Bench {
 public:
  explicit Bench(const BenchConfig& config);
  ~Bench();

  // -- control inputs (the future UART register file) --
  void set_mode(int mode);
  void set_duty(int duty_compare);
  void set_target_speed(int rad_s);
  void set_open_loop(std::uint32_t freq_word, std::uint32_t ramp_inc);
  void set_align_offset(int offset12);

  // -- run --
  void run_for(double seconds);
  void run_cycles(std::uint64_t cycles);

  // -- injections --
  void inject_drv_register_reset() { drv_.inject_register_reset(); }
  void inject_drv_latched_fault() { drv_.inject_latched_fault(); }
  void inject_drv_otw(bool active) { drv_.inject_otw(active); }
  void inject_magnet_loss(bool lost) { encoder_.inject_magnet_loss(lost); }
  void set_load_torque(double n_m) {
    base_load_nm_ = n_m;
    plant_.set_load_torque(n_m);
  }
  // E22: sinusoidal load-torque oscillation on top of the base load.
  void set_load_oscillation(double amplitude_nm, double freq_hz) {
    load_osc_amp_ = amplitude_nm;
    load_osc_hz_ = freq_hz;
  }

  // Digital-line corruption (realism 6.4): seeded random bit flips on the
  // selected lines until end_t. Mask bits: 0 drv_miso, 1 adc_miso,
  // 2 angle_pwm, 3 drv_mosi (RTL->model), 4 drv_sclk (RTL->model).
  void set_line_glitches(unsigned mask, double rate_hz, double end_t_s,
                         unsigned seed);

  // FPGA-configuration window (realism 6.5): before the RTL runs, drive the
  // DRV model inputs with seeded random toggling for `seconds`, modeling
  // floating FPGA pins. With the external EN_GATE pull-down present the
  // power stage must stay dead. Call before any run_* method.
  void run_config_window(double seconds, bool en_gate_pulldown,
                         unsigned seed);

  // UART host link (realism 6.1-6.3).
  void uart_send(const std::vector<int>& bytes);
  std::vector<int> uart_take_received();

  // -- probes --
  double time_s() const { return time_s_; }
  const ThreePhasePlant& plant() const { return plant_; }
  const Drv8301& drv() const { return drv_; }
  std::array<double, 3> currents() const { return plant_.state().current_a; }
  double omega() const { return plant_.state().omega_rad_s; }
  double theta() const { return plant_.state().theta_rad; }
  int dbg_sector() const;
  int dbg_duty() const;
  bool dbg_configured() const;
  int dbg_fault_count() const;
  int dbg_mismatch_count() const;
  int dbg_speed() const;
  int dbg_angle() const;
  bool dbg_angle_valid() const;
  int dbg_offset_a() const;
  int dbg_offset_b() const;
  int dbg_noctw_count() const;

  long shoot_through_violations() const { return shoot_through_; }
  double min_dead_time_s() const { return min_dead_time_; }
  long plant_shoot_through_requests() const {
    return plant_.shoot_through_requests();
  }

  // Realism probes.
  double bus_v() const { return plant_.bus_v(); }
  double bus_v_min() const { return bus_v_min_; }
  double bus_v_max() const { return bus_v_max_; }
  bool supply_in_cc() const { return plant_.supply_in_cc(); }
  bool drv_pvdd_uv() const { return drv_.pvdd_uv_active(); }
  long pvdd_uv_events() const { return pvdd_uv_events_; }
  double fet_tj_max_c() const { return thermal_.fet_tj_max_c(); }
  double drv_t_c() const { return thermal_.drv_t_c(); }
  double motor_t_c() const { return thermal_.motor_t_c(); }
  long config_window_gate_activity() const {
    return config_window_gate_activity_;
  }
  // Edge-case telemetry flags (E13/E20/E21/E14/E10/E5).
  bool drv_dead() const;
  bool locked_out() const;
  bool stalled() const;
  bool adc_stuck() const;
  bool offset_fault() const;
  bool speed_reverse() const;

  const std::vector<BenchTraceRow>& trace() const { return trace_; }
  const std::vector<AdcSampleLog>& adc_samples() const { return adc_log_; }
  void clear_trace() { trace_.clear(); adc_log_.clear(); }

  void open_vcd(const std::string& path);
  void close_vcd();

 private:
  void tick();
  void sync_plant();
  bool glitched(unsigned bit, bool value);

  BenchConfig config_;
  std::unique_ptr<VerilatedContext> ctx_;
  std::unique_ptr<Vcontroller_top> top_;
  std::unique_ptr<VerilatedVcdC> vcd_;

  ThreePhasePlant plant_;
  Drv8301 drv_;
  Mcp3208 adc_;
  As5600 encoder_;
  FeedbackChain chain_;
  ThermalModel thermal_;

  double time_s_ = 0.0;
  double plant_time_s_ = 0.0;
  double period_s_;
  std::uint64_t cycle_ = 0;
  double pre_time_s_ = 0.0;  // config-window time before the RTL runs

  // Disturbance bookkeeping.
  double last_bus_i_ = 0.0;
  double last_bus_i_t_ = -1.0;
  double bus_v_min_ = 1e9;
  double bus_v_max_ = -1e9;
  bool last_pvdd_uv_ = false;
  long pvdd_uv_events_ = 0;
  long config_window_gate_activity_ = 0;
  double base_load_nm_ = 0.0;
  double load_osc_amp_ = 0.0;
  double load_osc_hz_ = 0.0;

  // Glitch injector.
  unsigned glitch_mask_ = 0;
  double glitch_rate_hz_ = 0.0;
  double glitch_end_t_ = -1.0;
  std::mt19937 glitch_rng_{1};

  // UART host model (8N1 at rtl.uart_baud).
  double uart_bit_s_ = 0.0;
  std::vector<int> uart_tx_queue_;     // host -> RTL, bytes
  int uart_tx_bit_idx_ = -1;           // -1 idle, 0 start, 1..8 data, 9 stop
  double uart_tx_next_edge_ = 0.0;
  std::vector<int> uart_rx_bytes_;     // RTL -> host, decoded
  int uart_rx_bit_idx_ = -1;
  double uart_rx_next_sample_ = 0.0;
  int uart_rx_shift_ = 0;
  bool uart_prev_line_ = true;

  std::array<bool, 3> last_gh_{false, false, false};
  std::array<bool, 3> last_gl_{false, false, false};
  std::array<double, 3> gh_off_at_{-1.0, -1.0, -1.0};
  std::array<double, 3> gl_off_at_{-1.0, -1.0, -1.0};
  long shoot_through_ = 0;
  double min_dead_time_ = 1e9;

  long last_adc_conversions_ = 0;
  double next_trace_at_ = 0.0;
  std::vector<BenchTraceRow> trace_;
  std::vector<AdcSampleLog> adc_log_;
};

}  // namespace bldcsim
