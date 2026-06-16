// SPDX-License-Identifier: MIT
// Board-level synthesis wrapper for controller_top (platform-abstraction stage
// 15). controller_top is the *simulation* DUT: it exposes the whole ctrl_*
// command interface and dbg_* telemetry as ports (the bench's stand-in for the
// UART register file). On a real ULX3S those are NOT pins - control comes over
// UART and telemetry is read back over UART - so synthesizing controller_top
// directly exhausts the package I/O. This wrapper exposes only the genuine
// board-facing pins (clk, the gate outputs, the SPI/ADC/angle buses, UART) and
// keeps the ctrl_* inputs at reset constants (UART drives them at runtime).
// The dbg_* telemetry is OR-reduced into one test pin so the full datapath is
// kept (not optimized away) and the utilization/Fmax reflect the real logic.

module board_top (
    input  wire        clk,
    input  wire        rst_n,
    // DRV gate outputs + SPI
    output wire [2:0]  inh,
    output wire [2:0]  inl,
    output wire        en_gate,
    output wire        dc_cal,
    output wire        drv_sclk,
    output wire        drv_mosi,
    output wire        drv_ncs,
    input  wire        drv_miso,
    input  wire        nfault,
    input  wire        noctw,
    // MCP3208
    output wire        adc_sclk,
    output wire        adc_mosi,
    output wire        adc_ncs,
    input  wire        adc_miso,
    // ADS9224R
    output wire        adc2_convst,
    output wire        adc2_ncs,
    output wire        adc2_sclk,
    input  wire        adc2_sdo_a,
    input  wire        adc2_sdo_b,
    input  wire        adc2_ready,
    // Angle (AS5600 PWM in, AS5047P SPI)
    input  wire        angle_pwm,
    output wire        angle_sclk,
    output wire        angle_mosi,
    output wire        angle_ncs,
    input  wire        angle_miso,
    // UART
    input  wire        uart_rx_pin,
    output wire        uart_tx_pin,
    // Board-strap selects (a few real pins / config bits)
    input  wire        drv_hw_mode,
    input  wire        angle_spi_mode,
    input  wire        adc_dual_mode,
    input  wire [2:0]  cur_norm_shift,
    // Single aggregated telemetry pin (keeps the dbg_* datapath live)
    output wire        test_pin
);

  // Telemetry nets (kept, OR-reduced into test_pin).
  wire [2:0]  dbg_sector;
  wire [15:0] dbg_duty;
  wire        dbg_configured;
  wire [7:0]  dbg_fault_count, dbg_mismatch_count;
  wire [11:0] dbg_angle;
  wire        dbg_angle_valid;
  wire [15:0] dbg_speed;
  wire [11:0] dbg_cur_code, dbg_emf_code, dbg_vbus_code, dbg_offset_a, dbg_offset_b;
  wire [15:0] dbg_noctw_count, dbg_pwm_counter;
  wire        dbg_pwm_up, dbg_drv_dead, dbg_locked_out, dbg_stall;
  wire        dbg_adc_stuck, dbg_offset_fault, dbg_reverse;
  wire signed [17:0] dbg_foc_cur_a, dbg_foc_cur_b, dbg_foc_id, dbg_foc_iq;
  wire signed [17:0] dbg_foc_vd, dbg_foc_vq;

  controller_top u_ctrl (
      .clk(clk), .rst_n(rst_n),
      .inh(inh), .inl(inl), .en_gate(en_gate), .dc_cal(dc_cal),
      .drv_sclk(drv_sclk), .drv_mosi(drv_mosi), .drv_ncs(drv_ncs),
      .drv_miso(drv_miso), .nfault(nfault), .noctw(noctw),
      .adc_sclk(adc_sclk), .adc_mosi(adc_mosi), .adc_ncs(adc_ncs),
      .adc_miso(adc_miso),
      .adc2_convst(adc2_convst), .adc2_ncs(adc2_ncs), .adc2_sclk(adc2_sclk),
      .adc2_sdo_a(adc2_sdo_a), .adc2_sdo_b(adc2_sdo_b), .adc2_ready(adc2_ready),
      .angle_pwm(angle_pwm),
      .angle_sclk(angle_sclk), .angle_mosi(angle_mosi), .angle_ncs(angle_ncs),
      .angle_miso(angle_miso),
      .uart_rx_pin(uart_rx_pin), .uart_tx_pin(uart_tx_pin),
      // ctrl_* at reset constants; UART (or the straps below) drives runtime.
      .ctrl_mode(2'd0), .ctrl_duty(16'd0), .ctrl_target_speed(16'd0),
      .ctrl_ol_freq_word(32'd0), .ctrl_ol_ramp_inc(32'd0),
      .ctrl_align_offset(12'd0), .ctrl_foc_sample(1'b0),
      .ctrl_id_target(18'sd0), .ctrl_iq_target(18'sd0),
      .ctrl_foc_speed_loop(1'b0), .ctrl_foc_extrap(1'b1),
      .ctrl_drv_hw_mode(drv_hw_mode), .ctrl_angle_spi_mode(angle_spi_mode),
      .ctrl_cur_norm_shift(cur_norm_shift), .ctrl_adc_dual_mode(adc_dual_mode),
      .dbg_sector(dbg_sector), .dbg_duty(dbg_duty),
      .dbg_configured(dbg_configured), .dbg_fault_count(dbg_fault_count),
      .dbg_mismatch_count(dbg_mismatch_count), .dbg_angle(dbg_angle),
      .dbg_angle_valid(dbg_angle_valid), .dbg_speed(dbg_speed),
      .dbg_cur_code(dbg_cur_code), .dbg_emf_code(dbg_emf_code),
      .dbg_vbus_code(dbg_vbus_code), .dbg_offset_a(dbg_offset_a),
      .dbg_offset_b(dbg_offset_b), .dbg_noctw_count(dbg_noctw_count),
      .dbg_pwm_counter(dbg_pwm_counter), .dbg_pwm_up(dbg_pwm_up),
      .dbg_drv_dead(dbg_drv_dead), .dbg_locked_out(dbg_locked_out),
      .dbg_stall(dbg_stall), .dbg_adc_stuck(dbg_adc_stuck),
      .dbg_offset_fault(dbg_offset_fault), .dbg_reverse(dbg_reverse),
      .dbg_foc_cur_a(dbg_foc_cur_a), .dbg_foc_cur_b(dbg_foc_cur_b),
      .dbg_foc_valid(/* folded below */),
      .dbg_foc_id(dbg_foc_id), .dbg_foc_iq(dbg_foc_iq),
      .dbg_foc_vd(dbg_foc_vd), .dbg_foc_vq(dbg_foc_vq)
  );

  assign test_pin =
      ^{dbg_sector, dbg_duty, dbg_configured, dbg_fault_count,
        dbg_mismatch_count, dbg_angle, dbg_angle_valid, dbg_speed,
        dbg_cur_code, dbg_emf_code, dbg_vbus_code, dbg_offset_a, dbg_offset_b,
        dbg_noctw_count, dbg_pwm_counter, dbg_pwm_up, dbg_drv_dead,
        dbg_locked_out, dbg_stall, dbg_adc_stuck, dbg_offset_fault, dbg_reverse,
        dbg_foc_cur_a, dbg_foc_cur_b, dbg_foc_id, dbg_foc_iq,
        dbg_foc_vd, dbg_foc_vq};

endmodule
