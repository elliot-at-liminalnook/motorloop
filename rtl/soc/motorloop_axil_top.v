// SPDX-License-Identifier: MIT
// Reference-SoC integration wrapper (tier2-adoption-checklist §2): the motorloop
// controller as an AXI4-Lite peripheral. Connects `axil_regfile` (the formally
// proven AXI-Lite slave) to `controller_top` - the register file's control
// surface drives the controller's `ctrl_*`, and the controller's telemetry
// drives the read registers - so a SoC bus master (e.g. a RISC-V core) spins
// the motor and reads telemetry over AXI-Lite, exactly as it would over the
// UART register file.
//
// Platform straps are tied to the default BOM (zonri_drv8301: DRV8301 SPI gate
// driver, MCP3208 ADC, AS5600 PWM angle). The FOC torque comes from the speed
// PI (`ctrl_foc_speed_loop=1`), so writing {mode=3 (FOC), target_speed} commands
// closed-loop speed. UART is tied idle, so the controller takes its commands
// from `ctrl_*` (= the AXI registers).

module motorloop_axil_top #(
    parameter integer ADDR_W = 8     // AXI-Lite byte-address width (>= 7)
) (
    input  wire              clk,
    input  wire              rst_n,
    // AXI4-Lite slave (SoC side)
    input  wire [ADDR_W-1:0] s_awaddr,
    input  wire              s_awvalid,
    output wire              s_awready,
    input  wire [31:0]       s_wdata,
    input  wire [3:0]        s_wstrb,
    input  wire              s_wvalid,
    output wire              s_wready,
    output wire [1:0]        s_bresp,
    output wire              s_bvalid,
    input  wire              s_bready,
    input  wire [ADDR_W-1:0] s_araddr,
    input  wire              s_arvalid,
    output wire              s_arready,
    output wire [31:0]       s_rdata,
    output wire [1:0]        s_rresp,
    output wire              s_rvalid,
    input  wire              s_rready,
    // Controller peripheral pins (board side)
    output wire [2:0]        inh,
    output wire [2:0]        inl,
    output wire              en_gate,
    output wire              dc_cal,
    output wire              drv_sclk,
    output wire              drv_mosi,
    output wire              drv_ncs,
    input  wire              drv_miso,
    input  wire              nfault,
    input  wire              noctw,
    output wire              adc_sclk,
    output wire              adc_mosi,
    output wire              adc_ncs,
    input  wire              adc_miso,
    input  wire              angle_pwm
);
  // axil_regfile control surface + telemetry nets.
  wire        use_axi;
  wire [1:0]  r_mode;
  wire [15:0] r_duty, r_target_speed;
  wire [11:0] r_align;
  wire [31:0] r_ol_freq_word, r_ol_ramp_inc;

  wire [2:0]  dbg_sector;
  wire [15:0] dbg_speed, dbg_noctw_count;
  wire [11:0] dbg_angle;
  wire        dbg_configured;
  wire [7:0]  dbg_fault_count, dbg_mismatch_count;
  wire        dbg_adc_stuck, dbg_offset_fault, dbg_stall,
              dbg_locked_out, dbg_drv_dead, dbg_reverse;

  axil_regfile #(.ADDR_W(ADDR_W)) u_regs (
      .clk(clk), .rst_n(rst_n),
      .s_awaddr(s_awaddr), .s_awvalid(s_awvalid), .s_awready(s_awready),
      .s_wdata(s_wdata), .s_wstrb(s_wstrb), .s_wvalid(s_wvalid),
      .s_wready(s_wready), .s_bresp(s_bresp), .s_bvalid(s_bvalid),
      .s_bready(s_bready), .s_araddr(s_araddr), .s_arvalid(s_arvalid),
      .s_arready(s_arready), .s_rdata(s_rdata), .s_rresp(s_rresp),
      .s_rvalid(s_rvalid), .s_rready(s_rready),
      .use_axi(use_axi),
      .r_mode(r_mode), .r_duty(r_duty), .r_target_speed(r_target_speed),
      .r_align(r_align), .r_ol_freq_word(r_ol_freq_word),
      .r_ol_ramp_inc(r_ol_ramp_inc),
      .t_speed(dbg_speed), .t_fault_count(dbg_fault_count),
      .t_mismatch_count(dbg_mismatch_count), .t_angle(dbg_angle),
      .t_noctw_count(dbg_noctw_count), .t_sector(dbg_sector),
      .t_configured(dbg_configured),
      .t_flags({2'd0, dbg_adc_stuck, dbg_offset_fault, dbg_stall,
                dbg_locked_out, dbg_drv_dead, dbg_reverse}));

  controller_top u_ctrl (
      .clk(clk), .rst_n(rst_n),
      .inh(inh), .inl(inl), .en_gate(en_gate), .dc_cal(dc_cal),
      .drv_sclk(drv_sclk), .drv_mosi(drv_mosi), .drv_ncs(drv_ncs),
      .drv_miso(drv_miso), .nfault(nfault), .noctw(noctw),
      .adc_sclk(adc_sclk), .adc_mosi(adc_mosi), .adc_ncs(adc_ncs),
      .adc_miso(adc_miso),
      // ADS9224R unused on this BOM (MCP3208 path)
      .adc2_convst(), .adc2_ncs(), .adc2_sclk(),
      .adc2_sdo_a(1'b0), .adc2_sdo_b(1'b0), .adc2_ready(1'b0),
      .angle_pwm(angle_pwm),
      // AS5047P SPI angle unused on this BOM (AS5600 PWM path)
      .angle_sclk(), .angle_mosi(), .angle_ncs(), .angle_miso(1'b0),
      // UART tied idle: the controller takes commands from ctrl_* (= AXI regs)
      .uart_rx_pin(1'b1), .uart_tx_pin(),
      .ctrl_mode(r_mode), .ctrl_duty(r_duty),
      .ctrl_target_speed(r_target_speed),
      .ctrl_ol_freq_word(r_ol_freq_word), .ctrl_ol_ramp_inc(r_ol_ramp_inc),
      .ctrl_align_offset(r_align),
      .ctrl_foc_sample(1'b0),
      .ctrl_id_target(18'sd0), .ctrl_iq_target(18'sd0),
      .ctrl_foc_speed_loop(1'b1),    // FOC torque from the speed PI
      .ctrl_foc_extrap(1'b0),
      .ctrl_drv_hw_mode(1'b0),       // DRV8301: SPI config
      .ctrl_angle_spi_mode(1'b0),    // AS5600 PWM angle
      .ctrl_cur_norm_shift(3'd0),
      .ctrl_adc_dual_mode(1'b0),     // MCP3208
      .dbg_sector(dbg_sector), .dbg_duty(), .dbg_configured(dbg_configured),
      .dbg_fault_count(dbg_fault_count), .dbg_mismatch_count(dbg_mismatch_count),
      .dbg_angle(dbg_angle), .dbg_angle_valid(), .dbg_speed(dbg_speed),
      .dbg_cur_code(), .dbg_emf_code(), .dbg_vbus_code(),
      .dbg_offset_a(), .dbg_offset_b(),
      .dbg_noctw_count(dbg_noctw_count), .dbg_pwm_counter(), .dbg_pwm_up(),
      .dbg_drv_dead(dbg_drv_dead), .dbg_locked_out(dbg_locked_out),
      .dbg_stall(dbg_stall), .dbg_adc_stuck(dbg_adc_stuck),
      .dbg_offset_fault(dbg_offset_fault), .dbg_reverse(dbg_reverse),
      .dbg_foc_cur_a(), .dbg_foc_cur_b(), .dbg_foc_valid(),
      .dbg_foc_id(), .dbg_foc_iq(), .dbg_foc_vd(), .dbg_foc_vq());

  // use_axi is informational here (UART idle -> ctrl_* always drives); sink it.
  wire _unused_ok = &{1'b0, use_axi};
endmodule
