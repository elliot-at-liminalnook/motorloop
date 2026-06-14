// BLDC controller top level.
//
// Modes (ctrl_mode):
//   0 idle        - gates off, DRV stays configured
//   1 open loop   - forced commutation ramp, duty = ctrl_duty
//   2 closed loop - sensored six-step from the AS5600 angle, speed PI
//
// The ctrl_* ports are the bench's stand-in for the future UART/telemetry
// register file; dbg_* ports expose internal state for assertions.

`include "rtl_params.vh"

module controller_top (
    input  wire        clk,
    input  wire        rst_n,
    // DRV8301
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
    // AS5600
    input  wire        angle_pwm,
    // UART command/telemetry link (overrides ctrl_* when enabled via reg 8)
    input  wire        uart_rx_pin,
    output wire        uart_tx_pin,
    // Control interface (bench-driven)
    input  wire [1:0]  ctrl_mode,
    input  wire [15:0] ctrl_duty,            // open-loop duty compare
    input  wire [15:0] ctrl_target_speed,    // rad/s
    input  wire [31:0] ctrl_ol_freq_word,    // open-loop final freq word
    input  wire [31:0] ctrl_ol_ramp_inc,     // freq word step per 256 clk
    input  wire [11:0] ctrl_align_offset,    // elec angle -> sector offset
    input  wire        ctrl_foc_sample,      // force FOC current sampling (test)
    input  wire signed [17:0] ctrl_id_target,  // FOC d-axis current command
    input  wire signed [17:0] ctrl_iq_target,  // FOC q-axis (torque) command
    input  wire        ctrl_foc_speed_loop,    // 1: iq* from speed PI, 0: direct
    input  wire        ctrl_foc_extrap,        // 1: angle-latency extrapolation
    // Debug
    output wire [2:0]  dbg_sector,
    output wire [15:0] dbg_duty,
    output wire        dbg_configured,
    output wire [7:0]  dbg_fault_count,
    output wire [7:0]  dbg_mismatch_count,
    output wire [11:0] dbg_angle,
    output wire        dbg_angle_valid,
    output wire [15:0] dbg_speed,
    output wire [11:0] dbg_cur_code,
    output wire [11:0] dbg_emf_code,
    output wire [11:0] dbg_vbus_code,
    output wire [11:0] dbg_offset_a,
    output wire [11:0] dbg_offset_b,
    output wire [15:0] dbg_noctw_count,
    output wire [15:0] dbg_pwm_counter,
    output wire        dbg_pwm_up,
    output wire        dbg_drv_dead,
    output wire        dbg_locked_out,
    output wire        dbg_stall,
    output wire        dbg_adc_stuck,
    output wire        dbg_offset_fault,
    output wire        dbg_reverse,
    // FOC current sampling (foc-checklist stage 4)
    output wire signed [17:0] dbg_foc_cur_a,
    output wire signed [17:0] dbg_foc_cur_b,
    output wire        dbg_foc_valid,
    // FOC datapath (foc-checklist stage 5/6)
    output wire signed [17:0] dbg_foc_id,
    output wire signed [17:0] dbg_foc_iq,
    output wire signed [17:0] dbg_foc_vd,
    output wire signed [17:0] dbg_foc_vq
);

  // ---- Fault pin synchronizers ---------------------------------------------
  reg [1:0] nfault_sync_r, noctw_sync_r;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      nfault_sync_r <= 2'b11;
      noctw_sync_r <= 2'b11;
    end else begin
      nfault_sync_r <= {nfault_sync_r[0], nfault};
      noctw_sync_r <= {noctw_sync_r[0], noctw};
    end
  end
  wire nfault_sync = nfault_sync_r[1];

  reg [15:0] noctw_count;
  reg noctw_prev;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      noctw_count <= 16'd0;
      noctw_prev <= 1'b1;
    end else begin
      noctw_prev <= noctw_sync_r[1];
      if (!noctw_sync_r[1] && noctw_prev) noctw_count <= noctw_count + 16'd1;
    end
  end
  assign dbg_noctw_count = noctw_count;

  // ---- DRV8301 manager + SPI ------------------------------------------------
  wire        drv_spi_start, drv_spi_busy, drv_spi_done;
  wire [15:0] drv_spi_tx, drv_spi_rx;
  wire        configured, gate_kill;

  spi_drv_master u_drv_spi (
      .clk(clk), .rst_n(rst_n),
      .start(drv_spi_start), .tx(drv_spi_tx),
      .busy(drv_spi_busy), .done(drv_spi_done), .rx(drv_spi_rx),
      .sclk(drv_sclk), .mosi(drv_mosi), .ncs(drv_ncs), .miso(drv_miso)
  );

  wire lockout_clear;
  drv_manager u_drv_mgr (
      .clk(clk), .rst_n(rst_n),
      .nfault_sync(nfault_sync),
      .lockout_clear(lockout_clear),
      .en_gate(en_gate), .dc_cal(dc_cal),
      .configured(configured), .gate_kill(gate_kill),
      .drv_dead(dbg_drv_dead), .locked_out(dbg_locked_out),
      .fault_count(dbg_fault_count), .mismatch_count(dbg_mismatch_count),
      .spi_start(drv_spi_start), .spi_tx(drv_spi_tx),
      .spi_busy(drv_spi_busy), .spi_done(drv_spi_done), .spi_rx(drv_spi_rx)
  );
  assign dbg_configured = configured;

  // ---- Angle capture ---------------------------------------------------------
  wire [11:0] angle;
  wire        angle_valid, angle_sample;
  as5600_pwm_capture u_angle (
      .clk(clk), .rst_n(rst_n), .pwm_in(angle_pwm),
      .angle(angle), .angle_valid(angle_valid), .new_sample(angle_sample)
  );
  assign dbg_angle = angle;
  assign dbg_angle_valid = angle_valid;

  // ---- UART register file + control source mux --------------------------------
  wire        use_uart;
  wire [1:0]  u_mode;
  wire [15:0] u_duty, u_target_speed;
  wire [11:0] u_align;
  wire [31:0] u_ol_freq_word, u_ol_ramp_inc;

  wire [1:0]  eff_mode = use_uart ? u_mode : ctrl_mode;
  wire [15:0] eff_duty_in = use_uart ? u_duty : ctrl_duty;
  wire [15:0] eff_target = use_uart ? u_target_speed : ctrl_target_speed;
  wire [11:0] eff_align = use_uart ? u_align : ctrl_align_offset;
  wire [31:0] eff_freq_word = use_uart ? u_ol_freq_word : ctrl_ol_freq_word;
  wire [31:0] eff_ramp_inc = use_uart ? u_ol_ramp_inc : ctrl_ol_ramp_inc;

  // ---- Sector sources ---------------------------------------------------------
  wire [2:0] ol_sector;
  open_loop_ramp u_ramp (
      .clk(clk), .rst_n(rst_n),
      .enable(eff_mode == 2'd1 && configured),
      .target_freq_word(eff_freq_word),
      .ramp_increment(eff_ramp_inc),
      .sector(ol_sector)
  );

  // Sensored sector: electrical angle = mech angle * pole pairs (+ offset),
  // sector = elec12 * 6 / 4096, with hysteresis (E1): the registered sector
  // only advances to an ADJACENT sector once the position has penetrated
  // SECTOR_HYST counts past the shared boundary; bigger jumps (fast motion,
  // wrap) are accepted immediately.
  wire [31:0] elec_raw = {20'd0, angle} * `POLE_PAIRS
                         + {20'd0, eff_align};
  wire [11:0] elec12 = elec_raw[11:0];
  wire [31:0] sector_scaled = {20'd0, elec12} * 32'd6;
  wire [2:0] sector_cand = sector_scaled[14:12];
  wire [11:0] pos_in_sector = sector_scaled[11:0];  // 0..4095 within sector
  localparam [11:0] HYST_SCALED = `SECTOR_HYST * 6;

  reg [2:0] sensored_sector;
  wire [2:0] sect_next = (sensored_sector == 3'd5) ? 3'd0
                                                   : sensored_sector + 3'd1;
  wire [2:0] sect_prev = (sensored_sector == 3'd0) ? 3'd5
                                                   : sensored_sector - 3'd1;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sensored_sector <= 3'd0;
    end else if (sector_cand != sensored_sector) begin
      if (sector_cand == sect_next) begin
        if (pos_in_sector >= HYST_SCALED) sensored_sector <= sector_cand;
      end else if (sector_cand == sect_prev) begin
        if (pos_in_sector <= 12'd4095 - HYST_SCALED)
          sensored_sector <= sector_cand;
      end else begin
        sensored_sector <= sector_cand;  // non-adjacent: take it
      end
    end
  end

  reg [2:0] sector;
  always @(*) begin
    case (eff_mode)
      2'd1: sector = ol_sector;
      2'd2: sector = sensored_sector;
      default: sector = 3'd0;
    endcase
  end
  assign dbg_sector = sector;

  // ---- Commutation + PWM -----------------------------------------------------
  wire [5:0] leg_mode;
  wire [1:0] low_phase, float_phase;
  commutation u_comm (
      .sector(sector), .leg_mode(leg_mode),
      .low_phase(low_phase), .float_phase(float_phase)
  );

  // Modes 2 (sensored six-step) and 3 (FOC) need a valid rotor angle.
  wire need_angle = (eff_mode == 2'd2) || (eff_mode == 2'd3);
  wire run_gates = (eff_mode != 2'd0) && configured
                   && (!need_angle || angle_valid);

  wire [15:0] pi_duty;
  reg [15:0] duty;
  always @(*) begin
    case (eff_mode)
      2'd1: duty = eff_duty_in;
      2'd2: duty = pi_duty;
      default: duty = 16'd0;
    endcase
  end
  assign dbg_duty = duty;

  // ---- Stall detection (E21): near-max duty with zero measured speed for
  // STALL_CYC = stall fault; safe-off until the host idles (mode 0).
  localparam [15:0] STALL_DUTY = (`PWM_HALF_PERIOD * 9) / 10;
  reg        stall_latched;
  reg [31:0] stall_timer;
  // Angle-motion discriminator: a sensor-aliased fast rotor reads zero
  // SPEED at max duty (looks like a stall) but its ANGLE keeps churning;
  // a true stall freezes the angle. Sampled every ~10 ms.
  reg [11:0] stall_angle_ref;
  reg [17:0] stall_sample_cnt;
  wire [11:0] stall_angle_diff = (angle - stall_angle_ref) & 12'hFFF;
  wire angle_moved = stall_angle_diff > 12'd64
                     && stall_angle_diff < 12'd4032;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      stall_latched <= 1'b0;
      stall_timer <= 32'd0;
      stall_angle_ref <= 12'd0;
      stall_sample_cnt <= 18'd0;
    end else if (eff_mode == 2'd0) begin
      stall_latched <= 1'b0;   // host acknowledged by idling
      stall_timer <= 32'd0;
    end else if (eff_mode == 2'd2 && duty >= STALL_DUTY && speed == 16'd0
                 && dbg_configured) begin
      stall_sample_cnt <= stall_sample_cnt + 18'd1;
      if (stall_sample_cnt == 18'd0) begin
        if (angle_moved) stall_timer <= 32'd0;  // rotor moving: not a stall
        stall_angle_ref <= angle;
      end
      if (stall_timer >= `STALL_CYC) stall_latched <= 1'b1;
      else stall_timer <= stall_timer + 32'd1;
    end else begin
      stall_timer <= 32'd0;
      stall_sample_cnt <= 18'd0;
      stall_angle_ref <= angle;
    end
  end
  assign dbg_stall = stall_latched;

  // ---- Lockout/dead clear: host holds mode 0 for HEALTHY_RUN_CYC.
  reg [31:0] idle_clear_timer;
  reg        lockout_clear_r;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      idle_clear_timer <= 32'd0;
      lockout_clear_r <= 1'b0;
    end else begin
      lockout_clear_r <= 1'b0;
      if (eff_mode == 2'd0 && (dbg_locked_out || dbg_drv_dead)) begin
        if (idle_clear_timer >= `HEALTHY_RUN_CYC) begin
          idle_clear_timer <= 32'd0;
          lockout_clear_r <= 1'b1;
        end else begin
          idle_clear_timer <= idle_clear_timer + 32'd1;
        end
      end else begin
        idle_clear_timer <= 32'd0;
      end
    end
  end
  assign lockout_clear = lockout_clear_r;

  wire [15:0] pwm_counter;
  wire        pwm_up, period_start;
  wire [2:0]  gate_high, gate_low;
  // ---- FOC datapath (mode 3) -------------------------------------------------
  // Electrical angle to 16-bit (elec12 is 0..4095 = 0..2pi), optionally
  // advanced by omega_e*t_latency to undo the AS5600 frame+filter lag (Q22).
  wire signed [17:0] speed_signed_for_extrap =
      speed_reverse ? -$signed({2'b00, speed}) : $signed({2'b00, speed});
  wire signed [31:0] extrap_counts =
      (speed_signed_for_extrap * `EXTRAP_NUM) >>> `EXTRAP_SH;
  wire [11:0] elec12_foc = ctrl_foc_extrap
                         ? (elec12 + extrap_counts[11:0])
                         : elec12;
  wire [15:0] theta_e16 = {elec12_foc, 4'd0};
  wire [47:0] foc_duty3;
  wire foc_enable = (eff_mode == 2'd3) && run_gates;
  // A UART host commanding FOC (mode 3) always drives the outer speed loop
  // (it sends a speed target, not a raw current); the bench-direct path can
  // also command iq* directly (stage-5 fixed-angle current control).
  wire eff_foc_speed_loop = use_uart ? (eff_mode == 2'd3)
                                     : ctrl_foc_speed_loop;
  wire signed [17:0] speed_iq_cmd;
  speed_iq_pi u_speed_iq (
      .clk(clk), .rst_n(rst_n),
      .enable(foc_enable && eff_foc_speed_loop),
      .update(speed_update),
      .target_speed(eff_target),
      .speed(speed),
      .reverse(speed_reverse),
      .iq_target(speed_iq_cmd)
  );
  wire signed [17:0] foc_iq_target =
      eff_foc_speed_loop ? speed_iq_cmd : ctrl_iq_target;

  foc_core u_foc (
      .clk(clk), .rst_n(rst_n),
      .enable(foc_enable), .update(dbg_foc_valid),
      .cur_a(dbg_foc_cur_a), .cur_b(dbg_foc_cur_b),
      .theta_e(theta_e16),
      .id_target(ctrl_id_target), .iq_target(foc_iq_target),
      .duty3(foc_duty3),
      .dbg_id(dbg_foc_id), .dbg_iq(dbg_foc_iq),
      .dbg_vd(dbg_foc_vd), .dbg_vq(dbg_foc_vq)
  );

  // PWM input mux: FOC drives three independent duties with all legs in PWM
  // mode; six-step replicates the single duty and floats/grounds two legs.
  wire [47:0] pwm_duty3 = (eff_mode == 2'd3) ? foc_duty3 : {duty, duty, duty};
  wire [5:0]  pwm_leg_mode =
      !run_gates ? 6'b000000
                 : (eff_mode == 2'd3) ? 6'b010101 : leg_mode;

  pwm_generator u_pwm (
      .clk(clk), .rst_n(rst_n),
      .kill(gate_kill || !run_gates || stall_latched),
      .duty3(pwm_duty3),
      .leg_mode(pwm_leg_mode),
      .gate_high(gate_high), .gate_low(gate_low),
      .counter_out(pwm_counter), .counting_up(pwm_up),
      .period_start(period_start)
  );
  assign inh = gate_high;
  assign inl = gate_low;
  assign dbg_pwm_counter = pwm_counter;
  assign dbg_pwm_up = pwm_up;

  // ---- Speed measurement + PI -------------------------------------------------
  // Speed is measured from the sensored sector (the real rotor position) in
  // every mode, so the FOC speed loop (mode 3) has a measurement even though
  // it does not use the commutation sector.
  wire [15:0] speed;
  wire        speed_valid, speed_update, speed_reverse;
  speed_meter u_speed (
      .clk(clk), .rst_n(rst_n), .sector(sensored_sector),
      .speed(speed), .reverse(speed_reverse),
      .speed_valid(speed_valid), .update(speed_update)
  );
  assign dbg_speed = speed;
  assign dbg_reverse = speed_reverse;

  speed_pi u_pi (
      .clk(clk), .rst_n(rst_n),
      .enable(eff_mode == 2'd2 && run_gates),
      .update(speed_update),
      .target_speed(eff_target),
      .speed(speed),
      .reverse(speed_reverse),
      .duty_compare(pi_duty)
  );

  // ---- ADC ----------------------------------------------------------------------
  wire        adc_start, adc_busy, adc_done;
  wire [2:0]  adc_channel;
  wire [11:0] adc_code;
  adc_spi_master u_adc_spi (
      .clk(clk), .rst_n(rst_n),
      .start(adc_start), .channel(adc_channel),
      .busy(adc_busy), .done(adc_done), .code(adc_code),
      .sclk(adc_sclk), .mosi(adc_mosi), .ncs(adc_ncs), .miso(adc_miso)
  );

  // FOC current sampling is active in FOC mode (3) or when forced for tests.
  wire foc_sample = (eff_mode == 2'd3) || ctrl_foc_sample;

  wire [11:0] offset_c_unused;
  adc_sequencer u_adc_seq (
      .clk(clk), .rst_n(rst_n),
      .pwm_counter(pwm_counter), .pwm_up(pwm_up),
      .period_start(period_start),
      .duty_compare(duty),
      .low_phase(low_phase), .float_phase(float_phase),
      .dc_cal_window(dc_cal),
      .cur_code(dbg_cur_code), .emf_code(dbg_emf_code),
      .vbus_code(dbg_vbus_code),
      .offset_a(dbg_offset_a), .offset_b(dbg_offset_b),
      .offset_c(offset_c_unused),
      .offset_fault(dbg_offset_fault), .adc_stuck(dbg_adc_stuck),
      .cur_valid(), .emf_valid(),
      .adc_start(adc_start), .adc_channel(adc_channel),
      .adc_busy(adc_busy), .adc_done(adc_done), .adc_code(adc_code),
      .foc_mode(foc_sample),
      .foc_cur_a(dbg_foc_cur_a), .foc_cur_b(dbg_foc_cur_b),
      .foc_valid(dbg_foc_valid)
  );

  // ---- UART register file instance ---------------------------------------------
  uart_regfile u_uart (
      .clk(clk), .rst_n(rst_n),
      .uart_rx_pin(uart_rx_pin), .uart_tx_pin(uart_tx_pin),
      .use_uart(use_uart),
      .r_mode(u_mode), .r_duty(u_duty), .r_target_speed(u_target_speed),
      .r_align(u_align), .r_ol_freq_word(u_ol_freq_word),
      .r_ol_ramp_inc(u_ol_ramp_inc),
      .t_speed(speed),
      .t_fault_count(dbg_fault_count),
      .t_mismatch_count(dbg_mismatch_count),
      .t_angle(angle),
      .t_noctw_count(noctw_count),
      .t_sector(sector),
      .t_configured(configured),
      .t_flags({2'd0, dbg_adc_stuck, dbg_offset_fault, stall_latched,
                dbg_locked_out, dbg_drv_dead, speed_reverse})
  );

endmodule
