// SPDX-License-Identifier: MIT
// DRV8301 management state machine:
//  - power-up + EN_GATE ready sequencing,
//  - DC_CAL window for amplifier offset capture,
//  - SPI configuration of CR1/CR2 with readback-verify,
//  - periodic register refresh watchdog: reread CR1/CR2 every refresh
//    period; on mismatch (slva552 silent brownout reset) rewrite and count,
//  - nFAULT response: kill gates, quick-reset pulse on EN_GATE (< 10 us),
//    reconfigure, resume; faults counted.
//
// Reads are pipelined N+1: each verify sends two read frames and checks the
// second frame's response word {F0, addr, data}.

module drv_manager #(
    parameter integer CLK_HZ          = 25000000,
    parameter integer EN_READY_CYCLES = 300000,
    parameter integer QUICK_RESET_CYC = 100,
    parameter integer DRV_REFRESH_CYC = 25000,
    parameter integer HEALTHY_RUN_CYC = 2500000,
    parameter [3:0]   LOCKOUT_N       = 4'd4,
    parameter [3:0]   DRV_DEAD_N      = 4'd8,
    parameter integer OC_ADJ_CODE     = 16,
    parameter integer AMP_GAIN_CODE   = 0
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        nfault_sync,     // synchronized, active low
    input  wire        lockout_clear,   // host acknowledgement (>=100ms idle)
    input  wire        hw_mode,         // DRV8302: hardware-configured, no SPI
    output reg         en_gate,
    output reg         dc_cal,
    output reg         configured,
    output wire        gate_kill,
    output reg         drv_dead,        // E13: peripheral declared dead
    output reg         locked_out,      // E20: repeated-fault lockout
    output reg  [7:0]  fault_count,
    output reg  [7:0]  mismatch_count,
    // spi_drv_master interface
    output reg         spi_start,
    output reg  [15:0] spi_tx,
    input  wire        spi_busy,
    input  wire        spi_done,
    input  wire [15:0] spi_rx
);

  localparam [31:0] CR1_FULL = OC_ADJ_CODE << 6;
  localparam [31:0] CR2_FULL = AMP_GAIN_CODE << 2;
  localparam [10:0] CR1_VALUE = CR1_FULL[10:0];       // current-limit, 6-PWM
  localparam [10:0] CR2_VALUE = CR2_FULL[10:0];       // report OT+OC

  localparam [31:0] PWRUP_CYC = CLK_HZ / 1000;        // 1 ms settle
  localparam [31:0] DCCAL_CYC = CLK_HZ / 5000;        // 200 us
  localparam [31:0] FLT_WAIT_CYC = CLK_HZ / 5000;     // 200 us post-reset

  localparam [4:0]
      S_PWRUP   = 5'd0,  S_EN_WAIT = 5'd1,  S_DCCAL   = 5'd2,
      S_CFG_W1  = 5'd3,  S_CFG_W2  = 5'd4,
      S_VER1A   = 5'd5,  S_VER1B   = 5'd6,
      S_VER2A   = 5'd7,  S_VER2B   = 5'd8,
      S_CLR1    = 5'd9,  S_CLR2    = 5'd10,
      S_RUN     = 5'd11,
      S_FLT_KILL = 5'd12, S_FLT_QR = 5'd13, S_FLT_WAIT = 5'd14,
      S_DEAD    = 5'd15,  S_LOCKOUT = 5'd16;

  reg [4:0]  state;
  reg [31:0] timer;
  reg [31:0] refresh_timer;
  reg        frame_sent;
  reg [3:0]  consec_mismatch;   // E13: verify failures without a success
  reg [3:0]  consec_faults;     // E20: recoveries without a healthy run
  reg [31:0] healthy_timer;

  assign gate_kill = !configured;

  // One-frame helper: send tx once, wait for done.
  task automatic send_frame(input [15:0] tx, input [4:0] next_state);
    begin
      if (!frame_sent) begin
        if (!spi_busy && !spi_start) begin
          spi_tx <= tx;
          spi_start <= 1'b1;
          frame_sent <= 1'b1;
        end
      end else if (spi_done) begin
        frame_sent <= 1'b0;
        state <= next_state;
      end
    end
  endtask

  wire [15:0] frame_write_cr1 = {1'b0, 4'h2, CR1_VALUE};
  wire [15:0] frame_write_cr2 = {1'b0, 4'h3, CR2_VALUE};
  wire [15:0] frame_read_cr1 = {1'b1, 4'h2, 11'd0};
  wire [15:0] frame_read_cr2 = {1'b1, 4'h3, 11'd0};
  wire [15:0] frame_read_sr1 = {1'b1, 4'h0, 11'd0};
  wire [15:0] frame_read_sr2 = {1'b1, 4'h1, 11'd0};

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_PWRUP;
      en_gate <= 1'b0;
      dc_cal <= 1'b0;
      configured <= 1'b0;
      drv_dead <= 1'b0;
      locked_out <= 1'b0;
      fault_count <= 8'd0;
      mismatch_count <= 8'd0;
      spi_start <= 1'b0;
      spi_tx <= 16'd0;
      timer <= 32'd0;
      refresh_timer <= 32'd0;
      frame_sent <= 1'b0;
      consec_mismatch <= 4'd0;
      consec_faults <= 4'd0;
      healthy_timer <= 32'd0;
    end else begin
      spi_start <= 1'b0;

      case (state)
        S_PWRUP: begin
          en_gate <= 1'b0;
          configured <= 1'b0;
          timer <= timer + 32'd1;
          if (timer >= PWRUP_CYC) begin
            timer <= 32'd0;
            en_gate <= 1'b1;
            state <= S_EN_WAIT;
          end
        end
        S_EN_WAIT: begin
          timer <= timer + 32'd1;
          if (timer >= EN_READY_CYCLES) begin
            timer <= 32'd0;
            dc_cal <= 1'b1;
            state <= S_DCCAL;
          end
        end
        S_DCCAL: begin
          timer <= timer + 32'd1;
          if (timer >= DCCAL_CYC) begin
            timer <= 32'd0;
            dc_cal <= 1'b0;
            // Hardware-configured drivers (DRV8302) have no SPI register step:
            // go straight to RUN after EN_GATE + DC_CAL.
            state <= hw_mode ? S_RUN : S_CFG_W1;
          end
        end
        S_CFG_W1: send_frame(frame_write_cr1, S_CFG_W2);
        S_CFG_W2: send_frame(frame_write_cr2, S_VER1A);
        S_VER1A: send_frame(frame_read_cr1, S_VER1B);
        S_VER1B: begin
          if (!frame_sent) begin
            if (!spi_busy && !spi_start) begin
              spi_tx <= frame_read_cr1;
              spi_start <= 1'b1;
              frame_sent <= 1'b1;
            end
          end else if (spi_done) begin
            frame_sent <= 1'b0;
            if (spi_rx == {1'b0, 4'h2, CR1_VALUE}) begin
              state <= S_VER2A;
            end else begin
              if (mismatch_count != 8'hFF)        // E25: saturate
                mismatch_count <= mismatch_count + 8'd1;
              configured <= 1'b0;
              if (!nfault_sync) begin
                // Mismatch WITH a fault asserted = brownout/reset in
                // progress (E18), not dead hardware: take the fault path.
                if (fault_count != 8'hFF) fault_count <= fault_count + 8'd1;
                timer <= 32'd0;
                healthy_timer <= 32'd0;
                if (consec_faults >= LOCKOUT_N - 1) state <= S_LOCKOUT;
                else begin
                  consec_faults <= consec_faults + 4'd1;
                  state <= S_FLT_KILL;
                end
              end else if (consec_mismatch >= DRV_DEAD_N - 1) begin
                // E13: never verifies while otherwise healthy = dead.
                state <= S_DEAD;
              end else begin
                consec_mismatch <= consec_mismatch + 4'd1;
                state <= S_CFG_W1;  // rewrite both registers
              end
            end
          end
        end
        S_VER2A: send_frame(frame_read_cr2, S_VER2B);
        S_VER2B: begin
          if (!frame_sent) begin
            if (!spi_busy && !spi_start) begin
              spi_tx <= frame_read_cr2;
              spi_start <= 1'b1;
              frame_sent <= 1'b1;
            end
          end else if (spi_done) begin
            frame_sent <= 1'b0;
            if (spi_rx == {1'b0, 4'h3, CR2_VALUE}) begin
              consec_mismatch <= 4'd0;  // full verify pair succeeded
              state <= S_CLR1;
            end else begin
              if (mismatch_count != 8'hFF)
                mismatch_count <= mismatch_count + 8'd1;
              configured <= 1'b0;
              if (!nfault_sync) begin
                if (fault_count != 8'hFF) fault_count <= fault_count + 8'd1;
                timer <= 32'd0;
                healthy_timer <= 32'd0;
                if (consec_faults >= LOCKOUT_N - 1) state <= S_LOCKOUT;
                else begin
                  consec_faults <= consec_faults + 4'd1;
                  state <= S_FLT_KILL;
                end
              end else if (consec_mismatch >= DRV_DEAD_N - 1) begin
                state <= S_DEAD;
              end else begin
                consec_mismatch <= consec_mismatch + 4'd1;
                state <= S_CFG_W1;
              end
            end
          end
        end
        S_CLR1: send_frame(frame_read_sr1, S_CLR2);  // read clears SR1
        S_CLR2: send_frame(frame_read_sr2, S_RUN);
        S_RUN: begin
          configured <= 1'b1;
          refresh_timer <= refresh_timer + 32'd1;
          // E20: a sustained healthy run clears the repeated-fault counter.
          if (healthy_timer < HEALTHY_RUN_CYC) begin
            healthy_timer <= healthy_timer + 32'd1;
          end else begin
            consec_faults <= 4'd0;
          end
          if (!nfault_sync) begin
            configured <= 1'b0;
            if (fault_count != 8'hFF) fault_count <= fault_count + 8'd1;
            timer <= 32'd0;
            healthy_timer <= 32'd0;
            // E20: too many recoveries without a healthy interval = latch
            // lockout instead of retrying forever.
            if (consec_faults >= LOCKOUT_N - 1) begin
              state <= S_LOCKOUT;
            end else begin
              consec_faults <= consec_faults + 4'd1;
              state <= S_FLT_KILL;
            end
          end else if (!hw_mode && refresh_timer >= DRV_REFRESH_CYC) begin
            refresh_timer <= 32'd0;
            state <= S_VER1A;  // periodic SPI verify (not in hardware mode)
          end
        end
        S_DEAD: begin
          // E13: unrecoverable without host intervention.
          configured <= 1'b0;
          en_gate <= 1'b0;
          drv_dead <= 1'b1;
          if (lockout_clear) begin
            drv_dead <= 1'b0;
            consec_mismatch <= 4'd0;
            timer <= 32'd0;
            state <= S_PWRUP;  // full retry from scratch
          end
        end
        S_LOCKOUT: begin
          configured <= 1'b0;
          en_gate <= 1'b0;
          locked_out <= 1'b1;
          if (lockout_clear) begin
            locked_out <= 1'b0;
            consec_faults <= 4'd0;
            timer <= 32'd0;
            state <= S_PWRUP;
          end
        end
        S_FLT_KILL: begin
          timer <= timer + 32'd1;
          if (timer >= 32'd256) begin  // ~10 us: let gates die first
            timer <= 32'd0;
            en_gate <= 1'b0;
            state <= S_FLT_QR;
          end
        end
        S_FLT_QR: begin
          timer <= timer + 32'd1;
          if (timer >= QUICK_RESET_CYC) begin
            timer <= 32'd0;
            en_gate <= 1'b1;
            state <= S_FLT_WAIT;
          end
        end
        S_FLT_WAIT: begin
          timer <= timer + 32'd1;
          if (timer >= FLT_WAIT_CYC) begin
            timer <= 32'd0;
            // Hardware-configured drivers need no reconfigure: resume directly.
            state <= hw_mode ? S_RUN : S_CFG_W1;
          end
        end
        default: state <= S_PWRUP;
      endcase
    end
  end

endmodule
