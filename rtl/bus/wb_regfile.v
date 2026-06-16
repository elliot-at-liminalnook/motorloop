// SPDX-License-Identifier: MIT
// Wishbone B4 (classic, registered-ack) register file for the motorloop
// controller (robotics-ip-checklist stage 5). The open-SoC default bus
// (LiteX / RISC-V robotics stacks). Same register map as axil_regfile /
// uart_regfile. Handshake legality is formally proven (wb_regfile_fv).

module wb_regfile #(
    parameter integer ADDR_W = 8     // word-address width
) (
    input  wire              clk,
    input  wire              rst_n,
    // Wishbone B4 slave (classic)
    input  wire [ADDR_W-1:0] wb_adr,    // word address (register index)
    input  wire [31:0]       wb_dat_w,
    output reg  [31:0]       wb_dat_r,
    input  wire              wb_we,
    input  wire              wb_stb,
    input  wire              wb_cyc,
    output reg               wb_ack,
    // Control outputs (same surface as axil_regfile)
    output reg               use_wb,
    output reg  [1:0]        r_mode,
    output reg  [15:0]       r_duty,
    output reg  [15:0]       r_target_speed,
    output reg  [11:0]       r_align,
    output reg  [31:0]       r_ol_freq_word,
    output reg  [31:0]       r_ol_ramp_inc,
    // Telemetry inputs
    input  wire [15:0]       t_speed,
    input  wire [7:0]        t_fault_count,
    input  wire [7:0]        t_mismatch_count,
    input  wire [11:0]       t_angle,
    input  wire [15:0]       t_noctw_count,
    input  wire [2:0]        t_sector,
    input  wire              t_configured,
    input  wire [7:0]        t_flags
);

  // A strobe is accepted when cyc & stb and no ack is currently outstanding.
  wire access = wb_cyc && wb_stb && !wb_ack;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wb_ack <= 1'b0; wb_dat_r <= 32'd0;
      use_wb <= 1'b0; r_mode <= 2'd0; r_duty <= 16'd0; r_target_speed <= 16'd0;
      r_align <= 12'd0; r_ol_freq_word <= 32'd0; r_ol_ramp_inc <= 32'd0;
    end else begin
      wb_ack <= 1'b0;                 // single-cycle ack pulse
      if (access) begin
        wb_ack <= 1'b1;
        if (wb_we) begin
          case (wb_adr)
            0:  r_mode         <= wb_dat_w[1:0];
            1:  r_duty         <= wb_dat_w[15:0];
            2:  r_target_speed <= wb_dat_w[15:0];
            3:  r_align        <= wb_dat_w[11:0];
            4:  r_ol_freq_word[31:16] <= wb_dat_w[15:0];
            5:  r_ol_freq_word[15:0]  <= wb_dat_w[15:0];
            6:  r_ol_ramp_inc[31:16]  <= wb_dat_w[15:0];
            7:  r_ol_ramp_inc[15:0]   <= wb_dat_w[15:0];
            8:  use_wb         <= wb_dat_w[0];
            default: ;
          endcase
        end else begin
          case (wb_adr)
            0:  wb_dat_r <= {30'd0, r_mode};
            1:  wb_dat_r <= {16'd0, r_duty};
            2:  wb_dat_r <= {16'd0, r_target_speed};
            3:  wb_dat_r <= {20'd0, r_align};
            4:  wb_dat_r <= {16'd0, r_ol_freq_word[31:16]};
            5:  wb_dat_r <= {16'd0, r_ol_freq_word[15:0]};
            6:  wb_dat_r <= {16'd0, r_ol_ramp_inc[31:16]};
            7:  wb_dat_r <= {16'd0, r_ol_ramp_inc[15:0]};
            8:  wb_dat_r <= {31'd0, use_wb};
            16: wb_dat_r <= {16'd0, t_speed};
            17: wb_dat_r <= {16'd0, t_fault_count, t_mismatch_count};
            18: wb_dat_r <= {20'd0, t_angle};
            19: wb_dat_r <= {16'd0, t_noctw_count};
            20: wb_dat_r <= {28'd0, t_configured, t_sector};
            21: wb_dat_r <= {24'd0, t_flags};
            default: wb_dat_r <= 32'hDEADBEEF;
          endcase
        end
      end
    end
  end

  // Upper address bits + upper write-data bits (regs are <= 16-bit) are unused.
  wire _unused_ok = &{1'b0, wb_adr[ADDR_W-1:5], wb_dat_w[31:16], 1'b0};

endmodule
