// SPDX-License-Identifier: MIT
// AXI4-Lite register file for the motorloop controller (robotics-ip-checklist
// stage 5). Memory-maps the same register set as uart_regfile so a robotics SoC
// can drive the controller over AXI-Lite instead of UART. Word-addressed; one
// outstanding transaction (classic AXI-Lite slave). The AXI handshake legality
// is formally proven (formal/bind/axil_regfile_fv.sv).
//
// Register map (byte address = index * 4):
//   W/R 0 mode[1:0]  1 duty  2 target_speed  3 align[11:0]
//       4/5 ol_freq_word hi/lo   6/7 ol_ramp_inc hi/lo   8 control[0]=use_axi
//   R   16 speed  17 {fault,mismatch}  18 angle  19 noctw  20 {cfg,sector}
//       21 flags

module axil_regfile #(
    parameter integer ADDR_W = 8     // byte-address width (>= 7 for index 0..21)
) (
    input  wire              clk,
    input  wire              rst_n,
    // AXI4-Lite slave
    input  wire [ADDR_W-1:0] s_awaddr,
    input  wire              s_awvalid,
    output reg               s_awready,
    input  wire [31:0]       s_wdata,
    input  wire [3:0]        s_wstrb,
    input  wire              s_wvalid,
    output reg               s_wready,
    output reg  [1:0]        s_bresp,
    output reg               s_bvalid,
    input  wire              s_bready,
    input  wire [ADDR_W-1:0] s_araddr,
    input  wire              s_arvalid,
    output reg               s_arready,
    output reg  [31:0]       s_rdata,
    output reg  [1:0]        s_rresp,
    output reg               s_rvalid,
    input  wire              s_rready,
    // Control outputs (same surface as uart_regfile)
    output reg               use_axi,
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

  localparam [1:0] OKAY = 2'b00;

  // This is a word-only register slave: the AXI byte offset, write strobes, and
  // the upper 16 data bits are intentionally unused (a common, documented
  // simplification). Sink them so lint stays clean.
  wire _unused_ok = &{1'b0, s_wstrb, s_awaddr[1:0], s_araddr[1:0],
                      s_wdata[31:16], 1'b0};

  // ---- Write channel: accept (awaddr, wdata) together, then respond. --------
  wire        wr_fire = s_awvalid && s_wvalid && !s_bvalid
                        && !(s_awready && s_wready);
  wire [ADDR_W-3:0] waddr_idx = s_awaddr[ADDR_W-1:2];

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s_awready <= 1'b0; s_wready <= 1'b0; s_bvalid <= 1'b0; s_bresp <= OKAY;
      use_axi <= 1'b0; r_mode <= 2'd0; r_duty <= 16'd0; r_target_speed <= 16'd0;
      r_align <= 12'd0; r_ol_freq_word <= 32'd0; r_ol_ramp_inc <= 32'd0;
    end else begin
      s_awready <= 1'b0;
      s_wready  <= 1'b0;
      if (wr_fire) begin
        s_awready <= 1'b1;
        s_wready  <= 1'b1;
        s_bvalid  <= 1'b1;
        s_bresp   <= OKAY;
        case (waddr_idx)
          0:  r_mode         <= s_wdata[1:0];
          1:  r_duty         <= s_wdata[15:0];
          2:  r_target_speed <= s_wdata[15:0];
          3:  r_align        <= s_wdata[11:0];
          4:  r_ol_freq_word[31:16] <= s_wdata[15:0];
          5:  r_ol_freq_word[15:0]  <= s_wdata[15:0];
          6:  r_ol_ramp_inc[31:16]  <= s_wdata[15:0];
          7:  r_ol_ramp_inc[15:0]   <= s_wdata[15:0];
          8:  use_axi        <= s_wdata[0];
          default: ;  // ignore writes to RO / undefined regs (resp still OKAY)
        endcase
      end else if (s_bvalid && s_bready) begin
        s_bvalid <= 1'b0;
      end
    end
  end

  // ---- Read channel: accept araddr, then drive rdata/rvalid. ----------------
  wire        rd_fire = s_arvalid && !s_rvalid && !s_arready;
  wire [ADDR_W-3:0] raddr_idx = s_araddr[ADDR_W-1:2];

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s_arready <= 1'b0; s_rvalid <= 1'b0; s_rresp <= OKAY; s_rdata <= 32'd0;
    end else begin
      s_arready <= 1'b0;
      if (rd_fire) begin
        s_arready <= 1'b1;
        s_rvalid  <= 1'b1;
        s_rresp   <= OKAY;
        case (raddr_idx)
          0:  s_rdata <= {30'd0, r_mode};
          1:  s_rdata <= {16'd0, r_duty};
          2:  s_rdata <= {16'd0, r_target_speed};
          3:  s_rdata <= {20'd0, r_align};
          4:  s_rdata <= {16'd0, r_ol_freq_word[31:16]};
          5:  s_rdata <= {16'd0, r_ol_freq_word[15:0]};
          6:  s_rdata <= {16'd0, r_ol_ramp_inc[31:16]};
          7:  s_rdata <= {16'd0, r_ol_ramp_inc[15:0]};
          8:  s_rdata <= {31'd0, use_axi};
          16: s_rdata <= {16'd0, t_speed};
          17: s_rdata <= {16'd0, t_fault_count, t_mismatch_count};
          18: s_rdata <= {20'd0, t_angle};
          19: s_rdata <= {16'd0, t_noctw_count};
          20: s_rdata <= {28'd0, t_configured, t_sector};
          21: s_rdata <= {24'd0, t_flags};
          default: s_rdata <= 32'hDEADBEEF;
        endcase
      end else if (s_rvalid && s_rready) begin
        s_rvalid <= 1'b0;
      end
    end
  end

endmodule
