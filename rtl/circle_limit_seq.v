// SPDX-License-Identifier: MIT
// Sequential voltage-circle limiter (robotics-ip-checklist stage 6.5): the
// multi-cycle, bit-exact equivalent of circle_limit.v. The combinational
// limiter's 16-iteration isqrt and two 32-bit divisions are the FOC datapath's
// longest arc (post-route critical path); doing them one-op-per-clock lets the
// design close timing at speed (the unpipelined foc_core capped at ~3.3 MHz).
//
// Bit-exact to circle_limit.v for EVERY input, so foc_core's behaviour is
// unchanged (only later in time): same mag2, the same floor-isqrt recurrence
// (one iteration/clock), and the same truncate-toward-zero rescale. The signed
// divisions vd*VLIM/|v| reuse the unsigned `divider32`: |v| > 0 always, so the
// quotient's sign is the numerator's; we divide magnitudes and reapply the
// sign, which equals Verilog signed division (truncation toward zero).
//
//   start -> busy -> done (1-cycle pulse); outputs hold until the next start.
//   |v| = floor(sqrt(vd^2 + vq^2)); sat = (vd^2+vq^2) > VLIM^2.
//   vd_out = sat ? trunc(vd*VLIM / |v|) : vd   (likewise vq).
//
// Proven equivalent to circle_limit.v in formal/ (bounded miter).

module circle_limit_seq #(
    parameter integer V_CIRCLE_LIMIT = 594
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start,
    input  wire signed [17:0] vd_in,
    input  wire signed [17:0] vq_in,
    output reg  signed [17:0] vd_out,
    output reg  signed [17:0] vq_out,
    output reg                sat,
    output reg                busy,
    output reg                done
);
  localparam [31:0] VLIM  = V_CIRCLE_LIMIT;
  localparam [31:0] VLIM2 = V_CIRCLE_LIMIT * V_CIRCLE_LIMIT;

  localparam [2:0] S_IDLE  = 3'd0,
                   S_MUL   = 3'd1,   // register mag2 = vd^2 + vq^2 (hard mults)
                   S_CMP   = 3'd2,   // sat compare + isqrt init
                   S_SQRT  = 3'd3,   // 16-iteration floor isqrt, one per clock
                   S_DIV   = 3'd4,   // launch the two divisions
                   S_WAIT  = 3'd5;   // await both quotients, reapply sign
  reg [2:0] state;

  // Latched inputs (the 18-bit values; mag2 and the division operands use them
  // directly so the multiplies map to single hard multipliers).
  reg signed [17:0] vd_l, vq_l;

  // isqrt recurrence state (rem, x, b) - identical to circle_limit's function.
  reg [31:0] rem, xacc, bbit;
  reg [4:0]  iter;                  // counts 0..16
  wire [31:0] xb = xacc + bbit;
  wire        take = (rem >= xb);

  // mag2 = vd^2 + vq^2, low 32b - bit-exact to circle_limit's d*d+q*q, but the
  // operands are the 18-bit clamped inputs, so each square maps to ONE hard
  // multiplier (18x18) instead of a 32x32 - this was the limiter's longest arc.
  wire signed [63:0] dq2  = vd_l * vd_l + vq_l * vq_l;
  wire [31:0]        mag2 = dq2[31:0];
  reg  [31:0]        mag2_r;

  // division operands: |vd|*VLIM, |vq|*VLIM from the 18-bit magnitudes - the
  // narrow variable operand (vs the old 32-bit dabs) lets the constant multiply
  // fold to shift-adds over 18 bits, off the critical path. Bit-exact to
  // circle_limit's |d|*VLIM (|v| <= V_RAW_MAX = 2500).
  reg [31:0] safe_mag;
  reg [31:0] dnum, qnum;
  reg        dsign, qsign;
  reg        div_start;

  wire [17:0] vd_abs = vd_l[17] ? (~vd_l + 18'd1) : vd_l;
  wire [17:0] vq_abs = vq_l[17] ? (~vq_l + 18'd1) : vq_l;
  wire [63:0] dnum_full = vd_abs * VLIM;
  wire [63:0] qnum_full = vq_abs * VLIM;

  wire        div_busy_d, div_done_d, div_busy_q, div_done_q;
  wire [31:0] quot_d, quot_q;
  divider32 u_div_d (
      .clk(clk), .rst_n(rst_n), .start(div_start),
      .numerator(dnum), .denominator(safe_mag),
      .busy(div_busy_d), .done(div_done_d), .quotient(quot_d));
  divider32 u_div_q (
      .clk(clk), .rst_n(rst_n), .start(div_start),
      .numerator(qnum), .denominator(safe_mag),
      .busy(div_busy_q), .done(div_done_q), .quotient(quot_q));

  // Signed quotient (truncate toward zero), sliced to 18b like circle_limit.
  wire signed [31:0] vds = dsign ? -$signed(quot_d) : $signed(quot_d);
  wire signed [31:0] vqs = qsign ? -$signed(quot_q) : $signed(quot_q);

  // The dividers complete in lockstep (started together); we gate on `done`, so
  // their `busy` lines are unused. Sink them so lint stays honest.
  wire _unused_ok = &{1'b0, div_busy_d, div_busy_q};

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE;
      busy  <= 1'b0; done <= 1'b0; sat <= 1'b0;
      vd_out <= 18'sd0; vq_out <= 18'sd0;
      div_start <= 1'b0;
    end else begin
      done <= 1'b0;
      div_start <= 1'b0;
      case (state)
        S_IDLE:
          if (start) begin
            vd_l <= vd_in; vq_l <= vq_in;
            busy  <= 1'b1;
            state <= S_MUL;
          end
        S_MUL: begin
          mag2_r <= mag2;            // the squares + add (18x18 hard multipliers)
          state  <= S_CMP;
        end
        S_CMP: begin
          rem  <= mag2_r;
          xacc <= 32'd0;
          bbit <= 32'h4000_0000;
          iter <= 5'd0;
          if (mag2_r > VLIM2) begin
            sat   <= 1'b1;
            state <= S_SQRT;
          end else begin
            // not saturated: pass through (bit-exact with circle_limit)
            sat    <= 1'b0;
            vd_out <= vd_l;
            vq_out <= vq_l;
            busy   <= 1'b0;
            done   <= 1'b1;
            state  <= S_IDLE;
          end
        end
        S_SQRT: begin
          if (take) begin
            rem  <= rem - xb;
            xacc <= (xacc >> 1) + bbit;
          end else begin
            xacc <= xacc >> 1;
          end
          bbit <= bbit >> 2;
          iter <= iter + 5'd1;
          if (iter == 5'd15) state <= S_DIV;
        end
        S_DIV: begin
          // mag = xacc[15:0]; safe_mag = (mag==0)?1:mag.
          safe_mag <= (xacc[15:0] == 16'd0) ? 32'd1 : {16'd0, xacc[15:0]};
          dnum  <= dnum_full[31:0];
          qnum  <= qnum_full[31:0];
          dsign <= vd_l[17];
          qsign <= vq_l[17];
          div_start <= 1'b1;        // one-cycle launch (both dividers)
          state <= S_WAIT;
        end
        S_WAIT:
          if (div_done_d && div_done_q) begin
            vd_out <= vds[17:0];
            vq_out <= vqs[17:0];
            busy   <= 1'b0;
            done   <= 1'b1;
            state  <= S_IDLE;
          end
        default: state <= S_IDLE;
      endcase
    end
  end
endmodule
