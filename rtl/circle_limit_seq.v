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
                   S_MAG   = 3'd1,   // mag2 + sat decision
                   S_SQRT  = 3'd2,   // 16-iteration floor isqrt, one per clock
                   S_DIV   = 3'd3,   // launch the two divisions
                   S_WAIT  = 3'd4;   // await both quotients, reapply sign
  reg [2:0] state;

  // Latched inputs (sign-extended to 32b, matching circle_limit.v).
  reg signed [17:0] vd_l, vq_l;
  reg signed [31:0] d, q;

  // isqrt recurrence state (rem, x, b) - identical to circle_limit's function.
  reg [31:0] rem, xacc, bbit;
  reg [4:0]  iter;                  // counts 0..16
  wire [31:0] xb = xacc + bbit;
  wire        take = (rem >= xb);

  // mag2 = d*d + q*q, low 32b (inputs are clamped so it fits) - as circle_limit.
  wire signed [63:0] dq2  = d * d + q * q;
  wire [31:0]        mag2 = dq2[31:0];

  // division operands. |d|*VLIM, |q|*VLIM fit 32b (|v| <= V_RAW_MAX = 2500).
  reg [31:0] safe_mag;
  reg [31:0] dnum, qnum;
  reg        dsign, qsign;
  reg        div_start;

  wire [31:0] dabs = d[31] ? (~d + 32'd1) : d;
  wire [31:0] qabs = q[31] ? (~q + 32'd1) : q;
  wire [63:0] dnum_full = dabs * VLIM;
  wire [63:0] qnum_full = qabs * VLIM;

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
            d <= {{14{vd_in[17]}}, vd_in};
            q <= {{14{vq_in[17]}}, vq_in};
            busy  <= 1'b1;
            state <= S_MAG;
          end
        S_MAG: begin
          rem  <= mag2;
          xacc <= 32'd0;
          bbit <= 32'h4000_0000;
          iter <= 5'd0;
          if (mag2 > VLIM2) begin
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
          dsign <= d[31];
          qsign <= q[31];
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
