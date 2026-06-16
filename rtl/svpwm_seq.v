// SPDX-License-Identifier: MIT
// Sequential SVPWM (FOC Fmax-optimization checklist, task 4): the multi-cycle,
// bit-exact equivalent of svpwm.v. The combinational min/max common-mode
// injection + per-leg clamp was foc_core's longest arc (post-route critical
// path); doing it over registered stages shortens it so foc_core closes timing
// well above ~42 MHz.
//
// Bit-exact to svpwm.v for every input: identical arithmetic (the (sqrt3/2)*vb
// product, the ra/rb/rc references, the 3-way min/max common-mode, and the
// per-leg sat_duty), just split across stages:
//   S1: products + references (ra, rb, rc)
//   S2: min/max -> common-mode cm = (max+min)>>1
//   S3: per-leg clamp -> duty3
// start -> ... -> done (1-cycle pulse); duty3 holds until the next start.
// Verified equivalent to svpwm.v by the cocotb test tb_svpwm_seq.

module svpwm_seq #(
    parameter integer PWM_HALF_PERIOD = 625
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start,
    input  wire signed [17:0] valpha,
    input  wire signed [17:0] vbeta,
    output reg  [47:0]        duty3,
    output reg                busy,
    output reg                done
);
  localparam signed [31:0] SQRT3_OVER_2 = 32'sd28378;  // round(sqrt(3)/2 * 2^15)
  localparam signed [31:0] HALF   = PWM_HALF_PERIOD;
  localparam signed [31:0] CENTER = PWM_HALF_PERIOD >>> 1;

  localparam [1:0] S_IDLE = 2'd0, S_REF = 2'd1, S_CM = 2'd2, S_CLAMP = 2'd3;
  reg [1:0] state;

  reg signed [17:0] va_l, vb_l;                 // latched inputs
  reg signed [31:0] ra_r, rb_r, rc_r, cm_r;     // pipeline registers

  // Stage 1 (S_REF): references from the latched inputs.
  wire signed [31:0] va = {{14{va_l[17]}}, va_l};
  wire signed [31:0] vb = {{14{vb_l[17]}}, vb_l};
  wire signed [31:0] half_alpha = va >>> 1;
  wire signed [31:0] s3b        = (SQRT3_OVER_2 * vb) >>> 15;
  wire signed [31:0] ra_c = va;
  wire signed [31:0] rb_c = -half_alpha + s3b;
  wire signed [31:0] rc_c = -half_alpha - s3b;

  // Stage 2 (S_CM): common-mode from the registered references.
  wire signed [31:0] mx = (ra_r >= rb_r) ? ((ra_r >= rc_r) ? ra_r : rc_r)
                                         : ((rb_r >= rc_r) ? rb_r : rc_r);
  wire signed [31:0] mn = (ra_r <= rb_r) ? ((ra_r <= rc_r) ? ra_r : rc_r)
                                         : ((rb_r <= rc_r) ? rb_r : rc_r);
  wire signed [31:0] cm_c = (mx + mn) >>> 1;

  // Stage 3 (S_CLAMP): per-leg duty (same sat as svpwm.v).
  function [15:0] sat_duty(input signed [31:0] x);
    begin
      if (x < 0) sat_duty = 16'd0;
      else if (x > HALF) sat_duty = HALF[15:0];
      else sat_duty = x[15:0];
    end
  endfunction

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE; busy <= 1'b0; done <= 1'b0; duty3 <= 48'd0;
    end else begin
      done <= 1'b0;
      case (state)
        S_IDLE:
          if (start) begin
            va_l <= valpha; vb_l <= vbeta;
            busy <= 1'b1;
            state <= S_REF;
          end
        S_REF: begin
          ra_r <= ra_c; rb_r <= rb_c; rc_r <= rc_c;
          state <= S_CM;
        end
        S_CM: begin
          cm_r <= cm_c;
          state <= S_CLAMP;
        end
        S_CLAMP: begin
          duty3 <= {sat_duty(CENTER + (rc_r - cm_r)),
                    sat_duty(CENTER + (rb_r - cm_r)),
                    sat_duty(CENTER + (ra_r - cm_r))};
          busy <= 1'b0;
          done <= 1'b1;
          state <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end
endmodule
