// SPDX-License-Identifier: MIT
// Space-vector PWM by min/max common-mode injection. Given the stationary
// frame voltage (valpha, vbeta) in duty units (signed, +-PWM_HALF_PERIOD for
// full half-bus phase drive), produce three per-leg duty compares in
// [0, HALF] centered at HALF/2.
//
// Identical line-to-line output to sector-dwell SVPWM, with no sector-detect
// state: compute the three phase references, subtract the common-mode
// (max+min)/2, and bias around the 50%-duty zero-voltage point. The injection
// buys ~15% more linear modulation range than pure sine PWM while keeping
// every leg inside [0, HALF].
//
// Combinational. Arithmetic mirrors sim/scripts/foc_reference.svpwm_fx
// exactly (see notes/foc-fixed-point.md); test_foc_math.py enforces parity.

module svpwm #(
    parameter integer PWM_HALF_PERIOD = 625   // half PWM period in clocks
) (
    input  wire signed [17:0] valpha,
    input  wire signed [17:0] vbeta,
    output wire [47:0]        duty3     // {leg C, leg B, leg A}, 16 bits each
);

  localparam signed [31:0] SQRT3_OVER_2 = 32'sd28378;  // round(sqrt(3)/2 * 2^15)
  localparam signed [31:0] HALF   = PWM_HALF_PERIOD;
  localparam signed [31:0] CENTER = PWM_HALF_PERIOD >>> 1;  // 50% duty

  wire signed [31:0] va = {{14{valpha[17]}}, valpha};
  wire signed [31:0] vb = {{14{vbeta[17]}}, vbeta};

  wire signed [31:0] half_alpha = va >>> 1;
  wire signed [31:0] s3b        = (SQRT3_OVER_2 * vb) >>> 15;  // (sqrt3/2)*vbeta

  wire signed [31:0] ra = va;
  wire signed [31:0] rb = -half_alpha + s3b;
  wire signed [31:0] rc = -half_alpha - s3b;

  wire signed [31:0] mx = (ra >= rb) ? ((ra >= rc) ? ra : rc)
                                     : ((rb >= rc) ? rb : rc);
  wire signed [31:0] mn = (ra <= rb) ? ((ra <= rc) ? ra : rc)
                                     : ((rb <= rc) ? rb : rc);
  wire signed [31:0] cm = (mx + mn) >>> 1;

  function [15:0] sat_duty(input signed [31:0] x);
    begin
      if (x < 0) sat_duty = 16'd0;
      else if (x > HALF) sat_duty = HALF[15:0];
      else sat_duty = x[15:0];
    end
  endfunction

  wire [15:0] da = sat_duty(CENTER + (ra - cm));
  wire [15:0] db = sat_duty(CENTER + (rb - cm));
  wire [15:0] dc = sat_duty(CENTER + (rc - cm));

  assign duty3 = {dc, db, da};

endmodule
