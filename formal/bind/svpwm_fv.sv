// SPDX-License-Identifier: MIT
// Formal wrapper for svpwm (combinational): every per-leg duty stays within
// [0, PWM_HALF_PERIOD] for all inputs in the legal range. The legal range is
// an ASSUMPTION (the design feeds svpwm from inv_park of the circle-limited
// voltage, so |valpha|,|vbeta| <= V_RAW_MAX); without it the internal
// fixed-point arithmetic would overflow. The composition proof (stage 9)
// discharges this against the circle limiter + PI saturation.

`include "rtl_params.vh"

module svpwm_fv (
    input wire               clk,
    input wire signed [17:0] valpha,
    input wire signed [17:0] vbeta
);
  wire [47:0] duty3;
  svpwm dut (.valpha(valpha), .vbeta(vbeta), .duty3(duty3));

  wire [15:0] da = duty3[15:0];
  wire [15:0] db = duty3[31:16];
  wire [15:0] dc = duty3[47:32];

  always @(posedge clk) begin
    // Assumption: inputs are within the modulation range.
    assume (valpha >= -18'sd2500 && valpha <= 18'sd2500);
    assume (vbeta  >= -18'sd2500 && vbeta  <= 18'sd2500);

    // P_DUTY_BOUNDS: every leg duty is a legal compare in [0, HALF].
    assert (da <= `PWM_HALF_PERIOD);
    assert (db <= `PWM_HALF_PERIOD);
    assert (dc <= `PWM_HALF_PERIOD);

    // Non-vacuity: a non-centered (actively modulating) duty is reachable.
    cover (da != (`PWM_HALF_PERIOD >> 1));
    cover (da == 16'd0);
    cover (da == `PWM_HALF_PERIOD);
  end
endmodule
