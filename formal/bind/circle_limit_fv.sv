// SPDX-License-Identifier: MIT
// Formal wrapper for circle_limit (combinational): the limited voltage vector
// magnitude never exceeds the inscribed-circle radius (plus a small slack from
// the integer isqrt/divide). Input range is an ASSUMPTION - the design clamps
// each PI output to V_RAW_MAX before the limiter, and circle_limit's internal
// mag2 = vd^2+vq^2 is a 32-bit wire that would overflow beyond that. Stage-9
// composition discharges the assumption against current_pi saturation.

`include "rtl_params.vh"

module circle_limit_fv (
    input wire               clk,
    input wire signed [17:0] vd_in,
    input wire signed [17:0] vq_in
);
  wire signed [17:0] vd_out, vq_out;
  wire               sat;
  circle_limit dut (.vd_in(vd_in), .vq_in(vq_in),
                    .vd_out(vd_out), .vq_out(vq_out), .sat(sat));

  wire signed [31:0] d = {{14{vd_out[17]}}, vd_out};
  wire signed [31:0] q = {{14{vq_out[17]}}, vq_out};
  wire [63:0] out_mag2 = $unsigned(d * d) + $unsigned(q * q);
  // Allow a few LSB of slack: the rescale uses isqrt (floor) then an integer
  // divide, so the limited magnitude can sit a little above VLIM.
  localparam [63:0] BOUND =
      (`V_CIRCLE_LIMIT + 64) * (`V_CIRCLE_LIMIT + 64);

  always @(posedge clk) begin
    // Assumption: each axis is within the PI raw-output clamp.
    assume (vd_in >= -18'sd2500 && vd_in <= 18'sd2500);
    assume (vq_in >= -18'sd2500 && vq_in <= 18'sd2500);

    // P_CIRCLE_BOUND: the limited vector lies within the modulation circle.
    assert (out_mag2 <= BOUND);

    // Non-vacuity: the limiter actually engages, and also passes through.
    cover (sat);
    cover (!sat);
  end
endmodule
