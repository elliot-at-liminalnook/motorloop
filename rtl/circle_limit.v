// SPDX-License-Identifier: MIT
// Voltage-circle limiter: cap the (vd, vq) vector magnitude to the SVPWM
// inscribed-circle radius (V_CIRCLE_LIMIT, duty units), prioritizing nothing
// (uniform scale) - field-weakening priority on vd is handled by the caller
// choosing id_target. Emits `sat` so the current PIs freeze their integrators
// while clamped (anti-windup).
//
// Magnitude via an integer square root (combinational, bounded loop), then a
// uniform rescale: vd_out = vd*VLIM/|v|. The per-axis inputs are pre-clamped
// to V_RAW_MAX by current_pi.v so vd^2+vq^2 fits 32 bits.

module circle_limit #(
    parameter integer V_CIRCLE_LIMIT = 594   // SVPWM inscribed-circle radius (duty units)
) (
    input  wire signed [17:0] vd_in,
    input  wire signed [17:0] vq_in,
    output wire signed [17:0] vd_out,
    output wire signed [17:0] vq_out,
    output wire               sat
);
  localparam signed [31:0] VLIM = V_CIRCLE_LIMIT;

  wire signed [31:0] d = {{14{vd_in[17]}}, vd_in};
  wire signed [31:0] q = {{14{vq_in[17]}}, vq_in};
  wire [31:0] mag2 = d * d + q * q;       // non-negative

  assign sat = mag2 > (VLIM * VLIM);

  // Integer sqrt (floor) of a 32-bit value; fixed 16-iteration bit-by-bit
  // algorithm (synthesizable and formal-friendly - no data-dependent loop
  // bounds). Equivalent to floor(sqrt(n)) for all 32-bit n.
  function [15:0] isqrt(input [31:0] n);
    reg [31:0] rem, x, b;
    integer i;
    begin
      rem = n;
      x = 0;
      b = 32'h40000000;   // 2^30, the largest power of 4 in 32 bits
      for (i = 0; i < 16; i = i + 1) begin
        if (rem >= x + b) begin
          rem = rem - (x + b);
          x = (x >> 1) + b;
        end else begin
          x = x >> 1;
        end
        b = b >> 2;
      end
      isqrt = x[15:0];
    end
  endfunction

  wire [15:0] mag = isqrt(mag2);
  wire [31:0] safe_mag = (mag == 16'd0) ? 32'd1 : {16'd0, mag};

  wire signed [31:0] vds = (d * VLIM) / $signed(safe_mag);
  wire signed [31:0] vqs = (q * VLIM) / $signed(safe_mag);

  assign vd_out = sat ? vds[17:0] : vd_in;
  assign vq_out = sat ? vqs[17:0] : vq_in;
endmodule
