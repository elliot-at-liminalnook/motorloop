// SPDX-License-Identifier: MIT
// Reusable formal checker: no complementary gate pair is ever both asserted.
// Bind to any N-leg half-bridge driver. Parameterized by leg count.
//
//   bind <driver> no_shoot_through #(.N(3)) nst_i (
//       .clk(clk), .rst_n(rst_n), .gate_high(gate_high), .gate_low(gate_low));
//
// Assumptions: rst_n is the design reset (the property is gated on rst_n).
// Bundled non-vacuity cover: each high-side gate can actually enable.

module no_shoot_through #(
    parameter integer N = 3
) (
    input wire         clk,
    input wire         rst_n,
    input wire [N-1:0] gate_high,
    input wire [N-1:0] gate_low
);
  genvar k;
  generate
    for (k = 0; k < N; k = k + 1) begin : leg
      always @(posedge clk) begin
        if (rst_n) assert (!(gate_high[k] && gate_low[k]));
      end
      always @(posedge clk) begin
        if (rst_n) cover (gate_high[k]);   // non-vacuity: the leg drives
      end
    end
  endgenerate
endmodule
