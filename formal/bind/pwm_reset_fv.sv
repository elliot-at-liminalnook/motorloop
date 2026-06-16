// SPDX-License-Identifier: MIT
// Reset / initialization safety for pwm_generator (formal-checklist stage 2).
// During reset and on the first cycle after reset deasserts, every gate is
// off - the power stage can never glitch into an illegal drive out of reset.

module pwm_reset_fv (
    input wire       clk,
    input wire       rst_n,
    input wire [2:0] gate_high,
    input wire [2:0] gate_low
);
  // During reset: all gates off (async reset holds the gate FFs at 0).
  always @(posedge clk) begin
    if (!rst_n) assert (gate_high == 3'd0 && gate_low == 3'd0);
  end
  // Immediately after reset deasserts: still off (the FFs were reset to 0).
  always @(posedge clk) begin
    if (rst_n && !$past(rst_n))
      assert (gate_high == 3'd0 && gate_low == 3'd0);
  end
  // Non-vacuity: reset actually occurs from a running state, then releases.
  always @(posedge clk) begin
    cover (!rst_n);
    cover (rst_n && !$past(rst_n));
  end
endmodule

bind pwm_generator pwm_reset_fv rst_i (
    .clk(clk), .rst_n(rst_n), .gate_high(gate_high), .gate_low(gate_low));
