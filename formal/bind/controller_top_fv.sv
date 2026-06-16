// SPDX-License-Identifier: MIT
// Top-level composition (formal-checklist stage 9): the integrated controller
// never commands shoot-through at its DRV-gate outputs (inh/inl), across every
// mode and the FOC/six-step duty/leg-mode muxing. This confirms the
// per-module pwm_generator guarantee survives wiring - nothing downstream
// bypasses or inverts it.

module controller_top_fv (
    input wire       clk,
    input wire       rst_n,
    input wire [2:0] inh,
    input wire [2:0] inl
);
  always @(posedge clk) begin
    if (rst_n) begin
      assert (!(inh[0] && inl[0]));
      assert (!(inh[1] && inl[1]));
      assert (!(inh[2] && inl[2]));
    end
  end
  // Non-vacuity: the gates actually drive at the top level.
  always @(posedge clk) begin
    if (rst_n) cover (inh != 3'd0);
  end
endmodule

bind controller_top controller_top_fv comp_i (
    .clk(clk), .rst_n(rst_n), .inh(inh), .inl(inl));
