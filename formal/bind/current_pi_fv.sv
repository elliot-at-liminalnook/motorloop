// SPDX-License-Identifier: MIT
// Formal checker for current_pi: the PI output never leaves its declared
// saturation clamp [-V_RAW_MAX, V_RAW_MAX], for any error / integrator state.
// This is what lets the downstream circle_limit assume a bounded input (the
// assumption discharged by stage-9 composition).
//
// PARAMETER-GENERIC (envelope): the bound is the module's own V_RAW_MAX
// parameter, passed through the bind - not the motorloop global config. The
// design clamps to V_RAW_MAX and the checker asserts |v_out| <= V_RAW_MAX with
// the SAME parameter, so the property holds for ANY V_RAW_MAX a reuser picks
// (trusted-library-checklist stage 1.6).

module current_pi_fv #(
    parameter signed [31:0] V_RAW_MAX = 2500
) (
    input wire               clk,
    input wire               rst_n,
    input wire signed [17:0] v_out
);
  localparam signed [17:0] VMAX = V_RAW_MAX[17:0];
  always @(posedge clk) begin
    if (rst_n) begin
      assert (v_out >= -VMAX);
      assert (v_out <=  VMAX);
    end
  end
  // Non-vacuity: the clamp is actually reachable (the PI can demand max drive).
  always @(posedge clk) begin
    if (rst_n) begin
      cover (v_out ==  VMAX);
      cover (v_out == -VMAX);
    end
  end
endmodule

bind current_pi current_pi_fv #(.V_RAW_MAX(V_RAW_MAX)) fv_i (
    .clk(clk), .rst_n(rst_n), .v_out(v_out));
