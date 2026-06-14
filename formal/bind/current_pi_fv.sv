// Formal checker for current_pi: the PI output never leaves its declared
// saturation clamp [-V_RAW_MAX, V_RAW_MAX], for any error / integrator state.
// This is what lets the downstream circle_limit assume a bounded input (the
// assumption discharged by stage-9 composition).

`include "rtl_params.vh"

module current_pi_fv (
    input wire               clk,
    input wire               rst_n,
    input wire signed [17:0] v_out
);
  localparam signed [17:0] VMAX = `V_RAW_MAX;
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

bind current_pi current_pi_fv fv_i (.clk(clk), .rst_n(rst_n), .v_out(v_out));
