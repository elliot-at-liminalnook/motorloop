// Formal checker for speed_iq_pi: the outer-loop torque command iq_target
// never leaves its clamp [-IQ_MAX, IQ_MAX], for any speed error / integrator
// state - the FOC current command is bounded by construction.

`include "rtl_params.vh"

module speed_iq_pi_fv (
    input wire               clk,
    input wire               rst_n,
    input wire signed [17:0] iq_target
);
  localparam signed [17:0] IQMAX = `IQ_MAX;
  always @(posedge clk) begin
    if (rst_n) begin
      assert (iq_target >= -IQMAX);
      assert (iq_target <=  IQMAX);
    end
  end
  always @(posedge clk) begin
    if (rst_n) begin
      cover (iq_target ==  IQMAX);
      cover (iq_target == -IQMAX);
    end
  end
endmodule

bind speed_iq_pi speed_iq_pi_fv fv_i (
    .clk(clk), .rst_n(rst_n), .iq_target(iq_target));
