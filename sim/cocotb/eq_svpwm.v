// SPDX-License-Identifier: MIT
// Equivalence harness (FOC Fmax-optimization checklist, task 4): the
// combinational `svpwm` reference and the sequential `svpwm_seq` side by side,
// fed identical inputs, so a cocotb test can assert they agree bit-for-bit when
// the sequential block finishes. This is svpwm_seq's correctness claim - it
// justifies foc_core's behaviour being unchanged by the pipelining.
module eq_svpwm #(
    parameter integer PWM_HALF_PERIOD = 625
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start,
    input  wire signed [17:0] valpha,
    input  wire signed [17:0] vbeta,
    output wire [47:0]        seq_duty3,
    output wire               seq_busy,
    output wire               seq_done,
    output wire [47:0]        ref_duty3
);
  svpwm #(.PWM_HALF_PERIOD(PWM_HALF_PERIOD)) u_ref (
      .valpha(valpha), .vbeta(vbeta), .duty3(ref_duty3));

  svpwm_seq #(.PWM_HALF_PERIOD(PWM_HALF_PERIOD)) u_seq (
      .clk(clk), .rst_n(rst_n), .start(start),
      .valpha(valpha), .vbeta(vbeta),
      .duty3(seq_duty3), .busy(seq_busy), .done(seq_done));
endmodule
