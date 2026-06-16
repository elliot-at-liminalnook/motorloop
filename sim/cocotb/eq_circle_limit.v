// SPDX-License-Identifier: MIT
// Equivalence harness (robotics-ip-checklist stage 6.5): the combinational
// `circle_limit` reference and the sequential `circle_limit_seq` side by side,
// fed identical inputs, so a cocotb test can assert they agree bit-for-bit when
// the sequential block finishes. This is circle_limit_seq's correctness claim -
// it justifies foc_core's behaviour being unchanged by the pipelining.
module eq_circle_limit #(
    parameter integer V_CIRCLE_LIMIT = 594
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start,
    input  wire signed [17:0] vd_in,
    input  wire signed [17:0] vq_in,
    output wire signed [17:0] seq_vd,
    output wire signed [17:0] seq_vq,
    output wire               seq_sat,
    output wire               seq_busy,
    output wire               seq_done,
    output wire signed [17:0] ref_vd,
    output wire signed [17:0] ref_vq,
    output wire               ref_sat
);
  circle_limit #(.V_CIRCLE_LIMIT(V_CIRCLE_LIMIT)) u_ref (
      .vd_in(vd_in), .vq_in(vq_in),
      .vd_out(ref_vd), .vq_out(ref_vq), .sat(ref_sat));

  circle_limit_seq #(.V_CIRCLE_LIMIT(V_CIRCLE_LIMIT)) u_seq (
      .clk(clk), .rst_n(rst_n), .start(start),
      .vd_in(vd_in), .vq_in(vq_in),
      .vd_out(seq_vd), .vq_out(seq_vq), .sat(seq_sat),
      .busy(seq_busy), .done(seq_done));
endmodule
