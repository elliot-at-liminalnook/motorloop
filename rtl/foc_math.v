// Combinational test harness wiring the FOC math primitives so they can be
// exercised bit-for-bit against sim/scripts/foc_reference.py (test_foc_math).
// Not part of the controller; build target only.

module foc_math (
    input  wire [15:0]        theta,
    input  wire signed [17:0] ia,
    input  wire signed [17:0] ib,
    input  wire signed [17:0] vd,
    input  wire signed [17:0] vq,
    input  wire signed [17:0] valpha_in,
    input  wire signed [17:0] vbeta_in,
    output wire signed [15:0] sin_out,
    output wire signed [15:0] cos_out,
    output wire signed [17:0] ialpha,
    output wire signed [17:0] ibeta,
    output wire signed [17:0] id,
    output wire signed [17:0] iq,
    output wire signed [17:0] valpha_out,
    output wire signed [17:0] vbeta_out,
    output wire [47:0]        duty3
);
  sincos u_sincos (
      .theta(theta), .sin_out(sin_out), .cos_out(cos_out));

  clarke u_clarke (
      .ia(ia), .ib(ib), .ialpha(ialpha), .ibeta(ibeta));

  park u_park (
      .ialpha(ialpha), .ibeta(ibeta),
      .cos_q15(cos_out), .sin_q15(sin_out), .id(id), .iq(iq));

  inv_park u_inv_park (
      .vd(vd), .vq(vq),
      .cos_q15(cos_out), .sin_q15(sin_out),
      .valpha(valpha_out), .vbeta(vbeta_out));

  svpwm u_svpwm (
      .valpha(valpha_in), .vbeta(vbeta_in), .duty3(duty3));
endmodule
