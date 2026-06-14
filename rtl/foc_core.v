// FOC datapath core: Clarke -> Park -> {id,iq} PI -> circle limit ->
// inverse Park -> SVPWM, all combinational, with the PI integrators and the
// output duties registered once per current sample (`update` = the
// sequencer's foc_valid). Between samples the duties hold.
//
//   id/iq measured from the sampled phase currents at the rotor angle;
//   PIs drive (vd, vq) toward (id_target, iq_target);
//   the voltage-circle limiter bounds the vector and freezes the integrators
//   (anti-windup) while clamped;
//   inverse Park + SVPWM produce the three per-leg duties.
//
// Surface-PMSM convention: id_target = 0 (no reluctance torque); iq_target is
// the torque command. See notes/foc-fixed-point.md.

`include "rtl_params.vh"

module foc_core (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               enable,
    input  wire               update,        // FOC current-sample strobe
    input  wire signed [17:0] cur_a,
    input  wire signed [17:0] cur_b,
    input  wire [15:0]        theta_e,
    input  wire signed [17:0] id_target,
    input  wire signed [17:0] iq_target,
    output reg  [47:0]        duty3,
    output reg signed [17:0]  dbg_id,
    output reg signed [17:0]  dbg_iq,
    output reg signed [17:0]  dbg_vd,
    output reg signed [17:0]  dbg_vq
);
  localparam [15:0] CENTER = `PWM_HALF_PERIOD >> 1;

  // Forward measurement path (combinational).
  wire signed [17:0] ialpha, ibeta;
  clarke u_clarke (.ia(cur_a), .ib(cur_b), .ialpha(ialpha), .ibeta(ibeta));

  wire signed [15:0] cos_q15, sin_q15;
  sincos u_sincos (.theta(theta_e), .sin_out(sin_q15), .cos_out(cos_q15));

  wire signed [17:0] id_m, iq_m;
  park u_park (.ialpha(ialpha), .ibeta(ibeta),
               .cos_q15(cos_q15), .sin_q15(sin_q15),
               .id(id_m), .iq(iq_m));

  // Current PIs (frozen together when the voltage vector saturates).
  wire sat;
  wire signed [17:0] vd_raw, vq_raw, vd_lim, vq_lim;
  current_pi u_pi_d (
      .clk(clk), .rst_n(rst_n), .enable(enable), .update(update),
      .freeze(sat), .target(id_target), .meas(id_m), .v_out(vd_raw));
  current_pi u_pi_q (
      .clk(clk), .rst_n(rst_n), .enable(enable), .update(update),
      .freeze(sat), .target(iq_target), .meas(iq_m), .v_out(vq_raw));

  circle_limit u_limit (
      .vd_in(vd_raw), .vq_in(vq_raw),
      .vd_out(vd_lim), .vq_out(vq_lim), .sat(sat));

  // Reverse modulation path (combinational).
  wire signed [17:0] valpha, vbeta;
  inv_park u_inv_park (.vd(vd_lim), .vq(vq_lim),
                       .cos_q15(cos_q15), .sin_q15(sin_q15),
                       .valpha(valpha), .vbeta(vbeta));

  wire [47:0] duty3_comb;
  svpwm u_svpwm (.valpha(valpha), .vbeta(vbeta), .duty3(duty3_comb));

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      duty3 <= {CENTER, CENTER, CENTER};
      dbg_id <= 18'sd0; dbg_iq <= 18'sd0;
      dbg_vd <= 18'sd0; dbg_vq <= 18'sd0;
    end else if (!enable) begin
      duty3 <= {CENTER, CENTER, CENTER};   // 50% all legs = zero voltage
    end else if (update) begin
      duty3 <= duty3_comb;
      dbg_id <= id_m; dbg_iq <= iq_m;
      dbg_vd <= vd_lim; dbg_vq <= vq_lim;
    end
  end
endmodule
