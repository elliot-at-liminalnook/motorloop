// SPDX-License-Identifier: MIT
// FOC datapath core, PIPELINED (robotics-ip-checklist stage 6.5). The
// Clarke -> sincos -> Park -> {id,iq} PI -> circle-limit -> inverse-Park ->
// SVPWM chain is walked over six registered stages, one per clock, triggered
// by `update` (the sequencer's foc_valid current-sample strobe). Because
// `update` fires once per current sample - thousands of clocks apart - the
// multi-cycle walk costs no throughput but shortens the longest combinational
// arc (the circle_limit 16-iteration isqrt + two divisions, the unpipelined
// Fmax bottleneck), so the design closes timing far above the ~3.3 MHz cap.
//
// Behaviour is preserved exactly. The PI integrators advance once per `update`
// with the SAME error and freeze(sat) as the combinational version: the
// measurements and targets are latched on `update` and held across the walk,
// and the integrators move only in the LIMIT stage, using the registered sat
// from the same integrator state that produced the raw voltages. Only the
// duty/dbg outputs appear FOC_LATENCY clocks later; with `update` sparse this
// is invisible to the control loop (the PWM latches the new duty at the next
// period boundary regardless). foc_core is sim-only (latency-aware), not
// formally proven; the leaf blocks it sequences keep their own proofs.
//
//   id/iq measured from the sampled phase currents at the rotor angle;
//   PIs drive (vd, vq) toward (id_target, iq_target);
//   the voltage-circle limiter bounds the vector and freezes the integrators
//   (anti-windup) while clamped; inverse Park + SVPWM produce the per-leg duties.
//   Latency: `update` -> duty3 valid after ~14 clocks (unsaturated) or ~62
//   (saturated: the sequential isqrt + divisions + the sequential SVPWM); both
//   << the sample period.
//
// Surface-PMSM convention: id_target = 0 (no reluctance torque); iq_target is
// the torque command. See notes/foc-fixed-point.md.

module foc_core #(
    parameter integer PWM_HALF_PERIOD  = 625,
    parameter integer SINCOS_TABLE_BITS = 8,
    parameter integer V_CIRCLE_LIMIT   = 594,
    parameter integer CUR_PI_KP        = 2,
    parameter integer CUR_PI_KI_SHIFT  = 4,
    parameter integer V_RAW_MAX        = 2500
) (
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
  localparam [31:0] CENTER = PWM_HALF_PERIOD >> 1;  // 50% duty (sliced at use)

  // Sequencer: one combinational stage settles per state; `update` (sparse)
  // starts a walk from IDLE. The LIMIT stage is the multi-cycle sequential
  // circle limiter (~2 clocks unsaturated, ~50 saturated); a walk always
  // finishes far inside the sample period, so mid-walk strobes are ignored.
  localparam [3:0] S_IDLE    = 4'd0,
                   S_CLARKE  = 4'd1,   // clarke + sincos
                   S_PARK    = 4'd2,   // park -> id, iq
                   S_PI      = 4'd3,   // current PIs -> vd_raw, vq_raw (integ held)
                   S_LIMSTRT = 4'd4,   // launch the sequential circle limiter
                   S_LIMWAIT = 4'd5,   // await limiter done + integrator update
                   S_INVP    = 4'd6,   // inverse Park
                   S_SVPSTRT = 4'd7,   // launch the sequential SVPWM
                   S_SVPWAIT = 4'd8;   // await SVPWM done -> duty3
  reg [3:0] state;

  // Inputs latched on `update`, held stable across the walk.
  reg signed [17:0] in_cur_a, in_cur_b, in_idt, in_iqt;
  reg [15:0]        in_theta;

  // Pipeline registers between stages.
  reg signed [17:0] ialpha_r, ibeta_r;
  reg signed [15:0] cos_r, sin_r;
  reg signed [17:0] id_r, iq_r;
  reg signed [17:0] vd_raw_r, vq_raw_r;
  reg signed [17:0] vd_lim_r, vq_lim_r;
  reg signed [17:0] valpha_r, vbeta_r;

  // Stage 1: Clarke + sincos (combinational from the latched inputs).
  wire signed [17:0] ialpha, ibeta;
  clarke u_clarke (.ia(in_cur_a), .ib(in_cur_b), .ialpha(ialpha), .ibeta(ibeta));

  wire signed [15:0] cos_q15, sin_q15;
  sincos #(.BITS(SINCOS_TABLE_BITS)) u_sincos (
      .theta(in_theta), .sin_out(sin_q15), .cos_out(cos_q15));

  // Stage 2: Park (from the stage-1 registers).
  wire signed [17:0] id_m, iq_m;
  park u_park (.ialpha(ialpha_r), .ibeta(ibeta_r),
               .cos_q15(cos_r), .sin_q15(sin_r),
               .id(id_m), .iq(iq_m));

  // Stage 3: current PIs. meas/target are the held stage-2 regs / latched
  // targets, so v_out is stable from S_PI until the limiter finishes. The
  // integrators move only on pi_update (= S_LIMWAIT && limiter done) with
  // freeze = sat - identical (err, freeze) per `update` as the combinational
  // core, hence bit-identical integrator state.
  wire sat;
  wire signed [17:0] vd_raw, vq_raw, vd_lim, vq_lim;
  wire lim_busy, lim_done;
  wire lim_start = (state == S_LIMSTRT) && !lim_busy;
  wire pi_update = (state == S_LIMWAIT) && lim_done;
  current_pi #(.CUR_PI_KP(CUR_PI_KP), .CUR_PI_KI_SHIFT(CUR_PI_KI_SHIFT),
               .V_RAW_MAX(V_RAW_MAX)) u_pi_d (
      .clk(clk), .rst_n(rst_n), .enable(enable), .update(pi_update),
      .freeze(sat), .target(in_idt), .meas(id_r), .v_out(vd_raw));
  current_pi #(.CUR_PI_KP(CUR_PI_KP), .CUR_PI_KI_SHIFT(CUR_PI_KI_SHIFT),
               .V_RAW_MAX(V_RAW_MAX)) u_pi_q (
      .clk(clk), .rst_n(rst_n), .enable(enable), .update(pi_update),
      .freeze(sat), .target(in_iqt), .meas(iq_r), .v_out(vq_raw));

  // Stage 4: voltage-circle limit (sequential, bit-exact to circle_limit.v).
  // It was the unpipelined Fmax bottleneck (combinational isqrt + two 32-bit
  // divisions); doing it one-op-per-clock shortens the longest arc. The
  // integrators have not moved since S_PI, so when the limiter signals done the
  // live v_out still equals vd_raw_r/vq_raw_r and sat is consistent with the
  // integrator state that produced them - exactly the combinational case.
  circle_limit_seq #(.V_CIRCLE_LIMIT(V_CIRCLE_LIMIT)) u_limit (
      .clk(clk), .rst_n(rst_n), .start(lim_start),
      .vd_in(vd_raw_r), .vq_in(vq_raw_r),
      .vd_out(vd_lim), .vq_out(vq_lim), .sat(sat),
      .busy(lim_busy), .done(lim_done));

  // Stage 5: inverse Park (from the stage-4 registers).
  wire signed [17:0] valpha, vbeta;
  inv_park u_inv_park (.vd(vd_lim_r), .vq(vq_lim_r),
                       .cos_q15(cos_r), .sin_q15(sin_r),
                       .valpha(valpha), .vbeta(vbeta));

  // Stage 6: SVPWM (sequential, bit-exact to svpwm.v - was foc_core's longest
  // combinational arc / Fmax cap). Same start/done handshake as the limiter.
  wire [47:0] duty3_seq;
  wire svp_busy, svp_done;
  wire svp_start = (state == S_SVPSTRT) && !svp_busy;
  svpwm_seq #(.PWM_HALF_PERIOD(PWM_HALF_PERIOD)) u_svpwm (
      .clk(clk), .rst_n(rst_n), .start(svp_start),
      .valpha(valpha_r), .vbeta(vbeta_r),
      .duty3(duty3_seq), .busy(svp_busy), .done(svp_done));

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state  <= S_IDLE;
      duty3  <= {CENTER[15:0], CENTER[15:0], CENTER[15:0]};
      dbg_id <= 18'sd0; dbg_iq <= 18'sd0;
      dbg_vd <= 18'sd0; dbg_vq <= 18'sd0;
    end else if (!enable) begin
      state <= S_IDLE;
      duty3 <= {CENTER[15:0], CENTER[15:0], CENTER[15:0]};  // 50% all legs = zero V
    end else begin
      case (state)
        S_IDLE:
          if (update) begin
            in_cur_a <= cur_a; in_cur_b <= cur_b; in_theta <= theta_e;
            in_idt   <= id_target; in_iqt <= iq_target;
            state <= S_CLARKE;
          end
        S_CLARKE: begin
            ialpha_r <= ialpha; ibeta_r <= ibeta;
            cos_r <= cos_q15; sin_r <= sin_q15;
            state <= S_PARK;
          end
        S_PARK: begin
            id_r <= id_m; iq_r <= iq_m;
            state <= S_PI;
          end
        S_PI: begin
            vd_raw_r <= vd_raw; vq_raw_r <= vq_raw;
            state <= S_LIMSTRT;
          end
        S_LIMSTRT:
            // lim_start pulses here (guarded by !lim_busy); latch v in.
            if (!lim_busy) state <= S_LIMWAIT;
        S_LIMWAIT:
            // pi_update advances the integrators on the done cycle (freeze=sat).
            if (lim_done) begin
              vd_lim_r <= vd_lim; vq_lim_r <= vq_lim;
              state <= S_INVP;
            end
        S_INVP: begin
            valpha_r <= valpha; vbeta_r <= vbeta;
            state <= S_SVPSTRT;
          end
        S_SVPSTRT:
            if (!svp_busy) state <= S_SVPWAIT;
        S_SVPWAIT:
            if (svp_done) begin
              duty3  <= duty3_seq;
              dbg_id <= id_r;     dbg_iq <= iq_r;
              dbg_vd <= vd_lim_r; dbg_vq <= vq_lim_r;
              state  <= S_IDLE;
            end
        default: state <= S_IDLE;
      endcase
    end
  end
endmodule
