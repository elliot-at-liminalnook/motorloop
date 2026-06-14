// Single-axis current PI (one instance for id, one for iq). Parallel form
// with conditional-integration anti-windup: the integrator is frozen by the
// `freeze` input, which the FOC core drives from the voltage-circle limiter's
// saturation flag so neither axis winds up while the vector is clamped.
//
//   err   = target - meas                       (current LSB)
//   v_out = sat( KP*err + (KP*integ >> KI_SHIFT) )   (duty units)
//
// v_out is the raw, per-axis-clamped command; the joint magnitude is bounded
// downstream by circle_limit.v. Gains are placeholder-grade (foc.cur_pi_*,
// blocked by Q1). Mirrors the current-PI math in notes/foc-fixed-point.md.

`include "rtl_params.vh"

module current_pi (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               enable,   // reset integrator when low
    input  wire               update,   // one strobe per FOC current sample
    input  wire               freeze,   // anti-windup: hold integrator
    input  wire signed [17:0] target,
    input  wire signed [17:0] meas,
    output wire signed [17:0] v_out
);
  localparam signed [31:0] KP   = `CUR_PI_KP;
  localparam integer       KISH = `CUR_PI_KI_SHIFT;
  localparam signed [31:0] VMAX = `V_RAW_MAX;

  reg signed [31:0] integ;

  wire signed [31:0] err = {{14{target[17]}}, target} - {{14{meas[17]}}, meas};
  wire signed [31:0] praw = KP * err + ((KP * integ) >>> KISH);
  wire signed [31:0] vsat = (praw > VMAX) ? VMAX
                          : (praw < -VMAX) ? -VMAX : praw;

  assign v_out = vsat[17:0];

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) integ <= 32'sd0;
    else if (!enable) integ <= 32'sd0;
    else if (update && !freeze) integ <= integ + err;
  end
endmodule
