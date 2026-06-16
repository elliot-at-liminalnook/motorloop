// SPDX-License-Identifier: MIT
// FOC outer speed loop: speed error -> q-axis current command iq* (signed
// LSB). Parallel-form PI with conditional-integration anti-windup and a
// symmetric clamp to +-IQ_MAX. The signed output lets the loop command
// braking torque (negative iq) when overspeed - unlike the six-step duty PI,
// which is unsigned. Updated on each speed measurement.
//
//   err   = target - measured            (rad/s, signed)
//   iq*   = clamp( KP*err + (KP*integ >> KISH), +-IQ_MAX )
//
// Gains are placeholder-grade (foc.speed_*, blocked by Q1).

module speed_iq_pi #(
    parameter integer SPEED_IQ_KP   = 4,
    parameter integer SPEED_IQ_KISH = 5,
    parameter integer IQ_MAX        = 300
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               enable,        // reset integrator when low
    input  wire               update,        // new speed sample
    input  wire [15:0]        target_speed,  // rad/s
    input  wire [15:0]        speed,         // rad/s magnitude
    input  wire               reverse,       // measured rotation is backward
    output reg signed [17:0]  iq_target
);
  localparam signed [31:0] KP   = SPEED_IQ_KP;
  localparam integer       KISH = SPEED_IQ_KISH;
  localparam signed [31:0] IQMAX = IQ_MAX;

  reg signed [31:0] integ;

  wire signed [31:0] measured = reverse ? -$signed({16'd0, speed})
                                        :  $signed({16'd0, speed});
  wire signed [31:0] err = $signed({16'd0, target_speed}) - measured;
  wire signed [31:0] raw = KP * err + ((KP * integ) >>> KISH);

  wire sat_hi = raw > IQMAX;
  wire sat_lo = raw < -IQMAX;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      integ <= 32'sd0;
      iq_target <= 18'sd0;
    end else if (!enable) begin
      integ <= 32'sd0;
      iq_target <= 18'sd0;
    end else if (update) begin
      // Conditional integration: do not wind further into the active clamp.
      if (!(sat_hi && err > 0) && !(sat_lo && err < 0)) begin
        integ <= integ + err;
      end
      iq_target <= sat_hi ? IQMAX[17:0]
                 : sat_lo ? -IQMAX[17:0]
                 : raw[17:0];
    end
  end
endmodule
