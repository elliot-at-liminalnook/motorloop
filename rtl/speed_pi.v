// Speed PI controller, updated on each speed measurement.
// duty = clamp(KP * (err + integ >> KI_SHIFT)), conditional integration
// for anti-windup. Gains are placeholder-grade until motor identification
// (rtl.speed_pi_* in params.toml, blocked by Q1).

`include "rtl_params.vh"

module speed_pi (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,        // freeze + reset integrator when low
    input  wire        update,        // new speed sample
    input  wire [15:0] target_speed,  // rad/s
    input  wire [15:0] speed,         // rad/s, magnitude
    input  wire        reverse,       // E5: measured rotation is backward
    output reg  [15:0] duty_compare
);

  localparam signed [31:0] KP = `SPEED_PI_KP;
  localparam [15:0] DUTY_MAX = (`PWM_HALF_PERIOD * 49) / 50;  // 98%

  reg signed [31:0] integ;

  // Signed measurement: a backdriven (reverse-spinning) rotor must read as
  // negative speed, or the loop would believe it has reached target while
  // spinning the wrong way (E5).
  wire signed [17:0] measured = reverse ? -$signed({2'b00, speed})
                                        : $signed({2'b00, speed});
  wire signed [17:0] err = $signed({2'b00, target_speed}) - measured;
  wire signed [31:0] p_term = KP * err;
  wire signed [31:0] i_term = (KP * integ) >>> `SPEED_PI_KI_SHIFT;
  wire signed [31:0] raw = p_term + i_term;

  wire saturated_high = raw > $signed({16'd0, DUTY_MAX});
  wire saturated_low = raw < 32'sd0;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      integ <= 32'sd0;
      duty_compare <= 16'd0;
    end else if (!enable) begin
      integ <= 32'sd0;
      duty_compare <= 16'd0;
    end else if (update) begin
      // Conditional integration: stop winding further into saturation.
      if (!(saturated_high && err > 0) && !(saturated_low && err < 0)) begin
        integ <= integ + {{14{err[17]}}, err};
      end
      // Downward duty slew limit: bounds regenerative current during
      // deceleration so braking cannot pump the bus (realism 1.7). Upward
      // steps are unrestricted (acceleration draws from the supply, which
      // current-limits on its own).
      if (saturated_high) duty_compare <= clamp_down(DUTY_MAX);
      else if (saturated_low) duty_compare <= clamp_down(16'd0);
      else duty_compare <= clamp_down(raw[15:0]);
    end
  end

  function [15:0] clamp_down(input [15:0] requested);
    begin
      if (requested + `DUTY_DOWN_SLEW < duty_compare)
        clamp_down = duty_compare - `DUTY_DOWN_SLEW;
      else
        clamp_down = requested;
    end
  endfunction

endmodule
