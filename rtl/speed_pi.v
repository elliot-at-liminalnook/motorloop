// SPDX-License-Identifier: MIT
// Speed PI controller (six-step duty loop), updated on each speed measurement.
// duty = clamp(KP * (err + integ >> KI_SHIFT)), conditional integration for
// anti-windup, with a downward duty slew limit. Gains are placeholder-grade
// until motor identification (rtl.speed_pi_* in params.toml, blocked by Q1).
//
// PIPELINED (FOC Fmax-optimization checklist; surfaced as the system cap once
// the FOC datapath was sped up - speed_pi standalone was 47.6 MHz, exactly the
// system Fmax). `update` is sparse (once per speed sample), so the single-cycle
// 32-bit MAC + clamp is walked over a small sequencer - err (S_ERR), the MAC
// (S_MAC), then the clamp + conditional integrator update + down-slew (S_FIN).
// The integrator and duty_compare evolve EXACTLY as the combinational version
// (same err, saturation, and prior duty for the down-slew), so behaviour is
// identical and duty_compare merely appears ~3 clocks later (<< sample period).

module speed_pi #(
    parameter integer SPEED_PI_KP       = 12,
    parameter integer SPEED_PI_KI_SHIFT = 4,
    parameter integer PWM_HALF_PERIOD   = 625,
    parameter [15:0]  DUTY_DOWN_SLEW    = 16'd40
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,        // freeze + reset integrator when low
    input  wire        update,        // new speed sample
    input  wire [15:0] target_speed,  // rad/s
    input  wire [15:0] speed,         // rad/s, magnitude
    input  wire        reverse,       // E5: measured rotation is backward
    output reg  [15:0] duty_compare
);

  localparam signed [31:0] KP = SPEED_PI_KP;
  localparam [31:0] DUTY_MAX_FULL = (PWM_HALF_PERIOD * 49) / 50;  // 98%
  localparam [15:0] DUTY_MAX = DUTY_MAX_FULL[15:0];

  localparam [1:0] S_IDLE = 2'd0,   // await update
                   S_ERR  = 2'd1,   // err = target - measured
                   S_MAC  = 2'd2,   // raw = KP*err + (KP*integ >> KI_SHIFT)
                   S_FIN  = 2'd3;   // clamp + down-slew + conditional integration
  reg [1:0] state;

  reg signed [31:0] integ;
  reg [15:0]        t_r, sp_r;      // inputs latched on update, held across walk
  reg               rev_r;
  reg signed [17:0] err_r;          // pipeline registers
  reg signed [31:0] raw_r;

  // Signed measurement: a backdriven (reverse-spinning) rotor must read as
  // negative speed, or the loop would believe it has reached target while
  // spinning the wrong way (E5).
  wire signed [17:0] measured = rev_r ? -$signed({2'b00, sp_r})
                                      :  $signed({2'b00, sp_r});
  wire signed [17:0] err_c = $signed({2'b00, t_r}) - measured;
  wire signed [31:0] raw_c = KP * err_r + ((KP * integ) >>> SPEED_PI_KI_SHIFT);

  wire saturated_high = raw_r > $signed({16'd0, DUTY_MAX});
  wire saturated_low  = raw_r < 32'sd0;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      integ <= 32'sd0;
      duty_compare <= 16'd0;
      state <= S_IDLE;
    end else if (!enable) begin
      integ <= 32'sd0;
      duty_compare <= 16'd0;
      state <= S_IDLE;
    end else begin
      case (state)
        S_IDLE:
          if (update) begin
            t_r <= target_speed; sp_r <= speed; rev_r <= reverse;
            state <= S_ERR;
          end
        S_ERR: begin
          err_r <= err_c;
          state <= S_MAC;
        end
        S_MAC: begin
          raw_r <= raw_c;            // integ stable here (updates only in S_FIN)
          state <= S_FIN;
        end
        S_FIN: begin
          // Conditional integration: stop winding further into saturation.
          if (!(saturated_high && err_r > 0) && !(saturated_low && err_r < 0)) begin
            integ <= integ + {{14{err_r[17]}}, err_r};
          end
          // Downward duty slew limit: bounds regenerative current during
          // deceleration so braking cannot pump the bus (realism 1.7). Upward
          // steps are unrestricted (acceleration current-limits on its own).
          if (saturated_high) duty_compare <= clamp_down(DUTY_MAX);
          else if (saturated_low) duty_compare <= clamp_down(16'd0);
          else duty_compare <= clamp_down(raw_r[15:0]);
          state <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end

  function [15:0] clamp_down(input [15:0] requested);
    begin
      if (requested + DUTY_DOWN_SLEW < duty_compare)
        clamp_down = duty_compare - DUTY_DOWN_SLEW;
      else
        clamp_down = requested;
    end
  endfunction

endmodule
