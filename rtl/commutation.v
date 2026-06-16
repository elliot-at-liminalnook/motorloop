// SPDX-License-Identifier: MIT
// Six-step commutation table. Sector -> per-leg PWM modes plus the analog
// bookkeeping the ADC sequencer needs (which phase floats, which low-side
// shunt conducts solidly).
//
// Sector table (high, low): 0:AB 1:AC 2:BC 3:BA 4:CA 5:CB.
// The driven-high leg chops (complementary PWM); the driven-low leg is
// solid on; the third floats for BEMF observation.

module commutation (
    input  wire [2:0] sector,      // 0..5
    output reg  [5:0] leg_mode,    // {C,B,A} 2 bits each: 00 off 01 pwm 10 low
    output reg  [1:0] low_phase,   // phase index with solid low-side current
    output reg  [1:0] float_phase  // floating phase index
);

  localparam [1:0] MODE_OFF = 2'b00, MODE_PWM = 2'b01, MODE_LOW = 2'b10;

  always @(*) begin
    case (sector)
      3'd0: begin leg_mode = {MODE_OFF, MODE_LOW, MODE_PWM}; low_phase = 2'd1; float_phase = 2'd2; end
      3'd1: begin leg_mode = {MODE_LOW, MODE_OFF, MODE_PWM}; low_phase = 2'd2; float_phase = 2'd1; end
      3'd2: begin leg_mode = {MODE_LOW, MODE_PWM, MODE_OFF}; low_phase = 2'd2; float_phase = 2'd0; end
      3'd3: begin leg_mode = {MODE_OFF, MODE_PWM, MODE_LOW}; low_phase = 2'd0; float_phase = 2'd2; end
      3'd4: begin leg_mode = {MODE_PWM, MODE_OFF, MODE_LOW}; low_phase = 2'd0; float_phase = 2'd1; end
      3'd5: begin leg_mode = {MODE_PWM, MODE_LOW, MODE_OFF}; low_phase = 2'd1; float_phase = 2'd0; end
      default: begin leg_mode = 6'b000000; low_phase = 2'd0; float_phase = 2'd0; end
    endcase
  end

endmodule
