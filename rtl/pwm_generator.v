// Center-aligned three-phase PWM with per-leg complementary drive, dead-time
// insertion, and minimum-pulse enforcement.
//
// Leg modes:
//   00 OFF      - both gates off (floating leg)
//   01 PWM      - complementary chop: high FET on while counter < duty,
//                 low FET on otherwise, dead time inserted on both edges
//   10 LOW_ON   - low FET solid on
//   11 HIGH_ON  - high FET solid on (unused by six-step, kept for symmetry)
//
// The up/down counter peaks at PWM_HALF_PERIOD; the high-side on-window is
// centered on counter == 0, so the off-window is centered on the peak
// (where the ADC samples the floating-phase EMF).

`include "rtl_params.vh"

module pwm_generator (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        kill,            // force all gates off immediately
    input  wire [15:0] duty_compare,    // high-side on while counter < this
    input  wire [5:0]  leg_mode,        // {leg C, leg B, leg A}, 2 bits each
    output reg  [2:0]  gate_high,
    output reg  [2:0]  gate_low,
    output wire [15:0] counter_out,
    output wire        counting_up,
    output reg         period_start     // pulse at counter wrap through 0
);

  localparam [15:0] HALF = `PWM_HALF_PERIOD;
  localparam [15:0] DEAD = `DEAD_CYCLES;
  localparam [15:0] MINP = `MIN_PULSE_CYCLES;

  reg [15:0] counter;
  reg up;

  assign counter_out = counter;
  assign counting_up = up;

  // Min-pulse enforcement: snap tiny on/off slivers to full off/on.
  reg [15:0] duty_eff;
  always @(*) begin
    if (duty_compare < MINP) duty_eff = 16'd0;
    else if (duty_compare > HALF - MINP) duty_eff = HALF;
    else duty_eff = duty_compare;
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      counter <= 16'd0;
      up <= 1'b1;
      period_start <= 1'b0;
    end else begin
      period_start <= 1'b0;
      if (up) begin
        if (counter == HALF - 1) up <= 1'b0;
        counter <= counter + 16'd1;
      end else begin
        if (counter == 16'd1) begin
          up <= 1'b1;
          period_start <= 1'b1;
        end
        counter <= counter - 16'd1;
      end
    end
  end

  wire pwm_on = counter < duty_eff;

  // Per-leg complementary drive with dead time.
  genvar g;
  generate
    for (g = 0; g < 3; g = g + 1) begin : leg
      wire [1:0] mode = leg_mode[2*g+1:2*g];
      reg desired_h, desired_l;
      always @(*) begin
        case (mode)
          2'b00: begin desired_h = 1'b0;   desired_l = 1'b0; end
          2'b01: begin desired_h = pwm_on; desired_l = !pwm_on; end
          2'b10: begin desired_h = 1'b0;   desired_l = 1'b1; end
          default: begin desired_h = 1'b1; desired_l = 1'b0; end
        endcase
      end

      // Saturating off-time counters per gate.
      reg [15:0] off_time_h, off_time_l;
      always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          off_time_h <= 16'hFFFF;
          off_time_l <= 16'hFFFF;
        end else begin
          if (gate_high[g]) off_time_h <= 16'd0;
          else if (off_time_h != 16'hFFFF) off_time_h <= off_time_h + 16'd1;
          if (gate_low[g]) off_time_l <= 16'd0;
          else if (off_time_l != 16'hFFFF) off_time_l <= off_time_l + 16'd1;
        end
      end

      always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          gate_high[g] <= 1'b0;
          gate_low[g] <= 1'b0;
        end else if (kill) begin
          gate_high[g] <= 1'b0;
          gate_low[g] <= 1'b0;
        end else begin
          gate_high[g] <= desired_h && !gate_low[g] && !desired_l
                          && (off_time_l >= DEAD);
          gate_low[g]  <= desired_l && !gate_high[g] && !desired_h
                          && (off_time_h >= DEAD);
        end
      end
    end
  endgenerate

endmodule
