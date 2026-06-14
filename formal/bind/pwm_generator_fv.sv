// Formal checker for pwm_generator, bound into the DUT (formal-checklist
// stages 3-4). Properties + non-vacuity covers.
//
// Bound (not embedded) so the design stays clean and the checker can reach
// internal signals. Assumptions: synchronous reset pulse handled by sby; all
// inputs free (no constraint needed - the safety properties hold for any
// duty3/leg_mode/kill).

`include "rtl_params.vh"

module pwm_generator_fv (
    input wire        clk,
    input wire        rst_n,
    input wire        kill,
    input wire [2:0]  gate_high,
    input wire [2:0]  gate_low,
    input wire [15:0] counter_out,
    input wire        counting_up
);
  // ---- Safety properties ----------------------------------------------------
  genvar k;
  generate
    for (k = 0; k < 3; k = k + 1) begin : st
      // P_SHOOT_THROUGH: post-dead-time gates of a leg never both asserted.
      always @(posedge clk) begin
        if (rst_n) assert (!(gate_high[k] && gate_low[k]));
      end
    end
  endgenerate

  // P_COUNTER_RANGE: the up/down counter stays within [0, PWM_HALF_PERIOD].
  // Strengthening: the peak is only reached heading down (cannot increment
  // past it) - needed for k-induction.
  always @(posedge clk) begin
    if (rst_n) begin
      assert (counter_out <= `PWM_HALF_PERIOD);
      if (counter_out == `PWM_HALF_PERIOD) assert (!counting_up);
    end
  end

  // P_KILL_DOMINANCE: while kill holds, no gate is ever asserted. (Combined
  // with kill being registered into the gate FFs, this is the one-cycle
  // safety form; the bounded "within K cycles" form is checked in BMC mode.)
  always @(posedge clk) begin
    if (rst_n && $past(rst_n) && $past(kill))
      assert (gate_high == 3'd0 && gate_low == 3'd0);
  end

  // ---- Non-vacuity covers ---------------------------------------------------
  // Guard against a vacuous no-shoot-through proof (e.g. if some assumption
  // made the gates never enable): prove the interesting states are reachable.
  always @(posedge clk) begin
    if (rst_n) begin
      cover (gate_high[0]);                 // high-side actually enables
      cover (gate_low[0]);                  // low-side actually enables
      cover (gate_high[0] && counting_up);  // active high-side drive
      cover ($past(kill) && !kill);         // kill occurs during operation
    end
  end
endmodule

bind pwm_generator pwm_generator_fv fv_i (
    .clk(clk), .rst_n(rst_n), .kill(kill),
    .gate_high(gate_high), .gate_low(gate_low),
    .counter_out(counter_out), .counting_up(counting_up));
