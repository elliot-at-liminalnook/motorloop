// Dead-time minimum (formal-checklist 4.2). For each leg, a gate may only
// assert once its complement has been off for >= DEAD_CYCLES. The design
// enforces this with per-leg off-time counters (leg[k].off_time_h/l); by
// referencing those real counters in the bind, the edge property is
// 1-inductive (a gate rises only when the relevant off-time counter was
// already >= DEAD). Plant-independent safety, paired with shoot-through.

`include "rtl_params.vh"

module pwm_deadtime_fv (
    input wire        clk,
    input wire        rst_n,
    input wire [2:0]  gate_high,
    input wire [2:0]  gate_low,
    input wire [15:0] off_h0, input wire [15:0] off_l0,
    input wire [15:0] off_h1, input wire [15:0] off_l1,
    input wire [15:0] off_h2, input wire [15:0] off_l2
);
  localparam [15:0] DEAD = `DEAD_CYCLES;

  // When a high-side gate rises, its low-side complement had been off >= DEAD.
  always @(posedge clk) begin
    if (rst_n && $past(rst_n)) begin
      if (gate_high[0] && !$past(gate_high[0])) assert ($past(off_l0) >= DEAD);
      if (gate_high[1] && !$past(gate_high[1])) assert ($past(off_l1) >= DEAD);
      if (gate_high[2] && !$past(gate_high[2])) assert ($past(off_l2) >= DEAD);
      // Symmetric: a low-side gate rises only after its high-side was off.
      if (gate_low[0] && !$past(gate_low[0]))   assert ($past(off_h0) >= DEAD);
      if (gate_low[1] && !$past(gate_low[1]))   assert ($past(off_h1) >= DEAD);
      if (gate_low[2] && !$past(gate_low[2]))   assert ($past(off_h2) >= DEAD);
    end
  end

  // Non-vacuity: a complementary handoff actually happens (a gate rises).
  always @(posedge clk) begin
    if (rst_n) cover (gate_high[0] && !$past(gate_high[0]));
  end
endmodule

bind pwm_generator pwm_deadtime_fv dt_i (
    .clk(clk), .rst_n(rst_n), .gate_high(gate_high), .gate_low(gate_low),
    .off_h0(leg[0].off_time_h), .off_l0(leg[0].off_time_l),
    .off_h1(leg[1].off_time_h), .off_l1(leg[1].off_time_l),
    .off_h2(leg[2].off_time_h), .off_l2(leg[2].off_time_l));
