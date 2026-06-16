// SPDX-License-Identifier: MIT
// Framing/datapath integrity for as5047p_spi_master (platform-abstraction
// Phase C, stage 10.5). Two safety claims, both k-inductive:
//   1. FSM legality: the 3-bit state register only ever holds one of the 6
//      defined states S_IDLE(0)..S_GAP(5) - every case arm assigns a defined
//      next state and reset starts at S_IDLE.
//   2. new_sample is a well-formed single-cycle pulse - it defaults to 0 every
//      cycle and is set for at most one cycle (in the S_GAP terminal), so a
//      downstream consumer sees exactly one event per latched angle.
//
// Non-vacuity covers are shallow (a frame starts; the clock toggles); the full
// 16-bit read that drives new_sample is ~280 cycles deep, exercised by the
// simulation tier (test_as5047p / test_platforms) rather than BMC here.

`include "rtl_params.vh"

module as5047p_spi_master_fv (
    input wire       clk,
    input wire       rst_n,
    input wire [2:0] state,
    input wire       new_sample,
    input wire       ncs,
    input wire       sclk
);
  localparam [2:0] S_GAP = 3'd5;   // highest defined state
  always @(posedge clk) begin
    if (rst_n) assert (state <= S_GAP);                 // FSM legality
    if (rst_n && $past(rst_n))
      assert (!(new_sample && $past(new_sample)));      // 1-cycle pulse
  end
  // Non-vacuity: a frame actually starts (ncs low) and the clock asserts.
  always @(posedge clk) begin
    if (rst_n) cover (!ncs);
    if (rst_n) cover (sclk);
  end
endmodule

bind as5047p_spi_master as5047p_spi_master_fv fv_i (
    .clk(clk), .rst_n(rst_n), .state(state),
    .new_sample(new_sample), .ncs(ncs), .sclk(sclk));
