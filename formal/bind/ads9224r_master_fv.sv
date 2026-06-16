// SPDX-License-Identifier: MIT
// Framing/datapath integrity for ads9224r_master (platform-abstraction Phase C,
// stage 13.5). Two k-inductive safety claims:
//   1. FSM legality: the 3-bit state register only ever holds one of the 7
//      defined states S_IDLE(0)..S_DONE(6).
//   2. foc_valid is a well-formed single-cycle pulse (set only in the S_DONE
//      terminal), so the FOC core sees one current update per conversion.
//
// Non-vacuity covers are shallow (a conversion triggers; a readout frames); the
// full CONVST -> READY -> 16-bit read that drives foc_valid is exercised by the
// simulation tier (test_ads9224r / test_platforms).

`include "rtl_params.vh"

module ads9224r_master_fv (
    input wire       clk,
    input wire       rst_n,
    input wire [2:0] state,
    input wire       foc_valid,
    input wire       convst,
    input wire       ncs
);
  localparam [2:0] S_DONE = 3'd6;   // highest defined state
  always @(posedge clk) begin
    if (rst_n) assert (state <= S_DONE);                // FSM legality
    if (rst_n && $past(rst_n))
      assert (!(foc_valid && $past(foc_valid)));        // 1-cycle pulse
  end
  always @(posedge clk) begin
    if (rst_n) cover (convst);    // a conversion triggers
    if (rst_n) cover (!ncs);      // a readout frames
  end
endmodule

bind ads9224r_master ads9224r_master_fv fv_i (
    .clk(clk), .rst_n(rst_n), .state(state),
    .foc_valid(foc_valid), .convst(convst), .ncs(ncs));
