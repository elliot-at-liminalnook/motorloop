// SPDX-License-Identifier: MIT
// Wishbone B4 slave legality for wb_regfile (robotics-ip-checklist stage 5):
//   - ACK is asserted only in response to a fresh strobe (no spurious ACK);
//   - ACK is a single-cycle pulse (classic, registered ack).
// k-inductive via $past on the ack register.

module wb_regfile_fv (
    input wire clk,
    input wire rst_n,
    input wire wb_cyc,
    input wire wb_stb,
    input wire wb_ack
);
  always @(posedge clk) begin
    if (rst_n && $past(rst_n)) begin
      // ACK only follows an accepted strobe.
      if (wb_ack) assert ($past(wb_cyc && wb_stb && !wb_ack));
      // ACK never two cycles in a row (single-beat).
      assert (!(wb_ack && $past(wb_ack)));
    end
  end
  always @(posedge clk) begin
    if (rst_n) cover (wb_ack);
  end
endmodule

bind wb_regfile wb_regfile_fv fv_i (
    .clk(clk), .rst_n(rst_n),
    .wb_cyc(wb_cyc), .wb_stb(wb_stb), .wb_ack(wb_ack));
