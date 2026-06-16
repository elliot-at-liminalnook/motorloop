// SPDX-License-Identifier: MIT
// Datapath integrity for adc_sequencer (formal-checklist stage 6): the sample
// strobes are well-formed single-cycle pulses - foc_valid / cur_valid / the
// adc_start request never stay asserted two cycles in a row, so a downstream
// consumer (the FOC core, the current loop) sees exactly one event per sample.

`include "rtl_params.vh"

module adc_sequencer_fv (
    input wire clk,
    input wire rst_n,
    input wire foc_valid,
    input wire cur_valid,
    input wire emf_valid,
    input wire adc_start
);
  always @(posedge clk) begin
    if (rst_n && $past(rst_n)) begin
      assert (!(foc_valid && $past(foc_valid)));   // 1-cycle pulse
      assert (!(cur_valid && $past(cur_valid)));
      assert (!(emf_valid && $past(emf_valid)));
      assert (!(adc_start && $past(adc_start)));    // single-cycle start
    end
  end
  // Non-vacuity: a FOC sample strobe actually occurs.
  always @(posedge clk) begin
    if (rst_n) cover (foc_valid);
  end
endmodule

bind adc_sequencer adc_sequencer_fv fv_i (
    .clk(clk), .rst_n(rst_n),
    .foc_valid(foc_valid), .cur_valid(cur_valid), .emf_valid(emf_valid),
    .adc_start(adc_start));
