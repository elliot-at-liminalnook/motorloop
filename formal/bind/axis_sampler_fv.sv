// SPDX-License-Identifier: MIT
// AXI4-Stream master legality for axis_sampler (robotics-ip-checklist stage 5):
// once TVALID is asserted it holds until TREADY, and TDATA is stable while the
// beat is unaccepted (TVALID && !TREADY). This is the AXI-Stream contract a
// downstream sink/DMA relies on - the backpressure-safe drop logic must never
// break it. k-inductive via $past.

module axis_sampler_fv (
    input wire        clk,
    input wire        rst_n,
    input wire        m_tvalid,
    input wire        m_tready,
    input wire [31:0] m_tdata
);
  always @(posedge clk) begin
    if (rst_n && $past(rst_n)) begin
      if ($past(m_tvalid) && !$past(m_tready)) begin
        assert (m_tvalid);                      // held until accepted
        assert (m_tdata == $past(m_tdata));     // stable while unaccepted
      end
    end
  end
  always @(posedge clk) begin
    if (rst_n) cover (m_tvalid && m_tready);    // a beat actually streams
  end
endmodule

bind axis_sampler axis_sampler_fv fv_i (
    .clk(clk), .rst_n(rst_n),
    .m_tvalid(m_tvalid), .m_tready(m_tready), .m_tdata(m_tdata));
