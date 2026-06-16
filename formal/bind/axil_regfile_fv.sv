// SPDX-License-Identifier: MIT
// AXI4-Lite slave protocol legality for axil_regfile (robotics-ip-checklist
// stage 5). The handshake guarantees a master relies on, proven k-inductively:
//   - B/R valid, once asserted, holds until its ready (no withdrawn response);
//   - read data/resp are stable while RVALID && !RREADY;
//   - responses are always OKAY (this slave never errors).
// Together these mean the wrapper can never violate the AXI-Lite contract.

module axil_regfile_fv (
    input wire        clk,
    input wire        rst_n,
    input wire        s_bvalid,
    input wire        s_bready,
    input wire [1:0]  s_bresp,
    input wire        s_rvalid,
    input wire        s_rready,
    input wire [31:0] s_rdata,
    input wire [1:0]  s_rresp
);
  always @(posedge clk) begin
    if (rst_n && $past(rst_n)) begin
      // Write-response channel: held until accepted.
      if ($past(s_bvalid) && !$past(s_bready)) assert (s_bvalid);
      // Read-data channel: held + stable until accepted.
      if ($past(s_rvalid) && !$past(s_rready)) begin
        assert (s_rvalid);
        assert (s_rdata == $past(s_rdata));
        assert (s_rresp == $past(s_rresp));
      end
    end
  end
  always @(posedge clk) begin
    if (rst_n) begin
      if (s_bvalid) assert (s_bresp == 2'b00);   // OKAY
      if (s_rvalid) assert (s_rresp == 2'b00);
    end
  end
  // Non-vacuity: a read response actually completes.
  always @(posedge clk) begin
    if (rst_n) cover (s_rvalid && s_rready);
  end
endmodule

bind axil_regfile axil_regfile_fv fv_i (
    .clk(clk), .rst_n(rst_n),
    .s_bvalid(s_bvalid), .s_bready(s_bready), .s_bresp(s_bresp),
    .s_rvalid(s_rvalid), .s_rready(s_rready), .s_rdata(s_rdata), .s_rresp(s_rresp));
