// SPDX-License-Identifier: MIT
// AXI4-Stream telemetry sampler (robotics-ip-checklist stage 5). On each
// `sample` strobe (the FOC foc_valid / six-step cadence) it emits one packed
// telemetry beat on an AXI-Stream master port, so a robotics SoC can DMA the
// controller's state. Backpressure-safe: while a beat is unaccepted (TVALID &&
// !TREADY) TDATA is held stable and any new sample is dropped (counted), so the
// stream protocol is never violated. Legality is formally proven
// (axis_sampler_fv).
//
// Beat layout (TDATA[31:0]): {sector[2:0], configured, angle[11:0], speed[15:0]}

module axis_sampler (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        sample,          // 1-cycle strobe: latch a new beat
    input  wire [15:0] in_speed,
    input  wire [11:0] in_angle,
    input  wire [2:0]  in_sector,
    input  wire        in_configured,
    // AXI4-Stream master
    output reg  [31:0] m_tdata,
    output reg         m_tvalid,
    input  wire        m_tready,
    output wire        m_tlast,         // every beat is a complete sample
    output reg  [15:0] overflow_count   // dropped samples (consumer too slow)
);
  assign m_tlast = 1'b1;

  wire [31:0] beat = {in_sector, in_configured, in_angle, in_speed};
  wire        accept = m_tvalid && m_tready;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      m_tdata <= 32'd0; m_tvalid <= 1'b0; overflow_count <= 16'd0;
    end else begin
      // Beat is consumed this cycle.
      if (accept) m_tvalid <= 1'b0;

      if (sample) begin
        if (!m_tvalid || accept) begin
          // Free (or being consumed now): present the new beat.
          m_tdata  <= beat;
          m_tvalid <= 1'b1;
        end else begin
          // Held beat not yet accepted: drop the new sample (keep TDATA stable).
          if (overflow_count != 16'hFFFF) overflow_count <= overflow_count + 16'd1;
        end
      end
    end
  end
endmodule
