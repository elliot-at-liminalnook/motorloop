// SPDX-License-Identifier: MIT
// Sequential restoring divider, 32/32 -> 32, one quotient bit per cycle.
//
// Classic formulation: acc = {remainder, working}; each cycle shift left,
// conditionally subtract the denominator from the top half, and set the new
// LSB as the quotient bit. After 32 cycles acc[31:0] is the quotient.

module divider32 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [31:0] numerator,
    input  wire [31:0] denominator,
    output reg         busy,
    output reg         done,       // 1-cycle pulse, quotient valid
    output reg  [31:0] quotient
);

  reg [63:0] acc;
  reg [31:0] den;
  reg [5:0]  count;

  wire [63:0] shifted = {acc[62:0], 1'b0};
  wire        ge = shifted[63:32] >= den;
  wire [31:0] top_next = ge ? (shifted[63:32] - den) : shifted[63:32];

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      busy <= 1'b0; done <= 1'b0; quotient <= 32'd0;
      acc <= 64'd0; den <= 32'd0; count <= 6'd0;
    end else begin
      done <= 1'b0;
      if (start && !busy) begin
        if (denominator == 32'd0) begin
          quotient <= 32'hFFFFFFFF;
          done <= 1'b1;
        end else begin
          busy <= 1'b1;
          acc <= {32'd0, numerator};
          den <= denominator;
          count <= 6'd32;
        end
      end else if (busy) begin
        acc <= {top_next, shifted[31:1], ge};
        count <= count - 6'd1;
        if (count == 6'd1) begin
          busy <= 1'b0;
          done <= 1'b1;
          quotient <= {shifted[31:1], ge};  // final acc[31:0]
        end
      end
    end
  end

endmodule
