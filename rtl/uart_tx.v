// 8N1 UART transmitter at `UART_DIV clocks per bit.

`include "rtl_params.vh"

module uart_tx (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       start,
    input  wire [7:0] data,
    output reg        busy,
    output reg        tx
);

  localparam [15:0] DIV = `UART_DIV;

  reg [15:0] cnt;
  reg [3:0]  bit_idx;   // 0 start, 1..8 data, 9 stop
  reg [7:0]  shreg;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      busy <= 1'b0; tx <= 1'b1; cnt <= 16'd0;
      bit_idx <= 4'd0; shreg <= 8'd0;
    end else if (!busy) begin
      tx <= 1'b1;
      if (start) begin
        busy <= 1'b1;
        shreg <= data;
        bit_idx <= 4'd0;
        cnt <= DIV - 1;
        tx <= 1'b0;  // start bit
      end
    end else begin
      if (cnt == 16'd0) begin
        cnt <= DIV - 1;
        if (bit_idx < 4'd8) begin
          tx <= shreg[0];
          shreg <= {1'b0, shreg[7:1]};
          bit_idx <= bit_idx + 4'd1;
        end else if (bit_idx == 4'd8) begin
          tx <= 1'b1;  // stop bit
          bit_idx <= 4'd9;
        end else begin
          busy <= 1'b0;
        end
      end else begin
        cnt <= cnt - 16'd1;
      end
    end
  end

endmodule
