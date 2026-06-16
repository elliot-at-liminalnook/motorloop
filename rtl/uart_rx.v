// SPDX-License-Identifier: MIT
// 8N1 UART receiver, mid-bit sampling at UART_DIV clocks per bit.

module uart_rx #(
    parameter [15:0] UART_DIV = 16'd217   // clocks per bit (clk / baud)
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx,
    output reg  [7:0] data,
    output reg        valid      // 1-cycle pulse
);

  localparam [15:0] DIV = UART_DIV;

  reg [1:0]  sync;
  reg [15:0] cnt;
  reg [3:0]  bit_idx;   // 0 = start, 1..8 data, 9 = stop
  reg        busy;
  reg [7:0]  shreg;

  wire rx_s = sync[1];

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sync <= 2'b11; cnt <= 16'd0; bit_idx <= 4'd0;
      busy <= 1'b0; shreg <= 8'd0; data <= 8'd0; valid <= 1'b0;
    end else begin
      sync <= {sync[0], rx};
      valid <= 1'b0;
      if (!busy) begin
        if (!rx_s) begin            // start-bit edge
          busy <= 1'b1;
          bit_idx <= 4'd0;
          cnt <= DIV >> 1;          // sample mid-bit
        end
      end else begin
        if (cnt == 16'd0) begin
          cnt <= DIV - 16'd1;
          if (bit_idx == 4'd0) begin
            if (rx_s) busy <= 1'b0;  // false start
            else bit_idx <= 4'd1;
          end else if (bit_idx <= 4'd8) begin
            shreg <= {rx_s, shreg[7:1]};  // LSB first
            bit_idx <= bit_idx + 4'd1;
          end else begin
            busy <= 1'b0;
            if (rx_s) begin          // valid stop bit
              data <= shreg;
              valid <= 1'b1;
            end
          end
        end else begin
          cnt <= cnt - 16'd1;
        end
      end
    end
  end

endmodule
