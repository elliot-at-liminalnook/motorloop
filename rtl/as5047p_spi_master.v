// SPDX-License-Identifier: MIT
// SPI master for the AS5047P magnetic angle sensor: mode 1 (CPOL=0, CPHA=1),
// 16-bit frames, MSB first - the same timing class as the DRV8301 SPI master,
// so it reuses DRV_SPI_DIV. It continuously reads ANGLECOM (register 0x3FFF,
// the DAEC-compensated angle). The AS5047P returns the requested angle on the
// NEXT frame (pipelined read), so the master streams the same read command and
// latches each response.
//
// Response word: PARD[15] | EF[14] | ANGLE[13:0]; accepted when the even parity
// over bits[14:0] matches PARD and EF=0. The 14-bit angle is truncated to the
// 12-bit angle bus (top 12 bits) shared with the AS5600 PWM-capture path, so
// the downstream sector/FOC math is unchanged.

module as5047p_spi_master #(
    parameter [7:0] DRV_SPI_DIV = 8'd16   // SCLK = clk / DRV_SPI_DIV (<= 255)
) (
    input  wire        clk,
    input  wire        rst_n,
    output reg  [11:0] angle,
    output reg         angle_valid,
    output reg         new_sample,    // 1-cycle pulse on a fresh valid angle
    output reg         sclk,
    output reg         mosi,
    output reg         ncs,
    input  wire        miso
);

  localparam [7:0]  HALF_DIV = DRV_SPI_DIV >> 1;
  // Read ANGLECOM (0x3FFF): R/W=1, ADDR=0x3FFF -> bits[14:0]=0x7FFF (15 ones,
  // odd) so the even-parity PARC bit is 1 -> command word = 0xFFFF.
  localparam [15:0] READ_ANGLECOM = 16'hFFFF;

  localparam [2:0] S_IDLE = 3'd0, S_LEAD = 3'd1, S_LOW = 3'd2,
                   S_HIGH = 3'd3, S_TAIL = 3'd4, S_GAP = 3'd5;

  reg [2:0]  state;
  reg [7:0]  div;
  reg [4:0]  bit_idx;
  reg [15:0] shreg;
  reg [15:0] rx;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE; sclk <= 1'b0; mosi <= 1'b0; ncs <= 1'b1;
      div <= 8'd0; bit_idx <= 5'd0; shreg <= 16'd0; rx <= 16'd0;
      angle <= 12'd0; angle_valid <= 1'b0; new_sample <= 1'b0;
    end else begin
      new_sample <= 1'b0;
      case (state)
        S_IDLE: begin
          sclk <= 1'b0;
          shreg <= READ_ANGLECOM;
          bit_idx <= 5'd0;
          ncs <= 1'b0;            // start a frame immediately (free-running)
          div <= 8'd0;
          state <= S_LEAD;        // nSCS setup with SCLK low
        end
        S_LEAD: begin
          div <= div + 8'd1;
          if (div == HALF_DIV - 1) begin div <= 8'd0; state <= S_LOW; end
        end
        S_LOW: begin              // SCLK low half
          sclk <= 1'b0;
          div <= div + 8'd1;
          if (div == HALF_DIV - 1) begin
            div <= 8'd0;
            // CPHA=1: launch MOSI on the rising (leading) edge.
            sclk <= 1'b1;
            mosi <= shreg[15];
            shreg <= {shreg[14:0], 1'b0};
            state <= S_HIGH;
          end
        end
        S_HIGH: begin             // SCLK high half: slave drove SDO on rising
          div <= div + 8'd1;
          if (div == HALF_DIV - 1) begin
            div <= 8'd0;
            rx <= {rx[14:0], miso};   // sample at the trailing edge
            sclk <= 1'b0;             // falling edge: slave samples MOSI
            if (bit_idx == 5'd15) begin
              state <= S_TAIL;
            end else begin
              bit_idx <= bit_idx + 5'd1;
              state <= S_LOW;
            end
          end
        end
        S_TAIL: begin             // nSCS hold with SCLK low
          div <= div + 8'd1;
          if (div == HALF_DIV - 1) begin div <= 8'd0; ncs <= 1'b1; state <= S_GAP; end
        end
        S_GAP: begin              // minimum nSCS high time between frames
          div <= div + 8'd1;
          if (div == HALF_DIV - 1) begin
            div <= 8'd0;
            // Even parity over rx[14:0] must equal PARD (rx[15]) and EF=0.
            if ((^rx[14:0]) == rx[15] && rx[14] == 1'b0) begin
              angle <= rx[13:2];        // top 12 of the 14-bit angle
              angle_valid <= 1'b1;
              new_sample <= 1'b1;
            end else begin
              angle_valid <= 1'b0;      // parity/EF error (e.g. magnet loss)
            end
            state <= S_IDLE;
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
