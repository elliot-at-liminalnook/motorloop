// SPI master for the DRV8301: mode 1 (CPOL=0, CPHA=1), 16-bit frames,
// MSB first. The slave shifts SDO out on the SCLK rising edge and samples
// SDI on the falling edge, so the master drives MOSI before the rising edge
// and samples MISO at the end of the high half-period.

`include "rtl_params.vh"

module spi_drv_master (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [15:0] tx,
    output reg         busy,
    output reg         done,        // 1-cycle pulse, rx valid
    output reg  [15:0] rx,
    output reg         sclk,
    output reg         mosi,
    output reg         ncs,
    input  wire        miso
);

  localparam [7:0] HALF_DIV = `DRV_SPI_DIV / 2;

  localparam [2:0] S_IDLE = 3'd0, S_LEAD = 3'd1, S_LOW = 3'd2,
                   S_HIGH = 3'd3, S_TAIL = 3'd4, S_GAP = 3'd5;

  reg [2:0]  state;
  reg [7:0]  div;
  reg [4:0]  bit_idx;
  reg [15:0] shreg;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE; busy <= 1'b0; done <= 1'b0;
      sclk <= 1'b0; mosi <= 1'b0; ncs <= 1'b1;
      div <= 8'd0; bit_idx <= 5'd0; shreg <= 16'd0; rx <= 16'd0;
    end else begin
      done <= 1'b0;
      case (state)
        S_IDLE: begin
          sclk <= 1'b0;
          ncs <= 1'b1;
          if (start) begin
            busy <= 1'b1;
            shreg <= tx;
            bit_idx <= 5'd0;
            ncs <= 1'b0;
            div <= 8'd0;
            state <= S_LEAD;   // nSCS setup time with SCLK low
          end
        end
        S_LEAD: begin
          div <= div + 8'd1;
          if (div == HALF_DIV - 1) begin
            div <= 8'd0;
            state <= S_LOW;
          end
        end
        S_LOW: begin           // SCLK low half
          sclk <= 1'b0;
          div <= div + 8'd1;
          if (div == HALF_DIV - 1) begin
            div <= 8'd0;
            // CPHA=1: launch MOSI on the rising (leading) edge; the slave
            // samples it on the falling (trailing) edge.
            sclk <= 1'b1;
            mosi <= shreg[15];
            shreg <= {shreg[14:0], 1'b0};
            state <= S_HIGH;
          end
        end
        S_HIGH: begin          // SCLK high half: slave drove SDO on rising
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
        S_TAIL: begin          // nSCS hold time with SCLK low
          div <= div + 8'd1;
          if (div == HALF_DIV - 1) begin
            div <= 8'd0;
            ncs <= 1'b1;
            state <= S_GAP;
          end
        end
        S_GAP: begin           // minimum nSCS high time between frames
          div <= div + 8'd1;
          if (div == HALF_DIV - 1) begin
            div <= 8'd0;
            busy <= 1'b0;
            done <= 1'b1;
            state <= S_IDLE;
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
