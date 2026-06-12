// SPI master for the MCP3208: mode 0,0. The device samples DIN on rising
// edges and shifts DOUT on falling edges, so the master drives MOSI during
// the low half and samples MISO at the end of the high half (data set up by
// the previous falling edge).
//
// Frame: start(1), SGL=1, D2, D1, D0, then zeros; 20 clocks total. The
// device's hold aperture is the falling edge of the 5th clock after start;
// data bits B11..B0 are collected on clocks 7..18 (1-based).

`include "rtl_params.vh"

module adc_spi_master (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [2:0]  channel,
    output reg         busy,
    output reg         done,         // 1-cycle pulse, code valid
    output reg  [11:0] code,
    output reg         sclk,
    output reg         mosi,
    output reg         ncs,
    input  wire        miso
);

  localparam [7:0] HALF_DIV = `ADC_SPI_DIV / 2;
  localparam [4:0] TOTAL_CLOCKS = 5'd20;

  localparam [2:0] S_IDLE = 3'd0, S_LEAD = 3'd1, S_LOW = 3'd2,
                   S_HIGH = 3'd3, S_GAP = 3'd4;

  reg [2:0]  state;
  reg [7:0]  div;
  reg [4:0]  cyc;          // 1-based clock cycle index
  reg [4:0]  cmd;          // start, SGL, D2, D1, D0

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE; busy <= 1'b0; done <= 1'b0;
      sclk <= 1'b0; mosi <= 1'b0; ncs <= 1'b1;
      div <= 8'd0; cyc <= 5'd0; cmd <= 5'd0; code <= 12'd0;
    end else begin
      done <= 1'b0;
      case (state)
        S_IDLE: begin
          sclk <= 1'b0;
          ncs <= 1'b1;
          if (start) begin
            busy <= 1'b1;
            cmd <= {1'b1, 1'b1, channel};  // start, single-ended, channel
            cyc <= 5'd1;
            ncs <= 1'b0;
            div <= 8'd0;
            state <= S_LEAD;
          end
        end
        S_LEAD: begin
          div <= div + 8'd1;
          if (div == HALF_DIV - 1) begin
            div <= 8'd0;
            mosi <= 1'b1;  // start bit for cycle 1
            state <= S_LOW;
          end
        end
        S_LOW: begin
          sclk <= 1'b0;
          div <= div + 8'd1;
          if (div == HALF_DIV - 1) begin
            div <= 8'd0;
            sclk <= 1'b1;          // rising: device samples MOSI
            state <= S_HIGH;
          end
        end
        S_HIGH: begin
          div <= div + 8'd1;
          if (div == HALF_DIV - 1) begin
            div <= 8'd0;
            // Hold/null at falling F5 (end of clock cycle 6); B11..B0 are
            // launched on the fallings of cycles 7..18, and with the
            // one-cycle bench feedback delay the master collects them
            // during the high halves of cycles 8..19.
            if (cyc >= 5'd8 && cyc <= 5'd19) begin
              code <= {code[10:0], miso};
            end
            sclk <= 1'b0;          // falling: device shifts next DOUT
            if (cyc == TOTAL_CLOCKS) begin
              ncs <= 1'b1;
              state <= S_GAP;
            end else begin
              cyc <= cyc + 5'd1;
              // Command bits on cycles 1..5 (set up during low half).
              mosi <= (cyc < 5'd5) ? cmd[3'd4 - cyc[2:0]] : 1'b0;
              state <= S_LOW;
            end
          end
        end
        S_GAP: begin               // CS high >= 500 ns between conversions
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
