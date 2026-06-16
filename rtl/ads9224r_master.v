// SPDX-License-Identifier: MIT
// Master for the ADS9224R 16-bit dual *simultaneous*-sampling SAR ADC. In FOC
// mode it triggers one conversion near the PWM-counter peak (where every
// low-side shunt conducts), so BOTH phase currents are sampled at the same
// instant - the hardware resolution of Q21 (no sequential inter-channel skew).
// After READY it clocks 16 bits out of the two data lines SDO_A / SDO_B
// (MSB-first, the same shift/sample timing as the DRV/MCP SPI masters) and
// emits the two signed currents with a strobe. The codes are two's-complement
// (zero = no current), so no offset subtraction is needed downstream.

module ads9224r_master #(
    parameter [7:0]  ADC_SPI_DIV     = 8'd26,    // SCLK = clk / ADC_SPI_DIV (<= 255)
    parameter [15:0] PWM_HALF_PERIOD = 16'd625,
    parameter [15:0] ADC_EMF_LEAD    = 16'd160   // cycles before the counter peak
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] pwm_counter,
    input  wire        pwm_up,
    input  wire        foc_mode,
    // ADS9224R pins
    output reg         convst,
    output reg         ncs,
    output reg         sclk,
    input  wire        sdo_a,
    input  wire        sdo_b,
    input  wire        ready,
    // FOC current outputs (signed, two's-complement from the ADC)
    output reg signed [17:0] foc_cur_a,
    output reg signed [17:0] foc_cur_b,
    output reg               foc_valid
);

  localparam [7:0]  HALF_DIV = ADC_SPI_DIV >> 1;
  localparam [15:0] CONV_LAUNCH = PWM_HALF_PERIOD - ADC_EMF_LEAD;

  localparam [2:0] S_IDLE = 3'd0, S_CONV = 3'd1, S_WAITRDY = 3'd2,
                   S_LOW = 3'd3, S_HIGH = 3'd4, S_TAIL = 3'd5, S_DONE = 3'd6;

  reg [2:0]  state;
  reg [7:0]  div;
  reg [4:0]  bit_idx;
  reg [15:0] rx_a, rx_b;
  reg [3:0]  conv_cnt;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE; convst <= 1'b0; ncs <= 1'b1; sclk <= 1'b0;
      div <= 8'd0; bit_idx <= 5'd0; rx_a <= 16'd0; rx_b <= 16'd0;
      conv_cnt <= 4'd0;
      foc_cur_a <= 18'sd0; foc_cur_b <= 18'sd0; foc_valid <= 1'b0;
    end else begin
      foc_valid <= 1'b0;
      case (state)
        S_IDLE: begin
          convst <= 1'b0; ncs <= 1'b1; sclk <= 1'b0;
          // Trigger one conversion per period at the off-window center.
          if (foc_mode && pwm_up && pwm_counter == CONV_LAUNCH) begin
            convst <= 1'b1;        // CONVST rising: both S/H latch together
            conv_cnt <= 4'd0;
            state <= S_CONV;
          end
        end
        S_CONV: begin
          conv_cnt <= conv_cnt + 4'd1;   // hold CONVST high briefly
          if (conv_cnt >= 4'd2) begin
            convst <= 1'b0;
            state <= S_WAITRDY;
          end
        end
        S_WAITRDY: begin
          if (ready) begin
            ncs <= 1'b0;           // frame the readout
            bit_idx <= 5'd0;
            div <= 8'd0;
            state <= S_LOW;
          end
        end
        S_LOW: begin
          sclk <= 1'b0;
          div <= div + 8'd1;
          if (div == HALF_DIV - 1) begin
            div <= 8'd0;
            sclk <= 1'b1;          // rising (leading): slave presents next bit
            state <= S_HIGH;
          end
        end
        S_HIGH: begin
          div <= div + 8'd1;
          if (div == HALF_DIV - 1) begin
            div <= 8'd0;
            rx_a <= {rx_a[14:0], sdo_a};   // sample at the trailing edge
            rx_b <= {rx_b[14:0], sdo_b};
            sclk <= 1'b0;
            if (bit_idx == 5'd15) state <= S_TAIL;
            else begin bit_idx <= bit_idx + 5'd1; state <= S_LOW; end
          end
        end
        S_TAIL: begin
          ncs <= 1'b1;
          state <= S_DONE;
        end
        S_DONE: begin
          // Sign-extend the 16-bit two's-complement codes to 18-bit.
          foc_cur_a <= $signed({{2{rx_a[15]}}, rx_a});
          foc_cur_b <= $signed({{2{rx_b[15]}}, rx_b});
          foc_valid <= 1'b1;
          state <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
