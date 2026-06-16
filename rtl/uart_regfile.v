// SPDX-License-Identifier: MIT
// UART command/telemetry register file (realism stage 6).
//
// Protocol (host -> device):
//   write: {1'b1, addr[6:0]}, data[15:8], data[7:0]
//   read:  {1'b0, addr[6:0]}            -> device replies data[15:8], data[7:0]
//
// Write registers: 0 mode, 1 duty, 2 target speed, 3 align offset,
//   4/5 open-loop freq word hi/lo, 6/7 ramp increment hi/lo,
//   8 control (bit0: UART overrides the direct ctrl_* ports).
// Read registers: 0-8 echo, 16 speed, 17 {fault,mismatch} counts, 18 angle,
//   19 nOCTW count, 20 {configured, sector}.

module uart_regfile #(
    parameter [15:0]  UART_DIV         = 16'd217,
    parameter integer UART_TIMEOUT_CYC = 50000
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        uart_rx_pin,
    output wire        uart_tx_pin,
    // control outputs (muxed against the direct ports in the top)
    output reg         use_uart,
    output reg  [1:0]  r_mode,
    output reg  [15:0] r_duty,
    output reg  [15:0] r_target_speed,
    output reg  [11:0] r_align,
    output reg  [31:0] r_ol_freq_word,
    output reg  [31:0] r_ol_ramp_inc,
    // telemetry inputs
    input  wire [15:0] t_speed,
    input  wire [7:0]  t_fault_count,
    input  wire [7:0]  t_mismatch_count,
    input  wire [11:0] t_angle,
    input  wire [15:0] t_noctw_count,
    input  wire [2:0]  t_sector,
    input  wire        t_configured,
    input  wire [7:0]  t_flags  // {adc_stuck, offset_fault, stall,
                                //  locked_out, drv_dead, reverse, ...}
);

  wire [7:0] rx_data;
  wire       rx_valid;
  uart_rx #(.UART_DIV(UART_DIV)) u_rx (.clk(clk), .rst_n(rst_n), .rx(uart_rx_pin),
                .data(rx_data), .valid(rx_valid));

  reg        tx_start;
  reg [7:0]  tx_data;
  wire       tx_busy;
  uart_tx #(.UART_DIV(UART_DIV)) u_tx (.clk(clk), .rst_n(rst_n), .start(tx_start),
                .data(tx_data), .busy(tx_busy), .tx(uart_tx_pin));

  localparam [2:0] S_CMD = 3'd0, S_WHI = 3'd1, S_WLO = 3'd2,
                   S_RHI = 3'd3, S_RLO = 3'd4;

  reg [2:0]  state;
  reg [6:0]  addr;
  reg [7:0]  data_hi;
  reg [15:0] read_value;
  reg [7:0]  read_lo_latch;  // LO byte latched with HI: no torn telemetry
  reg [31:0] idle_timer;     // E16: torn-frame recovery

  always @(*) begin
    case (addr)
      7'd0:  read_value = {14'd0, r_mode};
      7'd1:  read_value = r_duty;
      7'd2:  read_value = r_target_speed;
      7'd3:  read_value = {4'd0, r_align};
      7'd4:  read_value = r_ol_freq_word[31:16];
      7'd5:  read_value = r_ol_freq_word[15:0];
      7'd6:  read_value = r_ol_ramp_inc[31:16];
      7'd7:  read_value = r_ol_ramp_inc[15:0];
      7'd8:  read_value = {15'd0, use_uart};
      7'd16: read_value = t_speed;
      7'd17: read_value = {t_fault_count, t_mismatch_count};
      7'd18: read_value = {4'd0, t_angle};
      7'd19: read_value = t_noctw_count;
      7'd20: read_value = {12'd0, t_configured, t_sector};
      7'd21: read_value = {8'd0, t_flags};
      default: read_value = 16'hDEAD;
    endcase
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_CMD; addr <= 7'd0; data_hi <= 8'd0; read_lo_latch <= 8'd0;
      use_uart <= 1'b0; r_mode <= 2'd0; r_duty <= 16'd0;
      r_target_speed <= 16'd0; r_align <= 12'd0;
      r_ol_freq_word <= 32'd0; r_ol_ramp_inc <= 32'd0;
      tx_start <= 1'b0; tx_data <= 8'd0;
      idle_timer <= 32'd0;
    end else begin
      tx_start <= 1'b0;
      // E16: inter-byte timeout - a torn frame (missing data bytes) must
      // not leave the FSM waiting forever, misinterpreting every byte that
      // follows. Mid-frame silence resets to command state.
      if (state == S_WHI || state == S_WLO) begin
        if (rx_valid) idle_timer <= 32'd0;
        else if (idle_timer >= UART_TIMEOUT_CYC) begin
          idle_timer <= 32'd0;
          state <= S_CMD;
        end else begin
          idle_timer <= idle_timer + 32'd1;
        end
      end else begin
        idle_timer <= 32'd0;
      end
      case (state)
        S_CMD: if (rx_valid) begin
          addr <= rx_data[6:0];
          state <= rx_data[7] ? S_WHI : S_RHI;
        end
        S_WHI: if (rx_valid) begin
          data_hi <= rx_data;
          state <= S_WLO;
        end
        S_WLO: if (rx_valid) begin
          case (addr)
            7'd0: r_mode <= rx_data[1:0];
            7'd1: r_duty <= {data_hi, rx_data};
            7'd2: r_target_speed <= {data_hi, rx_data};
            7'd3: r_align <= {data_hi[3:0], rx_data};
            7'd4: r_ol_freq_word[31:16] <= {data_hi, rx_data};
            7'd5: r_ol_freq_word[15:0] <= {data_hi, rx_data};
            7'd6: r_ol_ramp_inc[31:16] <= {data_hi, rx_data};
            7'd7: r_ol_ramp_inc[15:0] <= {data_hi, rx_data};
            7'd8: use_uart <= rx_data[0];
            default: ;
          endcase
          state <= S_CMD;
        end
        S_RHI: if (!tx_busy && !tx_start) begin
          tx_data <= read_value[15:8];
          read_lo_latch <= read_value[7:0];
          tx_start <= 1'b1;
          state <= S_RLO;
        end
        S_RLO: if (!tx_busy && !tx_start) begin
          tx_data <= read_lo_latch;
          tx_start <= 1'b1;
          state <= S_CMD;
        end
        default: state <= S_CMD;
      endcase
    end
  end

endmodule
