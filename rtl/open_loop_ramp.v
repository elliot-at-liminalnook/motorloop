// SPDX-License-Identifier: MIT
// Open-loop forced-commutation sector source: 32-bit electrical phase
// accumulator with a linear frequency ramp. The accumulator maps one
// electrical revolution to 2^32; sector = floor(phase * 6 / 2^32).

module open_loop_ramp (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,            // reset phase when low
    input  wire [31:0] target_freq_word,  // phase increment per clk at speed
    input  wire [31:0] ramp_increment,    // freq word increase per 4096 clk
    output wire [2:0]  sector
);

  reg [31:0] phase;
  reg [31:0] freq_word;
  reg [11:0] ramp_tick;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      phase <= 32'd0;
      freq_word <= 32'd0;
      ramp_tick <= 12'd0;
    end else if (!enable) begin
      phase <= 32'd0;
      freq_word <= 32'd0;
      ramp_tick <= 12'd0;
    end else begin
      phase <= phase + freq_word;
      ramp_tick <= ramp_tick + 12'd1;
      if (ramp_tick == 12'd0 && freq_word < target_freq_word) begin
        freq_word <= (target_freq_word - freq_word > ramp_increment)
                     ? freq_word + ramp_increment
                     : target_freq_word;
      end
    end
  end

  // sector = phase / (2^32 / 6) via comparators against k * floor(2^32/6).
  localparam [31:0] T1 = 32'd715827883;   // 2^32/6 * 1
  localparam [31:0] T2 = 32'd1431655765;  // * 2
  localparam [31:0] T3 = 32'd2147483648;  // * 3
  localparam [31:0] T4 = 32'd2863311531;  // * 4
  localparam [31:0] T5 = 32'd3579139413;  // * 5

  assign sector = (phase >= T5) ? 3'd5 :
                  (phase >= T4) ? 3'd4 :
                  (phase >= T3) ? 3'd3 :
                  (phase >= T2) ? 3'd2 :
                  (phase >= T1) ? 3'd1 : 3'd0;

endmodule
