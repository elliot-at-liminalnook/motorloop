// AS5600 PWM-output capture: measures the high time and period of the
// sensor's PWM frame and decodes the 12-bit angle.
//
// Frame: 4351 units per period, high for (128 + angle) units, so
// angle = high_cycles * 4351 / period_cycles - 128.
//
// Validity: period must be within +/-25% of the expected carrier and edges
// must keep arriving (magnet loss / disconnection -> invalid after timeout).

`include "rtl_params.vh"

module as5600_pwm_capture (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        pwm_in,
    output reg  [11:0] angle,
    output reg         angle_valid,
    output reg         new_sample    // 1-cycle pulse per decoded frame
);

  localparam [31:0] CARRIER = `ANGLE_CARRIER_CYC;
  // E15: validity hysteresis - while valid, tolerate +/-25% carrier drift;
  // once invalid, require +/-15% to revalidate, so a carrier sitting at the
  // boundary cannot flap the loop between run and coast.
  localparam [31:0] PERIOD_MIN = CARRIER - CARRIER / 4;
  localparam [31:0] PERIOD_MAX = CARRIER + CARRIER / 4;
  localparam [31:0] PERIOD_MIN_STRICT = CARRIER - (CARRIER * 3) / 20;
  localparam [31:0] PERIOD_MAX_STRICT = CARRIER + (CARRIER * 3) / 20;
  localparam [31:0] TIMEOUT = 2 * CARRIER;

  // Input synchronizer.
  reg [2:0] sync;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) sync <= 3'b000;
    else sync <= {sync[1:0], pwm_in};
  end
  wire level = sync[1];
  wire rising = sync[1] && !sync[2];
  wire falling = !sync[1] && sync[2];

  reg [31:0] cnt;          // cycles since last rising edge
  reg [31:0] high_cnt;     // latched high time of current period
  reg [31:0] idle_cnt;     // cycles since any edge
  reg        have_rise;

  reg        div_start;
  reg [31:0] div_num;
  reg [31:0] div_den;
  wire       div_busy;
  wire       div_done;
  wire [31:0] div_q;

  divider32 u_div (
      .clk(clk), .rst_n(rst_n), .start(div_start),
      .numerator(div_num), .denominator(div_den),
      .busy(div_busy), .done(div_done), .quotient(div_q)
  );

  function [11:0] angle_from_quotient(input [31:0] q);
    reg [31:0] adj;
    begin
      adj = (q >= 32'd128) ? (q - 32'd128) : 32'd0;
      angle_from_quotient = (adj > 32'd4095) ? 12'd4095 : adj[11:0];
    end
  endfunction

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cnt <= 32'd0; high_cnt <= 32'd0; idle_cnt <= 32'd0;
      have_rise <= 1'b0; angle <= 12'd0; angle_valid <= 1'b0;
      new_sample <= 1'b0; div_start <= 1'b0;
      div_num <= 32'd0; div_den <= 32'd1;
    end else begin
      new_sample <= 1'b0;
      div_start <= 1'b0;
      cnt <= cnt + 32'd1;

      if (rising || falling) idle_cnt <= 32'd0;
      else if (idle_cnt != 32'hFFFFFFFF) idle_cnt <= idle_cnt + 32'd1;
      if (idle_cnt > TIMEOUT) angle_valid <= 1'b0;

      if (falling && have_rise) begin
        high_cnt <= cnt;  // high time = cycles from rise to fall
      end

      if (rising) begin
        if (have_rise && !div_busy
            && ((angle_valid && cnt >= PERIOD_MIN && cnt <= PERIOD_MAX)
                || (!angle_valid && cnt >= PERIOD_MIN_STRICT
                    && cnt <= PERIOD_MAX_STRICT))) begin
          // angle+128 = high * 4351 / period
          div_num <= high_cnt * 32'd4351;
          div_den <= cnt;
          div_start <= 1'b1;
        end else if (have_rise
                     && (cnt < PERIOD_MIN || cnt > PERIOD_MAX)) begin
          angle_valid <= 1'b0;  // out of even the loose band: invalidate
        end
        cnt <= 32'd0;
        have_rise <= 1'b1;
      end

      if (div_done) begin
        angle <= angle_from_quotient(div_q);
        angle_valid <= 1'b1;
        new_sample <= 1'b1;
      end
    end
  end

endmodule
