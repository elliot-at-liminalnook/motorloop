// SPDX-License-Identifier: MIT
// Speed measurement from commutation-sector edges.
//
// The AS5600's PWM-frame latency (~1 ms) quantizes individual sector-edge
// timing to frame boundaries, so a single-sector period measurement is
// useless at speed (it reads SPEED_NUM/(n*frame)). Instead the period is
// measured across a full electrical revolution (6 edges), which averages
// the frame quantization down to a few percent.
//
// speed [rad/s] = 6 * SPEED_NUM / period_cycles,
// SPEED_NUM = CLK_HZ * 2*pi / (6 * PP).
// Times out to zero speed at standstill so the PI keeps updating.

module speed_meter #(
    parameter integer CLK_HZ    = 25000000,
    parameter integer SPEED_NUM = 6544985   // clk * 2pi / (6 * pole_pairs)
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [2:0]  sector,
    output reg  [15:0] speed,        // rad/s, saturated
    output reg         reverse,      // E5: sector sequence ran backward
    output reg         speed_valid,
    output reg         update        // 1-cycle pulse on new measurement
);

  // Quick timeout declares standstill only when NO edges arrived (so a slow
  // but completing 6-edge window is never corrupted); the hard timeout
  // abandons a stalled partial window.
  localparam [31:0] TIMEOUT = CLK_HZ / 50;        // 20 ms, edge_count == 0
  localparam [31:0] HARD_TIMEOUT = CLK_HZ / 4;    // 250 ms, any state
  localparam [31:0] NUM6 = 32'd6 * SPEED_NUM;

  reg [2:0]  prev_sector;
  reg [2:0]  edge_count;
  reg [31:0] cnt;
  reg [2:0]  fwd_edges, bwd_edges;  // direction vote over each 6-edge window

  reg        div_start;
  reg [31:0] div_den;
  wire       div_busy;
  wire       div_done;
  wire [31:0] div_q;

  divider32 u_div (
      .clk(clk), .rst_n(rst_n), .start(div_start),
      .numerator(NUM6), .denominator(div_den),
      .busy(div_busy), .done(div_done), .quotient(div_q)
  );

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      prev_sector <= 3'd0; edge_count <= 3'd0; cnt <= 32'd0;
      speed <= 16'd0; reverse <= 1'b0; speed_valid <= 1'b0; update <= 1'b0;
      div_start <= 1'b0; div_den <= 32'd1;
      fwd_edges <= 3'd0; bwd_edges <= 3'd0;
    end else begin
      update <= 1'b0;
      div_start <= 1'b0;
      cnt <= cnt + 32'd1;

      if (sector != prev_sector) begin
        prev_sector <= sector;
        // Direction vote: +1 step = forward, -1 step (mod 6) = backward;
        // larger jumps are ambiguous and not counted.
        if (sector == ((prev_sector == 3'd5) ? 3'd0 : prev_sector + 3'd1)) begin
          if (fwd_edges != 3'd7) fwd_edges <= fwd_edges + 3'd1;
        end else if (prev_sector ==
                     ((sector == 3'd5) ? 3'd0 : sector + 3'd1)) begin
          if (bwd_edges != 3'd7) bwd_edges <= bwd_edges + 3'd1;
        end
        if (edge_count == 3'd5) begin
          edge_count <= 3'd0;
          // Latch direction once per measurement window (hysteresis against
          // boundary flicker - E1/E5).
          reverse <= bwd_edges > fwd_edges;
          fwd_edges <= 3'd0;
          bwd_edges <= 3'd0;
          if (!div_busy && cnt != 32'd0) begin
            div_den <= cnt;
            div_start <= 1'b1;
          end
          cnt <= 32'd0;
        end else begin
          edge_count <= edge_count + 3'd1;
        end
      end else if ((cnt > TIMEOUT && edge_count == 3'd0)
                   || cnt > HARD_TIMEOUT) begin
        speed <= 16'd0;
        reverse <= 1'b0;  // standstill defaults to forward
        speed_valid <= 1'b1;
        update <= 1'b1;
        cnt <= 32'd0;
        edge_count <= 3'd0;
        fwd_edges <= 3'd0;
        bwd_edges <= 3'd0;
      end

      if (div_done) begin
        speed <= (div_q > 32'd65535) ? 16'd65535 : div_q[15:0];
        speed_valid <= 1'b1;
        update <= 1'b1;
      end
    end
  end

endmodule
