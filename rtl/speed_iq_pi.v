// SPDX-License-Identifier: MIT
// FOC outer speed loop: speed error -> q-axis current command iq* (signed
// LSB). Parallel-form PI with conditional-integration anti-windup and a
// symmetric clamp to +-IQ_MAX. The signed output lets the loop command
// braking torque (negative iq) when overspeed - unlike the six-step duty PI,
// which is unsigned. Updated on each speed sample.
//
//   err   = target - measured            (rad/s, signed)
//   iq*   = clamp( KP*err + (KP*integ >> KISH), +-IQ_MAX )
//
// PIPELINED (FOC Fmax-optimization checklist, task 1): the single-cycle 32-bit
// MAC + clamp was the system critical path (routing-bound). `update` is sparse
// (once per speed sample), so the computation is walked over a small sequencer -
// err (S_ERR), the MAC (S_MAC), then clamp + the integrator update (S_FIN). The
// integrator advances exactly once per `update` with the SAME err and freeze
// (sat) as the combinational version, so the loop behaviour is identical and
// iq_target merely appears ~3 clocks later (<< the speed-sample period).
//
// Gains are placeholder-grade (foc.speed_*, blocked by Q1).

module speed_iq_pi #(
    parameter integer SPEED_IQ_KP   = 4,
    parameter integer SPEED_IQ_KISH = 5,
    parameter integer IQ_MAX        = 300
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               enable,        // reset integrator when low
    input  wire               update,        // new speed sample
    input  wire [15:0]        target_speed,  // rad/s
    input  wire [15:0]        speed,         // rad/s magnitude
    input  wire               reverse,       // measured rotation is backward
    output reg signed [17:0]  iq_target
);
  localparam signed [31:0] KP   = SPEED_IQ_KP;
  localparam integer       KISH = SPEED_IQ_KISH;
  localparam signed [31:0] IQMAX = IQ_MAX;

  localparam [1:0] S_IDLE = 2'd0,   // await update
                   S_ERR  = 2'd1,   // err = target - measured
                   S_MAC  = 2'd2,   // raw = KP*err + (KP*integ >> KISH)
                   S_FIN  = 2'd3;   // clamp + conditional integrator update
  reg [1:0] state;

  reg signed [31:0] integ;
  reg [15:0]        t_r, sp_r;      // inputs latched on update, held across walk
  reg               rev_r;
  reg signed [31:0] err_r, raw_r;   // pipeline registers

  // Combinational stage functions (each consumes the prior stage's register).
  wire signed [31:0] measured = rev_r ? -$signed({16'd0, sp_r})
                                      :  $signed({16'd0, sp_r});
  wire signed [31:0] err_c = $signed({16'd0, t_r}) - measured;
  wire signed [31:0] raw_c = KP * err_r + ((KP * integ) >>> KISH);
  wire sat_hi = raw_r > IQMAX;
  wire sat_lo = raw_r < -IQMAX;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      integ <= 32'sd0;
      iq_target <= 18'sd0;
      state <= S_IDLE;
    end else if (!enable) begin
      integ <= 32'sd0;
      iq_target <= 18'sd0;
      state <= S_IDLE;
    end else begin
      case (state)
        S_IDLE:
          if (update) begin
            t_r <= target_speed; sp_r <= speed; rev_r <= reverse;
            state <= S_ERR;
          end
        S_ERR: begin
          err_r <= err_c;
          state <= S_MAC;
        end
        S_MAC: begin
          raw_r <= raw_c;            // integ is stable here (updates only in S_FIN)
          state <= S_FIN;
        end
        S_FIN: begin
          // Conditional integration: do not wind further into the active clamp.
          if (!(sat_hi && err_r > 0) && !(sat_lo && err_r < 0)) begin
            integ <= integ + err_r;
          end
          iq_target <= sat_hi ? IQMAX[17:0]
                     : sat_lo ? -IQMAX[17:0]
                     : raw_r[17:0];
          state <= S_IDLE;
        end
        default: state <= S_IDLE;
      endcase
    end
  end
endmodule
