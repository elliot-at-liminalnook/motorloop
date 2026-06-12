// Sector-aware ADC schedule, two conversions per PWM period (the 3.3 V
// MCP3208 budget, see architecture.md):
//
//  slot 1 (early in the period): IOUT of the solidly-conducting low-side
//          phase - its shunt carries the motor current all period;
//  slot 2 (launched ADC_EMF_LEAD cycles before the counter peak): floating
//          phase EMF, timed so the device hold aperture lands in the PWM
//          off-window, which is centered on the counter peak. Every 8th
//          period slot 2 samples the bus voltage instead.
//
// During the DC_CAL window the sequencer instead loops over the current
// channels and stores their offsets (amp inputs shorted -> raw offset code).
//
// Channels: 0..2 IOUTA/B/C, 3..5 EMF-A/B/C, 6 VBUS.

`include "rtl_params.vh"

module adc_sequencer (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] pwm_counter,
    input  wire        pwm_up,
    input  wire        period_start,
    input  wire [15:0] duty_compare,  // E4: skip EMF when off-window too narrow
    input  wire [1:0]  low_phase,
    input  wire [1:0]  float_phase,
    input  wire        dc_cal_window,
    output reg  [11:0] cur_code,      // latest current sample (offset NOT removed)
    output reg  [11:0] emf_code,
    output reg  [11:0] vbus_code,
    output reg  [11:0] offset_a,
    output reg  [11:0] offset_b,
    output reg  [11:0] offset_c,
    output reg         offset_fault,  // E10: an implausible offset was rejected
    output wire        adc_stuck,     // E14: conversions pinned at 0 or 4095
    output reg         cur_valid,
    output reg         emf_valid,
    // adc_spi_master interface
    output reg         adc_start,
    output reg  [2:0]  adc_channel,
    input  wire        adc_busy,
    input  wire        adc_done,
    input  wire [11:0] adc_code
);

  // Slot timing: the EMF conversion launches on the UP slope so its hold
  // aperture lands at the off-window center (counter peak); it stays busy
  // into the early down slope. The current conversion launches on the DOWN
  // slope after that, completing before the next EMF launch. (A conversion
  // takes ~560 clk; the period is 2*PWM_HALF_PERIOD = 1250 clk.)
  localparam [15:0] CUR_LAUNCH = 16'd150;  // on the down slope
  localparam [15:0] EMF_LAUNCH = `PWM_HALF_PERIOD - `ADC_EMF_LEAD;

  localparam [1:0] PEND_NONE = 2'd0, PEND_CUR = 2'd1, PEND_EMF = 2'd2,
                   PEND_VBUS = 2'd3;

  // E10: plausibility window for DC_CAL offsets (amp midpoint = 2048).
  localparam [11:0] OFFSET_LO = 12'd2048 - `DC_CAL_TOL;
  localparam [11:0] OFFSET_HI = 12'd2048 + `DC_CAL_TOL;

  reg [1:0] pending;
  reg [2:0] period_count;
  reg [1:0] cal_channel;
  reg       cal_pending;
  reg [7:0] stuck_count;  // E14

  assign adc_stuck = stuck_count >= `ADC_STUCK_N;

  wire offset_plausible = adc_code >= OFFSET_LO && adc_code <= OFFSET_HI;
  // E4: the hold aperture must land in the off-window; above this duty the
  // off-window is too narrow, so the EMF sample is skipped this period.
  wire emf_window_ok =
      duty_compare <= (`PWM_HALF_PERIOD - `EMF_SKIP_MARGIN);

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cur_code <= 12'd0; emf_code <= 12'd0; vbus_code <= 12'd0;
      offset_a <= 12'd2048; offset_b <= 12'd2048; offset_c <= 12'd2048;
      offset_fault <= 1'b0;
      cur_valid <= 1'b0; emf_valid <= 1'b0;
      adc_start <= 1'b0; adc_channel <= 3'd0;
      pending <= PEND_NONE; period_count <= 3'd0;
      cal_channel <= 2'd0; cal_pending <= 1'b0;
      stuck_count <= 8'd0;
    end else begin
      // E14: count consecutive rail-pinned conversions on any channel.
      if (adc_done) begin
        if (adc_code == 12'd0 || adc_code == 12'd4095) begin
          if (stuck_count != 8'hFF) stuck_count <= stuck_count + 8'd1;
        end else begin
          stuck_count <= 8'd0;
        end
      end
      adc_start <= 1'b0;
      cur_valid <= 1'b0;
      emf_valid <= 1'b0;

      if (period_start) period_count <= period_count + 3'd1;

      if (dc_cal_window) begin
        // Offset capture: round-robin the three current channels.
        if (!adc_busy && !cal_pending) begin
          adc_channel <= {1'b0, cal_channel};
          adc_start <= 1'b1;
          cal_pending <= 1'b1;
        end else if (adc_done && cal_pending) begin
          // E10: reject implausible offsets (e.g. captured mid-brownout);
          // keep the previous value and raise the flag instead.
          if (offset_plausible) begin
            case (cal_channel)
              2'd0: offset_a <= adc_code;
              2'd1: offset_b <= adc_code;
              default: offset_c <= adc_code;
            endcase
          end else begin
            offset_fault <= 1'b1;
          end
          cal_channel <= (cal_channel == 2'd2) ? 2'd0 : cal_channel + 2'd1;
          cal_pending <= 1'b0;
        end
      end else begin
        cal_pending <= 1'b0;

        // Slot 1: current of the conducting low-side phase.
        if (!pwm_up && pwm_counter == CUR_LAUNCH && !adc_busy
            && pending == PEND_NONE) begin
          adc_channel <= {1'b0, low_phase};
          adc_start <= 1'b1;
          pending <= PEND_CUR;
        end

        // Slot 2: floating-phase EMF (or bus voltage every 8th period).
        // E4: at very high duty the off-window is narrower than the hold
        // aperture - the EMF sample is skipped (emf_code holds last valid).
        if (pwm_up && pwm_counter == EMF_LAUNCH && !adc_busy
            && pending == PEND_NONE) begin
          if (period_count == 3'd0) begin
            adc_channel <= 3'd6;
            adc_start <= 1'b1;
            pending <= PEND_VBUS;
          end else if (emf_window_ok) begin
            adc_channel <= 3'd3 + {1'b0, float_phase};
            adc_start <= 1'b1;
            pending <= PEND_EMF;
          end
        end

        if (adc_done) begin
          case (pending)
            PEND_CUR: begin cur_code <= adc_code; cur_valid <= 1'b1; end
            PEND_EMF: begin emf_code <= adc_code; emf_valid <= 1'b1; end
            PEND_VBUS: vbus_code <= adc_code;
            default: ;
          endcase
          pending <= PEND_NONE;
        end
      end
    end
  end

endmodule
