// Quarter-wave sin/cos lookup. 16-bit electrical angle (0..65535 = 0..2pi)
// in; Q15 signed sin and cos out. A 2^SINCOS_TABLE_BITS-entry table holds
// sin over [0, pi/2); the top two angle bits fold the quadrant, cos reuses
// the table at theta + pi/2. The table is generated from the same rounding
// rule as sim/scripts/foc_reference.sincos_table (gen_rtl_params.py writes
// rtl/gen/sincos_init.vh), so RTL and reference agree bit-for-bit.

`include "rtl_params.vh"

module sincos #(
    parameter integer BITS = `SINCOS_TABLE_BITS
) (
    input  wire [15:0]        theta,
    output wire signed [15:0] sin_out,
    output wire signed [15:0] cos_out
);
  localparam integer N = (1 << BITS);

  reg [14:0] tab [0:N-1];
  initial begin
`include "sincos_init.vh"
  end

  function signed [15:0] lookup(input [15:0] a);
    reg [1:0]         quad;
    reg [BITS-1:0]    idx;
    reg [BITS-1:0]    ridx;     // mirror index = (N-1) - idx = ~idx (N=2^BITS)
    reg signed [15:0] mval;
    begin
      quad = a[15:14];
      idx  = a[13 -: BITS];
      ridx = ~idx;
      // quad[0] picks ascending vs mirrored within the quarter; quad[1] sign.
      mval = quad[0] ? $signed({1'b0, tab[ridx]})
                     : $signed({1'b0, tab[idx]});
      lookup = quad[1] ? -mval : mval;
    end
  endfunction

  assign sin_out = lookup(theta);
  assign cos_out = lookup(theta + 16'h4000);  // cos = sin(theta + pi/2)
endmodule
