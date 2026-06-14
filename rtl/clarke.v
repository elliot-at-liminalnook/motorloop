// Amplitude-invariant Clarke transform. Balanced three-phase => ic = -ia-ib,
// so only ia, ib are needed. Currents in ADC-LSB units (signed).
//   ialpha = ia
//   ibeta  = (ia + 2*ib) / sqrt(3)
// Arithmetic mirrors sim/scripts/foc_reference.clarke_fx (Q15 constant).

module clarke (
    input  wire signed [17:0] ia,
    input  wire signed [17:0] ib,
    output wire signed [17:0] ialpha,
    output wire signed [17:0] ibeta
);
  localparam signed [31:0] ONE_OVER_SQRT3 = 32'sd18919;  // round(1/sqrt3 * 2^15)

  wire signed [31:0] ia32 = {{14{ia[17]}}, ia};
  wire signed [31:0] ib32 = {{14{ib[17]}}, ib};
  wire signed [31:0] beta = ((ia32 + (ib32 <<< 1)) * ONE_OVER_SQRT3) >>> 15;

  assign ialpha = ia;
  assign ibeta  = beta[17:0];
endmodule
