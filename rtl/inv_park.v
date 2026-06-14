// Inverse Park: rotor (d,q) voltage -> stationary (alpha,beta).
//   valpha = vd*cos + vq*sin
//   vbeta  = vd*sin - vq*cos
// The Park matrix [[c,s],[s,-c]] is its own inverse, so this reuses the same
// form as park.v. Mirrors foc_reference.inv_park_fx.

module inv_park (
    input  wire signed [17:0] vd,
    input  wire signed [17:0] vq,
    input  wire signed [15:0] cos_q15,
    input  wire signed [15:0] sin_q15,
    output wire signed [17:0] valpha,
    output wire signed [17:0] vbeta
);
  wire signed [31:0] d = {{14{vd[17]}}, vd};
  wire signed [31:0] q = {{14{vq[17]}}, vq};
  wire signed [31:0] c = {{16{cos_q15[15]}}, cos_q15};
  wire signed [31:0] s = {{16{sin_q15[15]}}, sin_q15};

  wire signed [31:0] va = (d * c + q * s) >>> 15;
  wire signed [31:0] vb = (d * s - q * c) >>> 15;

  assign valpha = va[17:0];
  assign vbeta  = vb[17:0];
endmodule
