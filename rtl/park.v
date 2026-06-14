// Park transform: stationary (alpha,beta) -> rotor (d,q) frame.
//   id =  ialpha*cos + ibeta*sin
//   iq =  ialpha*sin - ibeta*cos
// cos/sin are Q15 signed. This convention puts the torque-producing current
// (in phase with the e_a = Ke*omega*sin(theta_e) back-EMF) on +iq with id = 0
// (see notes/foc-fixed-point.md). Mirrors foc_reference.park_fx.

module park (
    input  wire signed [17:0] ialpha,
    input  wire signed [17:0] ibeta,
    input  wire signed [15:0] cos_q15,
    input  wire signed [15:0] sin_q15,
    output wire signed [17:0] id,
    output wire signed [17:0] iq
);
  wire signed [31:0] a = {{14{ialpha[17]}}, ialpha};
  wire signed [31:0] b = {{14{ibeta[17]}}, ibeta};
  wire signed [31:0] c = {{16{cos_q15[15]}}, cos_q15};
  wire signed [31:0] s = {{16{sin_q15[15]}}, sin_q15};

  wire signed [31:0] id32 = (a * c + b * s) >>> 15;
  wire signed [31:0] iq32 = (a * s - b * c) >>> 15;

  assign id = id32[17:0];
  assign iq = iq32[17:0];
endmodule
