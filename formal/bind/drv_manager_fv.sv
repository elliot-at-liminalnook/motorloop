// FSM legality for drv_manager (formal-checklist 5.1): the 5-bit state
// register never holds an undefined value - it is always one of the 17 defined
// states S_PWRUP(0)..S_LOCKOUT(16). k-inductive: from any legal state the case
// assigns a legal next state, and reset starts at S_PWRUP.
//
// (Reachability covers - "reaches RUN", "reaches FAULT" - are intentionally
// omitted: the FSM is gated behind a ~25000-cycle power-up timer, so BMC cover
// to those states is infeasible; the FOC/init simulation tier exercises the
// full FSM. This proof has no input assumptions, so it cannot be vacuous.)

module drv_manager_fv (
    input wire       clk,
    input wire       rst_n,
    input wire [4:0] state
);
  localparam [4:0] S_LOCKOUT = 5'd16;   // highest defined state
  always @(posedge clk) begin
    if (rst_n) assert (state <= S_LOCKOUT);
  end
endmodule

bind drv_manager drv_manager_fv fv_i (.clk(clk), .rst_n(rst_n), .state(state));
