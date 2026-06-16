// SPDX-License-Identifier: MIT
// Dogfood: prove pwm_generator shoot-through freedom using the *reusable*
// formal/lib/no_shoot_through checker (formal-checklist 8.2), demonstrating the
// library checker binds to a real driver and carries its own non-vacuity
// cover. The same one line binds to any N-leg half-bridge.

bind pwm_generator no_shoot_through #(.N(3)) nst_i (
    .clk(clk), .rst_n(rst_n), .gate_high(gate_high), .gate_low(gate_low));
