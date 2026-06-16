<!-- SPDX-License-Identifier: MIT -->
# Reusable Formal Checkers

Parameterized, bindable property checkers — the building blocks of the
"library of well-proven HDL" vision. Each is a plain module you `bind` into a
DUT; the checker brings its own assertions **and** its bundled non-vacuity
covers, so a proof using it cannot be silently vacuous.

See [`../../notes/formal-checklist.md`](../../notes/formal-checklist.md) for
the methodology and [`../proof_report.md`](../proof_report.md) for what is
currently proven.

## Catalog

### `no_shoot_through #(N)`

No complementary gate pair of an N-leg half-bridge is ever both asserted — the
fundamental power-stage safety property, parameter-generic over leg count.

- **Proves:** `!(gate_high[k] && gate_low[k])` for every leg, gated on `rst_n`.
- **Assumptions:** `rst_n` is the design reset.
- **Bundled cover:** each leg's high-side can actually enable (non-vacuity).
- **Bind:**
  ```systemverilog
  bind my_driver no_shoot_through #(.N(3)) nst_i (
      .clk(clk), .rst_n(rst_n),
      .gate_high(gate_high), .gate_low(gate_low));
  ```
- **Proven on:** `pwm_generator` (`lib_no_shoot_through`), and the same
  property holds at the `controller_top` boundary
  (`controller_top_composition`).

## Adding a checker

A library checker is generic when it references no design constants (only its
parameters and the bound signals). Keep the property and its non-vacuity cover
together in the module so every binding gets both. Document it here with:
proves / assumptions / bundled cover / a one-line bind example / where it is
proven. Roadmap checkers (planned in the checklist): `legal_state #(MAX)`,
`counter_no_overflow`, `valid_ready_handshake`, `latch_clear`, `no_deadlock`,
`reset_safe`.
