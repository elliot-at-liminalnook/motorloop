<!-- SPDX-License-Identifier: MIT -->
# `divider32` — 32-bit sequential divider

Unsigned 32/32→32 restoring divider, one quotient bit per clock. Classic
formulation: `acc = {remainder, working}` is shifted left each cycle, the
denominator is conditionally subtracted from the top half, and the new LSB
becomes that quotient bit; after 32 iterations `acc[31:0]` is the (truncated,
toward-zero) quotient. Divide-by-zero returns `0xFFFFFFFF` in a single cycle.
The reusable long-latency primitive behind `circle_limit_seq` (the two
voltage-rescale divides) and `speed_meter` (the period→rad/s divide).

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset; clears state, `quotient`→0 |
| `start` | in | 1 | — | — | pulse to latch operands and launch (honored only when `!busy`) |
| `numerator` | in | 32 | no | — | dividend, latched on accepted `start` |
| `denominator` | in | 32 | no | — | divisor, latched on accepted `start`; 0 → saturated quotient |
| `busy` | out | 1 | — | 0 | high while iterating (not asserted for the divide-by-zero fast path) |
| `done` | out | 1 | — | 0 | 1-cycle pulse, `quotient` valid |
| `quotient` | out | 32 | no | 0 | `floor(numerator/denominator)`, or `0xFFFFFFFF` on divide-by-zero; holds until next `done` |

## Clocking & reset

- **Clock domains:** single `clk`; `start`→`done` handshake.
- **Reset:** async active-low `rst_n` → `busy/done=0`, `quotient=0`, internal
  `acc/den/count` cleared.
- **Latency:** **32 clocks** start→done for a normal divide (`busy` asserted the
  whole window); divide-by-zero pulses `done` the **next clock** with
  `0xFFFFFFFF` and never raises `busy`. `start` is ignored while `busy`.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| _(none)_ | | |

## Formal contract

- **Sim-only** (`formal/sim_only.toml`): sequential divider; covered by the speed-measurement sim tier
- Covered by the simulation tiers, not a machine proof.
- **Assumptions:** single clock; async active-low reset is the design reset;
  `start` pulsed only when `!busy`; operands stable on the accepted `start`.

## Synthesis fit

- **Device:** ECP5 (part of the system fit). Small: a 64-bit shift accumulator,
  one 32-bit subtract/compare, and a 6-bit counter; not on the system critical
  path (its callers walk it once per sparse `start`).

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — self-contained. (Used by `circle_limit_seq` and
  `speed_meter`.)
- **Pull it:** `fusesoc run motorloop:ip:divider32` (core at repo root).
