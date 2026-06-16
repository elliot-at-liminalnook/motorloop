<!-- SPDX-License-Identifier: MIT -->
# `circle_limit_seq` — Sequential voltage-circle limiter (multi-cycle, bit-exact to circle_limit)

Bounds the dq voltage vector to the SVPWM inscribed circle of radius
`V_CIRCLE_LIMIT`: if `vd²+vq² ≤ VLIM²` the inputs pass through unchanged
(`sat=0`); otherwise the vector is rescaled onto the circle by
`v_out = trunc(v·VLIM / |v|)` with `|v| = floor(sqrt(vd²+vq²))`, and `sat=1`
(used as the PI anti-windup freeze upstream). This is the **multi-cycle,
bit-exact equivalent of `circle_limit`** — same `mag2`, the same floor-isqrt
recurrence, and the same truncate-toward-zero rescale, just walked one op per
clock so it stops being `foc_core`'s critical path. `foc_core` instantiates it
in the inv-Park/SVPWM arc.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset; clears state, outputs to 0 |
| `start` | in | 1 | — | — | pulse to latch `vd_in`/`vq_in` and launch the sequencer (honored in S_IDLE) |
| `vd_in` | in | 18 | yes | — | d-axis voltage command (LSB), latched on `start` |
| `vq_in` | in | 18 | yes | — | q-axis voltage command (LSB), latched on `start` |
| `vd_out` | out | 18 | yes | 0 | limited d-axis voltage; holds until next `done` |
| `vq_out` | out | 18 | yes | 0 | limited q-axis voltage; holds until next `done` |
| `sat` | out | 1 | — | 0 | 1 when the vector was rescaled (`mag2 > VLIM²`); freeze for anti-windup |
| `busy` | out | 1 | — | 0 | high from `start` until `done` |
| `done` | out | 1 | — | 0 | 1-cycle pulse, outputs valid |

## Clocking & reset

- **Clock domains:** single `clk`; `start`→`done` handshake.
- **Reset:** async active-low `rst_n` → `state=S_IDLE`, `busy/done/sat=0`,
  `vd_out=vq_out=0`, `div_start=0`.
- **Latency:** variable, depends on saturation.
  - **Not saturated** (`mag2 ≤ VLIM²`): S_IDLE→S_MUL→S_CMP completes there —
    `done` ~**3 clocks** after `start`, outputs = inputs passed through.
  - **Saturated:** S_MUL (square+add) → S_CMP → S_SQRT (**16** isqrt iterations,
    one/clock) → S_DIV (launch both `divider32`) → S_WAIT (~**32** clocks for the
    two lockstep 32-bit divides) → `done`: roughly **53 clocks** after `start`.
  Outputs hold until the next `start`. `start` is sparse (1/PWM period), so the
  walk always finishes inside the sample period.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `V_CIRCLE_LIMIT` | `594` | inscribed-circle radius in duty LSB; `VLIM²` is the saturation threshold and `VLIM` the rescale numerator |

## Formal contract

- **Sim-only** (`formal/sim_only.toml`): sequential, multi-cycle limiter; bit-exact to circle_limit (itself DOCUMENTED - the isqrt+divide is intractable for the open SMT engines) by the cocotb equivalence test (tb_circle_limit_seq, ~1000 cases incl. the saturation boundary) + the in-system FOC regression
- Covered by the simulation tiers, not a machine proof.
- **Assumptions:** single clock; async active-low reset is the design reset;
  `start` pulsed only when idle; `|v| ≤ V_RAW_MAX` (= 2500) as bounded by the
  upstream per-axis PI clamps, so `mag2` and the division operands stay in range.

## Synthesis fit

- **Device:** ECP5. Standalone Fmax **82 MHz** (`synth/fmax_module.py
  circle_limit_seq`); `mag2` and the `|v|·VLIM` operands are built from the
  18-bit inputs so the squares map to single 18×18 hard multipliers (was 64 MHz
  before the 18-bit-operand + S_MUL/S_CMP split). Small: an isqrt recurrence
  plus two reused `divider32` instances.

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** instantiates `divider32` ×2 (the two signed-magnitude divides).
- **Pull it:** `fusesoc run motorloop:ip:circle_limit_seq` (core at repo root).
