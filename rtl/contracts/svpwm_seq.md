<!-- SPDX-License-Identifier: MIT -->
# `svpwm_seq` — Sequential SVPWM (multi-cycle, bit-exact to svpwm)

Space-vector PWM by min/max common-mode injection: from the stationary
`(valpha, vbeta)` it forms the three phase references, subtracts the common-mode
`cm = (max+min)>>1`, and clamps each leg to `[0, PWM_HALF_PERIOD]` to produce the
three 16-bit per-leg duty compares. This is the **multi-cycle, bit-exact
equivalent of `svpwm`** — identical arithmetic (the `(√3/2)·vβ` product, the
ra/rb/rc references, the 3-way min/max, the per-leg saturating clamp), just split
across three registered stages so it stops being `foc_core`'s longest arc.
`foc_core` instantiates it at the end of the datapath.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset; clears state, `duty3`→0 |
| `start` | in | 1 | — | — | pulse to latch `valpha`/`vbeta` and launch the sequencer (honored in S_IDLE) |
| `valpha` | in | 18 | yes | — | α-axis voltage (LSB), latched on `start` |
| `vbeta` | in | 18 | yes | — | β-axis voltage (LSB), latched on `start` |
| `duty3` | out | 48 | no | 0 | `{leg C, leg B, leg A}`, 16 bits each, each in `[0, HALF]`; holds until next `done` |
| `busy` | out | 1 | — | 0 | high from `start` until `done` |
| `done` | out | 1 | — | 0 | 1-cycle pulse, `duty3` valid |

## Clocking & reset

- **Clock domains:** single `clk`; `start`→`done` handshake.
- **Reset:** async active-low `rst_n` → `state=S_IDLE`, `busy/done=0`, `duty3=0`.
- **Latency:** fixed **3 clocks** start→done. `start` latches the inputs and
  enters S_REF (references) → S_CM (common-mode) → S_CLAMP (per-leg clamp,
  registers `duty3`, pulses `done`). `duty3` holds until the next `start`;
  `start` is sparse (1/PWM period), so the walk always finishes in-period.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `PWM_HALF_PERIOD` | `625` | half PWM period; the duty `CENTER` (`HALF>>1`) and the upper clamp on each leg |

## Formal contract

- **Sim-only** (`formal/sim_only.toml`): sequential, multi-cycle SVPWM (FOC Fmax task 4); bit-exact to the PROVEN svpwm by the cocotb equivalence test (tb_svpwm_seq, ~670 cases incl. the clamp boundary) + the in-system FOC regression
- Covered by the simulation tiers, not a machine proof.
- **Assumptions:** single clock; async active-low reset is the design reset;
  `start` pulsed only when idle; `valpha`/`vbeta` are post-limit voltages (within
  the inscribed circle), so the per-leg references stay in the clamp's range.

## Synthesis fit

- **Device:** ECP5. Standalone Fmax **95 MHz** (`synth/fmax_module.py
  svpwm_seq`); this block was previously `foc_core`'s cap as a combinational
  path. Small: one 18×16 product, a 3-way min/max, and a per-leg clamp over
  three pipeline registers.

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — self-contained.
- **Pull it:** `fusesoc run motorloop:ip:svpwm_seq` (core at repo root).
