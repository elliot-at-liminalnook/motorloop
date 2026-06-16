<!-- SPDX-License-Identifier: MIT -->
# `uart_rx` — 8N1 UART receiver

8N1 UART receiver (8 data bits, no parity, 1 stop) with `UART_DIV` clocks per
bit. A 2-FF synchronizer cleans `rx`; on the start-bit falling edge it arms a
bit counter offset by half a bit so each subsequent bit is **sampled at
mid-bit**. It rejects a false start (line high at the start-bit mid-point),
shifts 8 LSB-first data bits into a register, and on a valid stop bit (line
high) presents the byte on `data` and pulses `valid` for one cycle. **One
`valid` pulse per correctly-framed byte.** Self-contained (no includes, no child
modules); instantiated inside `uart_regfile`.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset; idle, `valid`=0 |
| `rx` | in | 1 | — | — | serial input (idles high; synchronized internally) |
| `data` | out | 8 | — | 0 | received byte, LSB-first on the wire |
| `valid` | out | 1 | — | 0 | 1-cycle pulse on a byte with a good stop bit |

## Clocking & reset

- **Clock domains:** single `clk`; `rx` is asynchronous and passed through a
  2-FF synchronizer before edge detection.
- **Reset:** async active-low `rst_n` → synchronizer high, not busy, `valid`=0.
- **Latency:** one byte = start + 8 data + stop = 10 bit-times, each `UART_DIV`
  clocks; `data`/`valid` appear at the stop-bit sample. `valid` is single-cycle.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `UART_DIV` | `16'd217   // clocks per bit (clk / baud` | clocks per bit = `clk / baud` (e.g. 25 MHz / 115200 ≈ 217) |

## Formal contract

- **Sim-only** (`formal/sim_only.toml`): 8N1 receiver; exercised end-to-end against the host UART model
- Covered by the simulation tiers, not a machine proof.
- **Assumptions:** single `clk`; async `rst_n` is the design reset; the host
  drives standard 8N1 framing at a baud matching `UART_DIV` (within the usual
  mid-bit sampling tolerance); `rx` idles high.

## Synthesis fit

- **Device:** ECP5. Tiny: a 2-FF synchronizer, a 16-bit bit-timer, a 4-bit bit
  index and an 8-bit shift register (`synth/fmax_module.py uart_rx`).

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — drop in `rtl/uart_rx.v`, set `UART_DIV`.
- **Pull it:** `fusesoc run motorloop:ip:uart_rx` (core at repo root).
