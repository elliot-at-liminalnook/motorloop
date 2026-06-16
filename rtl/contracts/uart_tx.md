<!-- SPDX-License-Identifier: MIT -->
# `uart_tx` — 8N1 UART transmitter

8N1 UART transmitter (8 data bits, no parity, 1 stop) with `UART_DIV` clocks per
bit. `tx` idles high; on `start` (accepted only while idle) it latches `data`,
drives the start bit low, then shifts 8 LSB-first data bits and a high stop bit,
each held for `UART_DIV` clocks, and drops `busy`. **One frame per accepted
`start`; `start` is ignored while `busy`.** Self-contained (no includes, no
child modules); instantiated inside `uart_regfile`.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset; `tx`=1, `busy`=0 |
| `start` | in | 1 | — | — | latch `data` and begin a frame (ignored while `busy`) |
| `data` | in | 8 | — | — | byte to send, LSB-first on the wire |
| `busy` | out | 1 | — | 0 | high from `start` accept until the stop bit completes |
| `tx` | out | 1 | — | 1 | serial output (idles high) |

## Clocking & reset

- **Clock domains:** single `clk`; `start`/`busy` handshake.
- **Reset:** async active-low `rst_n` → `tx`=1 (idle), `busy`=0.
- **Latency:** one byte = start + 8 data + stop = 10 bit-times, each `UART_DIV`
  clocks; `busy` falls one bit-time after the stop bit is driven.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `UART_DIV` | `16'd217   // clocks per bit (clk / baud` | clocks per bit = `clk / baud` (e.g. 25 MHz / 115200 ≈ 217) |

## Formal contract

- **Sim-only** (`formal/sim_only.toml`): 8N1 transmitter; exercised end-to-end against the host UART model
- Covered by the simulation tiers, not a machine proof.
- **Assumptions:** single `clk`; async `rst_n` is the design reset; the host
  receives standard 8N1 framing at a baud matching `UART_DIV`; `start` is not
  pulsed while `busy`.

## Synthesis fit

- **Device:** ECP5. Tiny: a 16-bit bit-timer, a 4-bit bit index and an 8-bit
  shift register (`synth/fmax_module.py uart_tx`).

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — drop in `rtl/uart_tx.v`, set `UART_DIV`.
- **Pull it:** `fusesoc run motorloop:ip:uart_tx` (core at repo root).
