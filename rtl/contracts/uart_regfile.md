<!-- SPDX-License-Identifier: MIT -->
# `uart_regfile` — UART command/telemetry register file

Host-facing command/telemetry register file over an 8N1 UART link (realism
stage 6). It owns a `uart_rx` and a `uart_tx` and runs a small FSM on received
bytes. **Write:** `{1'b1, addr[6:0]}`, `data[15:8]`, `data[7:0]` — updates a
control register. **Read:** `{1'b0, addr[6:0]}` — the device replies
`data[15:8]`, `data[7:0]`. Write registers: 0 mode, 1 duty, 2 target speed,
3 align offset, 4/5 open-loop freq word hi/lo, 6/7 ramp increment hi/lo, 8
control (bit0 = `use_uart`, letting UART override the direct ctrl ports). Read
registers echo 0–8 and expose telemetry: 16 speed, 17 {fault,mismatch}, 18
angle, 19 nOCTW count, 20 {configured,sector}, 21 flags (other addresses read
`0xDEAD`). The 16-bit telemetry value is selected combinationally; on a read the
LO byte is **latched with the HI byte** so telemetry can't tear mid-reply. An
inter-byte timeout (E16, `UART_TIMEOUT_CYC`) resets a torn write frame back to
the command state so it can't misread every following byte. **One write applied
or one 2-byte reply per command frame.**

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset → `S_CMD`, registers cleared |
| `uart_rx_pin` | in | 1 | — | — | UART serial in (to internal `uart_rx`) |
| `uart_tx_pin` | out | 1 | — | 1 | UART serial out (from internal `uart_tx`) |
| `use_uart` | out | 1 | — | 0 | reg 8 bit0: UART overrides the direct ctrl ports |
| `r_mode` | out | 2 | — | 0 | reg 0: control mode |
| `r_duty` | out | 16 | — | 0 | reg 1: commanded duty |
| `r_target_speed` | out | 16 | — | 0 | reg 2: target speed |
| `r_align` | out | 12 | — | 0 | reg 3: align offset |
| `r_ol_freq_word` | out | 32 | — | 0 | regs 4/5: open-loop frequency word (hi/lo) |
| `r_ol_ramp_inc` | out | 32 | — | 0 | regs 6/7: open-loop ramp increment (hi/lo) |
| `t_speed` | in | 16 | — | — | telemetry: measured speed (read reg 16) |
| `t_fault_count` | in | 8 | — | — | telemetry: fault count (read reg 17 hi) |
| `t_mismatch_count` | in | 8 | — | — | telemetry: mismatch count (read reg 17 lo) |
| `t_angle` | in | 12 | — | — | telemetry: rotor angle (read reg 18) |
| `t_noctw_count` | in | 16 | — | — | telemetry: nOCTW count (read reg 19) |
| `t_sector` | in | 3 | — | — | telemetry: commutation sector (read reg 20) |
| `t_configured` | in | 1 | — | — | telemetry: DRV configured (read reg 20) |
| `t_flags` | in | 8 | — | — | telemetry: status flags (read reg 21) |

## Clocking & reset

- **Clock domains:** single `clk`. RX/TX sit at one UART baud derived from
  `UART_DIV`; the FSM steps on `rx_valid` and the `tx_busy`/`tx_start`
  handshake.
- **Reset:** async active-low `rst_n` → `S_CMD`, all control registers and
  `use_uart` cleared, idle timer cleared.
- **Latency:** command-byte then 2 data/reply bytes, each a full UART frame
  (10 bit-times × `UART_DIV`); a stalled write frame is recovered after
  `UART_TIMEOUT_CYC` idle clocks.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `UART_DIV` | `16'd217` | clocks per bit = `clk / baud`; threaded to both `uart_rx`/`uart_tx` |
| `UART_TIMEOUT_CYC` | `50000` | E16 inter-byte idle clocks before a torn write frame resets to `S_CMD` |

## Formal contract

- **Sim-only** (`formal/sim_only.toml`): command/telemetry FSM; exercised end-to-end over the UART link
- Covered by the simulation tiers, not a machine proof.
- **Assumptions:** single `clk`; async `rst_n` is the design reset; the host
  obeys the write/read framing above at a baud matching `UART_DIV`; the child
  `uart_rx`/`uart_tx` meet their own contracts (one `valid` per byte, `start`
  honoured only while not busy).

## Synthesis fit

- **Device:** ECP5. Small: a 5-state FSM, the control registers, the
  combinational telemetry mux and a 32-bit timeout counter, plus the `uart_rx`/
  `uart_tx` children (`synth/fmax_module.py uart_regfile`).

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** instantiates `uart_rx` and `uart_tx` (both in the core).
- **Pull it:** `fusesoc run motorloop:ip:uart_regfile` (core at repo root).
