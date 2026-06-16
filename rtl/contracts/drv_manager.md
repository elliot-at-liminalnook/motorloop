<!-- SPDX-License-Identifier: MIT -->
# `drv_manager` — Gate-driver bring-up/SPI-config/fault FSM (legality PROVEN)

The **DRV8301** (and hardware-configured DRV8302) management state machine. It
sequences power-up: a settle window, `en_gate` assert with an EN-ready wait,
then a `dc_cal` window for amplifier offset capture. It then SPI-configures CR1
(current-limit / 6-PWM) and CR2 (amp gain / report OT+OC) and **verifies them by
read-back**, clears SR1/SR2, and enters RUN with `configured` high. In RUN it
runs a refresh watchdog — re-reading CR1/CR2 every `DRV_REFRESH_CYC`; a mismatch
(slva552 silent-brownout reset) is rewritten and counted. `nfault_sync` (or a
mismatch *while* a fault is asserted) triggers the fault path: kill gates, a
quick-reset `en_gate` pulse (`< 10 µs`), reconfigure, resume — faults counted.
Two latching escalations: **E13 dead** (too many verify failures while otherwise
healthy → `drv_dead`) and **E20 lockout** (too many recoveries without a healthy
run → `locked_out`); both clear only on `lockout_clear`. `gate_kill =
!configured`, so gates are forced off until fully configured. Reads are
pipelined N+1: each verify checks the *second* read frame's response.
**The state register is proven to hold only the 17 defined states.** Drives an
external `spi_drv_master`.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset → `S_PWRUP`, gates off |
| `nfault_sync` | in | 1 | — | — | synchronized DRV nFAULT, active low; low = fault |
| `lockout_clear` | in | 1 | — | — | host ack (`>= 100 ms` idle); clears DEAD/LOCKOUT |
| `hw_mode` | in | 1 | — | — | DRV8302 hardware-config: skip SPI register steps |
| `en_gate` | out | 1 | — | 0 | DRV EN_GATE; pulses low for the quick-reset |
| `dc_cal` | out | 1 | — | 0 | DC-calibration window (amp offset capture) |
| `configured` | out | 1 | — | 0 | registers verified; in RUN |
| `gate_kill` | out | 1 | — | — | `= !configured`; forces PWM gates off until configured |
| `drv_dead` | out | 1 | — | 0 | E13: peripheral declared dead (latched) |
| `locked_out` | out | 1 | — | 0 | E20: repeated-fault lockout (latched) |
| `fault_count` | out | 8 | — | 0 | saturating count of fault events |
| `mismatch_count` | out | 8 | — | 0 | saturating count of verify mismatches |
| `spi_start` | out | 1 | — | 0 | 1-cycle start to `spi_drv_master` |
| `spi_tx` | out | 16 | — | 0 | command/data word for the SPI master |
| `spi_busy` | in | 1 | — | — | SPI master busy |
| `spi_done` | in | 1 | — | — | SPI master frame-done strobe |
| `spi_rx` | in | 16 | — | — | SPI response word (for read-back verify) |

## Clocking & reset

- **Clock domains:** single `clk`; the SPI side is a
  `spi_start`/`spi_busy`/`spi_done`/`spi_rx` handshake to `spi_drv_master`.
- **Reset:** async active-low `rst_n` → `S_PWRUP`, `en_gate`=0, `configured`=0,
  all counters/flags cleared.
- **Latency:** seconds-scale at power-up (settle `CLK_HZ/1000`, EN-ready
  `EN_READY_CYCLES`, DC-cal `CLK_HZ/5000`), then per-frame SPI handshakes for
  config/verify; the RUN refresh fires every `DRV_REFRESH_CYC`; the quick-reset
  sequence is ~10 µs kill (256 clk) + `QUICK_RESET_CYC` + `CLK_HZ/5000` settle.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `CLK_HZ` | `25000000` | system clock; derives the settle (1 ms), DC-cal (200 µs) and post-reset (200 µs) windows |
| `EN_READY_CYCLES` | `300000` | wait after EN_GATE before DC-cal (gate-supply ready) |
| `QUICK_RESET_CYC` | `100` | EN_GATE low time during the fault quick-reset pulse |
| `DRV_REFRESH_CYC` | `25000` | RUN-state interval between CR1/CR2 read-back refreshes |
| `HEALTHY_RUN_CYC` | `2500000` | sustained-healthy interval that clears the repeated-fault counter (E20) |
| `LOCKOUT_N` | `4'd4` | recoveries without a healthy run before latching LOCKOUT (E20) |
| `DRV_DEAD_N` | `4'd8` | consecutive verify failures before declaring DEAD (E13) |
| `OC_ADJ_CODE` | `16` | over-current adjust code written into CR1 |
| `AMP_GAIN_CODE` | `0` | current-amplifier gain code written into CR2 |

## Formal contract

- **PROVEN** (`formal/manifest.toml`, `formal/bind/drv_manager_fv.sv`): FSM legality: the state register only ever holds one of the 17 defined states (no undefined/illegal state).
- **Method:** prove, `engine smtbmc boolector`.
- **Assumptions:** single `clk`; async `rst_n` is the design reset; `spi_busy`/
  `spi_done`/`spi_rx` obey the `spi_drv_master` handshake (one `done` per
  `start`, `rx` valid at `done`); `nfault_sync` is already synchronized to
  `clk`; the bench DRV model echoes the written CR1/CR2 on a clean read-back.

## Synthesis fit

- **Device:** ECP5. Modest control logic: a 17-state FSM, several 32-bit timers,
  small saturating counters and the frame helpers — no datapath
  (`synth/fmax_module.py drv_manager`).

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none internally — pairs with an external `spi_drv_master`
  via the SPI handshake ports.
- **Pull it:** `fusesoc run motorloop:ip:drv_manager` (core at repo root).
