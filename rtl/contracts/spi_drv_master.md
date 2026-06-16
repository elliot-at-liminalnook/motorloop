<!-- SPDX-License-Identifier: MIT -->
# `spi_drv_master` — SPI master, mode-1 16-bit (gate driver)

SPI master for the **DRV8301** gate driver: **mode 1** (CPOL=0, CPHA=1),
16-bit MSB-first frames. On `start` it latches `tx`, drops `ncs`, then walks a
divided SCLK: MOSI is launched on each rising (leading) edge and MISO is
sampled on the high half (the slave drove SDO on the rising edge and samples
SDI on the falling edge). After 16 bits it raises `ncs` with hold time and
enforces a minimum inter-frame gap, then pulses `done` with `rx` valid. **One
frame per `start`; framing and inter-frame gap are guaranteed by the FSM.**
Self-contained (no includes, no child modules). Used by `drv_manager` to
write/read CR1/CR2/SR1/SR2.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset; idle, `ncs`=1, gates of SPI quiet |
| `start` | in | 1 | — | — | latch `tx` and begin a frame (ignored while `busy`) |
| `tx` | in | 16 | — | — | command/data word, MSB-first on the wire |
| `busy` | out | 1 | — | 0 | high from `start` accept until the inter-frame gap ends |
| `done` | out | 1 | — | 0 | 1-cycle pulse at frame end; `rx` is valid this cycle |
| `rx` | out | 16 | — | 0 | response word sampled on MISO, MSB-first |
| `sclk` | out | 1 | — | 0 | divided serial clock (idles low, CPOL=0) |
| `mosi` | out | 1 | — | 0 | data to DRV8301 SDI, launched on the leading edge |
| `ncs` | out | 1 | — | 1 | active-low chip select; low only during a frame |
| `miso` | in | 1 | — | — | data from DRV8301 SDO, sampled mid-high |

## Clocking & reset

- **Clock domains:** single `clk`; `start`/`done`/`busy` handshake.
- **Reset:** async active-low `rst_n` → `S_IDLE`, `ncs`=1, `sclk`=0, `busy`=0.
- **Latency:** start→done is one full frame: lead (`HALF_DIV`) + 16 bits of
  two half-periods each + tail + gap, i.e. roughly `(16+2)·DRV_SPI_DIV` clocks
  with `SCLK = clk / DRV_SPI_DIV`. `done` is a single-cycle strobe.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `DRV_SPI_DIV` | `8'd16   // SCLK = clk / DRV_SPI_DIV (<= 255` | SCLK divider; each half-period is `DRV_SPI_DIV>>1` clocks (`<= 255`) |

## Formal contract

- **Sim-only** (`formal/sim_only.toml`): SPI master; protocol exercised against the golden DRV8301 model in the co-sim
- Covered by the simulation tiers, not a machine proof.
- **Assumptions:** single `clk`; async `rst_n` is the design reset; the bench
  DRV8301 model supplies MISO with the mode-1 (CPHA=1) launch/sample timing and
  one-cycle feedback delay the master expects; `start` is not pulsed while
  `busy`; `DRV_SPI_DIV` is even and `>= 2`.

## Synthesis fit

- **Device:** ECP5. Small: a 16-bit shift register, 16-bit `rx`, an 8-bit
  divider and a 6-state FSM; comfortably fast standalone (`synth/fmax_module.py spi_drv_master`).

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — drop in `rtl/spi_drv_master.v`, set `DRV_SPI_DIV`.
- **Pull it:** `fusesoc run motorloop:ip:spi_drv_master` (core at repo root).
