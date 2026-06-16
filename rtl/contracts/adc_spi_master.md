<!-- SPDX-License-Identifier: MIT -->
# `adc_spi_master` — SPI master for the MCP3208 12-bit ADC

SPI master for the **MCP3208** 12-bit SAR ADC in **mode 0,0**. On `start` it
latches a single-ended command `{start=1, SGL=1, channel[2:0]}`, drops `ncs`,
and clocks a fixed **20-clock** frame: command bits are set up during the SCLK
low half (device samples DIN on rising edges), and the 12 result bits B11..B0
are shifted out by the device on falling edges and collected on the high halves
of clocks 8..19 (accounting for the one-cycle bench feedback delay). It then
raises `ncs`, holds a CS-high gap (`>= 500 ns` between conversions), and pulses
`done` with the 12-bit `code` valid. **One conversion per `start`; the 20-clock
frame and inter-conversion gap are guaranteed.** Driven by `adc_sequencer`.
Self-contained (no includes, no child modules).

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset; idle, `ncs`=1 |
| `start` | in | 1 | — | — | latch `channel` and begin a conversion (ignored while `busy`) |
| `channel` | in | 3 | — | — | MCP3208 single-ended input 0..7 |
| `busy` | out | 1 | — | 0 | high from `start` accept until the CS-high gap ends |
| `done` | out | 1 | — | 0 | 1-cycle pulse at frame end; `code` valid this cycle |
| `code` | out | 12 | — | 0 | latest 12-bit conversion result (B11..B0) |
| `sclk` | out | 1 | — | 0 | divided serial clock (idles low, CPOL=0) |
| `mosi` | out | 1 | — | 0 | command bits to MCP3208 DIN, set up on the low half |
| `ncs` | out | 1 | — | 1 | active-low chip select; low only during a frame |
| `miso` | in | 1 | — | — | DOUT from MCP3208, sampled on the high half |

## Clocking & reset

- **Clock domains:** single `clk`; `start`/`done`/`busy` handshake.
- **Reset:** async active-low `rst_n` → `S_IDLE`, `ncs`=1, `sclk`=0, `busy`=0.
- **Latency:** start→done is the fixed 20-clock SPI frame plus lead and CS-high
  gap, each SCLK half-period being `ADC_SPI_DIV>>1` clocks
  (`SCLK = clk / ADC_SPI_DIV`); `done` is a single-cycle strobe.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `ADC_SPI_DIV` | `8'd26   // SCLK = clk / ADC_SPI_DIV (<= 255` | SCLK divider; each half-period is `ADC_SPI_DIV>>1` clocks (`<= 255`) |

## Formal contract

- **Sim-only** (`formal/sim_only.toml`): SPI master; protocol exercised against the golden MCP3208 model in the co-sim
- Covered by the simulation tiers, not a machine proof.
- **Assumptions:** single `clk`; async `rst_n` is the design reset; the bench
  MCP3208 model presents DOUT with the mode-0,0 timing and one-cycle feedback
  delay this master assumes (B11..B0 launched on the fallings of clocks 7..18);
  `start` is not pulsed while `busy`; `ADC_SPI_DIV` is even and `>= 2`.

## Synthesis fit

- **Device:** ECP5. Small: a 12-bit result register, 5-bit command/cycle
  counters, an 8-bit divider and a 5-state FSM (`synth/fmax_module.py adc_spi_master`).

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — drop in `rtl/adc_spi_master.v`, set `ADC_SPI_DIV`.
- **Pull it:** `fusesoc run motorloop:ip:adc_spi_master` (core at repo root).
