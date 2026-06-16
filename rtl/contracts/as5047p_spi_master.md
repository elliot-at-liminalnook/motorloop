<!-- SPDX-License-Identifier: MIT -->
# `as5047p_spi_master` — AS5047P SPI angle reader (framing PROVEN)

Free-running SPI master for the **AS5047P** magnetic angle sensor: **mode 1**
(CPOL=0, CPHA=1), 16-bit MSB-first frames — the same timing class as the
DRV8301 master, so it reuses `DRV_SPI_DIV`. It continuously streams the read
command for `ANGLECOM` (register `0x3FFF`, the DAEC-compensated angle; the
command word with its even-parity PARC bit is `0xFFFF`) and latches each
response. The AS5047P is a pipelined-read device, so each frame returns the
angle requested by the previous frame; streaming the same command yields a
fresh angle every frame. A response is `PARD[15] | EF[14] | ANGLE[13:0]` and is
**accepted only when even parity over bits[14:0] equals PARD and EF=0**; the top
12 of the 14-bit angle drive the shared 12-bit `angle` bus, and `new_sample`
pulses once per accepted angle. On parity/EF error (e.g. magnet loss)
`angle_valid` drops and no pulse is emitted. **No `start` — it self-restarts;
FSM legality and `new_sample` well-formedness are proven.**

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset; idle, `ncs`=1, no valid angle |
| `angle` | out | 12 | — | 0 | top 12 bits of the 14-bit ANGLECOM (held on error) |
| `angle_valid` | out | 1 | — | 0 | last frame passed parity and EF=0 |
| `new_sample` | out | 1 | — | 0 | 1-cycle pulse on each freshly accepted angle |
| `sclk` | out | 1 | — | 0 | divided serial clock (idles low, CPOL=0) |
| `mosi` | out | 1 | — | 0 | read command to SDI, launched on the leading edge |
| `ncs` | out | 1 | — | 1 | active-low chip select; low only during a frame |
| `miso` | in | 1 | — | — | SDO from AS5047P, sampled on the high half |

## Clocking & reset

- **Clock domains:** single `clk`; no external handshake — the master loops
  S_IDLE→…→S_GAP→S_IDLE on its own, emitting `new_sample` per accepted frame.
- **Reset:** async active-low `rst_n` → `S_IDLE`, `ncs`=1, `sclk`=0,
  `angle_valid`=0.
- **Latency:** one frame ≈ `(16+2)·DRV_SPI_DIV` clocks
  (`SCLK = clk / DRV_SPI_DIV`); the angle decoded from a given frame is the
  command issued one frame earlier (pipelined read). `new_sample` is a
  single-cycle strobe.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `DRV_SPI_DIV` | `8'd16   // SCLK = clk / DRV_SPI_DIV (<= 255` | SCLK divider (shared with the DRV master); each half-period is `DRV_SPI_DIV>>1` clocks |

## Formal contract

- **PROVEN** (`formal/manifest.toml`, `formal/bind/as5047p_spi_master_fv.sv`): FSM legality: the SPI master state register only ever holds one of the 6 defined states S_IDLE..S_GAP; new_sample is a well-formed single-cycle pulse - one event per latched angle.
- **Method:** prove, `engine smtbmc boolector`.
- **Assumptions:** single `clk`; async `rst_n` is the design reset; the bench
  AS5047P model presents SDO with the mode-1 launch/sample timing and pipelined
  read (response = previous command) the master assumes; `DRV_SPI_DIV` even,
  `>= 2`.

## Synthesis fit

- **Device:** ECP5. Small: 16-bit shift/`rx` registers, an 8-bit divider, a
  6-state FSM plus the parity/EF check (`synth/fmax_module.py as5047p_spi_master`).

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — drop in `rtl/as5047p_spi_master.v`, set `DRV_SPI_DIV`.
- **Pull it:** `fusesoc run motorloop:ip:as5047p_spi_master` (core at repo root).
