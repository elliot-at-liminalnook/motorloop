<!-- SPDX-License-Identifier: MIT -->
# `wb_regfile` — Wishbone B4 register slave for the controller (protocol PROVEN)

A classic Wishbone B4 (registered-ack) register slave that memory-maps the
motorloop controller — the open-SoC default bus for LiteX / RISC-V robotics
stacks. It exposes the **same register map as `axil_regfile` and `uart_regfile`**
(here word-addressed: `wb_adr` is the register index), so a SoC can drive
mode/duty/speed/open-loop config and read back telemetry over Wishbone instead
of UART; `use_wb` selects the bus as the live control source. One outstanding
transaction; a strobe (`cyc && stb`) is accepted when no ack is pending and
answered with a single-cycle `ack`. **Wishbone handshake legality is FORMALLY
PROVEN**: `ack` only follows an accepted strobe (no spurious ack) and is a
single-cycle pulse — the wrapper can never violate the B4 contract.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset |
| `wb_adr` | in | [ADDR_W-1:0] | — | — | word address = register index; upper bits unused |
| `wb_dat_w` | in | 32 | — | — | write data; only `[15:0]` (or narrower) used per reg |
| `wb_dat_r` | out | 32 | — | 0 | read data, valid with `ack` on a read cycle |
| `wb_we` | in | 1 | — | — | 1 = write, 0 = read |
| `wb_stb` | in | 1 | — | — | strobe: this slave is selected |
| `wb_cyc` | in | 1 | — | — | bus cycle in progress |
| `wb_ack` | out | 1 | — | 0 | single-cycle ack pulse for an accepted strobe |
| `use_wb` | out | 1 | — | 0 | bit0 of `control`: Wishbone overrides direct `ctrl_*` |
| `r_mode` | out | 2 | — | 0 | control mode (0 idle, 1 OL, 2 six-step, 3 FOC) |
| `r_duty` | out | 16 | — | 0 | open-loop duty compare |
| `r_target_speed` | out | 16 | — | 0 | speed target (rad/s) |
| `r_align` | out | 12 | — | 0 | electrical-angle alignment offset |
| `r_ol_freq_word` | out | 32 | — | 0 | open-loop frequency word (hi/lo halves) |
| `r_ol_ramp_inc` | out | 32 | — | 0 | open-loop ramp increment (hi/lo halves) |
| `t_speed` | in | 16 | — | — | measured speed telemetry (rad/s) |
| `t_fault_count` | in | 8 | — | — | fault event count |
| `t_mismatch_count` | in | 8 | — | — | redundant-channel mismatch count |
| `t_angle` | in | 12 | — | — | rotor angle (12-bit) |
| `t_noctw_count` | in | 16 | — | — | nOCTW event count |
| `t_sector` | in | 3 | — | — | commutation sector |
| `t_configured` | in | 1 | — | — | controller configured flag |
| `t_flags` | in | 8 | — | — | stall/lockout/dead/reverse flags |

## Clocking & reset

- **Clock domains:** single `clk`; `ack` and `dat_r` are registered on its
  rising edge.
- **Reset:** async active-low `rst_n` → `ack` and `dat_r` cleared and every `r_*`
  control register cleared to 0 (`use_wb` = 0, so the bus does not override on
  reset).
- **Latency / handshake:** one outstanding transaction. A strobe is accepted
  when `cyc && stb && !ack`; the next cycle `ack` pulses for exactly one clock
  (classic registered ack), with the written register updated or `dat_r` driven
  for the read. No wait-states, no pipelining.

## Register map

`wb_adr` is the **word** address (register index, not byte offset). The **same
map** is shared by `axil_regfile` (byte-addressed, `index*4`) and `uart_regfile`.
W = read-write control, R = read-only telemetry; writes to RO/undefined indices
are ignored, and reads of undefined indices return `0xDEADBEEF`.

| Index | Byte off. | Name | Access | Fields |
| --- | --- | --- | --- | --- |
| 0 | `0x00` | `mode` | RW | `mode[1:0]` (0 idle, 1 OL, 2 six-step, 3 FOC) |
| 1 | `0x04` | `duty` | RW | `duty[15:0]` open-loop duty compare |
| 2 | `0x08` | `target_speed` | RW | `target_speed[15:0]` (rad/s) |
| 3 | `0x0C` | `align` | RW | `align[11:0]` electrical-angle alignment offset |
| 4 | `0x10` | `ol_freq_word_hi` | RW | open-loop freq word `[31:16]` |
| 5 | `0x14` | `ol_freq_word_lo` | RW | open-loop freq word `[15:0]` |
| 6 | `0x18` | `ol_ramp_inc_hi` | RW | open-loop ramp increment `[31:16]` |
| 7 | `0x1C` | `ol_ramp_inc_lo` | RW | open-loop ramp increment `[15:0]` |
| 8 | `0x20` | `control` | RW | `use_wb[0]`: Wishbone overrides direct `ctrl_*` |
| 16 | `0x40` | `speed` | RO | `speed[15:0]` measured speed (rad/s) |
| 17 | `0x44` | `fault_counts` | RO | `mismatch[7:0]`, `fault[15:8]` |
| 18 | `0x48` | `angle` | RO | `angle[11:0]` rotor angle |
| 19 | `0x4C` | `noctw_count` | RO | `noctw[15:0]` nOCTW event count |
| 20 | `0x50` | `status` | RO | `sector[2:0]`, `configured[3]` |
| 21 | `0x54` | `flags` | RO | `flags[7:0]` stall/lockout/dead/reverse |

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `ADDR_W` | `8` | Wishbone word-address width; must cover index 0..21 |

## Formal contract

- **PROVEN** (`formal/manifest.toml`, `formal/bind/wb_regfile_fv.sv`): Wishbone B4: ACK only follows an accepted strobe (no spurious ACK); ACK is a single-cycle pulse (classic registered ack).
- **Method:** prove, `engine smtbmc boolector`.
- **Assumptions:** single `clk` is the design clock; async active-low `rst_n` is
  the design reset; the master obeys Wishbone B4 (`stb` implies `cyc`). Proof
  checker: `formal/bind/wb_regfile_fv.sv` (k-inductive `$past` assertions: ack
  only after an accepted strobe, ack is a single-cycle pulse).

## Synthesis fit

- **Device:** ECP5. Very small classic Wishbone slave — one ack flop plus the
  register write decode and `dat_r` read mux; no multipliers, not on the
  critical path (well above the system clock). Measure standalone via
  `synth/fmax_module.py wb_regfile`.

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — standalone bus wrapper, no child modules. The matching
  IP-XACT component is `ip-xact/motorloop.wb_regfile.xml` (the register map is
  generated from the one source in `scripts/gen_ipxact.py`).
- **Pull it:** `fusesoc run motorloop:ip:wb_regfile` (core at repo root).
