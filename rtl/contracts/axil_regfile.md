<!-- SPDX-License-Identifier: MIT -->
# `axil_regfile` — AXI4-Lite register slave for the controller (protocol PROVEN)

A classic AXI4-Lite register slave that memory-maps the motorloop controller. It
exposes the **same register map as `uart_regfile`** (and `wb_regfile`), so a
robotics SoC can drive mode/duty/speed/open-loop config and read back telemetry
over AXI-Lite instead of UART — `use_axi` selects the bus as the live control
source. Word-addressed, one outstanding transaction; writes accept
`(awaddr,wdata)` together then respond, reads accept `araddr` then drive data.
**AXI handshake legality is FORMALLY PROVEN**: no withdrawn B/R response, read
data/resp held stable while `RVALID && !RREADY`, and every response is `OKAY` —
the wrapper can never violate the AXI-Lite contract.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset |
| `s_awaddr` | in | [ADDR_W-1:0] | — | — | write byte address; `[ADDR_W-1:2]` = register index, `[1:0]` ignored |
| `s_awvalid` | in | 1 | — | — | write-address valid |
| `s_awready` | out | 1 | — | 0 | write-address accepted (1-cycle, with `wready`) |
| `s_wdata` | in | 32 | — | — | write data; only `[15:0]` (or narrower) used per reg |
| `s_wstrb` | in | 4 | — | — | byte strobes — ignored (word-only slave) |
| `s_wvalid` | in | 1 | — | — | write-data valid |
| `s_wready` | out | 1 | — | 0 | write-data accepted (1-cycle, with `awready`) |
| `s_bresp` | out | 2 | — | OKAY | write response, always `OKAY` (`2'b00`) |
| `s_bvalid` | out | 1 | — | 0 | write response valid; held until `bready` |
| `s_bready` | in | 1 | — | — | master accepts write response |
| `s_araddr` | in | [ADDR_W-1:0] | — | — | read byte address; `[ADDR_W-1:2]` = register index, `[1:0]` ignored |
| `s_arvalid` | in | 1 | — | — | read-address valid |
| `s_arready` | out | 1 | — | 0 | read-address accepted (1-cycle) |
| `s_rdata` | out | 32 | — | 0 | read data; held stable while `rvalid && !rready` |
| `s_rresp` | out | 2 | — | OKAY | read response, always `OKAY` (`2'b00`) |
| `s_rvalid` | out | 1 | — | 0 | read data valid; held until `rready` |
| `s_rready` | in | 1 | — | — | master accepts read data |
| `use_axi` | out | 1 | — | 0 | bit0 of `control`: AXI overrides direct `ctrl_*` |
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

- **Clock domains:** single `clk`; everything is registered on its rising edge.
- **Reset:** async active-low `rst_n` → all handshake outputs deasserted
  (`awready`/`wready`/`bvalid`/`arready`/`rvalid` = 0) and every `r_*` control
  register cleared to 0 (`use_axi` = 0, so the bus does not override on reset).
- **Latency / handshake:** one outstanding transaction per channel. A write
  fires when `awvalid && wvalid` and no `bvalid` is pending: `awready`/`wready`
  pulse for one cycle and `bvalid` asserts, held until `bready`. A read fires on
  `arvalid` with no `rvalid` pending: `arready` pulses, `rdata`/`rvalid` register
  next cycle and hold (data stable) until `rready`.

## Register map

Byte address = `index * 4` (`addr[ADDR_W-1:2]` selects the register; the two LSBs
are ignored). The **same map** is shared by `wb_regfile` (word-addressed) and
`uart_regfile`. W = read-write control, R = read-only telemetry; writes to
RO/undefined offsets are ignored but still respond `OKAY`, and reads of undefined
offsets return `0xDEADBEEF`.

| Offset | Index | Name | Access | Fields |
| --- | --- | --- | --- | --- |
| `0x00` | 0 | `mode` | RW | `mode[1:0]` (0 idle, 1 OL, 2 six-step, 3 FOC) |
| `0x04` | 1 | `duty` | RW | `duty[15:0]` open-loop duty compare |
| `0x08` | 2 | `target_speed` | RW | `target_speed[15:0]` (rad/s) |
| `0x0C` | 3 | `align` | RW | `align[11:0]` electrical-angle alignment offset |
| `0x10` | 4 | `ol_freq_word_hi` | RW | open-loop freq word `[31:16]` |
| `0x14` | 5 | `ol_freq_word_lo` | RW | open-loop freq word `[15:0]` |
| `0x18` | 6 | `ol_ramp_inc_hi` | RW | open-loop ramp increment `[31:16]` |
| `0x1C` | 7 | `ol_ramp_inc_lo` | RW | open-loop ramp increment `[15:0]` |
| `0x20` | 8 | `control` | RW | `use_axi[0]`: AXI overrides direct `ctrl_*` |
| `0x40` | 16 | `speed` | RO | `speed[15:0]` measured speed (rad/s) |
| `0x44` | 17 | `fault_counts` | RO | `mismatch[7:0]`, `fault[15:8]` |
| `0x48` | 18 | `angle` | RO | `angle[11:0]` rotor angle |
| `0x4C` | 19 | `noctw_count` | RO | `noctw[15:0]` nOCTW event count |
| `0x50` | 20 | `status` | RO | `sector[2:0]`, `configured[3]` |
| `0x54` | 21 | `flags` | RO | `flags[7:0]` stall/lockout/dead/reverse |

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `ADDR_W` | `8` | AXI byte-address width; must be `>= 7` to cover index 0..21 |

## Formal contract

- **PROVEN** (`formal/manifest.toml`, `formal/bind/axil_regfile_fv.sv`): AXI4-Lite: B/R valid holds until its ready (no withdrawn response); Read data/resp stable while RVALID && !RREADY; Write/read responses are always OKAY.
- **Method:** prove, `engine smtbmc boolector`.
- **Assumptions:** single `clk` is the design clock; async active-low `rst_n` is
  the design reset; the master obeys AXI-Lite (no withdrawn `*valid`). Proof
  checker: `formal/bind/axil_regfile_fv.sv` (k-inductive `$past` assertions on
  the B/R channels, plus a non-vacuity cover that a read completes).

## Synthesis fit

- **Device:** ECP5. Small word-only register slave — per-channel handshake FSM
  bits plus the `r_*`/telemetry read mux; no multipliers or deep logic, so it is
  not on the critical path (well above the system clock). Measure standalone via
  `synth/fmax_module.py axil_regfile`.

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — standalone bus wrapper, no child modules. The matching
  IP-XACT component is `ip-xact/motorloop.axil_regfile.xml` (the register map is
  generated from the one source in `scripts/gen_ipxact.py`).
- **Pull it:** `fusesoc run motorloop:ip:axil_regfile` (core at repo root).
