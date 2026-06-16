<!-- SPDX-License-Identifier: MIT -->
# `axis_sampler` — AXI4-Stream telemetry sampler (protocol PROVEN)

An AXI4-Stream telemetry master: on each `sample` strobe (the FOC `foc_valid` /
six-step cadence) it latches one packed 32-bit beat of controller state and
presents it on an AXI-Stream master port, so a robotics SoC can DMA the
controller's telemetry. Every beat is a complete sample (`m_tlast = 1`).
Backpressure-safe by construction: while a beat is unaccepted
(`TVALID && !TREADY`) `TDATA` is held stable and any new `sample` is dropped and
counted in `overflow_count`, so the stream is never corrupted. **AXI4-Stream
handshake legality is FORMALLY PROVEN**: once `TVALID` asserts it holds until
`TREADY`, and `TDATA` is stable while the beat is unaccepted — the drop logic
can never break the contract a downstream sink/DMA relies on.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset |
| `sample` | in | 1 | — | — | 1-cycle strobe: latch a new telemetry beat |
| `in_speed` | in | 16 | — | — | measured speed (rad/s) → `TDATA[15:0]` |
| `in_angle` | in | 12 | — | — | rotor angle → `TDATA[27:16]` |
| `in_sector` | in | 3 | — | — | commutation sector → `TDATA[31:29]` |
| `in_configured` | in | 1 | — | — | configured flag → `TDATA[28]` |
| `m_tdata` | out | 32 | — | 0 | packed beat; stable while `tvalid && !tready` |
| `m_tvalid` | out | 1 | — | 0 | beat valid; held until `tready` |
| `m_tready` | in | 1 | — | — | downstream sink/DMA accepts the beat |
| `m_tlast` | out | 1 | — | — | tied 1 — every beat is a complete sample |
| `overflow_count` | out | 16 | — | 0 | saturating count of dropped samples (sink too slow) |

## Clocking & reset

- **Clock domains:** single `clk`; `TDATA`/`TVALID`/`overflow_count` are all
  registered on its rising edge.
- **Reset:** async active-low `rst_n` → `TDATA` = 0, `TVALID` = 0,
  `overflow_count` = 0 (no spurious beat after reset release).
- **Latency / handshake:** one beat in flight. A `sample` while the port is free
  (or being consumed this cycle) registers the new beat and asserts `TVALID` the
  next clock; a `sample` while a beat is still pending is dropped and bumps the
  saturating `overflow_count` (caps at `0xFFFF`). A beat is consumed when
  `TVALID && TREADY`.

## Telemetry beat

One 32-bit beat per accepted sample, `m_tlast` tied high:

| Bits | Field | Source |
| --- | --- | --- |
| `[15:0]` | `speed` | `in_speed` — measured speed (rad/s) |
| `[27:16]` | `angle` | `in_angle` — rotor angle (12-bit) |
| `[28]` | `configured` | `in_configured` |
| `[31:29]` | `sector` | `in_sector` — commutation sector |

Note this is an interface-only master (no memory map); its IP-XACT describes the
stream port alone.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| _(none)_ | | fixed 32-bit beat; no parameters |

## Formal contract

- **PROVEN** (`formal/manifest.toml`, `formal/bind/axis_sampler_fv.sv`): AXI4-Stream: TVALID holds until TREADY (no withdrawn beat); TDATA stable while TVALID && !TREADY (backpressure-safe drop).
- **Method:** prove, `engine smtbmc boolector`.
- **Assumptions:** single `clk` is the design clock; async active-low `rst_n` is
  the design reset; `TREADY` may go low at any time (backpressure is the case
  under test). Proof checker: `formal/bind/axis_sampler_fv.sv` (k-inductive
  `$past`: `TVALID` held until `TREADY`, `TDATA` stable while unaccepted, plus a
  non-vacuity cover that a beat actually streams).

## Synthesis fit

- **Device:** ECP5. Tiny — a 32-bit beat register, the `tvalid` flop, and a
  16-bit saturating overflow counter; no multipliers, not on the critical path
  (well above the system clock). Measure standalone via
  `synth/fmax_module.py axis_sampler`.

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — standalone bus wrapper, no child modules. The matching
  IP-XACT component is `ip-xact/motorloop.axis_sampler.xml` (an interface-only
  AXI-Stream master, no register map).
- **Pull it:** `fusesoc run motorloop:ip:axis_sampler` (core at repo root).
