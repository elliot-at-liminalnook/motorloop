<!-- SPDX-License-Identifier: MIT -->
# `speed_meter` — Sector-edge speed measurement

Measures rotor speed from commutation-sector edges. Because the AS5600's
PWM-frame latency (~1 ms) quantizes individual edge timing, the period is timed
across a **full electrical revolution (6 edges)** and converted by
`speed = 6·SPEED_NUM / period_cycles` (the divide is a reused `divider32`,
result saturated to 16 bits). A per-window direction vote sets `reverse` (E5),
and watchdog timeouts force zero speed at standstill so the speed PI keeps
updating. Drives the `speed`/`reverse`/`update` inputs of `speed_pi` /
`speed_iq_pi`.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset; all state/outputs → 0 |
| `sector` | in | 3 | no | — | current commutation sector 0..5 (e.g. from `open_loop_ramp` or the angle decode) |
| `speed` | out | 16 | no | 0 | measured speed magnitude (rad/s), saturated; updated on `update` |
| `reverse` | out | 1 | — | 0 | 1 when the 6-edge window ran backward (sector sequence decreasing) |
| `speed_valid` | out | 1 | — | 0 | latched once a measurement (or a timeout) has occurred |
| `update` | out | 1 | — | 0 | 1-cycle pulse on each new measurement (divide done, or timeout) |

## Clocking & reset

- **Clock domains:** single `clk`.
- **Reset:** async active-low `rst_n` → counters/votes cleared, `speed=0`,
  `reverse=0`, `speed_valid=0`, `update=0`, `div_den=1`.
- **Latency:** event-driven, not fixed. A free-running cycle counter `cnt`
  accumulates between sector edges; on every 6th edge the window period is sent
  to `divider32` and `update` pulses ~**32 clocks** later when the divide
  completes (`speed` then valid). Two watchdogs force an immediate zero-speed
  `update`: a quick **20 ms** timeout (`CLK_HZ/50`) when no edges arrived, and a
  hard **250 ms** timeout (`CLK_HZ/4`) abandoning a stalled partial window.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `CLK_HZ` | `25000000` | system clock in Hz; sets the timeout windows (`CLK_HZ/50`, `CLK_HZ/4`) |
| `SPEED_NUM` | `6544985` | speed-scale numerator = `CLK_HZ·2π / (6·pole_pairs)`; `6·SPEED_NUM` is the divide dividend |

## Formal contract

- **Sim-only** (`formal/sim_only.toml`): sector-edge speed measurement; covered by the speed sim tier
- Covered by the simulation tiers, not a machine proof.
- **Assumptions:** single clock; async active-low reset is the design reset;
  `sector` changes by ±1 (mod 6) at each edge (larger jumps are not voted);
  `CLK_HZ`/`SPEED_NUM` match the deployment clock and pole-pair count.

## Synthesis fit

- **Device:** ECP5 (part of the system fit). Small: a 32-bit cycle counter, edge
  bookkeeping, direction-vote counters, plus one reused `divider32`; not on the
  system critical path.

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** instantiates `divider32` (the period→rad/s divide).
- **Pull it:** `fusesoc run motorloop:ip:speed_meter` (core at repo root).
