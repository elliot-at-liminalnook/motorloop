<!-- SPDX-License-Identifier: MIT -->
# `open_loop_ramp` — Open-loop commutation frequency ramp

Open-loop forced-commutation source: a 32-bit electrical-phase accumulator
advanced each clock by a frequency word that ramps linearly toward
`target_freq_word`. One electrical revolution maps to 2³², and the six-step
`sector = floor(phase·6 / 2³²)` is decoded by comparators against the
sixth-of-turn thresholds. The frequency word steps toward target by at most
`ramp_increment` once every 4096 clocks (one `ramp_tick` wrap), giving a smooth
spin-up; it never overshoots target. Used to bring the motor up to speed before
closed-loop control engages.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset; `phase`/`freq_word`/`ramp_tick`→0 |
| `enable` | in | 1 | — | — | low → hold `phase`/`freq_word`/`ramp_tick` at 0 (no advance) |
| `target_freq_word` | in | 32 | no | — | steady-state phase increment per clk (sets final electrical speed) |
| `ramp_increment` | in | 32 | no | — | max `freq_word` increase applied per 4096-clk tick (ramp rate) |
| `sector` | out | 3 | no | 0 | current six-step commutation sector 0..5, combinational in `phase` |

## Clocking & reset

- **Clock domains:** single `clk`.
- **Reset:** async active-low `rst_n` → all state cleared (`sector=0`);
  `!enable` holds the same cleared state synchronously.
- **Latency:** free-running accumulator — `phase` advances by `freq_word` every
  clock; `sector` is a **combinational** decode of `phase`. `freq_word` increases
  by ≤ `ramp_increment` once per 4096 clocks until it reaches `target_freq_word`.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| _(none)_ | | |

## Formal contract

- **Sim-only** (`formal/sim_only.toml`): phase-accumulator ramp; covered by the open-loop spin sim tier
- Covered by the simulation tiers, not a machine proof.
- **Assumptions:** single clock; async active-low reset is the design reset;
  `target_freq_word`/`ramp_increment` are configuration constants.

## Synthesis fit

- **Device:** ECP5 (part of the system fit). Small: two 32-bit accumulators, a
  12-bit tick counter, and a five-comparator sector decode; not on the system
  critical path.

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — self-contained.
- **Pull it:** `fusesoc run motorloop:ip:open_loop_ramp` (core at repo root).
