<!-- SPDX-License-Identifier: MIT -->
# `pwm_generator` — center-aligned 3-phase complementary PWM with dead-time

Up/down (center-aligned) PWM for a three-phase half-bridge: three independent
per-leg duties, complementary high/low gates with an enforced dead-time, a
hard `kill`, and a sampling-window strobe. **Shoot-through-free by construction
and formally proven.** Self-contained (no includes, no child modules).

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset; all gates off |
| `kill` | in | 1 | — | — | force every gate off immediately (fault/disable) |
| `duty3` | in | 48 | no | — | `{leg C, leg B, leg A}`, 16 bits each, compare value in `[0,HALF]` |
| `leg_mode` | in | 6 | no | — | `{leg C, leg B, leg A}`, 2 bits each (drive/float/low) |
| `gate_high` | out | 3 | — | 0 | high-side gate per leg |
| `gate_low` | out | 3 | — | 0 | low-side gate per leg |
| `counter_out` | out | 16 | no | 0 | the up/down counter (0..HALF) |
| `counting_up` | out | 1 | — | — | counter direction |
| `period_start` | out | 1 | — | 0 | 1-cycle pulse at counter wrap through 0 |

## Clocking & reset

- **Clock domains:** single `clk`.
- **Reset:** async active-low `rst_n`; during and immediately after reset every
  gate is 0 (proven, no power-on glitch).
- **Latency:** registered gates; the counter advances one step/clk.

## Parameters

| Parameter | Default | Legal range | Meaning |
| --- | --- | --- | --- |
| `PWM_HALF_PERIOD` | `16'd625` | `[1, 65535]` | half PWM period in clocks (counter peak) |
| `DEAD_CYCLES` | `16'd25` | `[0, HALF]` | enforced dead-time between complementary edges |
| `MIN_PULSE_CYCLES` | `16'd2` | `[0, HALF]` | minimum pulse; below this a leg is held off/full |

## Formal contract

- **Proven** (`formal/manifest.toml`):
  - `pwm_generator`: no leg ever drives both gates; counter stays in `[0,HALF]`;
    while `kill`, every gate is off.
  - `pwm_deadtime`: a gate asserts only after its complement has been off
    ≥ `DEAD_CYCLES` (no cross-conduction window).
  - `pwm_reset`: gates off during reset and the first cycle after release.
  - `lib_no_shoot_through`: the reusable, parameter-generic shoot-through checker.
- **Assumptions:** async reset is the design reset; PWM constants are config.
- **Method:** prove (k-induction), `engine smtbmc boolector`.

## Synthesis fit

- **Device:** ECP5 (part of the system fit; standalone is small — a 16-bit
  counter + per-leg compare/dead-time logic). Combinational gate logic + one
  registered counter; comfortably > 25 MHz standalone.

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — drop in `rtl/pwm_generator.v`, set the three params.
- **Pull it:** `fusesoc run motorloop:ip:pwm_generator` (core at repo root).
