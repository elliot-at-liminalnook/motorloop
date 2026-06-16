<!-- SPDX-License-Identifier: MIT -->
# `as5600_pwm_capture` — AS5600 PWM-output angle capture

Decodes the **AS5600** magnetic encoder's PWM-output mode into a 12-bit angle.
The sensor emits a fixed-period frame of 4351 units, high for `(128 + angle)`
units, so `angle = high_cycles · 4351 / period_cycles − 128`. A 3-tap
synchronizer cleans `pwm_in`; rising/falling edges measure the high time and
the full period in `clk` cycles, and on each rising edge a `divider32` computes
`high·4351 / period`, from which `angle_from_quotient` subtracts 128 (clamped to
`[0, 4095]`). Validity uses **hysteresis (E15)**: while valid it tolerates ±25%
carrier drift, but must be within ±15% to (re)validate, so a carrier at the
boundary cannot flap run/coast; a 2-period edge timeout (magnet loss /
disconnection) drops `angle_valid`. `new_sample` pulses once per decoded frame.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset; counters cleared, no valid angle |
| `pwm_in` | in | 1 | — | — | AS5600 PWM output (synchronized internally) |
| `angle` | out | 12 | — | 0 | decoded angle 0..4095 (held on invalid/skip) |
| `angle_valid` | out | 1 | — | 0 | period in band and edges arriving (hysteresis E15) |
| `new_sample` | out | 1 | — | 0 | 1-cycle pulse per decoded frame (on divide completion) |

## Clocking & reset

- **Clock domains:** single `clk`; `pwm_in` is asynchronous and passed through a
  3-FF synchronizer before edge detection.
- **Reset:** async active-low `rst_n` → counters/`high_cnt`/`idle_cnt` cleared,
  `angle`=0, `angle_valid`=0.
- **Latency:** one decode per PWM frame. On each valid rising edge a divide is
  launched; `divider32` runs ~32 cycles, then `div_done` latches `angle`,
  raises `angle_valid` and pulses `new_sample`. An edge timeout of `2·CARRIER`
  cycles invalidates.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `ANGLE_CARRIER_CYC` | `27174   // AS5600 PWM frame period in clocks` | expected AS5600 PWM frame period in `clk` cycles; sets the ±25%/±15% bands and the 2× timeout |

## Formal contract

- **Sim-only** (`formal/sim_only.toml`): PWM duty capture; exercised against the AS5600 model in the co-sim
- Covered by the simulation tiers, not a machine proof.
- **Assumptions:** single `clk`; async `rst_n` is the design reset; the bench
  AS5600 model emits the 4351-unit PWM frame (high = 128 + angle) at a carrier
  near `ANGLE_CARRIER_CYC`; `pwm_in` glitches are absorbed by the synchronizer;
  `divider32` is the trusted-library divider.

## Synthesis fit

- **Device:** ECP5. Modest: several 32-bit counters, edge logic, the band
  comparators and one `divider32` instance (`synth/fmax_module.py as5600_pwm_capture`).

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** instantiates `divider32` (trusted library, in the core).
- **Pull it:** `fusesoc run motorloop:ip:as5600_pwm_capture` (core at repo root).
