<!-- SPDX-License-Identifier: MIT -->
# `speed_pi` — Six-step speed PI with down-slew limit

Outer speed loop for the **six-step** duty path: parallel-form PI
`duty = clamp( KP·err + (KP·integ >> KI_SHIFT) )` with conditional-integration
anti-windup and a downward duty slew limit. `err = target_speed − measured`,
where a `reverse`-flagged rotor reads as negative speed (so a backdriven motor
does not look on-target — E5). The output is clamped to `[0, DUTY_MAX]` (98 % of
`HALF`); the down-slew bounds how fast duty may fall (caps regenerative braking
current — realism 1.7), while upward steps are unrestricted. Updated once per
speed sample. The signed-output FOC analogue is `speed_iq_pi`.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset; `integ`→0, `duty_compare`→0, S_IDLE |
| `enable` | in | 1 | — | — | low → freeze + reset integrator and duty to 0, hold in S_IDLE |
| `update` | in | 1 | — | — | new speed sample; latches inputs and starts the walk (in S_IDLE) |
| `target_speed` | in | 16 | no | — | speed setpoint (rad/s), latched on `update` |
| `speed` | in | 16 | no | — | measured speed magnitude (rad/s), latched on `update` |
| `reverse` | in | 1 | — | — | measured rotation is backward → measured speed treated as negative |
| `duty_compare` | out | 16 | no | 0 | PWM compare value in `[0, DUTY_MAX]`; holds between samples |

## Clocking & reset

- **Clock domains:** single `clk`.
- **Reset:** async active-low `rst_n` → `integ=0`, `duty_compare=0`,
  `state=S_IDLE`; `!enable` does the same synchronously (held while low).
- **Latency:** **pipelined**, `update` starts a 3-stage walk
  S_IDLE→S_ERR→S_MAC→S_FIN: `err` (S_ERR), the 32-bit MAC (S_MAC), then clamp +
  conditional integration + down-slew (S_FIN, which also registers
  `duty_compare` and returns to S_IDLE). `duty_compare` appears **~3 clocks**
  after `update` and holds until the next sample. The integrator and the output
  evolve identically to the single-cycle version (same err, saturation, and
  prior duty for the slew) — only later in time. `update` is sparse (once/speed
  sample) so the walk always finishes in-period.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `SPEED_PI_KP` | `12` | proportional gain (also scales the integral term) |
| `SPEED_PI_KI_SHIFT` | `4` | integral right-shift; effective Ki = `KP / 2^KI_SHIFT` |
| `PWM_HALF_PERIOD` | `625` | half PWM period; `DUTY_MAX = HALF·49/50` (98 %) is the upper clamp |
| `DUTY_DOWN_SLEW` | `16'd40` | max downward duty step per update (regen/brake current limit) |

## Formal contract

- **Sim-only** (`formal/sim_only.toml`): six-step duty PI; clamp behaviour exercised by the closed-loop sim (the FOC analogue speed_iq_pi IS proven, param-generic)
- Covered by the simulation tiers, not a machine proof.
- **Assumptions:** single clock; async active-low reset is the design reset;
  `update` pulsed once per speed sample (sparse vs. the 3-clock walk); 16-bit
  unsigned `target_speed`/`speed`.

## Synthesis fit

- **Device:** ECP5. Standalone Fmax **94 MHz** (`synth/fmax_module.py
  speed_pi`), up from 48 MHz after pipelining the MAC+clamp+down-slew — it had
  surfaced as the *system* cap once the FOC datapath was sped up. Small: one
  32-bit MAC, a clamp, the down-slew compare, and a 32-bit integrator over a
  4-state sequencer.

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — self-contained.
- **Pull it:** `fusesoc run motorloop:ip:speed_pi` (core at repo root).
