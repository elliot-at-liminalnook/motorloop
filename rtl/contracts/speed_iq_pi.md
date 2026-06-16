<!-- SPDX-License-Identifier: MIT -->
# `speed_iq_pi` — FOC outer speed loop -> iq* with clamp (PROVEN)

FOC outer speed loop: turns speed error into a signed q-axis current command
`iq* = clamp( KP·err + (KP·integ >> KISH), ±IQ_MAX )`, parallel-form PI with
conditional-integration anti-windup. `err = target_speed − measured`, where a
`reverse`-flagged rotor reads as negative speed (E5). The **signed** output lets
the loop command braking torque (negative iq) on overspeed — unlike the unsigned
six-step `speed_pi`. The output magnitude stays within `IQ_MAX` for any state and
any `IQ_MAX` — **formally proven, parameter-generic.** Feeds the inner current
PIs as `iq_target`.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset; `integ`→0, `iq_target`→0, S_IDLE |
| `enable` | in | 1 | — | — | low → reset integrator and `iq_target` to 0, hold in S_IDLE |
| `update` | in | 1 | — | — | new speed sample; latches inputs and starts the walk (in S_IDLE) |
| `target_speed` | in | 16 | no | — | speed setpoint (rad/s), latched on `update` |
| `speed` | in | 16 | no | — | measured speed magnitude (rad/s), latched on `update` |
| `reverse` | in | 1 | — | — | measured rotation is backward → measured speed treated as negative |
| `iq_target` | out | 18 | yes | 0 | q-axis current command (LSB), in `[−IQ_MAX, IQ_MAX]`; holds between samples |

## Clocking & reset

- **Clock domains:** single `clk`.
- **Reset:** async active-low `rst_n` → `integ=0`, `iq_target=0`,
  `state=S_IDLE`; `!enable` does the same synchronously (held while low).
- **Latency:** **pipelined**, `update` starts a 3-stage walk
  S_IDLE→S_ERR→S_MAC→S_FIN: `err` (S_ERR), the 32-bit MAC (S_MAC), then clamp +
  conditional integration (S_FIN, which registers `iq_target` and returns to
  S_IDLE). `iq_target` appears **~3 clocks** after `update` and holds until the
  next sample. The integrator advances exactly once per `update` with the same
  err and freeze(sat) as the single-cycle version — behaviour-identical, only
  later in time. `update` is sparse (once/speed sample).

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `SPEED_IQ_KP` | `4` | proportional gain (also scales the integral term) |
| `SPEED_IQ_KISH` | `5` | integral right-shift; effective Ki = `KP / 2^KISH` |
| `IQ_MAX` | `300` | symmetric current-command clamp (LSB); the proven bound |

## Formal contract

- **PROVEN** (`formal/manifest.toml`, `formal/bind/speed_iq_pi_fv.sv`): FOC torque command iq_target stays within [-IQ_MAX, IQ_MAX] for any speed error/integrator state AND any IQ_MAX parameter (parameter-generic).
- **Method:** prove, `engine smtbmc boolector` (the proof asserts on the output
  register; depth covers the S_ERR→S_MAC→S_FIN walk).
- **Assumptions:** single clock; async active-low reset is the design reset;
  `update` pulsed once per speed sample (sparse vs. the 3-clock walk); 16-bit
  unsigned `target_speed`/`speed`.

## Synthesis fit

- **Device:** ECP5. Standalone Fmax **129 MHz** (`synth/fmax_module.py
  speed_iq_pi`), up from 75 MHz after pipelining the MAC+clamp (Fmax task 1, the
  original system bottleneck; re-proven PROVEN). Small: one 32-bit MAC, a
  symmetric clamp, and a 32-bit integrator over a 4-state sequencer.

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — self-contained.
- **Pull it:** `fusesoc run motorloop:ip:speed_iq_pi` (core at repo root).
