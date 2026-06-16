<!-- SPDX-License-Identifier: MIT -->
# `current_pi` — Single-axis current PI with anti-windup clamp (PROVEN)

Single-axis current PI (one instance for id, one for iq) in parallel form with
conditional-integration anti-windup: `v_out = sat( KP·err + (KP·integ >>
KI_SHIFT) )` where `err = target − meas`, clamped per-axis to `[−V_RAW_MAX,
+V_RAW_MAX]`. The integrator advances by `err` only on `update` while `!freeze`
(the FOC core drives `freeze` from the voltage-circle limiter's `sat` so neither
axis winds up while the vector is clamped). `v_out` is the raw per-axis command;
the joint magnitude is bounded downstream by `circle_limit`. The per-axis clamp
holds for any state and any `V_RAW_MAX` — **formally proven, parameter-generic.**

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset; integrator → 0 |
| `enable` | in | 1 | — | — | low → hold integrator at 0 |
| `update` | in | 1 | — | — | FOC current-sample strobe; integrates `err` (unless `freeze`) |
| `freeze` | in | 1 | — | — | anti-windup: skip the integrator update this `update` (driven by limiter `sat`) |
| `target` | in | 18 | yes | — | dq current command (LSB) |
| `meas` | in | 18 | yes | — | measured dq current (LSB) |
| `v_out` | out | 18 | yes | — | clamped voltage command (duty units), combinational |

## Clocking & reset

- **Clock domains:** single `clk`.
- **Reset:** async active-low `rst_n` → `integ = 0`; `!enable` likewise holds
  `integ = 0`.
- **Latency:** `v_out` is **combinational** in `target`/`meas`/`integ` (the
  KP·err + scaled-integral + clamp). The only register is `integ`, which updates
  on the clock after `update` when `enable && !freeze`; the new integral term
  then appears on `v_out` the cycle after.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `CUR_PI_KP` | `2` | proportional gain (also scales the integral term `KP·integ`) |
| `CUR_PI_KI_SHIFT` | `4` | integral right-shift; effective Ki = `KP / 2^KI_SHIFT` |
| `V_RAW_MAX` | `2500` | symmetric per-axis output clamp (duty units); the proven bound |

## Formal contract

- **PROVEN** (`formal/manifest.toml`, `formal/bind/current_pi_fv.sv`): Current-PI output stays within its clamp [-V_RAW_MAX, V_RAW_MAX] for any error/integrator state AND any V_RAW_MAX parameter (parameter-generic).
- **Method:** prove, `engine smtbmc boolector`.
- **Assumptions:** single clock; async active-low reset is the design reset;
  18-bit signed `target`/`meas`; the clamp is asserted on `v_out` for any
  integrator state.

## Synthesis fit

- **Device:** ECP5. Standalone Fmax **100 MHz** (`synth/fmax_module.py
  current_pi`); never the binding path, so it was left unpipelined through the
  Fmax campaign. One 32-bit MAC + clamp and a single 32-bit integrator register.

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — self-contained.
- **Pull it:** `fusesoc run motorloop:ip:current_pi` (core at repo root).
