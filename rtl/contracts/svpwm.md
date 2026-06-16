<!-- SPDX-License-Identifier: MIT -->
# `svpwm` — Space-vector PWM by min/max injection, combinational

Space-vector PWM by min/max common-mode injection: turns a stationary-frame
voltage `(valpha, vbeta)` (in duty units, ±`PWM_HALF_PERIOD` for full half-bus
phase drive) into three per-leg duty compares in `[0, HALF]` centered at
`HALF/2` (50% = zero voltage). It forms the three phase references
(`ra = valpha`, `rb = -valpha/2 + (sqrt3/2)*vbeta`, `rc = -valpha/2 -
(sqrt3/2)*vbeta`), subtracts the common-mode `(max+min)/2`, biases around
`CENTER`, and saturates each leg into `[0, HALF]`. This produces the same
line-to-line output as sector-dwell SVPWM with no sector-detect state, buying
~15% more linear modulation range than pure sine PWM. Final stage of the FOC
voltage path, after `inv_park`; `duty3` feeds `pwm_generator`. The `sqrt3/2`
factor is a Q15 constant (`28378`) rounded by `>>>15`, bit-exact against
`foc_reference.svpwm_fx`.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `valpha` | in | 18 | yes | — | α-axis voltage, duty units (from `inv_park`) |
| `vbeta` | in | 18 | yes | — | β-axis voltage, duty units (from `inv_park`) |
| `duty3` | out | 48 | no | — | `{leg C, leg B, leg A}`, 16 bits each; per-leg compare in `[0, HALF]`, centered at `HALF/2` |

## Clocking & reset

- **Clock domains:** none — pure combinational transform (no `clk`).
- **Reset:** none — stateless.
- **Latency:** combinational (zero cycles).

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `PWM_HALF_PERIOD` | `625` | half PWM period in clocks; sets `HALF` (duty full-scale) and `CENTER = HALF/2` (50% / zero-voltage point) |

## Formal contract

- **PROVEN** (`formal/manifest.toml`, `formal/bind/svpwm_fv.sv`): SVPWM: every per-leg duty stays within [0, PWM_HALF_PERIOD].
- **Method:** prove, `engine smtbmc boolector`.
- **Assumptions:** combinational (no clock/reset); `valpha`,`vbeta` are signed
  voltages in duty units that fit the internal 32-bit arithmetic. The `[0,HALF]`
  bound holds for any input (each leg is `sat_duty`-clamped), so the proof needs
  no input-range assumption.

## Synthesis fit

- **Device:** ECP5. Combinational, modest: one Q15 multiply for `(sqrt3/2)*vbeta`,
  the three references, a min/max/common-mode network, and three `sat_duty`
  saturating adders. The single multiply can map to a DSP slice; the rest is LUTs.
  No registers (the system uses the sequential `svpwm_seq` variant for Fmax).

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — leaf module (`rtl/svpwm.v`); set `PWM_HALF_PERIOD` to
  match `pwm_generator`.
- **Pull it:** `fusesoc run motorloop:ip:svpwm` (core at repo root).
