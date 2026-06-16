<!-- SPDX-License-Identifier: MIT -->
# `circle_limit` — Voltage-circle magnitude limiter (isqrt), combinational

Voltage-circle limiter: caps the `(vd, vq)` voltage-vector magnitude to the
SVPWM inscribed-circle radius `V_CIRCLE_LIMIT` (duty units) with a uniform
scale, so the modulation stays in the linear region. Computes
`mag2 = vd^2 + vq^2`, takes a floor integer square root, and when
`mag2 > VLIM^2` rescales both axes by `VLIM/|v|` (`vd_out = vd*VLIM/|v|`,
likewise `vq`); otherwise passes the inputs through unchanged. Emits `sat` (high
when clamping) so the upstream current PIs freeze their integrators for
anti-windup. Sits between the dq PIs and `inv_park` in the FOC voltage path.
The scale is a uniform (non-prioritized) shrink — field-weakening priority on
`vd` is left to the caller's choice of `id_target`.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `vd_in` | in | 18 | yes | — | direct-axis voltage from the d-PI (duty units) |
| `vq_in` | in | 18 | yes | — | quadrature-axis voltage from the q-PI (duty units) |
| `vd_out` | out | 18 | yes | — | limited d voltage: `vd_in*VLIM/|v|` if `sat`, else `vd_in` |
| `vq_out` | out | 18 | yes | — | limited q voltage: `vq_in*VLIM/|v|` if `sat`, else `vq_in` |
| `sat` | out | 1 | — | — | 1 when `vd^2+vq^2 > VLIM^2` (vector was clamped); drives PI anti-windup |

## Clocking & reset

- **Clock domains:** none — pure combinational transform (no `clk`).
- **Reset:** none — stateless.
- **Latency:** combinational (zero cycles); the isqrt is an unrolled fixed
  16-iteration bit-by-bit loop (no data-dependent bound), so it is a fixed
  combinational chain, not a sequencer.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `V_CIRCLE_LIMIT` | `594` | SVPWM inscribed-circle radius (duty units); the magnitude cap `VLIM`, and `VLIM^2` is the saturation threshold |

## Formal contract

- **DOCUMENTED** (`formal/manifest.toml`, `formal/bind/circle_limit_fv.sv`): Voltage-circle limiter: |(vd_out, vq_out)| <= V_CIRCLE_LIMIT + slack (modulation stays within the inscribed circle).
- **Method:** prove, `engine smtbmc boolector`.
- **Assumptions:** combinational (no clock/reset). The per-axis inputs are
  assumed pre-clamped to `V_RAW_MAX` by `current_pi.v` so `vd^2 + vq^2` fits 32
  bits. The bound is "documented, not machine-proven" — the integer
  divide + isqrt is intractable for the open SMT engines, so it is bounded by
  construction and validated by the FOC sim tier (the `+ slack` covers
  floor-isqrt and integer-divide rounding).

## Synthesis fit

- **Device:** ECP5. Combinational but the heaviest of the leaves: two squaring
  multiplies for `mag2`, a 16-iteration unrolled isqrt, and two 32-bit signed
  divides (`vd*VLIM/|v|`). The divides dominate area/delay — this is why the
  system instantiates the sequential `circle_limit_seq` (one op per clock,
  bit-exact) instead, to lift Fmax. No registers here.

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — self-contained leaf (`rtl/circle_limit.v`); the isqrt
  and divides are inline (the sequential variant `circle_limit_seq` instead
  instantiates `divider32`).
- **Pull it:** `fusesoc run motorloop:ip:circle_limit` (core at repo root).
