<!-- SPDX-License-Identifier: MIT -->
# `inv_park` — Inverse Park (d/q -> alpha/beta), combinational

Inverse Park transform: rotates the commanded rotor-frame voltage `(vd, vq)`
back to the stationary `(valpha, vbeta)` frame at electrical angle `theta_e`
(Q15 `cos`/`sin` from `sincos`). Computes `valpha = vd*cos + vq*sin` and
`vbeta = vd*sin - vq*cos`. Because the Park matrix `[[c,s],[s,-c]]` is its own
inverse, this reuses exactly the `park` arithmetic form. Output stage of the
FOC voltage path: takes the circle-limited PI outputs and produces the
stationary voltage vector that `svpwm` turns into per-leg duties. Products are
Q15, rounded by `>>>15`, bit-exact against `foc_reference.inv_park_fx`.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `vd` | in | 18 | yes | — | direct-axis voltage command (duty units) |
| `vq` | in | 18 | yes | — | quadrature-axis voltage command (duty units) |
| `cos_q15` | in | 16 | yes | — | `cos(theta_e)`, Q15 signed (from `sincos`) |
| `sin_q15` | in | 16 | yes | — | `sin(theta_e)`, Q15 signed (from `sincos`) |
| `valpha` | out | 18 | yes | — | α-axis voltage = `(vd*cos + vq*sin) >>> 15` |
| `vbeta` | out | 18 | yes | — | β-axis voltage = `(vd*sin - vq*cos) >>> 15` |

## Clocking & reset

- **Clock domains:** none — pure combinational transform (no `clk`).
- **Reset:** none — stateless.
- **Latency:** combinational (zero cycles).

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| _(none)_ | | |

## Formal contract

- **Sim-only** (`formal/sim_only.toml`): pure combinational transform; bit-exact (test_foc_math)
- Covered by the simulation tiers, not a machine proof.
- **Assumptions:** combinational (no clock/reset); `cos_q15`,`sin_q15` are a
  valid Q15 unit vector (as produced by `sincos`); inputs are signed and their
  sum-of-products fits the internal 32-bit sign-extended arithmetic.

## Synthesis fit

- **Device:** ECP5. Combinational, small: four signed 18×16 multiplies plus
  add/sub and a `>>>15` (same shape as `park`); products map to ECP5 DSP
  slices, the rest to LUTs. No registers.

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — leaf module (`rtl/inv_park.v`); pairs with `sincos`
  for the angle.
- **Pull it:** `fusesoc run motorloop:ip:inv_park` (core at repo root).
