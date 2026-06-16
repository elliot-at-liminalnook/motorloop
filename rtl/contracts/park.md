<!-- SPDX-License-Identifier: MIT -->
# `park` — Park transform (alpha/beta -> d/q), combinational

Park transform: rotates the stationary `(ialpha, ibeta)` currents into the
rotor `(d, q)` frame at electrical angle `theta_e` (supplied as precomputed Q15
`cos`/`sin` from `sincos`). Computes `id = ialpha*cos + ibeta*sin` and
`iq = ialpha*sin - ibeta*cos`. This sign convention places the torque-producing
current (in phase with the `e_a = Ke*omega*sin(theta_e)` back-EMF) on `+iq`
with `id = 0`. Second stage of the FOC measurement path, after `clarke`; its
`(id, iq)` are the feedback for the current PIs. Products are Q15 and rounded by
`>>>15`, bit-exact against `foc_reference.park_fx`.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `ialpha` | in | 18 | yes | — | α-axis current (from `clarke`) |
| `ibeta` | in | 18 | yes | — | β-axis current (from `clarke`) |
| `cos_q15` | in | 16 | yes | — | `cos(theta_e)`, Q15 signed (from `sincos`) |
| `sin_q15` | in | 16 | yes | — | `sin(theta_e)`, Q15 signed (from `sincos`) |
| `id` | out | 18 | yes | — | direct-axis current = `(ialpha*cos + ibeta*sin) >>> 15` |
| `iq` | out | 18 | yes | — | quadrature-axis current = `(ialpha*sin - ibeta*cos) >>> 15` |

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

- **Device:** ECP5. Combinational, small: two signed 18×16 multiplies per axis
  (four total) plus add/sub and a `>>>15`; the four products map to ECP5 DSP
  (`MULT18X18`) slices, the rest to LUTs. No registers.

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — leaf module (`rtl/park.v`); pairs with `sincos` for
  the angle.
- **Pull it:** `fusesoc run motorloop:ip:park` (core at repo root).
