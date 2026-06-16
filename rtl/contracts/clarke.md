<!-- SPDX-License-Identifier: MIT -->
# `clarke` — Clarke transform (3-phase -> alpha/beta), combinational

Amplitude-invariant Clarke transform: maps the two measured phase currents
`(ia, ib)` of a balanced three-phase set (`ic = -ia-ib` is implied) into the
stationary two-axis frame `(ialpha, ibeta)`, with `ialpha = ia` and
`ibeta = (ia + 2*ib)/sqrt(3)`. First stage of the FOC measurement path: feeds
`park` to obtain `(id, iq)`. The `1/sqrt(3)` factor is a Q15 constant
(`18919 = round(2^15/sqrt(3))`) and the product is rounded back by `>>>15`, so
the result is bit-exact against `foc_reference.clarke_fx`.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `ia` | in | 18 | yes | — | phase-A current, ADC-LSB units |
| `ib` | in | 18 | yes | — | phase-B current, ADC-LSB units |
| `ialpha` | out | 18 | yes | — | α-axis current = `ia` (pass-through) |
| `ibeta` | out | 18 | yes | — | β-axis current = `(ia + 2*ib)/sqrt(3)`, Q15 scaled then `>>>15` |

## Clocking & reset

- **Clock domains:** none — pure combinational transform (no `clk`).
- **Reset:** none — stateless.
- **Latency:** combinational (zero cycles).

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| _(none)_ | | |

## Formal contract

- **Sim-only** (`formal/sim_only.toml`): pure combinational transform; bit-exact vs the Python fixed-point reference (test_foc_math)
- Covered by the simulation tiers, not a machine proof.
- **Assumptions:** combinational (no clock/reset); `ia`,`ib` are signed 18-bit
  currents whose `ia + 2*ib` sum-of-products fits the internal 32-bit sign-
  extended arithmetic (true for the ADC-LSB ranges the system produces).

## Synthesis fit

- **Device:** ECP5. Combinational, tiny: one 32-bit signed multiply by the Q15
  constant plus an add/shift; maps to a handful of LUTs (no register, no DSP
  required for a constant multiply).

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — leaf module (`rtl/clarke.v`).
- **Pull it:** `fusesoc run motorloop:ip:clarke` (core at repo root).
