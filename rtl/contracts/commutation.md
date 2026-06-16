<!-- SPDX-License-Identifier: MIT -->
# `commutation` — Six-step sector -> per-leg drive decode, combinational

Six-step (trapezoidal) commutation decode: maps the electrical `sector` (0..5)
to the three per-leg PWM modes plus the analog bookkeeping the ADC sequencer
needs. Per sector exactly one leg chops (`MODE_PWM`, complementary high-side),
one is held solid low (`MODE_LOW`), and the third floats (`MODE_OFF`) for
back-EMF observation; the (high, low) pairing follows the standard sequence
0:AB 1:AC 2:BC 3:BA 4:CA 5:CB. It also reports `low_phase` (the phase carrying
solid low-side shunt current) and `float_phase` (the floating phase) so the
sampling/BEMF logic knows where to look. Used in the six-step drive mode; the
`leg_mode` it emits feeds `pwm_generator`.

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `sector` | in | 3 | — | — | commutation sector 0..5 (values ≥6 → all legs off) |
| `leg_mode` | out | 6 | — | — | `{C,B,A}` 2 bits each: `00` off / `01` pwm / `10` low |
| `low_phase` | out | 2 | — | — | phase index (0=A,1=B,2=C) carrying solid low-side current |
| `float_phase` | out | 2 | — | — | floating phase index (the BEMF-observation phase) |

## Clocking & reset

- **Clock domains:** none — pure combinational decode (`always @(*)`, no `clk`).
- **Reset:** none — stateless.
- **Latency:** combinational (zero cycles).

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| _(none)_ | | |

## Formal contract

- **Sim-only** (`formal/sim_only.toml`): combinational sector->drive decode; covered by the top no-shoot-through composition proof + six-step sim
- Covered by the simulation tiers, not a machine proof.
- **Assumptions:** combinational (no clock/reset); `sector` in 0..5 for valid
  drive (the `default` arm forces all legs off, so out-of-range is safe). No leg
  is ever both pwm and low, so the table never asks for shoot-through; the
  end-to-end no-shoot-through guarantee is the top `controller_top_composition`
  proof through the `pwm_generator` mux.

## Synthesis fit

- **Device:** ECP5. Combinational, trivial: a 6-entry case (3-bit input → 10
  output bits) that maps to a couple of LUTs. No registers, no arithmetic.

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — leaf module (`rtl/commutation.v`); `leg_mode` is sized
  to drive `pwm_generator`'s `leg_mode` input.
- **Pull it:** `fusesoc run motorloop:ip:commutation` (core at repo root).
