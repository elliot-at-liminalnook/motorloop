<!-- SPDX-License-Identifier: MIT -->
# `foc_core` — FOC datapath (Clarke → Park → PI → circle-limit → inv-Park → SVPWM)

The field-oriented-control inner datapath: measures `(id,iq)` from the sampled
phase currents at the rotor angle, runs the dq current PIs toward the targets,
bounds the voltage vector to the SVPWM inscribed circle (with integrator
anti-windup), and produces three per-leg duties. The PI integrators and the
output duties register once per current sample (`update`); between samples the
duties hold. Surface-PMSM convention (`id_target = 0`).

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk`,`rst_n` | in | 1 | — | — | clock / async active-low reset |
| `enable` | in | 1 | — | — | low → hold 50% duty, reset integrators |
| `update` | in | 1 | — | — | FOC current-sample strobe (1/PWM period) |
| `cur_a`,`cur_b` | in | 18 | yes | — | measured phase currents (LSB), offset-removed |
| `theta_e` | in | 16 | no | — | electrical angle (0..65535 = 0..2π) |
| `id_target`,`iq_target` | in | 18 | yes | — | dq current commands (LSB) |
| `duty3` | out | 48 | no | center | `{C,B,A}` 16-bit per-leg duty compares |
| `dbg_id`,`dbg_iq`,`dbg_vd`,`dbg_vq` | out | 18 | yes | 0 | datapath telemetry |

## Clocking & reset

- **Single clock**; async active-low reset → duty3 = 50% (zero voltage).
- **Latency:** the Clarke→…→SVPWM chain is **combinational**; results register
  on `update`. So the input-to-duty path is one long combinational chain (see
  Synthesis).

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `PWM_HALF_PERIOD` | `625` | duty center / SVPWM scale (threaded to `svpwm`) |
| `SINCOS_TABLE_BITS` | `8` | sin/cos LUT size (threaded to `sincos`; table regenerated if changed) |
| `V_CIRCLE_LIMIT` | `594` | inscribed-circle radius (threaded to `circle_limit`) |
| `CUR_PI_KP`,`CUR_PI_KI_SHIFT` | `2`,`4` | current-PI gains (threaded to both `current_pi`) |
| `V_RAW_MAX` | `2500` | per-axis PI clamp (threaded to `current_pi`) |

## Formal contract

- **Proven (sub-blocks):** `svpwm` (per-leg duty bound), `current_pi` (output
  clamp, **parameter-generic**), `speed_iq_pi` (outer clamp, parameter-generic).
  The top-level `controller_top_composition` proves no-shoot-through through this
  datapath's mux.
- **Documented, not machine-proven:** `circle_limit` magnitude bound (an integer
  divide + isqrt; intractable for the open SMT engines — bounded by construction
  + validated by the FOC sim tier).
- **Bit-exact sim:** `test_foc_math.py` checks Clarke/Park/inv-Park/SVPWM/sincos
  against the Python fixed-point reference.

## Synthesis fit

- **Device:** ECP5. **Finding:** the unpipelined combinational chain (notably
  `circle_limit`'s 16-iteration isqrt) caps **Fmax ≈ 3.3 MHz** in the system
  build — *below* the 25 MHz sim clock. A real high-speed board build must
  **pipeline this datapath** (registered stages). See `synth/synth_report.md`.
  The simulator is cycle-accurate regardless.

## Reuse notes

- **Language:** Verilog-2005. **Dependencies:** instantiates `sincos`, `clarke`,
  `park`, `inv_park`, `current_pi`×2, `circle_limit`, `svpwm` (all in the core).
- **Pull it:** `fusesoc run motorloop:ip:foc_core` (core in `cores/`).
