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
- **Latency (pipelined, stage 6.5):** `update` starts a sequencer that walks the
  Clarke→Park→PI→limit→inv-Park→SVPWM chain over **registered stages**, one op
  per clock. `duty3`/`dbg_*` update **~14 clocks** after `update` (or **~62**
  when the limiter saturates: the sequential isqrt + two divides + the
  sequential SVPWM). `update` is sparse (1/PWM period, hundreds–thousands of
  clocks), so the walk always finishes well inside the sample period and the
  loop is unaffected; the PI integrators still advance exactly once per `update`,
  with the **same error and freeze(sat)** as the combinational core (bit-identical
  integrator state).

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `PWM_HALF_PERIOD` | `625` | duty center / SVPWM scale (threaded to `svpwm_seq`) |
| `SINCOS_TABLE_BITS` | `8` | sin/cos LUT size (threaded to `sincos`; table regenerated if changed) |
| `V_CIRCLE_LIMIT` | `594` | inscribed-circle radius (threaded to `circle_limit_seq`) |
| `CUR_PI_KP`,`CUR_PI_KI_SHIFT` | `2`,`4` | current-PI gains (threaded to both `current_pi`) |
| `V_RAW_MAX` | `2500` | per-axis PI clamp (threaded to `current_pi`) |

## Formal contract

- **Proven (sub-blocks):** `svpwm` (per-leg duty bound), `current_pi` (output
  clamp, **parameter-generic**), `speed_iq_pi` (outer clamp, parameter-generic).
  The top-level `controller_top_composition` proves no-shoot-through through this
  datapath's mux.
- **Documented, not machine-proven:** the `circle_limit` magnitude bound (an
  integer divide + isqrt; intractable for the open SMT engines — bounded by
  construction + validated by the FOC sim tier).
- **Bit-exact equivalence:** `circle_limit_seq` and `svpwm_seq` (the sequential
  blocks foc_core uses) are proven bit-exact to the combinational `circle_limit`
  / `svpwm` by the cocotb tests `tb_circle_limit_seq` / `tb_svpwm_seq`.
- **Bit-exact sim:** `test_foc_math.py` checks Clarke/Park/inv-Park/SVPWM/sincos
  against the Python fixed-point reference.

## Synthesis fit

- **Device:** ECP5. **Finding:** pipelining the datapath took it from **Fmax ≈
  3.3 MHz** (unpipelined) to **≈ 79 MHz standalone** in stages: the sequential
  `circle_limit_seq` (bit-exact limiter, isqrt+divide one-op-per-clock, with the
  squares/products computed from the 18-bit inputs so they map to hard
  multipliers) and the sequential `svpwm_seq` (bit-exact SVPWM). LUT usage
  *dropped* vs the combinational mega-path. In the full system this contributed
  to **41 → 64 MHz** (alongside pipelining the speed PIs); see
  `notes/foc-fmax-optimization-checklist.md` and `synth/synth_report.md`. The
  simulator is cycle-accurate regardless.
- **Portability:** the Verilog-2005 RTL maps to Xilinx 7-series / Intel Cyclone /
  Gowin GW5A as well as ECP5 (`synth/portability_report.md`, `make portability`).

## Reuse notes

- **Language:** Verilog-2005. **Dependencies:** instantiates `sincos`, `clarke`,
  `park`, `inv_park`, `current_pi`×2, `circle_limit_seq` (→ `divider32`),
  `svpwm_seq` (all in the core).
- **Pull it:** `fusesoc run motorloop:ip:foc_core` (core at repo root).
