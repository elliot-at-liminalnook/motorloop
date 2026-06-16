<!-- SPDX-License-Identifier: MIT -->
# Module status matrix

One row per module: what's **formally proven**, what's **simulated**, whether a
**FuseSoC core** and **contract** exist, and the synthesis note. The audit
artifact a lab reads first. Hand-authored today; `notes/gen_status_matrix.py`
regenerates the proof column from `formal/work/results.json` (CI, stage 3.3).

Legend: ✅ proven/present · 🟡 documented-not-proven · ⚪ sim-only · — n/a

| Module | Formal | Simulated | Core | Contract | Synth note |
| --- | --- | --- | --- | --- | --- |
| `pwm_generator` | ✅ shoot-through, dead-time, reset | ✅ | ✅ | ✅ | small, fast |
| `current_pi` | ✅ clamp (**envelope**) | ✅ | ✅ | — | fast |
| `speed_iq_pi` | ✅ clamp (**envelope**) | ✅ | ✅ | — | fast |
| `svpwm` | ✅ per-leg duty bound | ✅ bit-exact | ✅ | — | comb. |
| `circle_limit` | 🟡 magnitude (isqrt; documented) | ✅ | ✅ | — | **isqrt = the Fmax limiter** |
| `drv_manager` | ✅ FSM legality | ✅ | ✅ | — | fast |
| `adc_sequencer` | ✅ pulse well-formedness | ✅ | ✅ | — | fast |
| `as5047p_spi_master` | ✅ framing + 1-pulse | ✅ | ✅ | — | fast |
| `ads9224r_master` | ✅ framing + 1-pulse | ✅ | ✅ | — | fast |
| `clarke`,`park`,`inv_park` | — | ✅ bit-exact | ✅ | — | comb. |
| `sincos` | — | ✅ bit-exact | ✅ | — | LUT |
| `commutation` | (via composition) | ✅ | ✅ | — | comb. |
| `speed_pi` | — | ✅ | ✅ | — | fast |
| `speed_meter` | — | ✅ | ✅ | — | fast |
| `spi_drv_master`,`adc_spi_master` | — | ✅ | ✅ | — | fast |
| `as5600_pwm_capture` | — | ✅ | ✅ | — | fast |
| `open_loop_ramp`,`divider32` | — | ✅ | ✅ | — | fast |
| `uart_rx`,`uart_tx`,`uart_regfile` | — | ✅ | ✅ | — | fast |
| `foc_core` | composes proven sub-blocks | ✅ bit-exact | ✅ | ✅ | **Fmax ≈ 3.3 MHz (pipeline needed)** |
| `controller_top` (system) | ✅ composition: no shoot-through | ✅ 401 tests | ✅ (`motorloop.core`) | — | fits ECP5-85F; Fmax ≈ 3.3 MHz |

**Totals:** 12 PROVEN + 1 DOCUMENTED formal; 401-test sim suite; 25 leaf cores +
1 system core; 2 contracts (template + the rest follow `module-contract-template.md`).

**Headline gaps (the honest ones):** (1) timing — the FOC datapath is unpipelined
(Fmax ≈ 3.3 MHz < 25 MHz); pipelining `foc_core`/`circle_limit` is the next RTL
work. (2) validation — verification only; no silicon-correlation tier yet. Both
are the library plan's stages 4–5.
