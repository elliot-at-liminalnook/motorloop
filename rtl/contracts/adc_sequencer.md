<!-- SPDX-License-Identifier: MIT -->
# `adc_sequencer` — Sector-aware ADC schedule + FOC current sampling (pulse well-formedness PROVEN)

Schedules the single MCP3208 across the PWM period (the 3.3 V budget) and
drives `adc_spi_master`. It runs in three modes. **Normal (six-step / sensored):
two conversions per period** — slot 1 (down slope, `CUR_LAUNCH`) samples the
solidly-conducting low-side phase current; slot 2 (up slope, `EMF_LAUNCH =
PWM_HALF_PERIOD − ADC_EMF_LEAD`) samples the floating-phase back-EMF, timed so
the hold aperture lands in the off-window centered on the counter peak — except
every 8th period it samples VBUS (ch6) instead. **DC_CAL window:** round-robins
the three current channels and stores their offsets (amp inputs shorted),
rejecting implausible codes (E10) within ±`DC_CAL_TOL` of midpoint 2048.
**FOC mode:** near the counter peak it reads ch0 then ch1, subtracts the stored
offsets, and emits the signed `(ia, ib)` with `foc_valid`. Guards: E4 skips the
EMF sample when the duty leaves the off-window too narrow; E14 raises `adc_stuck`
after `ADC_STUCK_N` rail-pinned (0/4095) conversions. Channels: 0..2 IOUTA/B/C,
3..5 EMF-A/B/C, 6 VBUS. **All sample strobes and `adc_start` are proven
single-cycle pulses.**

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset; offsets→2048, no pending conv |
| `pwm_counter` | in | 16 | — | — | up/down PWM counter, for slot timing |
| `pwm_up` | in | 1 | — | — | PWM direction (1 = up slope) |
| `period_start` | in | 1 | — | — | period-wrap pulse; advances the every-8th VBUS counter |
| `duty_compare` | in | 16 | — | — | active duty; E4 EMF-skip when off-window too narrow |
| `low_phase` | in | 2 | — | — | sector's solidly-conducting low-side phase (slot 1) |
| `float_phase` | in | 2 | — | — | sector's floating phase (slot 2 EMF channel) |
| `dc_cal_window` | in | 1 | — | — | DC-cal active: capture amp offsets instead of sampling |
| `cur_code` | out | 12 | — | 0 | latest current sample (offset NOT removed) |
| `emf_code` | out | 12 | — | 0 | latest floating-phase EMF sample (held on skip) |
| `vbus_code` | out | 12 | — | 0 | latest bus-voltage sample |
| `offset_a` | out | 12 | — | 2048 | captured phase-A current offset |
| `offset_b` | out | 12 | — | 2048 | captured phase-B current offset |
| `offset_c` | out | 12 | — | 2048 | captured phase-C current offset |
| `offset_fault` | out | 1 | — | 0 | E10: an implausible DC-cal offset was rejected |
| `adc_stuck` | out | 1 | — | — | E14: `stuck_count >= ADC_STUCK_N` (codes pinned at rail) |
| `cur_valid` | out | 1 | — | 0 | 1-cycle pulse; `cur_code` updated |
| `emf_valid` | out | 1 | — | 0 | 1-cycle pulse; `emf_code` updated |
| `adc_start` | out | 1 | — | 0 | 1-cycle start to `adc_spi_master` |
| `adc_channel` | out | 3 | — | 0 | selected MCP3208 channel |
| `adc_busy` | in | 1 | — | — | ADC master busy |
| `adc_done` | in | 1 | — | — | ADC master conversion-done strobe |
| `adc_code` | in | 12 | — | — | latest ADC result |
| `foc_mode` | in | 1 | — | — | enable FOC ch0/ch1 current sampling |
| `foc_cur_a` | out | 18 | yes | 0 | phase-A current, offset-removed (FOC mode) |
| `foc_cur_b` | out | 18 | yes | 0 | phase-B current, offset-removed (FOC mode) |
| `foc_valid` | out | 1 | — | 0 | 1-cycle pulse; `foc_cur_a/b` valid |

## Clocking & reset

- **Clock domains:** single `clk`. Conversions go out via the
  `adc_start`/`adc_busy`/`adc_done`/`adc_code` handshake to `adc_spi_master`;
  slots are gated on `pwm_counter`/`pwm_up`.
- **Reset:** async active-low `rst_n` → no pending conversion, offsets = 2048,
  all strobes/flags cleared.
- **Latency:** event-driven per PWM period. A conversion takes ~560 clk and the
  period is `2·PWM_HALF_PERIOD` (1250 clk by default), so the EMF (up-slope) and
  current (down-slope, `CUR_LAUNCH`) slots — or the FOC ch0→ch1 pair — fit
  within one period; outputs strobe as each conversion completes.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `PWM_HALF_PERIOD` | `16'd625` | counter peak; sets `EMF_LAUNCH` and the period length |
| `ADC_EMF_LEAD` | `16'd160` | lead before the peak at which the EMF/FOC conversion launches |
| `DC_CAL_TOL` | `12'd256` | E10 plausibility half-window around midpoint 2048 for offsets |
| `ADC_STUCK_N` | `8'd32` | E14 consecutive rail-pinned conversions before `adc_stuck` |
| `EMF_SKIP_MARGIN` | `16'd24` | E4 off-window margin; skip EMF when `duty_compare > HALF − margin` |

## Formal contract

- **PROVEN** (`formal/manifest.toml`, `formal/bind/adc_sequencer_fv.sv`): Sample strobes (foc_valid, cur_valid, emf_valid) and adc_start are well-formed single-cycle pulses - one event per sample.
- **Method:** prove, `engine smtbmc boolector`.
- **Assumptions:** single `clk`; async `rst_n` is the design reset;
  `adc_busy`/`adc_done`/`adc_code` obey the `adc_spi_master` handshake (one
  `done` per `start`); `pwm_counter`/`pwm_up`/`period_start` come from the
  shared center-aligned PWM; `low_phase`/`float_phase` are the current sector's
  phases; `ADC_EMF_LEAD < PWM_HALF_PERIOD`.

## Synthesis fit

- **Device:** ECP5. Modest control logic: slot/FOC/DC-cal FSMs, the offset and
  code registers, small counters and comparators — no datapath
  (`synth/fmax_module.py adc_sequencer`).

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none internally — pairs with an external `adc_spi_master`
  via the ADC handshake ports.
- **Pull it:** `fusesoc run motorloop:ip:adc_sequencer` (core at repo root).
