<!-- SPDX-License-Identifier: MIT -->
# `ads9224r_master` — ADS9224R dual-simultaneous 16-bit ADC master (framing PROVEN)

Master for the **ADS9224R** 16-bit dual *simultaneous*-sampling SAR ADC. In FOC
mode it pulses `CONVST` once per PWM period, `ADC_EMF_LEAD` cycles before the
counter peak (where every low-side shunt conducts), so **both phase currents
are sampled at the same instant** — the hardware resolution of inter-channel
skew (Q21). It holds `CONVST` high briefly, waits for `ready`, then frames the
readout (`ncs` low) and clocks 16 MSB-first bits out of the two data lines
`sdo_a`/`sdo_b` in parallel (same shift/sample timing as the DRV/MCP masters).
The 16-bit two's-complement codes (zero = no current, so no offset subtraction)
are sign-extended to 18 bits and emitted on `foc_cur_a`/`foc_cur_b` with a
single-cycle `foc_valid`. **One CONVST and one current update per period; FSM
legality and `foc_valid` well-formedness are proven.**

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async active-low reset; idle, `ncs`=1, `convst`=0 |
| `pwm_counter` | in | 16 | — | — | up/down PWM counter (0..HALF), for CONVST timing |
| `pwm_up` | in | 1 | — | — | PWM counter direction (1 = counting up) |
| `foc_mode` | in | 1 | — | — | enable FOC conversions; idle when low |
| `convst` | out | 1 | — | 0 | conversion-start; rising edge latches both S/H together |
| `ncs` | out | 1 | — | 1 | active-low chip select; low only during readout |
| `sclk` | out | 1 | — | 0 | divided readout clock (idles low) |
| `sdo_a` | in | 1 | — | — | channel-A serial data, sampled on the high half |
| `sdo_b` | in | 1 | — | — | channel-B serial data, sampled on the high half |
| `ready` | in | 1 | — | — | conversion-complete from the device; gates readout |
| `foc_cur_a` | out | 18 | yes | 0 | phase-A current, sign-extended two's-complement |
| `foc_cur_b` | out | 18 | yes | 0 | phase-B current, sign-extended two's-complement |
| `foc_valid` | out | 1 | — | 0 | 1-cycle pulse; `foc_cur_a/b` valid this cycle |

## Clocking & reset

- **Clock domains:** single `clk`. CONVST is triggered off the PWM counter;
  readout is gated by `ready`; `foc_valid` is the output strobe.
- **Reset:** async active-low `rst_n` → `S_IDLE`, `ncs`=1, `convst`=0,
  currents 0.
- **Latency:** per period — CONVST pulse (`conv_cnt`), wait for `ready`, then
  16 bits of two half-periods (`SCLK = clk / ADC_SPI_DIV`), then a 1-cycle
  S_DONE that sign-extends the codes and pulses `foc_valid`. CONVST launches at
  `pwm_counter == PWM_HALF_PERIOD - ADC_EMF_LEAD` on the up-slope.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `ADC_SPI_DIV` | `8'd26` | readout SCLK divider; each half-period is `ADC_SPI_DIV>>1` clocks (`<= 255`) |
| `PWM_HALF_PERIOD` | `16'd625` | PWM counter peak; sets the off-window center for CONVST timing |
| `ADC_EMF_LEAD` | `16'd160   // cycles before the counter peak` | lead before the peak at which CONVST fires (`CONV_LAUNCH = HALF - LEAD`) |

## Formal contract

- **PROVEN** (`formal/manifest.toml`, `formal/bind/ads9224r_master_fv.sv`): FSM legality: the ADS9224R master state register only ever holds one of the 7 defined states S_IDLE..S_DONE; foc_valid is a well-formed single-cycle pulse - one current update per conversion.
- **Method:** prove, `engine smtbmc boolector`.
- **Assumptions:** single `clk`; async `rst_n` is the design reset; the bench
  ADS9224R model asserts `ready` after a CONVST and drives `sdo_a/sdo_b`
  MSB-first with the expected shift/sample timing; `pwm_counter`/`pwm_up` come
  from the shared center-aligned PWM; `ADC_EMF_LEAD < PWM_HALF_PERIOD`;
  `ADC_SPI_DIV` even, `>= 2`.

## Synthesis fit

- **Device:** ECP5. Small: two 16-bit shift registers, an 8-bit divider, small
  counters and a 7-state FSM (`synth/fmax_module.py ads9224r_master`).

## Reuse notes

- **Language:** Verilog-2005; accepted by Verilator/yosys/Vivado/Quartus.
- **Dependencies:** none — drop in `rtl/ads9224r_master.v`; thread
  `PWM_HALF_PERIOD`/`ADC_EMF_LEAD` to match the PWM generator.
- **Pull it:** `fusesoc run motorloop:ip:ads9224r_master` (core at repo root).
