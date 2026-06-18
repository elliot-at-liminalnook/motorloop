<!-- SPDX-License-Identifier: CERN-OHL-S-2.0 -->
# Open ADS9224R simultaneous current-sense module

A small **open-hardware** breakout for the TI **ADS9224R** — a dual-channel,
16-bit, 3 MSPS **simultaneous-sampling** SAR ADC — built as a motor
phase-current front-end for FPGA FOC. It exists because, as of this writing,
**no open-source ADS9224R board exists** (researched: TI's `ADS9224REVM-PDK` is
the only reference, and it ships no open EDA source). This module is designed off
the EVM's published topology and released as KiCad source + BOM + HDL driver +
cocotb + analog provenance.

> **Status: designed + simulated, NOT yet fabricated or bench-validated.**
> The schematic, scaling, and analog front-end are provenance-tracked
> (`sim/config/params.toml` `[circuit.ads9224r_module]`) and SPICE-cross-checked
> (`sim/circuits/ads9224r_*.cir`, `test_ads9224r_*`). Component values are
> `assumed` / datasheet-typical pending hardware bring-up (**Q23**). Do not treat
> any number here as a measured hardware result.

## Why this part

A SAR ADC's input is a switched cap-DAC — a dynamic load — so it needs a
low-impedance, low-distortion driver to settle within the acquisition window.
Each channel is driven by a **THS4551 fully-differential amplifier** into an
**RC charge-bucket** (`flt_r`/`flt_c`), with a **buffered low-noise reference**
and a reservoir cap (the SAR draws charge from REF every conversion). One
`CONVST` samples both phase currents at the **same instant** — the hardware
resolution of inter-channel skew (**Q21**), quantified in the part-comparison
study (`notes/part-comparison-report.md`, T3/T4).

## Signal chain

```
phase shunt (Kelvin) --diff--> THS4551 FDA (gain Rf/Rg) --> Rflt --+--> ADS9224R AINx+
                                                                   Cflt
                                              REF6041 4.096 V --buf--> REFx (+ reservoir)
ADS9224R: CONVST / CS / SCLK / SDO_A / SDO_B / READY --> FPGA header (3.3 V)
```

## Current scaling (build options)

Differential full-scale `±I_fs = ref_v / (shunt · gain)`, codes/A `= 32768 / I_fs`
(16-bit signed). Derived + SPICE-checked; pick shunt + FDA gain for your motor:

| Shunt | FDA gain (Rf/Rg) | Full-scale | codes/A | Use |
| --- | --- | --- | --- | --- |
| 2 mΩ | 20 (2k/100) | ±102 A | 320 | **default** |
| 1 mΩ | 20 (2k/100) | ±205 A | 160 | high current |
| 1 mΩ | 41 (4.02k/100) | ±100 A | 328 | high current, finer |
| 5 mΩ | 20 (2k/100) | ±41 A | 800 | low current |
| 10 mΩ | 20 (2k/100) | ±20 A | 1600 | precision / low current |

The default (2 mΩ, gain 20) is the value committed in `params.toml`; change
`circuit.ads9224r_module.{shunt,fda_rf,fda_rg}` and re-run
`derive_params.py --update` to retarget.

## FPGA wiring (header J1)

Pins map 1:1 to [`rtl/ads9224r_master.v`](../../rtl/ads9224r_master.v):
`convst, ncs, sclk, sdo_a, sdo_b, ready` + 3V3 + GND. The 3.3 V digital rail
mates the **ULX3S directly** — no level translator (it is not 5 V-tolerant).
Driver contract + scaling: [`rtl/contracts/ads9224r_master.md`](../../rtl/contracts/ads9224r_master.md).

## Files

- `bom.csv` — parts + MPNs + distributor PNs + provenance.
- `module.kicad_sch` — generated passive-network schematic (`gen_ads9224r_sch.py`).
- `contract.md` — the board datasheet (interface, scaling, validation status).
- analog model: `sim/circuits/ads9224r_frontend.cir` (scaling) +
  `ads9224r_settle.cir` (acquisition settling); figures in `figures/ads9224r-module/`.

## Ordering (after bring-up review)

Generate Gerbers/drill/pick-place from the KiCad PCB with `kicad-cli pcb export`
(see `Makefile` `ads9224r-fab` once the layout is routed). The layout/routing and
fabrication are the interactive-KiCad / maintainer step (checklist §4/§10).

## License & IP hygiene

- **Hardware** (`hw/ads9224r-module/`): **CERN-OHL-S-2.0** (see `LICENSES/`).
  Software in this repo stays MIT.
- Designed from TI's **public datasheet/EVM application topology** — the intended
  use of an application circuit. We author our **own** schematic + layout; we do
  **not** copy TI's EDA files, Gerbers, or layout artwork, and we don't
  redistribute TI's PDFs (kept as gitignored `proprietary-reference`). No TI
  endorsement is implied. For commercial volume production, do a
  freedom-to-operate / patent review first. Not legal advice.
