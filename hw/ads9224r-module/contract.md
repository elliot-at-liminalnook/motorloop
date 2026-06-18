<!-- SPDX-License-Identifier: CERN-OHL-S-2.0 -->
# Open ADS9224R current-sense module — board contract

The hardware datasheet for the module (mirrors `rtl/contracts/` for the RTL): the
interface, the analog transfer, provenance, and validation status. Source of
truth for the electrical values: `sim/config/params.toml`
`[circuit.ads9224r_module]` + the derived `[feedback.current_ads9224r]`.

## What it is

A dual-channel 16-bit **simultaneous-sampling** current-sense front-end for FPGA
FOC, built around the TI ADS9224R. Each phase: low-side shunt → THS4551
fully-differential driver → RC charge-bucket → ADS9224R differential input; a
buffered REF6041 4.096 V reference with a reservoir cap. One `CONVST` samples
both channels at the same instant (resolves Q21).

## Electrical transfer

| Quantity | Value | Source |
| --- | --- | --- |
| Shunt (default) | 2 mΩ | `circuit.ads9224r_module.shunt` |
| FDA differential gain | 20 (Rf/Rg = 2 kΩ/100 Ω) | derived `feedback.current_ads9224r.fda_gain` |
| Full-scale current | ±102.4 A | derived `feedback.current_ads9224r.full_scale_a` |
| Scale factor | 320 codes/A (16-bit signed) | derived `feedback.current_ads9224r.codes_per_amp` |
| Reference | 4.096 V | `circuit.ads9224r_module.ref_v` |
| Charge-bucket | Rflt 10 Ω, Cflt 1.5 nF | `circuit.ads9224r_module.flt_*` |
| Acq. settling estimate | 1.6e-6 (< 0.5 LSB) | derived `adc.acq_settle_residual_ads9224r` |

Scaling is SPICE-cross-checked (`test_ads9224r_frontend_dc`, slope = gain·shunt →
codes/A); settling is SPICE-cross-checked (`test_ads9224r_settle_transient`).
Re-target via `circuit.ads9224r_module.{shunt,fda_rf,fda_rg}` + `derive_params.py
--update`; other build options in `bom.csv` / `README.md`.

## Digital interface (FPGA header J1)

Drives [`rtl/ads9224r_master.v`](../../rtl/ads9224r_master.v) directly (1:1 pins):
`convst, ncs, sclk, sdo_a, sdo_b, ready` + 3V3 + GND. Codes are two's-complement,
MSB-first on the two SDO lanes; zero current = zero code (no offset subtraction).
3.3 V digital rail mates the ULX3S directly (no level translator). Driver timing
+ formal contract: [`rtl/contracts/ads9224r_master.md`](../../rtl/contracts/ads9224r_master.md).

## Provenance & validation status

- **Designed + simulated.** Values are `assumed` / datasheet-typical / EVM-topology
  baseline (blocked_by **Q23**) — the analog front-end is validated against the
  ngspice models (`sim/circuits/ads9224r_*.cir`), not silicon.
- **Pending (Q23, maintainer/lab — checklist §10):** confirm the reference IC/value,
  the FDA gain + shunt scaling (codes/A), and acquisition settling / ENOB at the
  conversion rate on a fabricated board; then promote the `assumed` values to
  `measured`. Measure inter-channel skew to confirm simultaneity on hardware.

## License

CERN-OHL-S-2.0 (hardware). Designed from TI's public datasheet/EVM application
topology; own schematic + layout (no TI EDA files copied). See `README.md` for IP
hygiene.
