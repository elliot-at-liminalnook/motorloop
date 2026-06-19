<!-- SPDX-License-Identifier: MIT -->
# TI Simulation Model Index

TI publishes several DRV8301 electrical model files. These are not FMUs and will not drop directly into the OpenModelica plant, but they are useful references for validating simplified behavioral models.

| File | Contents |
| --- | --- |
| `spnm068-drv8301-tina-ti-spice-model.zip` | TI DRV8301 TINA-TI/SPICE model package. |
| `spnm068-drv8301-tina-ti-spice-model/DRV8301/Release_TI/TINA/DRV8301_TINA_AIO/DRV8301_TINA_AIO_SPICE_MODEL/DRV8301.LIB` | Extracted SPICE library. |
| `spnm068-drv8301-tina-ti-spice-model/DRV8301/Release_TI/TINA/DRV8301_TINA_AIO/DRV8301_TINA_AIO_SPICE_MODEL/DRV8301.TSM` | TINA macro model. |
| `spnm068-drv8301-tina-ti-spice-model/DRV8301/Release_TI/TINA/DRV8301_TINA_AIO/DRV8301_TINA_AIO_REF_DESIGN/DRV8301.TSC` | TINA reference design included in the model ZIP. |
| `spnm069-drv8301-tina-ti-reference-design.tsc` | Separate TI TINA-TI reference design download. |
| `slom252-drv8301-ibis-model.zip` | TI DRV8301 IBIS model package. |
| `slom252-drv8301-ibis-model/DRV8301_IBIS/drv8301.ibs` | Extracted IBIS model. |

Source page:

- DRV8301 product/tools page: https://www.ti.com/product/DRV8301/toolssoftware

## Open ADS9224R module — Tier-3 vendor macromodels (sim-validation §3)

Portal-gated TI downloads (PSpice-for-TI / TINA-TI). Drop the extracted `.LIB`
at the path below and the Tier-3 tests (`sim/tests/test_ads9224r_vendor.py`)
cross-check automatically; absent, they skip (CI stays green). Confirm each
model's `.SUBCKT` pin order against the netlist instantiation when first added.

| File (place here) | Model | Source |
| --- | --- | --- |
| `ths4551/THS4551.LIB` | THS4551 fully-differential ADC driver | https://www.ti.com/product/THS4551 → Design & development → simulation models (PSpice-for-TI / TINA-TI) |
| `ref6041/REF6041.LIB` | REF6041 4.096 V reference + buffer (external-ref option, §3.4) | https://www.ti.com/product/REF6041 → Design & development → simulation models |
| (optional) `ads9224r/ADS9224R.TSM` | ADS9224R TINA model / input model | https://www.ti.com/product/ADS9224R → Design & development |
