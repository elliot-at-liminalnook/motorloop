<!-- SPDX-License-Identifier: MIT -->
# Vivado (Xilinx 7-series) — authoritative resource + timing

Proprietary, license-gated, **not in CI**. Provides the authoritative Xilinx
numbers to complement the open yosys resource estimate
(`synth/portability_report.md`) and the open ECP5 P&R Fmax.

```sh
vivado -mode batch -source synth/vivado/motorloop.tcl                 # xc7a35t
vivado -mode batch -source synth/vivado/motorloop.tcl -tclargs xc7a100tcsg324-1
```

Reads the same RTL as the open flows (`controller_top`, the vendor-neutral top),
out-of-context, with a 25 MHz clock; writes `utilization.rpt` + `timing.rpt`.
Put the headline LUT/FF/DSP/BRAM + WNS→Fmax into `synth/portability_report.md`'s
table, labelled **vendor**.
