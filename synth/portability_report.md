<!-- SPDX-License-Identifier: MIT -->
# Multi-vendor portability (resource estimates)

`controller_top` mapped to each FPGA family with **yosys vendor backends and no vendor tools** (release-and-portability §4.1) - proof the Verilog-2005 RTL ports off Lattice. **Pre-place&route estimates**; authoritative Fmax/utilisation: `synth/synth_report.md` (ECP5, real P&R), `synth/run_gowin.py` (Gowin, open P&R), `synth/vivado/` + `synth/quartus/` (vendor, where licensed).

| Family | Maps | LUTs | Flow | Full breakdown |
| --- | --- | --- | --- | --- |
| Lattice ECP5 | yes ✓ | 13104 | yosys synth (open-estimate) | `synth/work/portability_lattice.log` |
| Xilinx 7-series | yes ✓ | 12864 | yosys synth (open-estimate) | `synth/work/portability_xilinx.log` |
| Intel Cyclone | yes ✓ | 24971 | yosys synth (open-estimate) | `synth/work/portability_intel.log` |
| Gowin GW5A | yes ✓ | 13458 | yosys synth (open-estimate) | `synth/work/portability_gowin.log` |

**Result:** the RTL maps to 4/4 FPGA families.

_LUTs are the dominant, roughly-comparable resource (the design is LUT-heavy ~13k on every family). Mapped-cell totals and the full per-vendor FF/DSP/BRAM breakdown (primitive names differ per vendor) are in the linked logs. These are pre-P&R estimates - real timing is in the P&R/vendor flows._