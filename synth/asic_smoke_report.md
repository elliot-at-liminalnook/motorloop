<!-- SPDX-License-Identifier: MIT -->
# ASIC-readiness smoke (robotics-ip stage 10)

Each reusable block driven to gate-level synthesis with yosys and checked for ASIC cleanliness: maps to gates, **no inferred latches**, **no combinational loops / multiple drivers** (`check -assert`).

> **Honest framing.** This is an ASIC-*synthesizability* smoke, **not** a tapeout and **not** real-PDK place&route. Post-route timing/area need OpenROAD + the sky130 PDK (`synth/openlane/` wires that flow). The FPGA flow reports a real post-route system Fmax of **41.3 MHz** (stage 6.5).

| Block | Synthesizes | Inferred latches | Multidriver/loop | Gate cells |
| --- | --- | --- | --- | --- |
| `pwm_generator` | yes ✓ | 0 ✓ | none ✓ | 922 |
| `commutation` | yes ✓ | 0 ✓ | none ✓ | 15 |
| `svpwm` | yes ✓ | 0 ✓ | none ✓ | 2042 |
| `current_pi` | yes ✓ | 0 ✓ | none ✓ | 546 |
| `circle_limit_seq` | yes ✓ | 0 ✓ | none ✓ | 6029 |
| `divider32` | yes ✓ | 0 ✓ | none ✓ | 528 |
| `spi_drv_master` | yes ✓ | 0 ✓ | none ✓ | 156 |
| `as5047p_spi_master` | yes ✓ | 0 ✓ | none ✓ | 172 |
| `drv_manager` | yes ✓ | 0 ✓ | none ✓ | 791 |
| `adc_sequencer` | yes ✓ | 0 ✓ | none ✓ | 477 |
| `axil_regfile` | yes ✓ | 0 ✓ | none ✓ | 571 |
| `wb_regfile` | yes ✓ | 0 ✓ | none ✓ | 545 |
| `axis_sampler` | yes ✓ | 0 ✓ | none ✓ | 85 |
| `foc_core` | yes ✓ | 0 ✓ | none ✓ | 28491 |

**Result:** 14/14 blocks ASIC-synthesizable and structurally clean — all clean.
