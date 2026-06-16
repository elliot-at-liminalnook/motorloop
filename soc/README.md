<!-- SPDX-License-Identifier: MIT -->
# Reference SoC — a RISC-V drives the motorloop controller over AXI-Lite

This is the "show, don't tell" artifact (adoption-roadmap §2): not a bus wrapper
in isolation, but a complete **RISC-V SoC that spins the motor** by writing the
controller's registers over AXI-Lite and reading back telemetry.

```
  VexRiscv  ──Wishbone──▶ LiteX interconnect ──AXI-Lite──▶ motorloop_axil_top
  (firmware)                                                 ├─ axil_regfile (PROVEN slave)
                                                             └─ controller_top (the BLDC FOC controller)
```

## Why the ULX3S (Radiona, Lattice ECP5 LFE5U-85F)

- **Same device the open synth flow already targets** (`synth/`): the controller
  is known to fit and close timing at 64 MHz on this exact part.
- **Fully open toolchain** — yosys + nextpnr-ecp5 + ecppack (LiteX's `trellis`),
  the same stack in `toolchain.lock`; no vendor tools.
- **First-class LiteX support** (`litex_boards/targets/radiona_ulx3s.py`).
- **50 MHz default sys-clk** sits comfortably under the controller's 64 MHz Fmax.

(Wishbone alternative: LiteX is Wishbone-native, so `rtl/bus/wb_regfile.v` can
attach with no AXI bridge. This SoC leads with the AXI-Lite wrapper per the
roadmap; swapping in `wb_regfile` is a smaller, bridge-free variant.)

## Build the gateware

Needs LiteX + a RISC-V GCC + the OSS CAD Suite (yosys/nextpnr/ecppack) on PATH.

```sh
# LiteX (one time): see https://github.com/enjoy-digital/litex#quick-start
python3 -m venv ~/litex-venv && . ~/litex-venv/bin/activate
curl -sL https://raw.githubusercontent.com/enjoy-digital/litex/master/litex_setup.py -o /tmp/ls.py
python3 /tmp/ls.py --init --install && python3 /tmp/ls.py --gcc=riscv

source ~/oss-cad-suite/environment              # yosys/nextpnr-ecp5/ecppack
python3 soc/motorloop_soc.py --build            # -> build/radiona_ulx3s/gateware/*.bit
python3 soc/motorloop_soc.py --build --load     # + flash a connected ULX3S
```

## Build + run the firmware

```sh
make -C soc/firmware BUILD_DIR=$(pwd)/build/radiona_ulx3s   # -> firmware.bin
# load firmware.bin over the LiteX UART (litex_term) or pack it into the gateware
litex_term /dev/ttyUSB0 --kernel soc/firmware/firmware.bin
```

The firmware (`firmware/main.c`) writes `{mode=FOC, target_speed, use_axi}` and
then prints `speed / sector / configured / faults / flags` from the telemetry
registers — the controller spinning a motor, commanded from C over AXI-Lite.

## Pins

`motorloop_soc.py` maps the controller pins (3× `inh`/`inl`, `en_gate`, the
DRV8301 SPI, the MCP3208 SPI, `angle_pwm`, `nfault`/`noctw`) onto the ULX3S
`gp*`/`gn*` header in the `_motor_io` extension — **adjust those pin names to
your gate-driver wiring.** The default BOM is DRV8301 + MCP3208 + AS5600
(`rtl/soc/motorloop_axil_top.v` ties the platform straps accordingly).

## What's proven where (honest framing)

- **CI (no board, no LiteX needed):** `sim/cocotb/tb_motorloop_axil_top.py`
  drives `motorloop_axil_top` with a cocotbext-axi master (the role the RISC-V
  plays) — register round-trip + telemetry + the controller running. This is the
  CI-gated proof of the **CPU↔AXI↔controller plumbing**. Run it with
  `make cocotb` (or the cocotb venv).
- **This SoC, build-validated:** `python3 soc/motorloop_soc.py` (or `--sim`)
  generates the full gateware, compiles the BIOS with RISC-V GCC, and Verilates
  the whole SoC into `Vsim` — the SoC map shows `motor → motorloop_axil_top`
  ("AXI-Lite 32-bit → Wishbone"), confirming the integration elaborates and
  links end to end. To watch the RISC-V **boot live**, run `make soc-sim` on a
  real terminal (serial2console needs a TTY) and allow time for the ~one-off
  Verilator build of the full SoC (several minutes); the BIOS then prints over
  the sim serial.
- **On hardware:** the integration made real — a RISC-V issuing the bus
  transactions on the ULX3S (`--build --load`).
- **The motor actually spinning:** the cycle-accurate C++ co-sim (with the plant
  + golden device models) and a hardware bring-up — *not* the plumbing test.
