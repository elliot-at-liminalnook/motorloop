<!-- SPDX-License-Identifier: MIT -->
# Hardware Photo Inventory

Photo source: a local photo directory (not included in this repo); the
observations transcribed below are what matter.

## Confirmed Components

| Photo | Observed component | Markings and notes |
| --- | --- | --- |
| `PXL_20260531_145017259.MP.jpg` | Sipeed Tang Primer 25K Dock | Back side clearly reads `SiPEED TANG Primer 25K Dock`. Photo does not show FPGA top marking/package. |
| `PXL_20260520_225740902.jpg` | ZONRI DRV8301 3-phase gate driver/power board | Board brand `ZONRI`; IC marked `DRV8301`; MOSFETs marked `CRSS052N08N`; outputs silkscreened `OUTA`, `OUTB`, `OUTC`; supply input `VIN` and `GND`; LEDs labelled `FAULT` and `OCTW`. |
| `PXL_20260520_225748215.jpg` | ZONRI board backside | Bottom silkscreen includes separate DRV8302 and DRV8301 signal maps. |
| `PXL_20260531_152608915.jpg` | ZONRI board backside close-up | Clear DRV8301 table with `PWRGD`, `nFAULT`, `nOCTW`, `SDI`, `nSCS`, `SCLK`, `SDO`, `EN-GATE`, `DC-CAL`, `INL-*`, `INH-*`, `SO-*`, `EMF-*`, `EX-REF`, `REF+ 1.65V`, and `IOUT*`. |
| `PXL_20260520_225818801.jpg` | HW-221 level shifter module | Board labelled `HW-221`; chip marked `TXB0108E`; pins labelled `VA`, `VB`, `OE`, `A1..A8`, `B1..B8`, `GND`. |
| `PXL_20260520_225829452.jpg` | MCP3208 ADC DIP | Chip marked `MCP3208-CI/P`. |
| `PXL_20260520_225835974.jpg`, `PXL_20260531_153939766.jpg` | AS5600 magnetic encoder breakout | Breakout connector labels include `OUT`, `PGO`, `GND`, `VCC`, `SCL`, `SDA`. |

## Immediate Implications

- The ZONRI board is very likely a clone or derivative of the TI/D3 Engineering DRV8301/DRV8302 High Current EVM family rather than the small BOOSTXL-DRV8301 BoosterPack.
- Use the ZONRI silkscreen pin map as the physical truth for this board, then cross-check each net against TI's `515502~1.PDF` schematic before wiring.
- The level shifter is confirmed as TXB0108E. Use it cautiously for push-pull SPI/PWM-style signals. Do not use it as the default plan for AS5600 I2C.
- The AS5600 breakout exposes both `OUT` and I2C pins. For first bring-up, `OUT` is attractive because it avoids I2C level-shifting issues.
- The FPGA photo confirms the dock, but not the FPGA chip marking. Read the top-side chip/package marking before creating final Gowin constraints.
