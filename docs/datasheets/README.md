# Datasheet Index

Local copies were saved in this directory on 2026-06-08.

## FPGA Board

| Item | Local copy | Online source |
| --- | --- | --- |
| Sipeed Tang Primer 25K docs | See board docs below | https://wiki.sipeed.com/hardware/en/tang/tang-primer-25k/primer-25k.html |
| Tang Primer 25K Dock schematic | `sipeed-tang-primer-25k-dock-schematic.pdf` | https://dl.sipeed.com/fileList/TANG/Primer_25K/02_Schematic/Tang_Primer_25K_Dock_60033_Schematic.pdf |
| Tang Primer 25K core schematic | `sipeed-tang-primer-25k-core-schematic.pdf` | https://dl.sipeed.com/fileList/TANG/Primer_25K/02_Schematic/Tang_Primer_25K_52300_Schematic.pdf |
| Gowin GW5A family datasheet | `gowin-gw5a-datasheet-ds1103e.pdf` | https://cdn.gowinsemi.com.cn/DS1103E.pdf |

## Motor Driver / Power Stage

| Item | Local copy | Online source |
| --- | --- | --- |
| TI DRV8301 gate driver datasheet | `ti-drv8301-datasheet.pdf` | https://www.ti.com/lit/ds/symlink/drv8301.pdf |
| DRV8301 product page | Not a PDF | https://www.ti.com/product/DRV8301 |
| TI DRV8302 gate driver datasheet | `ti-drv8302-datasheet.pdf` | https://www.ti.com/lit/ds/symlink/drv8302.pdf |
| CR Micro CRSS052N08N MOSFET datasheet | `crmicro-crss052n08n-datasheet.pdf` | https://file2.dzsc.com/icpdf/25/02/19/52172_163600637.pdf |
| TI CSD18540Q5B MOSFET datasheet, reference-design comparison part | `ti-csd18540q5b-reference-design-mosfet-datasheet.pdf` | https://www.ti.com/lit/ds/symlink/csd18540q5b.pdf |
| DRV8301 module product notes | Not a PDF | https://www.thanksbuyer.com/drv8301-motor-drive-module-high-power-st-foc-vector-control-bldc-pmsm-drive-62948 |

For the ZONRI power board, prefer the TI reference-board files in `../ti-reference-boards/` over reseller notes. The bottom silkscreen pin maps and net names match TI's DRV8301/DRV8302 high-current EVM family closely.

## ADC

| Item | Local copy | Online source |
| --- | --- | --- |
| Microchip MCP3208 product page | Not a PDF | https://www.microchip.com/en-us/product/MCP3208 |
| Microchip MCP3204/3208 datasheet | `microchip-mcp3208-datasheet.pdf` | https://ww1.microchip.com/downloads/aemDocuments/documents/APID/ProductDocuments/DataSheets/21298e.pdf |

## Rotor Angle Sensor

| Item | Local copy | Online source |
| --- | --- | --- |
| ams OSRAM AS5600 product page | Not a PDF | https://ams-osram.com/products/sensor-solutions/position-sensors/ams-as5600-position-sensor |
| ams OSRAM AS5600 datasheet | `ams-osram-as5600-datasheet.pdf` | https://look.ams-osram.com/m/7059eac7531a86fd/original/AS5600-DS000365.pdf |

## Level Shifter

| Item | Local copy | Online source |
| --- | --- | --- |
| TI TXB0108 product page | Not a PDF | https://www.ti.com/product/TXB0108 |
| TI TXB0108 datasheet | `ti-txb0108-datasheet.pdf` | https://www.ti.com/lit/ds/symlink/txb0108.pdf |
| TI E2E note on TXB0108 and I2C | Not a PDF | https://e2e.ti.com/support/logic-group/logic/f/logic-forum/1319987/txb0108-txb-for-i2c |

## Open Items

- Confirm exact FPGA package/part marking before assigning pins.
- Physical photos confirm the level-shifter board is marked TXB0108E.
- Search for an original ZONRI board schematic or vendor manual if the board has a model number beyond the DRV8301 module description. Until then, use TI DRV830x-HC-C2-KIT collateral as the reference topology.
