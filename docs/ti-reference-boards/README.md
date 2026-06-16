<!-- SPDX-License-Identifier: MIT -->
# TI Reference Board Index

These files are the authoritative reference-board collateral most relevant to the ZONRI DRV8301 board.

## Primary Match: DRV830x-HC-C2-KIT / DRV8301-HC-EVM

The ZONRI board photos show a DRV8301 device, CRSS052N08N MOSFETs, and bottom silkscreen pin tables for both DRV8301 and DRV8302 variants. The signal names line up with TI's high-current DRV8301/DRV8302 EVM collateral, especially the `INH_*`, `INL_*`, `EN_GATE`, `DC-CAL`, `SDI`, `SDO`, `SCLK`, `nSCS`, `SO-1`, `SO-2`, `EMF-*`, and `IOUT*` nets.

TI's reference design page says TIDM-THREEPHASE-BLDC-HC-SPI is based on the DRV8301 evaluation kit and publishes the schematic, BOM, design ZIP, and support documents:

- Source page: https://www.ti.com/tool/TIDM-THREEPHASE-BLDC-HC-SPI
- Hardware guide: `drv830x-hc-c2-kit-hardware-reference-guide-tidu317.pdf`
- How-to-run guide: `tidu396-drv830x-hc-c2-kit-how-to-run-guide.pdf`
- Quick start guide: `spruhx4-drv830x-digital-motor-control-kit-quick-start-guide.pdf`
- PDF schematic: `tidm-threephase-bldc-hc-spi-schematic-tidr738.pdf`
- Revision/history schematic packet: `tidr741-drv8301-8302-high-current-evm-revision-history.pdf`
- BOM: `tidm-threephase-bldc-hc-spi-bom-tidr740a.pdf`
- Full design ZIP: `tidm-threephase-bldc-hc-spi-design-files-sloc292.zip`
- Extracted package: `tidm-threephase-bldc-hc-spi-design-files/DRV830x_RevD_HWDevPKG/`

Important extracted files:

- `tidm-threephase-bldc-hc-spi-design-files/DRV830x_RevD_HWDevPKG/Schematic/515502~1.PDF`
- `tidm-threephase-bldc-hc-spi-design-files/DRV830x_RevD_HWDevPKG/Schematic/515502~1.DSN`
- `tidm-threephase-bldc-hc-spi-design-files/DRV830x_RevD_HWDevPKG/BOM/DRV8301EVM-REVD1-RELEASE.pdf`
- `tidm-threephase-bldc-hc-spi-design-files/DRV830x_RevD_HWDevPKG/BOM/DRV8301EVM-REVD1-RELEASE.xls`
- `tidm-threephase-bldc-hc-spi-design-files/DRV830x_RevD_HWDevPKG/Layout/DRV830x_D1-RELEASE.brd`
- `tidm-threephase-bldc-hc-spi-design-files/DRV830x_RevD_HWDevPKG/Layout/DRV830x_D1-RELEASE_GERBER.zip`

Use these for topology, net names, SPI configuration expectations, analog feedback scaling clues, and connector mapping. Do not assume the ZONRI board has identical shunt values, MOSFETs, layout copper, connectors, current rating, or thermal performance.

## Secondary Reference: BOOSTXL-DRV8301

The BOOSTXL-DRV8301 is a smaller 10 A LaunchPad BoosterPack, not the closest physical match to the ZONRI board. It is still useful for DRV8301 bring-up practices, LaunchPad-era signal naming, current-sense behavior, and TI's safe-start recommendations.

- Source page: https://www.ti.com/tool/BOOSTXL-DRV8301
- User guide: `boostxl-drv8301-hardware-users-guide-slvu974.pdf`
- Hardware ZIP: `boostxl-drv8301-hardware-files-slvc539.zip`
- Extracted schematic: `boostxl-drv8301-hardware-files/BOOSTXL-DRV8301 Hardware Files/BOOSTXL-DRV8301_SCH.PDF`
- Extracted BOM: `boostxl-drv8301-hardware-files/BOOSTXL-DRV8301 Hardware Files/BOOSTXL-DRV8301_BOM.xls`

## Notes For Our Board

- Physical board brand: ZONRI.
- Gate driver IC marking: DRV8301.
- MOSFET marking: CRSS052N08N.
- Bottom silkscreen provides two pin tables, one for DRV8302 and one for DRV8301. Use the DRV8301 table for our current board.
- The TI high-current EVM docs mention up to 60 V bus operation and 60 A peak output current, but those are TI EVM statements. They are not validated ratings for the ZONRI clone.
