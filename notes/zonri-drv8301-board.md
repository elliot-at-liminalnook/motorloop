<!-- SPDX-License-Identifier: MIT -->
# ZONRI DRV8301 Power Board Notes

## Current Identification

The physical board appears to be a ZONRI clone or derivative of TI's D3 Engineering DRV8301/DRV8302 EVM High Current design.

Evidence:

- The top side is branded `ZONRI`.
- The main driver IC is marked `DRV8301`.
- The six power MOSFETs are marked `CRSS052N08N`.
- The bottom side has two silkscreen pin tables, labelled `DRV8302` and `DRV8301`.
- The signal names match TI's high-current DRV8301/DRV8302 EVM schematic family: `INH_A`, `INL_A`, `EN_GATE`, `DC-CAL`, `SDI`, `SDO`, `SCLK`, `nSCS`, `SO-1`, `SO-2`, `EMF-A`, `EMF-B`, `EMF-C`, `IOUTA`, `IOUTB`, and `IOUTC`.
- TI's TIDM-THREEPHASE-BLDC-HC-SPI schematic is titled `D3 Engineering - TI - DRV8301/DRV8302 EVM - High Current`.

Primary TI reference:

- `../docs/ti-reference-boards/tidm-threephase-bldc-hc-spi-schematic-tidr738.pdf`
- `../docs/ti-reference-boards/tidm-threephase-bldc-hc-spi-design-files/DRV830x_RevD_HWDevPKG/Schematic/515502~1.PDF`
- Source page: https://www.ti.com/tool/TIDM-THREEPHASE-BLDC-HC-SPI

## DRV8301 Silkscreen Pin Table

The close-up photo `PXL_20260531_152608915.jpg` shows this DRV8301 table. Each row is a two-pin pair:

| Left pin | Right pin |
| --- | --- |
| PWRGD | GND |
| nFAULT | nOCTW |
| SDI | nSCS |
| SCLK | SDO |
| EN-GATE | DC-CAL |
| INL-A | INH-A |
| INL-B | INH-B |
| INL-C | INH-C |
| GND | GND |
| GND | GND |
| SO-1 | SO-2 |
| GND | GND |
| EMF-A | VPDD-OUT, verify label/function |
| EMF-C | EMF-B |
| EX-REF | GND |
| REF+ 1.65V | IOUTA |
| GND | IOUTB |
| GND | IOUTC |

The board also includes a DRV8302 table. Ignore it for our installed DRV8301 board except as a clue that the PCB may support both DRV8301 and DRV8302 assembly variants.

## FPGA-Relevant Digital Signals

Likely FPGA outputs:

- `EN-GATE`
- `DC-CAL`
- `INH-A`, `INL-A`
- `INH-B`, `INL-B`
- `INH-C`, `INL-C`
- SPI master outputs: `SCLK`, `SDI`, `nSCS`

Likely FPGA inputs:

- SPI input: `SDO`
- Fault/status inputs: `nFAULT`, `nOCTW`, maybe `PWRGD`

Check voltage domain before direct connection. The DRV8301 supports 3.3 V and 5 V digital interfaces, but the ZONRI board may include pullups or local rails that need confirmation.

## Analog Signals To ADC

Likely MCP3208 channels:

- `IOUTA`, `IOUTB`, `IOUTC`: board-provided current feedback signals.
- `EMF-A`, `EMF-B`, `EMF-C`: board-provided phase/back-EMF feedback signals.
- Optional: the silkscreened `VPDD-OUT` node or a scaled bus voltage node, only if confirmed by schematic and measured safely.

Do not connect high-voltage motor phases directly to the ADC. Use only board-provided scaled analog outputs after confirming their range with a current-limited supply and DMM/scope.

## Safety Notes

- TI's DRV830x-HC-C2-KIT documentation mentions 8 V to 60 V DC input and high peak current for the TI EVM. Do not assume those ratings apply to the ZONRI board.
- The ZONRI board's actual MOSFETs, current shunts, connectors, copper, thermal path, and assembly quality determine the real safe envelope.
- First power should use a current-limited bench supply, no motor, `EN-GATE` held inactive, and scope/logic analyzer probes ready on PWM, SPI, `nFAULT`, `nOCTW`, and gate-drive status.
- Before connecting the FPGA, measure whether board-side digital pins are pulled to 3.3 V, 5 V, or another rail.
