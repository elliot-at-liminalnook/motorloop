<!-- SPDX-License-Identifier: MIT -->
# Using The TI Reference Files

This note explains what the downloaded TI collateral gives us, whether each format is an industry standard, and how to inspect or use it with open-source tools where possible.

## Big Picture

The TI files give us a reference implementation of a DRV8301/DRV8302 high-current BLDC/PMSM inverter. They are not proof that the ZONRI clone is electrically identical, but they give us the best authoritative baseline for:

- Power-stage topology.
- DRV8301 pin use and SPI configuration expectations.
- PWM, enable, fault, and status signal naming.
- Current-sense and back-EMF analog feedback paths.
- Gate-drive layout and high-current routing practices.
- Component values for the TI EVM.
- Safe bring-up sequence and incremental test strategy.
- Simulation reference models for parts of the DRV8301, especially analog/current-sense behavior and digital I/O buffers.

Use these files to build our wiring map and simulation model, then verify differences against the actual ZONRI board with photos, continuity checks, and measurements.

## File Types And What They Give Us

| File type | Examples | What it gives us | Standard/open? | Best open-source path |
| --- | --- | --- | --- | --- |
| PDF schematic | `515502~1.PDF`, `tidm-threephase-bldc-hc-spi-schematic-tidr738.pdf` | Readable schematic pages: nets, topology, connector maps, feedback circuits, power supplies. | PDF is a standard document format. | Any PDF reader; `pdftotext`/`pdfgrep` for search. |
| PDF hardware/user guides | `tidu317`, `tidu396`, `spruhx4` | Bring-up steps, ratings for the TI EVM, connector descriptions, safety warnings. | PDF. | PDF reader; `pdftotext`. |
| BOM spreadsheets | `.xls`, `.xlsx`, BOM PDFs | Component values and manufacturer part numbers. Useful for shunts, dividers, gate resistors, no-pop variants. | Office formats, common but not electronics-specific standards. | LibreOffice Calc; Python with `openpyxl`/`pandas` for `.xlsx`. |
| Gerbers and drill files | `DRV830x_D1-RELEASE_GERBER.zip`, `.art`, `.hol` | PCB copper, mask, silk, paste, fab/assembly artwork, drill holes. Useful for measuring placement/routing and high-current paths. | Gerber is the PCB fabrication de facto standard; Excellon/XNC-style drill files are common manufacturing formats. | KiCad GerbView or `gerbv`. |
| ODB++ | BOOSTXL `ODB.zip` | Rich PCB manufacturing/assembly dataset, often with layers, components, drill, net, and assembly info. | De facto PCB manufacturing exchange standard, but owned by Siemens. | Open-source support is weaker; use Gerbers first. |
| OrCAD schematic source | `515502~1.DSN` | Original schematic database if opened in Cadence/OrCAD. | Proprietary Cadence format. Also note: this is not the same as Specctra `.dsn` used by FreeRouting. | Use the PDF schematic instead; Cadence OrCAD X Free Viewer is a proprietary fallback. |
| Allegro board source | `DRV830x_D1-RELEASE.brd` | Original PCB layout database if opened in Cadence Allegro. | Proprietary Cadence format. | Use Gerbers and layout PDFs instead; Cadence viewer is a proprietary fallback. |
| Assembly/fab drawings | `*_ASSY_DWG.pdf`, `*_FAB_DWG.pdf`, `DRV830~3.PDF` | Board outline, drill/fab details, assembly views, layer stack clues. | PDF. | PDF reader. |
| TI SPICE library | `DRV8301.LIB` | Behavioral analog model. In this package the top-level subcircuit is `DRV8301 AGND AVDD VINM VINP VOUT`, so it is not a complete 56-pin gate-driver/power-stage model. It mostly helps with current-sense amplifier behavior. | SPICE is a de facto circuit-simulation standard, but vendor dialects differ. | Try `ngspice`, KiCad simulator, or Qucs-S; expect manual adaptation. |
| TINA-TI files | `.TSC`, `.TSM`, `.TLD` | TI/DesignSoft schematic and macro files for TINA-TI. Useful if we want to reproduce TI's intended analog simulation. | Proprietary TINA-TI format. | Use extracted `.LIB` first; TINA-TI is free but not open source. |
| IBIS model | `drv8301.ibs` | Digital I/O buffer model: package parasitics, input/output buffer electrical behavior. Useful for SPI/PWM edge integrity, not logic or motor behavior. | IBIS is an industry standard managed by the IBIS Open Forum. | Text inspection; specialized IBIS tools are often commercial. Use as a reference unless signal-integrity problems appear. |
| 3D/model files | `.bdf`, `.ldf` | Mechanical/EDA model collateral from the original package. | Vendor/EDA-specific. | Lower priority; use photos and fab drawings first. |

## What Is Immediately Useful For Our Simulation

For the plant models (C++ primary, Modelica oracle — see [architecture](architecture.md)):

- Use the schematic and BOM to parameterize the inverter topology, shunt values, sense amp gains, feedback dividers, fault pins, and any input filtering.
- Use the layout/app notes to decide which parasitics matter: gate resistance, switching-node capacitance, shunt Kelvin routing, bulk capacitance, and ground offsets.
- Use the DRV8301 datasheet plus app notes for SPI registers, overcurrent mode, dead time, gain settings, and fault/status behavior.
- Use the SPICE `.LIB` as a behavioral reference for analog amplifier limits, not as a complete DRV8301 model.

For Verilog/FPGA verification:

- Use the schematic and silkscreen table to define expected pins: `EN_GATE`, `DC_CAL`, `INH_*`, `INL_*`, `SDI`, `SDO`, `SCLK`, `nSCS`, `nFAULT`, `nOCTW`, `PWRGD`.
- Use the DRV8301 datasheet for SPI transaction format and reset/fault behavior.
- Use the IBIS model only if edge rates, level shifting, or cable length create signal-integrity concerns.

For physical bring-up:

- Use TI's how-to-run guide as a safety checklist, not as a guarantee that ZONRI can use the same current limits.
- Use Gerbers/layout PDFs to understand where high-current and sensitive analog paths are on the reference design.
- Measure the actual ZONRI board before connecting the FPGA: logic voltage pullups, analog output ranges, current-sense offsets, and bus-voltage feedback scaling.

## Recommended Open-Source Toolchain

On Ubuntu/Debian-style systems, the practical open-source install set is:

```bash
sudo apt update
sudo apt install kicad gerbv ngspice qucs-s libreoffice poppler-utils unzip
```

What each tool is for:

- `kicad`: schematic/PCB environment; includes GerbView on many installs and integrates ngspice for circuit simulation.
- `gerbv`: lightweight open-source Gerber/Excellon viewer.
- `ngspice`: command-line SPICE simulator.
- `qucs-s`: schematic-based circuit simulation front end that can use ngspice.
- `libreoffice`: BOM spreadsheet inspection.
- `poppler-utils`: `pdfinfo`, `pdftotext`, and related PDF inspection tools.
- `unzip`: unpack TI design packages.

Optional proprietary-but-free fallbacks:

- Cadence OrCAD X Free Viewer: can open OrCAD `.dsn` and Allegro `.brd` files read-only on Windows.
- TINA-TI: can open TI's `.TSC`/`.TSM` simulation files. It is free from TI but not open source.

## First Inspection Workflow

1. Read `515502~1.PDF` and compare its connector/signal names with the ZONRI silkscreen photos.
2. Open the DRV8301 BOM in LibreOffice and identify the shunt resistors, gate resistors, feedback dividers, op-amps, pullups/pulldowns, and no-pop parts.
3. Open the Gerber ZIP in KiCad GerbView or `gerbv`; inspect copper, shunt placement, gate-drive routing, and bulk capacitor placement.
4. Search the PDFs for each signal we plan to wire: `INH_A`, `INL_A`, `EN_GATE`, `DC_CAL`, `SDI`, `SDO`, `SCLK`, `nSCS`, `nFAULT`, `nOCTW`, `SO1`, `SO2`, `EMF`.
5. Build a simple ngspice test circuit for the current-sense amplifier `.LIB`, only if its behavior matters for ADC calibration.
6. Treat every ZONRI deviation as real until measured. The TI EVM is our reference design, not our board's datasheet.

## Current Local Tool Status

Installed on 2026-06-08:

| Tool | Command | Installed version/source |
| --- | --- | --- |
| KiCad | `kicad`, `kicad-cli`, `gerbview` | `kicad 9.0.8+dfsg-1` from Ubuntu `resolute/universe` |
| gerbv | `gerbv` | `gerbv 2.10.0-2build1` from Ubuntu `resolute/universe` |
| ngspice | `ngspice` | `ngspice 45.2+ds-1` from Ubuntu `resolute/universe` |
| Qucs-S | `qucs-s` | Upstream AppImage `Qucs-S-26.1.1-linux-x86_64.AppImage`, installed at `/opt/qucs-s/` with launcher `/usr/local/bin/qucs-s` |
| LibreOffice | `libreoffice`, `localc` | `LibreOffice 26.2.3.2` |
| PDF tools | `pdfinfo`, `pdftotext` | `poppler-utils 26.01.0-2build2` |
| Archive tools | `unzip` | `unzip 6.0-29ubuntu1` |

The Qucs-S AppImage SHA-256 was verified against the upstream `hashes.sha256` file from https://github.com/ra3xdh/qucs_s/releases/tag/26.1.1.
