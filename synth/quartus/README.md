<!-- SPDX-License-Identifier: MIT -->
# Quartus (Intel Cyclone) — authoritative resource + Fmax

Proprietary, license-gated, **not in CI**. Provides the authoritative Intel
numbers to complement the open yosys resource estimate
(`synth/portability_report.md`).

```sh
quartus_sh -t synth/quartus/motorloop.tcl     # map -> fit -> sta
```

Reads the same RTL as the open flows (`controller_top`), with
`motorloop.sdc` (25 MHz clock); reports land under
`synth/quartus/db/output_files/`. Edit `FAMILY`/`DEVICE` in `motorloop.tcl` for
your board. Put the headline ALM/FF/DSP/M9K + TimeQuest Fmax into
`synth/portability_report.md`'s table, labelled **vendor**.
