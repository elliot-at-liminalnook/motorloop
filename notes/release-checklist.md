<!-- SPDX-License-Identifier: MIT -->
# Release checklist (v0.1.0) — for the maintainer to execute

The repo prepares every release artifact (tier2-adoption §3B); the steps below
are the **git/GitHub/Zenodo actions only the maintainer runs**. Release tagging
and publishing stay deliberate maintainer actions.

## 0. Pre-flight (verify on a clean checkout)
- [ ] `make all` green — 400 passed / 2 skipped, 12 PROVEN, synth ~64 MHz, ASIC
      14/14, docs build, 30/30 contracts.
- [ ] `make cocotb` green (12 blocks incl. `motorloop_axil_top`).
- [ ] `reuse lint` compliant.
- [ ] `CITATION.cff` and `.zenodo.json` resolve (version `0.1.0`, correct repo URL).
- [ ] `CHANGELOG.md` has the `[0.1.0]` section with today's date.

## 1. Tag + push
```sh
git tag -a v0.1.0 -m "motorloop v0.1.0"
git push origin v0.1.0
```

## 2. GitHub release
- [ ] Create the release from tag `v0.1.0`.
- [ ] Attach assets: `notes/status-matrix-generated.md`, `formal/proof_report.md`,
      `synth/synth_report.md`, `synth/asic_smoke_report.md`, and the ECP5
      bitstream (`build/.../*.bit`) + the SoC bitstream if built.
- [ ] Paste the `[0.1.0]` CHANGELOG section as the release notes.

## 3. Zenodo DOI
- [ ] Enable the GitHub↔Zenodo integration for the repo (one-time, on zenodo.org).
- [ ] The `v0.1.0` GitHub release auto-creates a Zenodo deposition from
      `.zenodo.json`; publish it to mint the DOI.
- [ ] Put the DOI back into the repo:
  - `CITATION.cff`: uncomment + set `doi:`.
  - `README.md`: replace the `DOI-pending` badge target with the real DOI badge.
  - Commit as `docs: add Zenodo DOI for v0.1.0` (a post-release patch; or fold
    into v0.1.1).

## 4. Announce
- [ ] Link the docs site, the DOI, and the reference-SoC demo (`soc/README.md`).
