<!-- SPDX-License-Identifier: MIT -->
# Release (v0.1.0 + DOI) & multi-vendor portability checklist

Two adoption steps from [adoption-roadmap.md](adoption-roadmap.md) #3 + #5:
**ship a citable release** and **prove the RTL ports off Lattice**. Ordered,
precise. Two recurring splits:

- **Prep work vs maintainer-only actions** (release): CI/verifier files can be
  prepared in-tree; the `git tag` / GitHub release / Zenodo mint are the
  maintainer's actions.
- **Open vs vendor** (portability): yosys's vendor backends give resource numbers
  with *no licenses* (CI-able); authoritative Fmax/utilization comes from the
  vendor tools, run where installed (the OpenLane-honesty pattern).

North star unchanged: each item earns its place by making a real block easier to
*cite* or *adopt on a target board* — not by adding a badge for its own sake.

---

# §3 — Tag v0.1.0 + mint the Zenodo DOI

State: `CITATION.cff`, `.zenodo.json`, README badges, `CHANGELOG [0.1.0]`, and
`notes/release-checklist.md` are written. Versions agree at **0.1.0** (CITATION,
IP-XACT `<version>`, FuseSoC cores `…:0.1.0`). This checklist hardens + automates
the prep, then hands the maintainer a one-sitting tag-and-publish.

## §3.1 Pre-flight verification (repo-runnable)
- [ ] `make all` green on a clean checkout: 400 passed / 2 skipped, 12 PROVEN,
      synth ~64 MHz, ASIC 14/14, docs build, 30/30 contracts.
- [ ] `make cocotb` green (12 blocks incl. `motorloop_axil_top`); `reuse lint` ok.
- [ ] **Version consistency gate** — add `scripts/check_version.py` asserting one
      version string across `CITATION.cff`, `.zenodo.json` (none → tag-derived,
      document that), the IP-XACT `<ipxact:version>`, and every `*.core`
      `:X.Y.Z`. Fail CI on a mismatch (so a release can't ship inconsistent
      versions). Wire as `make version`.
- [ ] `CHANGELOG.md` `[0.1.0]` has the real release date (today on tag day).
- [ ] Confirm the release **assets exist + regenerate**: `notes/status-matrix-generated.md`,
      `formal/proof_report.md`, `synth/synth_report.md`, `synth/asic_smoke_report.md`,
      and a built `*.bit` (`make synth`).
- [ ] Badge sanity: the `ci.yml`/`formal.yml` workflow files exist (badge URLs
      match), the REUSE badge repo path is correct, the DOI badge is the
      `DOI-pending` placeholder linking `notes/release-checklist.md`.

## §3.2 Harden + automate the release (repo-preparable)
- [ ] **`.github/workflows/release.yml`** (on `push: tags: v*`): run `make verify`,
      then attach the reports + bitstream to the GitHub release automatically, so
      a tag produces a complete release without manual asset uploads.
- [ ] **Publish the docs site to GitHub Pages** — add a `gh-pages` deploy
      (`mkdocs gh-deploy` in a workflow, or a Pages action). The README/contract
      links + the "browsable catalog" claim only pay off if the site is live;
      add the Pages URL to the README.
- [ ] **GitHub "Cite this repository"** — confirm `CITATION.cff` parses (GitHub
      renders a citation widget from it automatically once it's on the default
      branch). Add an **ORCID** for the author to `CITATION.cff` + `.zenodo.json`
      if available (Zenodo links it).
- [ ] Add a `## Citing` section to the README pointing at `CITATION.cff` + the
      (pending) DOI.

## §3.3 Tag + GitHub release (MAINTAINER)
- [ ] `git tag -a v0.1.0 -m "motorloop v0.1.0"` && `git push origin v0.1.0`.
- [ ] Create the GitHub release from the tag; paste the `[0.1.0]` CHANGELOG as
      notes; `release.yml` (§3.2) attaches the assets, or attach them manually.

## §3.4 Zenodo DOI (MAINTAINER)
- [ ] One-time: enable the GitHub↔Zenodo integration for the repo on zenodo.org
      and flip the repo on.
- [ ] The `v0.1.0` release triggers a Zenodo deposition seeded from `.zenodo.json`;
      review + **publish** to mint the DOI.
- [ ] Note Zenodo issues **two** DOIs: a *concept* DOI (all versions) and a
      *version* DOI (v0.1.0). Use the **concept** DOI in the README badge and
      `CITATION.cff` so citations resolve to "latest".

## §3.5 Close the loop (repo prep + maintainer commit)
- [ ] Put the DOI back: uncomment `doi:` in `CITATION.cff`; swap the README
      `DOI-pending` badge for the real `https://zenodo.org/badge/DOI/…` badge.
- [ ] Roll the CHANGELOG: add a fresh `## [Unreleased]` above `[0.1.0]`; bump the
      cores/IP-XACT to `0.2.0` only when an interface changes (document the policy
      in a short `VERSIONING.md` — until 1.0.0, minor may break, contracts define
      the stable surface).
- [ ] Announce: docs-site URL + DOI + the reference-SoC demo (`soc/README.md`).
- **Done-when:** `v0.1.0` is tagged + released with assets, the concept DOI
  resolves and shows in the README badge + `CITATION.cff`, and the docs site is
  live. `notes/release-checklist.md` remains the maintainer's runbook.

---

# §4 — Multi-vendor portability (Xilinx / Intel / Gowin)

Goal: a **portability matrix** — resources (LUT/FF/DSP/BRAM) and Fmax per FPGA
family — proving the deliberate Verilog-2005 portability and reaching the ~90% of
developers not on ECP5. Build it open-first (no licenses, CI-able), then add
authoritative vendor numbers where the tools exist.

## §4.1 Open resource-portability table (yosys backends — CI-able, no licenses)
yosys ships `synth_xilinx`, `synth_intel_alm`, `synth_gowin` (all confirmed
present in the pinned OSS CAD Suite) — they map the RTL to each family's cells
**without** any vendor tool.
- [ ] **`synth/portability.py`** (mirror `synth/asic_smoke.py`): for each target
      `{xilinx (xc7), intel_alm (Cyclone V/10), gowin (GW5A)}`, run the matching
      `synth_<vendor>` on `controller_top` (+ the gen include), parse the mapped
      LUT/FF/DSP(MULT)/BRAM counts, and write `synth/portability_report.md` —
      a family × resource matrix alongside the existing ECP5 row.
- [ ] **Make it a gate** (`make portability`, in CI): `synth_<vendor>` *erroring*
      surfaces non-portable RTL (the high-value part). Likely offenders to fix to
      vendor-neutral form, re-verifying **byte-identical** against the sim:
  - memory init / `$readmemh` of `rtl/gen/sincos_init.vh` (ROM inference differs
    per vendor — confirm it infers a block ROM on all three, not LUTRAM/regs);
  - async vs sync reset inference (the design uses async active-low — ensure each
    backend maps it to the family's FF reset, no extra logic);
  - the `divider32`/isqrt shift-add patterns and the MULT18X18-style products
    (DSP inference differs: Xilinx DSP48, Intel DSP, Gowin DSP).
- [ ] Honest caveat in the report: yosys `synth_<vendor>` counts are **pre-P&R
      estimates**; they prove *mapping* + give ballpark resources, not timing.

## §4.2 Open place&route Fmax where toolable (real timing, still no licenses)
- [ ] **Gowin (Tang Primer 25K / GW5A)** — full open P&R is available
      (`synth_gowin` → `nextpnr-himbaechel` → `gowin_pack`, all in the OSS CAD
      Suite; install the **apicula** device DB). Add `synth/run_gowin.py`
      (mirror `run_synth.py`) → a real post-route Fmax for a second real board.
      *This also advances the project's existing Tang Primer 25K interest.*
- [ ] **Xilinx 7-series** — open P&R needs `nextpnr-xilinx` + the `prjxray` DB
      (not in the OSS CAD Suite). Optional heavier add; otherwise Xilinx Fmax
      comes from Vivado (§4.3). Document the choice.

## §4.3 Authoritative vendor numbers (where the tools are installed)
Proprietary, license-gated, **not** in CI — provide the project files + a results
template; numbers get filled where licensed (the OpenLane pattern).
- [ ] **`synth/vivado/motorloop.tcl`** — read the RTL set (reuse the
      `synth_ecp5.ys` file list), `synth_design` for an Artix-7 part (e.g.
      `xc7a35t`), report `report_utilization` + `report_timing_summary` at a
      stated clock; a `synth/vivado/README.md` with the `vivado -mode batch`
      invocation.
- [ ] **`synth/quartus/`** — a `.qsf` (part = a Cyclone V/10, the RTL list, an
      SDC clock) + a `quartus_sh` flow script + README; report Fitter resources +
      TimeQuest Fmax.
- [ ] A `synth/portability_report.md` table row per vendor flow, each labelled
      **open-estimate / open-P&R / vendor** so a reader knows the provenance.

## §4.4 Publish + wire in
- [ ] Add the portability matrix to the README (a small table) and to the docs
      site; add a **multi-vendor line to `rtl/contracts/foc_core.md`** (and the
      bus wrappers) "Synthesis fit" — e.g. "maps to Xilinx 7-series / Intel
      Cyclone / Gowin GW5A; resources in `synth/portability_report.md`".
- [ ] `make portability` in `make all`/CI (the open yosys table) so portability
      can't silently regress.
- **Done-when:** `synth/portability_report.md` shows the RTL mapping to ≥3
      families (open yosys estimates) + ≥1 extra real post-route Fmax (open Gowin
      and/or vendor), each labelled by provenance; CI gates the open table; the
      README/contracts cite it.

## What NOT to do
Don't gate CI on proprietary tools (keep Vivado/Quartus out of the pipeline —
provide files, run where licensed). Don't claim a vendor Fmax you didn't measure
— label every number open-estimate / open-P&R / vendor. Don't fork RTL per vendor;
fix to one vendor-neutral source and keep it byte-identical to the sim baseline.

## Implemented (results)

Both halves done; the repo-preparable release work + all open portability work
landed, with the maintainer-only tag/DOI/vendor runs documented.

- **§3 release-prep:** `scripts/check_version.py` (`make version`, CI) proves
  **0.1.0 consistent** across CITATION/cores/IP-XACT/CHANGELOG. `release.yml`
  (regenerate reports + attach to the GitHub release on a `v*` tag, with a
  tag↔version assertion), `docs.yml` (GitHub Pages deploy), README **Citing** +
  a **Portability** matrix, `VERSIONING.md`. The `git tag` / GitHub release /
  Zenodo mint stay the maintainer's (`notes/release-checklist.md`).
- **§4 portability:** `synth/portability.py` (`make portability`, CI) — the RTL
  **maps to 4/4 families** (ECP5 / Xilinx 7-series / Intel Cyclone / Gowin GW5A)
  via yosys backends, no licenses (~13k LUTs on the LUT4 families; breakdown in
  `synth/work/portability_*.log`). `synth/run_gowin.py` wires the **open Gowin
  GW5A** P&R (Tang Primer 25K) — `synth_gowin` maps it; a real Fmax needs a board
  `.cst` (himbaechel requires constrained I/O). `synth/vivado/` + `synth/quartus/`
  give authoritative numbers where licensed. ECP5 (64 MHz) is the authoritative
  open-P&R Fmax.
- **Honest provenance:** every number labelled open-estimate / open-P&R / vendor;
  the only unproduced numbers are vendor (license-gated) + the Gowin Fmax
  (board-`.cst`-gated) — documented, not faked.
