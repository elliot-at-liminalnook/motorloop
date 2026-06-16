<!-- SPDX-License-Identifier: MIT -->
# Versioning policy

motorloop follows [Semantic Versioning](https://semver.org). **One version
number** spans the whole project — it's carried in `CITATION.cff`, every FuseSoC
`*.core` VLNV (`motorloop:…:X.Y.Z`), and the IP-XACT `<ipxact:version>`, and
`scripts/check_version.py` (CI: `make version`) fails the build on any drift.

## What the number promises

- **Patch (`x.y.Z`)** — fixes, doc/test/tooling changes; no interface or timing
  change to any block. Drop-in.
- **Minor (`x.Y.0`)** — new blocks/features; **before 1.0.0, a minor may also
  carry breaking interface or timing changes** (the project is pre-1.0). Read the
  CHANGELOG.
- **Major (`X.0.0`)** — reserved for the post-1.0 stability promise.

## The stable surface

Until 1.0.0 the **per-block contracts** (`rtl/contracts/*.md`) define the surface
a release stands behind: a block's interface, parameters, timing/latency, and
proof-or-sim-only status. A change to any of those is a contract change — bump
the version and note it in `CHANGELOG.md`. Pipeline latency counts: e.g.
`foc_core`'s `update`→`duty3` latency is part of its contract.

## Cutting a release

See [`notes/release-checklist.md`](notes/release-checklist.md): bump the version
in `CITATION.cff` (regenerate cores/IP-XACT so `make version` passes), date the
`CHANGELOG.md` section, tag `vX.Y.Z`, and let `release.yml` + the Zenodo hook do
the rest. Zenodo's **concept DOI** (all versions) is the one to cite.
