#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Version-consistency gate (release-and-portability-checklist §3.1). One version
string must agree across the release surface so a tag can't ship inconsistent
metadata: CITATION.cff (the canonical source), every FuseSoC `*.core` VLNV, every
IP-XACT `<ipxact:version>`, and a matching `## [<ver>]` section in CHANGELOG.md.
`.zenodo.json` carries no version (Zenodo derives it from the git tag).

  python3 scripts/check_version.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    m = re.search(r"^version:\s*([0-9][^\s#]*)", (ROOT / "CITATION.cff").read_text(), re.M)
    if not m:
        print("  CITATION.cff: no `version:` field"); return 1
    canon = m.group(1).strip().strip('"')
    bad = []

    for core in sorted(ROOT.glob("*.core")):
        nm = re.search(r"^name:\s*\S+:([0-9][^\s]*)", core.read_text(), re.M)
        v = nm.group(1) if nm else "(none)"
        if v != canon:
            bad.append(f"{core.name}: VLNV version {v} != {canon}")

    for xml in sorted((ROOT / "ip-xact").glob("*.xml")):
        for v in re.findall(r"<ipxact:version>([^<]+)</ipxact:version>", xml.read_text()):
            if v != canon:
                bad.append(f"{xml.name}: <version> {v} != {canon}")

    if f"## [{canon}]" not in (ROOT / "CHANGELOG.md").read_text():
        bad.append(f"CHANGELOG.md: no `## [{canon}]` section")

    for b in bad:
        print(f"  MISMATCH: {b}")
    if bad:
        print(f"[version] FAIL - {len(bad)} mismatch(es); canonical is {canon}")
        return 1
    print(f"[version] OK - {canon} consistent across CITATION, cores, IP-XACT, CHANGELOG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
