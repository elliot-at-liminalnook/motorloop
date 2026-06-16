#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Contract coverage gate (tier2-adoption-checklist §4): every reusable core has
a finished datasheet in rtl/contracts/. Fails CI if a block ships without a
contract or with an unfinished one (a leftover `TODO:` marker). The §2 "manifest
as the truth source" discipline, applied to the per-block docs.

  python3 scripts/check_contracts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS = ROOT / "rtl" / "contracts"
SYSTEM = {"motorloop"}


def main() -> int:
    blocks = sorted(c.stem for c in ROOT.glob("*.core") if c.stem not in SYSTEM)
    missing, unfinished = [], []
    for b in blocks:
        f = CONTRACTS / f"{b}.md"
        if not f.exists():
            missing.append(b)
        elif "TODO:" in f.read_text():
            unfinished.append(b)
    for b in missing:
        print(f"  MISSING contract: rtl/contracts/{b}.md "
              f"(run scripts/gen_contract_stubs.py, then fill it)")
    for b in unfinished:
        print(f"  UNFINISHED contract (has TODO:): rtl/contracts/{b}.md")
    if missing or unfinished:
        print(f"[contracts] FAIL - {len(missing)} missing, "
              f"{len(unfinished)} unfinished of {len(blocks)} blocks")
        return 1
    print(f"[contracts] OK - {len(blocks)} blocks each have a finished datasheet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
