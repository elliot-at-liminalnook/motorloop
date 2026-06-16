#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Add an SPDX MIT header to every source file that lacks one
(trusted-library-checklist stage 2.1). Idempotent: skips files that already
have an SPDX-License-Identifier line. Comment syntax + placement per file type
(after a shebang or a required first line like `CAPI=2:`).

  python3 scripts/add_spdx.py            # add headers
  python3 scripts/add_spdx.py --check    # list files missing one (exit 1 if any)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# REUSE-IgnoreStart
TAG = "SPDX-License-Identifier: MIT"
# REUSE-IgnoreEnd

# extension -> (comment_prefix, comment_suffix)
SLASH = ("// ", "")
HASH = ("# ", "")
HTML = ("<!-- ", " -->")
COMMENT = {
    ".v": SLASH, ".sv": SLASH, ".vh": SLASH,
    ".cpp": SLASH, ".hpp": SLASH, ".h": SLASH, ".cc": SLASH,
    ".py": HASH, ".sh": HASH, ".toml": HASH, ".yml": HASH, ".yaml": HASH,
    ".ys": HASH, ".lpf": HASH, ".core": HASH, ".cmake": HASH,
    ".md": HTML,
}
SKIP_DIRS = {".git", "build", "work", "__pycache__", "oss-cad-suite",
             ".pytest_cache", "gen", "LICENSES"}  # LICENSES/ texts: reuse owns them
# gen/ is auto-generated (rtl_params.vh, sincos_init.vh) - headers belong on the
# generator, not the output.


def header(ext: str) -> str:
    pre, suf = COMMENT[ext]
    return f"{pre}{TAG}{suf}\n"


def first_line_must_stay(line: str) -> bool:
    s = line.strip()
    return s.startswith("#!") or s.startswith("CAPI=") or s.startswith("<?")


def process(path: Path, check: bool) -> bool:  # True if missing/added
    text = path.read_text()
    if TAG in text[:400]:
        return False
    if check:
        return True
    lines = text.splitlines(keepends=True)
    hdr = header(path.suffix)
    if lines and first_line_must_stay(lines[0]):
        lines.insert(1, hdr)
    else:
        lines.insert(0, hdr)
    path.write_text("".join(lines))
    return True


def main():
    check = "--check" in sys.argv
    missing = []
    for path in ROOT.rglob("*"):
        if any(p in SKIP_DIRS for p in path.parts):
            continue
        if not path.is_file() or path.suffix not in COMMENT:
            continue
        if process(path, check):
            missing.append(path.relative_to(ROOT))
    if check:
        for m in missing:
            print(f"missing SPDX: {m}")
        print(f"{len(missing)} file(s) missing SPDX")
        sys.exit(1 if missing else 0)
    print(f"added SPDX to {len(missing)} file(s)")


if __name__ == "__main__":
    main()
