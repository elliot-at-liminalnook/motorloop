# SPDX-License-Identifier: MIT
"""Stage 5.1: Verilator lint gate over the RTL."""

from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RTL = PROJECT_ROOT / "rtl"


def test_rtl_lints_clean():
    sources = sorted(RTL.glob("*.v"))
    assert sources, "no RTL sources found"
    result = subprocess.run(
        ["verilator", "--lint-only", f"-I{RTL}", f"-I{RTL / 'gen'}",
         "--top", "controller_top", *map(str, sources)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"verilator lint failed:\n{result.stdout}\n{result.stderr}"
    )
