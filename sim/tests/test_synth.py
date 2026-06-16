# SPDX-License-Identifier: MIT
"""ECP5 synthesis gate (platform-abstraction Phase C, stage 15).

Runs the open `synth_ecp5` flow on controller_top and asserts it maps to ECP5
primitives - the gate that catches non-synthesizable RTL (the class of bug the
circle_limit while-loop -> 16-iteration fix once hit). Skipped unless the OSS
CAD Suite is on PATH (`source ~/oss-cad-suite/environment`), so the default
simulation regression stays fast; CI/manual runs with the toolchain exercise it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    shutil.which("yosys") is None,
    reason="yosys not on PATH (source ~/oss-cad-suite/environment)")


def test_controller_top_synthesizes_to_ecp5():
    """yosys synth_ecp5 maps the full controller_top hierarchy to ECP5 cells."""
    rc = subprocess.run(
        ["python3", "synth/run_synth.py", "--check"], cwd=ROOT).returncode
    assert rc == 0, "ECP5 synthesis failed"
    stat = ROOT / "synth" / "work" / "stat.txt"
    assert stat.exists(), "no synthesis stat produced"
    text = stat.read_text()
    assert "LUT4" in text and "TRELLIS_FF" in text, (
        "synthesis produced no ECP5 LUT/FF cells")
