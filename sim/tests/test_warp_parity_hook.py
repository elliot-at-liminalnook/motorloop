# SPDX-License-Identifier: MIT
"""Component-suite hook for the plain-MuJoCo/MuJoCo-Warp parity gate."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sim/robot/test_parity.py"
WARP_PY = Path(os.environ.get("WARP_PY", ROOT / ".venv-warp/bin/python"))


def test_mujoco_warp_parity_script():
    assert WARP_PY.is_file(), (
        f"MuJoCo-Warp interpreter missing at {WARP_PY}; "
        "run the canonical scripts/run_pre_gpu_tests.sh setup")
    result = subprocess.run([str(WARP_PY), str(SCRIPT)], capture_output=True, text=True)
    print(result.stdout[-2000:])
    print(result.stderr[-1000:])
    assert result.returncode == 0, f"MuJoCo-Warp parity gate failed (rc={result.returncode})"
