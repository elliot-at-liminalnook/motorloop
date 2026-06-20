# SPDX-License-Identifier: MIT
"""Suite hook for the MJX<->MuJoCo parity gate (Phase 1 critical / Phase 9 suite test).

The full parity check lives in sim/robot/test_parity.py. On the CPU CI box (no JAX/MJX)
it SKIPs and returns 0; on the GPU box it runs for real and returns non-zero if the MJX
port diverges from the CPU reference. This wrapper just asserts the script exits cleanly,
so a broken port is caught by `make test` once run where JAX exists (and never red on CPU).
"""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "robot" / "test_parity.py"


def test_mjx_mujoco_parity():
    r = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    print(r.stdout[-2000:]); print(r.stderr[-1000:])
    assert r.returncode == 0, f"MJX<->MuJoCo parity gate failed (rc={r.returncode})"
