# SPDX-License-Identifier: MIT
"""Standalone plain-MuJoCo versus MuJoCo-Warp trajectory parity gate."""

from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def main():
    result = pytest.main([
        "-q",
        str(HERE / "test_pre_gpu_warp.py"),
        "-k", "active_walker_mujoco_warp_trajectory_parity",
    ])
    raise SystemExit(result)


if __name__ == "__main__":
    main()
