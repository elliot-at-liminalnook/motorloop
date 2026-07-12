# SPDX-License-Identifier: MIT
"""Converted compatibility entry point for MuJoCo-Warp commanded locomotion."""

import sys

from warp_train_cli import run


if __name__ == "__main__":
    geometry = "mesh" if "mesh" in sys.argv else "walker"
    run(geometry, sys.argv[1:], default_tag="commanded_warp")
