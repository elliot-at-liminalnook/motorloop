# SPDX-License-Identifier: MIT
"""Converted compatibility entry point for grouped MuJoCo-Warp co-design."""

import sys

from warp_train_cli import run


if __name__ == "__main__":
    run("universal", sys.argv[1:], default_tag="codesign_warp")
