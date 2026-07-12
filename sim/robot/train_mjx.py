# SPDX-License-Identifier: MIT
"""Converted compatibility entry point; training now uses MuJoCo-Warp."""

import sys

from warp_train_cli import run


if __name__ == "__main__":
    run("walker", [arg for arg in sys.argv[1:] if arg != "--smoke"],
        default_tag="walker_warp")
