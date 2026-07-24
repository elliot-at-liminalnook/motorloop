# SPDX-License-Identifier: MIT
"""Converted compatibility entry point for MuJoCo-Warp Hall-of-Fame self-play."""

import sys

from warp_train_cli import selfplay


if __name__ == "__main__":
    selfplay(sys.argv[1:])
