# SPDX-License-Identifier: MIT
"""Train a design-conditioned policy across actual MuJoCo-Warp model groups."""

import sys

from warp_train_cli import run


if __name__ == "__main__":
    run("universal", sys.argv[1:], default_tag="codesign_warp")
