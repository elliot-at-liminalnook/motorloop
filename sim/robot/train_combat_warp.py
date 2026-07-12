# SPDX-License-Identifier: MIT
"""Train or self-play the fused two-robot MuJoCo-Warp combat environment."""

import sys

from warp_train_cli import run, selfplay


if __name__ == "__main__":
    if "--selfplay" in sys.argv:
        selfplay([arg for arg in sys.argv[1:] if arg != "--selfplay"])
    else:
        run("combat", sys.argv[1:], default_tag="combat_warp")
