# SPDX-License-Identifier: MIT
"""Train command-conditioned locomotion with MuJoCo-Warp and Torch PPO."""

import sys

from warp_train_cli import run


if __name__ == "__main__":
    geometry = "mesh" if "--env=mesh" in sys.argv or (
        "--env" in sys.argv and sys.argv[sys.argv.index("--env") + 1] == "mesh") else "walker"
    run(geometry, sys.argv[1:], default_tag="commanded_warp")
