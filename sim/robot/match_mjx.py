# SPDX-License-Identifier: MIT
"""Converted compatibility entry point for fused MuJoCo-Warp combat."""

import sys

from combat_warp_env import CombatWarpEnv as MatchWarp
from warp_train_cli import run

MatchMjx = MatchWarp


if __name__ == "__main__":
    run("combat", sys.argv[1:], default_tag="match_warp")
