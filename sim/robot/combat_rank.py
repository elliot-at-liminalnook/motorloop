# SPDX-License-Identifier: MIT
"""Rank combat checkpoints with fixed MuJoCo-Warp scenarios."""
from warp_compat_cli import eval_cli
if __name__ == "__main__":
    eval_cli("combat", "rank")
