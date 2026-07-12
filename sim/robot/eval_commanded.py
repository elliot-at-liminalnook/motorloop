# SPDX-License-Identifier: MIT
"""Evaluate command-conditioned checkpoints in MuJoCo-Warp."""
from warp_compat_cli import eval_cli
if __name__ == "__main__":
    eval_cli("walker", "eval")
