# SPDX-License-Identifier: MIT
"""Evaluate a fixed combat seed through MuJoCo-Warp."""
from warp_compat_cli import eval_cli
if __name__ == "__main__":
    eval_cli("combat", "diagnose")
