# SPDX-License-Identifier: MIT
"""Run combat anti-cheat diagnostics through fused MuJoCo-Warp."""
from warp_compat_cli import eval_cli
if __name__ == "__main__":
    eval_cli("combat", "diagnose")
