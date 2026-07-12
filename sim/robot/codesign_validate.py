# SPDX-License-Identifier: MIT
"""Evaluate a design-conditioned policy on the Warp model ensemble."""
from warp_compat_cli import eval_cli
if __name__ == "__main__":
    eval_cli("universal", "eval")
