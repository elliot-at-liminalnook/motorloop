# SPDX-License-Identifier: MIT
"""Re-evaluate design rankings through the actual Warp model ensemble."""
from warp_compat_cli import eval_cli
if __name__ == "__main__":
    eval_cli("universal", "rank")
