# SPDX-License-Identifier: MIT
"""Train the design-conditioned MuJoCo-Warp policy."""
from warp_compat_cli import train_cli
if __name__ == "__main__":
    train_cli("universal", tag="codesign_warp")
