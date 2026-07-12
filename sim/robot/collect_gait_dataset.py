# SPDX-License-Identifier: MIT
"""Collect gait trajectories directly from MuJoCo-Warp."""
from warp_compat_cli import dataset_cli
if __name__ == "__main__":
    dataset_cli("walker", stem="gait_warp")
