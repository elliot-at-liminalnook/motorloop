# SPDX-License-Identifier: MIT
"""Collect route-correction trajectories directly from MuJoCo-Warp."""
from warp_compat_cli import dataset_cli
if __name__ == "__main__":
    dataset_cli("walker", stem="route_correction_warp")
