# SPDX-License-Identifier: MIT
"""Train a Torch route corrector from Warp rollout data."""
from warp_compat_cli import dataset_cli
if __name__ == "__main__":
    dataset_cli("walker", mode="clone", stem="route_corrector_warp")
