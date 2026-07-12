# SPDX-License-Identifier: MIT
"""Train a Torch behavior clone from a MuJoCo-Warp rollout dataset."""
from warp_compat_cli import dataset_cli
if __name__ == "__main__":
    dataset_cli("walker", mode="clone", stem="bootstrap_bc_warp")
