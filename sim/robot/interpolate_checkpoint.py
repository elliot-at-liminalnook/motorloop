# SPDX-License-Identifier: MIT
"""Interpolate compatible Torch policy checkpoints."""

import argparse
import copy

import torch


def _blend_tree(base, other, alpha):
    out = copy.deepcopy(base)
    for section in ("actor", "critic", "obs_norm", "priv_norm"):
        if section not in out or section not in other:
            continue
        for key, value in out[section].items():
            peer = other[section].get(key)
            if torch.is_tensor(value) and torch.is_tensor(peer) and value.shape == peer.shape:
                out[section][key] = (1.0 - alpha) * value + alpha * peer
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--other", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    other = torch.load(args.other, map_location="cpu", weights_only=False)
    torch.save(_blend_tree(base, other, args.alpha), args.out)
    print(f"saved {args.out} alpha={args.alpha}")


if __name__ == "__main__":
    main()
