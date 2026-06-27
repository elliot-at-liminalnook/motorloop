# SPDX-License-Identifier: MIT
"""Interpolate PPO checkpoint policy parameters for task-vector line search."""

from __future__ import annotations

import argparse
import pickle

import jax
import jax.numpy as jnp


def _blend_tree(base, other, alpha):
    def blend(a, b):
        if hasattr(a, "shape") and hasattr(b, "shape") and a.shape == b.shape:
            return (1.0 - alpha) * a + alpha * b
        return a
    return jax.tree_util.tree_map(blend, base, other)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blend-normalizer", action="store_true")
    ap.add_argument("--blend-value", action="store_true")
    args = ap.parse_args()

    base = list(pickle.load(open(args.base, "rb")))
    other = list(pickle.load(open(args.other, "rb")))
    out = list(base)
    if args.blend_normalizer:
        out[0] = _blend_tree(base[0], other[0], args.alpha)
    out[1] = _blend_tree(base[1], other[1], args.alpha)
    if args.blend_value and len(base) > 2 and len(other) > 2:
        out[2] = _blend_tree(base[2], other[2], args.alpha)
    pickle.dump(tuple(out), open(args.out, "wb"))
    print(f"saved {args.out} alpha={args.alpha}")


if __name__ == "__main__":
    main()
