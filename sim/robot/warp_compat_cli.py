# SPDX-License-Identifier: MIT
"""Small adapters keeping historical command names on the Warp workflows."""

from __future__ import annotations

import sys
from pathlib import Path


def _has(argv, *names):
    return any(arg in names or any(arg.startswith(name + "=") for name in names)
               for arg in argv)


def eval_cli(geometry="walker", mode="eval", argv=None):
    from warp_eval import main
    argv = list(sys.argv[1:] if argv is None else argv)
    return main([mode, "--geometry", geometry, *argv])


def search_cli(mode="cpg", argv=None):
    from warp_search import main
    return main([mode, *(sys.argv[1:] if argv is None else argv)])


def dataset_cli(geometry="walker", mode="collect", argv=None, stem="warp_dataset"):
    from warp_dataset import main
    argv = list(sys.argv[1:] if argv is None else argv)
    if not _has(argv, "--out"):
        suffix = ".pt" if mode == "clone" else ".npz"
        argv += ["--out", str(Path("sim/build/gpu/out") / f"{stem}{suffix}")]
    return main([mode, "--geometry", geometry, *argv])


def train_cli(geometry="walker", argv=None, tag=None):
    from warp_train_cli import run
    return run(geometry, sys.argv[1:] if argv is None else argv,
               default_tag=tag or f"{geometry}_warp")
