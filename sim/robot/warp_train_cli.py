# SPDX-License-Identifier: MIT
"""Shared CLI adapters for converted MuJoCo-Warp training workflows."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from train_mesh_warp import build_args, train


def run(default_geometry: str, argv=None, default_tag: str | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000_000)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--episode-length", type=int, default=800)
    parser.add_argument("--tag", default=default_tag or f"{default_geometry}_warp")
    parser.add_argument("--resume")
    parser.add_argument("--opponent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--preflight", choices=("strict", "warn", "off"), default="strict")
    args, _ = parser.parse_known_args(argv)
    if args.tiny:
        args.steps, args.envs, args.horizon = 512, 16, 16
        args.preflight = "off"
    out = [
        "--geometry", default_geometry, "--steps", str(args.steps),
        "--envs", str(args.envs), "--horizon", str(args.horizon),
        "--episode-length", str(args.episode_length), "--tag", args.tag,
        "--seed", str(args.seed), "--preflight", args.preflight,
    ]
    if args.device:
        out += ["--device", args.device]
    if args.resume:
        out += ["--resume", args.resume]
    if args.opponent:
        out += ["--opponent", args.opponent]
    return train(build_args(out))


def selfplay(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--tag", default="selfplay_warp")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tiny", action="store_true")
    args, _ = parser.parse_known_args(argv)
    if args.tiny:
        args.rounds, args.steps, args.envs, args.horizon = 2, 512, 16, 16
    hall: list[str] = []
    for round_index in range(args.rounds):
        tag = f"{args.tag}_r{round_index}"
        opponent = random.Random(round_index).choice(hall) if hall else None
        cli = ["--steps", str(args.steps), "--envs", str(args.envs),
               "--horizon", str(args.horizon), "--tag", tag,
               "--seed", str(round_index), "--preflight", "off"]
        if args.device:
            cli += ["--device", args.device]
        if opponent:
            cli += ["--opponent", opponent]
        result = run("combat", cli, default_tag=tag)
        hall.append(str(Path(result["ckpt"])))
        print(f"LEAGUE round={round_index} opponent={opponent or 'passive'} "
              f"checkpoint={result['ckpt']}", flush=True)
    return hall
