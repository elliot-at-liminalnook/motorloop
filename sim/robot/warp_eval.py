# SPDX-License-Identifier: MIT
"""Shared evaluation, ranking, diagnostics, and rendering for Warp policies."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp

from combat_warp_env import CombatWarpEnv
from codesign_warp_env import DesignEnsembleWarpEnv
from mesh_warp_env import MeshWarpEnv
from train_mesh_warp import load_policy
from walker_warp_env import WalkerWarpEnv

HERE = Path(__file__).resolve().parent
ENVIRONMENTS = {
    "walker": WalkerWarpEnv,
    "mesh": MeshWarpEnv,
    "combat": CombatWarpEnv,
    "universal": DesignEnsembleWarpEnv,
}


def resolve_checkpoint(value: str | Path) -> Path:
    path = Path(value)
    candidates = (path, path.with_suffix(".pt"),
                  Path(os.environ.get("CODESIGN_OUT", HERE.parent / "build/gpu/out")) / path,
                  Path(os.environ.get("CODESIGN_OUT", HERE.parent / "build/gpu/out")) /
                  path.with_suffix(".pt"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(value)


def make_env(geometry: str, nworld: int, seed: int, device: str | None,
             episode_length: int, lidar: bool = False):
    kwargs = dict(nworld=nworld, seed=seed, device=device,
                  episode_length=episode_length)
    if geometry == "combat":
        kwargs["lidar"] = lidar
    return ENVIRONMENTS[geometry](**kwargs)


@torch.no_grad()
def evaluate(checkpoint: str | Path | None, geometry="walker", episodes=4,
             steps=250, nworld=16, seed=0, device=None, command=None,
             opponent=None, lidar=False, record=False):
    env = make_env(geometry, nworld, seed, device, steps, lidar)
    policy = (load_policy(resolve_checkpoint(checkpoint), env.obs_dim, env.act_dim, env.device)
              if checkpoint else lambda obs: torch.zeros((len(obs), env.act_dim), device=env.device))
    if opponent:
        if geometry != "combat":
            raise ValueError("an opponent checkpoint requires --geometry combat")
        env.set_opponent(load_policy(resolve_checkpoint(opponent), env.obs_dim,
                                     env.act_dim, env.device))
    command_t = None if command is None else torch.as_tensor(
        command, dtype=torch.float32, device=env.device).reshape(1, 3)
    returns = torch.zeros(nworld, device=env.device)
    falls = torch.zeros(nworld, device=env.device)
    start_xy = None
    frames = []
    metrics = {key: torch.zeros((), device=env.device)
               for key in ("track", "verr", "align", "speed", "progress", "up", "height")}
    obs = env.reset()
    if hasattr(env, "xpos"):
        torso = env.layer.idx.At if geometry == "combat" else env._torso
        start_xy = env.xpos[:, torso, :2].clone()
    total_steps = int(episodes) * int(steps)
    for _ in range(total_steps):
        if command_t is not None and hasattr(env, "_cmd"):
            env._cmd.copy_(command_t.expand(nworld, -1))
            env._timer.zero_()
        obs, reward, done, info = env.step(policy(obs))
        returns += reward
        falls += done
        for key in metrics:
            metrics[key] += info[key].mean()
        if record:
            frames.append(env.qpos[0].detach().cpu().numpy().copy())
    result = {"checkpoint": str(checkpoint) if checkpoint else None,
              "geometry": geometry, "return_mean": float(returns.mean() / episodes),
              "done_rate": float(falls.mean() / episodes)}
    result.update({key: float(value / total_steps) for key, value in metrics.items()})
    if start_xy is not None:
        result["displacement"] = float(torch.linalg.vector_norm(
            env.xpos[:, torso, :2] - start_xy, dim=-1).mean())
    if geometry == "combat":
        result.update(dealt=float(wp.to_torch(env.layer.dealt_leg).mean()),
                      taken=float(wp.to_torch(env.layer.taken_leg).mean()),
                      penetration=float(wp.to_torch(env.layer.pen_peak).max()))
    return result, env, frames


def rank(checkpoints, **kwargs):
    rows = [evaluate(path, **kwargs)[0] for path in checkpoints]
    return sorted(rows, key=lambda row: row["return_mean"], reverse=True)


def render_video(model, qposes, output: Path, fps=50, width=960, height=540):
    """Render recorded Warp states through MuJoCo's reference renderer and ffmpeg."""
    output.parent.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    command = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
               "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert process.stdin is not None
    try:
        for qpos in qposes:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            process.stdin.write(renderer.render().tobytes())
    finally:
        process.stdin.close()
        code = process.wait()
        renderer.close()
    if code:
        raise RuntimeError(f"ffmpeg exited {code}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("eval", "rank", "render", "diagnose"),
                        default="eval")
    parser.add_argument("--checkpoint", "--ckpt", "--model", "--tag", "--a",
                        dest="checkpoint")
    parser.add_argument("--checkpoints", nargs="*")
    parser.add_argument("--opponent", "--b")
    parser.add_argument("--geometry", choices=tuple(ENVIRONMENTS), default="walker")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--command", default=None)
    parser.add_argument("--lidar", action="store_true")
    parser.add_argument("--out", default=None)
    args, _ = parser.parse_known_args(argv)
    command = tuple(map(float, args.command.split(","))) if args.command else None
    if command and len(command) == 2:
        command = (*command, 0.0)
    common = dict(geometry=args.geometry, episodes=args.episodes, steps=args.steps,
                  nworld=args.envs, seed=args.seed, device=args.device,
                  command=command, opponent=args.opponent, lidar=args.lidar)
    if args.mode == "rank":
        result = rank(args.checkpoints or ([args.checkpoint] if args.checkpoint else []), **common)
    else:
        result, env, frames = evaluate(args.checkpoint, record=args.mode == "render", **common)
        if args.mode == "render":
            output = Path(args.out or f"{args.geometry}_warp.mp4")
            model = env.layer.mjm if args.geometry == "combat" else env.mjm
            render_video(model, frames, output)
            result["video"] = str(output)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out and args.mode != "render":
        Path(args.out).write_text(text + "\n")
    return result


if __name__ == "__main__":
    main()
