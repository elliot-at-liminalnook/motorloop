# SPDX-License-Identifier: MIT
"""Shared evaluation, ranking, diagnostics, and rendering for Warp policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import mujoco
import torch
import warp as wp

from combat_warp_env import CombatWarpEnv
from codesign_warp_env import DesignEnsembleWarpEnv
from leg_attack_warp_env import LEG_NAMES, LegAttackWarpEnv
from ladder_warp_env import (LadderCombatWarpEnv, LadderLocomotionWarpEnv,
                             UniversalCommandWarpEnv, UniversalControlWarpEnv)
from mesh_warp_env import EvalTelemetry, MeshWarpEnv
from train_mesh_warp import (Actor, RunningNorm, _load_actor_normalizer_compatible,
                             expected_load_semantics, inherit_task_conditioning,
                             load_policy)
from walker_warp_env import WalkerWarpEnv

HERE = Path(__file__).resolve().parent
ENVIRONMENTS = {
    "walker": WalkerWarpEnv,
    "mesh": MeshWarpEnv,
    "combat": CombatWarpEnv,
    "leg_attack": LegAttackWarpEnv,
    "ladder_locomotion": LadderLocomotionWarpEnv,
    "ladder_combat": LadderCombatWarpEnv,
    "universal_control": UniversalControlWarpEnv,
    "universal_command": UniversalCommandWarpEnv,
    "universal": DesignEnsembleWarpEnv,
}
COMBAT_ENVIRONMENTS = frozenset(("combat", "leg_attack", "ladder_combat"))


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
             episode_length: int, lidar: bool = False, rung: int | None = None,
             power_model: str = "off"):
    kwargs = dict(nworld=nworld, seed=seed, device=device,
                  episode_length=episode_length)
    if geometry in COMBAT_ENVIRONMENTS:
        kwargs["lidar"] = lidar
    if power_model != "off":
        if geometry in COMBAT_ENVIRONMENTS:
            raise ValueError(
                "--power-model is not yet implemented for the fused combat layer")
        kwargs["power_model"] = power_model
    if geometry in ("ladder_locomotion", "ladder_combat", "universal_control",
                    "universal_command"):
        if rung is None:
            raise ValueError(f"--geometry {geometry} requires --rung")
        kwargs["rung"] = rung
    return ENVIRONMENTS[geometry](**kwargs)


@torch.no_grad()
def evaluate(checkpoint: str | Path | None, geometry="walker", episodes=4,
             steps=250, nworld=16, seed=0, device=None, command=None,
             opponent=None, lidar=False, record=False, attack_leg=None,
             attack_active=None, attack_switch=False, rung=None,
             power_model="off"):
    env = make_env(geometry, nworld, seed, device, steps, lidar, rung,
                   power_model=power_model)
    combat_domain = (geometry in COMBAT_ENVIRONMENTS or (
        geometry in ("universal_control", "universal_command")
        and getattr(env, "domain", None) == "combat"))
    policy = (load_policy(
        resolve_checkpoint(checkpoint), env.obs_dim, env.act_dim, env.device,
        task_dim=getattr(env, "architecture_task_dim", 0), morphology_source=env,
        expected_semantics=expected_load_semantics(env))
              if checkpoint else lambda obs: torch.zeros((len(obs), env.act_dim), device=env.device))
    if opponent:
        if not combat_domain:
            raise ValueError("an opponent checkpoint requires a combat-family geometry")
        env.set_opponent(load_policy(
            resolve_checkpoint(opponent), env.obs_dim, env.act_dim, env.device,
            task_dim=getattr(env, "architecture_task_dim", 0), morphology_source=env,
            expected_semantics=expected_load_semantics(env)))
    if attack_leg is not None:
        if not hasattr(env, "set_attack_command"):
            raise ValueError("--attack-leg requires --geometry leg_attack")
        env.set_attack_command(attack_leg, True if attack_active is None else attack_active)
    elif attack_active is not None:
        if not hasattr(env, "set_attack_enabled"):
            raise ValueError("--attack-off requires --geometry leg_attack")
        env.set_attack_enabled(attack_active)
    command_t = None if command is None else torch.as_tensor(
        command, dtype=torch.float32, device=env.device).reshape(1, 3)
    returns = torch.zeros(nworld, device=env.device)
    falls = torch.zeros(nworld, device=env.device)
    start_xy = None
    frames = []
    telemetry = EvalTelemetry(env.device)
    obs = env.reset()
    if hasattr(env, "xpos"):
        torso = env.layer.idx.At if combat_domain else env._torso
        start_xy = env.xpos[:, torso, :2].clone()
    total_steps = int(episodes) * int(steps)
    switch_interval = max(1, total_steps // 8)
    switch_count = 0
    for step_index in range(total_steps):
        if attack_switch and step_index % switch_interval == 0:
            env.set_attack_command(LEG_NAMES[switch_count % len(LEG_NAMES)], True)
            switch_count += 1
        if command_t is not None and hasattr(env, "_cmd"):
            env._cmd.copy_(command_t.expand(nworld, -1))
            env._timer.zero_()
        obs, reward, done, info = env.step(policy(obs))
        if hasattr(policy, "reset"):
            policy.reset(done)
        returns += reward
        falls += done
        telemetry.add(reward, info)
        if record:
            frames.append(env.qpos[0].detach().cpu().numpy().copy())
    result = {"checkpoint": str(checkpoint) if checkpoint else None,
              "geometry": geometry, "return_mean": float(returns.mean() / episodes),
              "done_rate": float(falls.mean() / episodes)}
    result.update(telemetry.result())
    if start_xy is not None:
        result["displacement"] = float(torch.linalg.vector_norm(
            env.xpos[:, torso, :2] - start_xy, dim=-1).mean())
    if combat_domain:
        result.update(dealt=float(wp.to_torch(env.layer.dealt_leg).mean()),
                      taken=float(wp.to_torch(env.layer.taken_leg).mean()),
                      penetration=float(wp.to_torch(env.layer.pen_peak).max()))
    if attack_switch:
        result.update(
            attack_switch_count=switch_count,
            attack_switch_fallrate=float(result.get("fallrate", result["done_rate"])))
    if geometry == "leg_attack":
        result.update(
            attack_leg=attack_leg,
            attack_active=(True if attack_active is None else bool(attack_active)),
            selected_hit=float(info["attack_selected_hit"].mean()),
            wrong_leg_hit=float(info["attack_wrong_hit"].mean()),
            support=float(info["attack_support"].mean()),
            kick_speed=float(info["attack_kick_speed"].mean()),
        )
    return result, env, frames


def rank(checkpoints, **kwargs):
    rows = [evaluate(path, **kwargs)[0] for path in checkpoints]
    return sorted(rows, key=lambda row: row["return_mean"], reverse=True)


@torch.no_grad()
def collect_replay(checkpoint: str | Path, geometry: str, rung: int,
                   steps: int, nworld: int, seed: int, device: str | None,
                   output: str | Path, opponent: str | None = None,
                   power_model: str = "off") -> dict:
    """Persist real prior-task states and accepted-policy actions for replay."""
    env = make_env(geometry, nworld, seed, device, steps, rung=rung,
                   power_model=power_model)
    policy = load_policy(
        resolve_checkpoint(checkpoint), env.obs_dim, env.act_dim, env.device,
        task_dim=getattr(env, "architecture_task_dim", 0), morphology_source=env,
        expected_semantics=expected_load_semantics(env))
    if opponent:
        env.set_opponent(load_policy(
            resolve_checkpoint(opponent), env.obs_dim, env.act_dim, env.device,
            task_dim=getattr(env, "architecture_task_dim", 0), morphology_source=env,
            expected_semantics=expected_load_semantics(env)))
    env._gen.manual_seed(seed)
    obs = env.reset()
    observations, actions, dones = [], [], []
    for _ in range(int(steps)):
        action = policy(obs)
        actor_obs = policy.preprocess(obs) if hasattr(policy, "preprocess") else obs
        observations.append(actor_obs.detach().cpu())
        actions.append(action.detach().cpu())
        obs, _, done, _ = env.step(action)
        dones.append(done.detach().cpu())
        if hasattr(policy, "reset"):
            policy.reset(done)
    checkpoint_path = resolve_checkpoint(checkpoint)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    artifact = {
        "schema_version": 2,
        "rung": int(rung),
        "geometry": geometry,
        "seed": int(seed),
        "steps": int(steps),
        "envs": int(nworld),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": digest,
        "opponent": str(resolve_checkpoint(opponent)) if opponent else None,
        "opponent_sha256": (hashlib.sha256(
            resolve_checkpoint(opponent).read_bytes()).hexdigest() if opponent else None),
        "observation_semantics": getattr(env, "observation_semantics", None),
        "observations": torch.stack(observations),
        "actions": torch.stack(actions),
        "dones": torch.stack(dones),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    metadata = {key: value for key, value in artifact.items()
                if key not in ("observations", "actions", "dones")}
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


@torch.no_grad()
def inherit_policy_checkpoint(checkpoint: str | Path, output: str | Path,
                              source_task: int, target_task: int) -> dict:
    """Create an immutable zero-shot candidate with exact task inheritance.

    The actor and observation normalizer are the only learned components reused
    by a ladder transition.  Critic/optimizer/runtime tensors remain in the
    artifact solely so it keeps the normal checkpoint shape; a later trainer
    warm-start deliberately ignores those stale task-specific tensors.
    """
    source = resolve_checkpoint(checkpoint)
    ck = torch.load(source, map_location="cpu", weights_only=True)
    if ck.get("contract") is None:
        # This function MANUFACTURES a promotion candidate; deriving one from
        # a contract-less legacy checkpoint would launder it into the gated
        # pipeline with no semantic identity attached.
        raise ValueError(
            f"cannot derive a zero-shot candidate from {source}: the source "
            "checkpoint carries no contract")
    saved_args = ck.get("args", {})
    hidden = tuple(int(value) for value in saved_args.get(
        "hidden", "512,256,128").split(","))
    architecture = saved_args.get("architecture", "mlp")
    prediction_decoder = saved_args.get("prediction_decoder", "recurrent")
    task_dim = int(saved_args.get("actor_task_dim", 0))
    obs_dim = int(ck["obs_norm"]["mean"].numel())
    act_dim = int(ck["actor"]["log_std"].numel())
    actor = Actor(obs_dim, act_dim, hidden, architecture=architecture,
                  task_dim=task_dim,
                  prediction_decoder=prediction_decoder)
    normalizer = RunningNorm(obs_dim)
    _load_actor_normalizer_compatible(ck, actor, normalizer)
    residual = inherit_task_conditioning(
        actor, normalizer, int(source_task), int(target_task))
    ck["actor"] = actor.state_dict()
    ck["obs_norm"] = normalizer.state_dict()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "kind": "zero_shot_task_inheritance",
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_hash,
        "source_task_index": int(source_task),
        "target_task_index": int(target_task),
        "max_pre_activation_error": residual,
        "architecture": architecture,
        "task_dim": task_dim,
    }
    ck["zero_shot_inheritance"] = metadata
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(ck, temporary)
    temporary.replace(output)
    metadata["checkpoint"] = str(output)
    metadata["checkpoint_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


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
    parser.add_argument("mode", nargs="?", choices=(
        "eval", "rank", "render", "diagnose", "collect-replay", "inherit-policy"),
                        default="eval")
    parser.add_argument("--checkpoint", "--ckpt", "--model", "--tag", "--a",
                        dest="checkpoint")
    parser.add_argument("--checkpoints", nargs="*")
    parser.add_argument("--opponent", "--b")
    parser.add_argument("--geometry", choices=tuple(ENVIRONMENTS), default="walker")
    parser.add_argument("--rung", type=int,
                        help="task number for ladder_locomotion/ladder_combat")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--command", default=None)
    parser.add_argument("--attack-leg", choices=LEG_NAMES,
                        help="lock the leg_attack policy to FL, FR, RL, or RR")
    parser.add_argument("--attack-off", action="store_true",
                        help="disable the attack channel while preserving balance")
    parser.add_argument("--attack-switch", action="store_true",
                        help="switch FL/FR/RL/RR repeatedly without resetting dynamics")
    parser.add_argument("--lidar", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--source-task", type=int)
    parser.add_argument("--target-task", type=int)
    parser.add_argument("--power-model", choices=("off", "shared_bus"),
                        default="off",
                        help="evaluate under the shared-bus electrical budget "
                             "(must match how the checkpoint trained)")
    args = parser.parse_args(argv)
    command = tuple(map(float, args.command.split(","))) if args.command else None
    if command and len(command) == 2:
        command = (*command, 0.0)
    common = dict(geometry=args.geometry, episodes=args.episodes, steps=args.steps,
                  nworld=args.envs, seed=args.seed, device=args.device,
                  command=command, opponent=args.opponent, lidar=args.lidar,
                  attack_leg=args.attack_leg,
                  attack_active=False if args.attack_off else None,
                  attack_switch=args.attack_switch,
                  rung=args.rung, power_model=args.power_model)
    if args.mode == "inherit-policy":
        if (not args.checkpoint or not args.out or args.source_task is None
                or args.target_task is None):
            parser.error(
                "inherit-policy requires --checkpoint, --out, --source-task, and --target-task")
        result = inherit_policy_checkpoint(
            args.checkpoint, args.out, args.source_task, args.target_task)
    elif args.mode == "collect-replay":
        if not args.checkpoint or args.rung is None or not args.out:
            parser.error("collect-replay requires --checkpoint, --rung, and --out")
        result = collect_replay(
            args.checkpoint, args.geometry, args.rung, args.steps, args.envs,
            args.seed, args.device, args.out, opponent=args.opponent,
            power_model=args.power_model)
    elif args.mode == "rank":
        result = rank(args.checkpoints or ([args.checkpoint] if args.checkpoint else []), **common)
    else:
        result, env, frames = evaluate(args.checkpoint, record=args.mode == "render", **common)
        if args.mode == "render":
            output = Path(args.out or f"{args.geometry}_warp.mp4")
            model = env.layer.mjm if args.geometry in COMBAT_ENVIRONMENTS else env.mjm
            render_video(model, frames, output)
            result["video"] = str(output)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out and args.mode not in ("render", "collect-replay", "inherit-policy"):
        Path(args.out).write_text(text + "\n")
    return result


if __name__ == "__main__":
    main()
