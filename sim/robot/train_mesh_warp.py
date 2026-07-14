# SPDX-License-Identifier: MIT
"""Torch PPO over the canonical MuJoCo-Warp environments.

PPO with GAE(lambda=0.95, gamma=0.99), clip 0.2, minibatch SGD, running obs
normalization (saved in the checkpoint), and an ASYMMETRIC critic: the actor
sees the 50-obs policy input, the critic sees obs + the env's privileged tensor
(contact/force proxies, qfrc_actuator, true root vel, loop slides). Actor is a
(512,256,128) tanh-gaussian (squashed, with the log-det correction) with small
final-layer init. Schedules, all linear in total env steps:

  * entropy coef 3e-2 -> 5e-3 over --steps;
  * curriculum alpha (servo derating) --alpha-start -> --alpha-end over the
    first --alpha-frac (default 0.6) of --steps;
  * imitation weight anneal 1 -> 0 over --imit-anneal-frac (default 0.7)
    of --steps (only bites if sim/robot/reference_gait.json exists).

Every eval interval a DETERMINISTIC (tanh-mean) pass on a separate eval env
prints exactly one METRIC line and writes {tag}.pt (actor+critic+norms+optimizer
+step; resumable via --resume). On moving tasks, duty > 0.98 past half the run
means the creep optimum won and triggers exit code 3. Static stance, balance,
pose, and height-control rungs deliberately keep all feet planted and therefore
do not use that tripwire.

--geometry {mesh,walker,combat,leg_attack} selects the slider-crank locomotor,
the 12-servo hardware-contract walker, the symmetric two-robot fight model, or
the commanded-leg kick curriculum. All share one Torch learner and MuJoCo-Warp
physics path.

Rollout, GAE, and PPO updates all live on the env's device; physics is CUDA-
graph-captured by the env when a GPU is present. The same code runs (slowly)
on CPU:

  .venv-warp/bin/python sim/robot/train_mesh_warp.py --geometry walker \
      --steps 200000 --envs 64 --horizon 64 --tag /tmp/walkerwarp_smoke --evals 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import mujoco_warp as mjwp  # noqa: E402
import warp as wp  # noqa: E402

from mesh_warp_env import EvalTelemetry, MeshWarpEnv  # noqa: E402
from walker_warp_env import WalkerWarpEnv  # noqa: E402
from combat_warp_env import CombatWarpEnv  # noqa: E402
from leg_attack_warp_env import LegAttackWarpEnv  # noqa: E402
from ladder_warp_env import LadderCombatWarpEnv, LadderLocomotionWarpEnv  # noqa: E402
from codesign_warp_env import DesignEnsembleWarpEnv  # noqa: E402
from training_diagnostics import (  # noqa: E402
    critic_calibration,
    diagnostic_alerts,
    gradient_clip_diagnostics,
    normalization_diagnostics,
    normalization_snapshot,
    objective_gradient_diagnostics,
    optimizer_diagnostics,
    parameter_snapshot,
    parameter_update_diagnostics,
    policy_trust_region_diagnostics,
    multi_seed_summary,
    scalar_metric_gap,
    tensor_stats,
)

# --geometry selects the batched env; both expose the SAME interface (obs_dim,
# priv_dim, act_dim=12, step/observe/privileged/reset, gait_loaded) and each
# defaults its own reference-gait path, so nothing else in the trainer changes.
GEOMETRIES = {
    "mesh": MeshWarpEnv,
    "walker": WalkerWarpEnv,
    "combat": CombatWarpEnv,
    "leg_attack": LegAttackWarpEnv,
    "ladder_locomotion": LadderLocomotionWarpEnv,
    "ladder_combat": LadderCombatWarpEnv,
    "universal": DesignEnsembleWarpEnv,
}

GAMMA, LAM, CLIP = 0.99, 0.95, 0.2
ENT_START, ENT_END = 3e-2, 5e-3
LOG2PI = math.log(2.0 * math.pi)


def sha256_file(path: str | Path | None) -> str | None:
    """Return a streaming SHA-256 for a diagnostic input, if it exists."""
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_hash(root: Path = HERE) -> str:
    """Fingerprint the executable Python source even on pods without ``.git``."""
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def append_jsonl(path: Path, record: dict) -> None:
    """Durably append one compact event so a killed pod retains its diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json_atomic(path: Path, record: dict) -> None:
    """Atomically replace a human-readable latest-result artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def training_provenance(args, env, contract: dict, device: torch.device) -> dict:
    """Capture enough immutable context to reproduce or compare an invocation."""
    input_names = (
        "resume", "init_policy", "anchor_policy", "transfer_policy",
        "action_prior_json", "opponent",
    )
    hashes = {}
    cache: dict[str, str | None] = {}
    for name in input_names:
        value = getattr(args, name, None)
        if not value:
            continue
        key = str(Path(value).resolve())
        if key not in cache:
            cache[key] = sha256_file(value)
        hashes[name] = {"path": str(value), "sha256": cache[key]}
    hardware = {"device": str(device)}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        hardware.update(
            gpu_name=properties.name,
            compute_capability=f"{properties.major}.{properties.minor}",
            gpu_memory_bytes=int(properties.total_memory),
        )
    return {
        "schema_version": 2,
        "run_id": f"{time.time_ns()}-{os.getpid()}",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "warp": getattr(wp, "__version__", "unknown"),
        "mujoco_warp": getattr(mjwp, "__version__", "unknown"),
        "hardware": hardware,
        "source_tree_sha256": source_tree_hash(),
        "contract": contract,
        "args": vars(args).copy(),
        "inputs": hashes,
        "model_hash": getattr(env, "model_hash", None),
    }


# ---------------------------------------------------------------------- nets
class RunningNorm(nn.Module):
    """Running mean/std (parallel Welford merge); state rides the checkpoint."""

    def __init__(self, dim: int, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), float(x.shape[0])
        delta = bm - self.mean
        tot = self.count + bc
        self.mean.add_(delta * (bc / tot))
        self.var.copy_((self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot)
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


@torch.no_grad()
def normalizer_from_snapshot(snapshot: dict[str, torch.Tensor], dim: int,
                             device: torch.device) -> RunningNorm:
    normalizer = RunningNorm(dim).to(device)
    normalizer.mean.copy_(snapshot["mean"].to(device))
    normalizer.var.copy_(snapshot["var"].to(device))
    normalizer.count.copy_(snapshot["count"].to(device))
    normalizer.eval()
    return normalizer


def _mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.SiLU()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden,
                 architecture: str = "mlp", task_dim: int = 0):
        super().__init__()
        self.architecture = architecture
        self.task_dim = int(task_dim)
        if architecture == "mlp":
            self.trunk = _mlp([obs_dim, *hidden])
            output_dim = hidden[-1]
        elif architecture == "task_film":
            if not 0 < self.task_dim < obs_dim:
                raise ValueError("task_film requires 0 < task_dim < obs_dim")
            width = int(hidden[0])
            embed = min(128, width)
            self.feature_in = nn.Linear(obs_dim - self.task_dim, width)
            self.task_encoder = nn.Sequential(
                nn.Linear(self.task_dim, embed), nn.SiLU(), nn.Linear(embed, embed), nn.SiLU())
            self.film = nn.ModuleList(nn.Linear(embed, 2 * width) for _ in hidden)
            self.blocks = nn.ModuleList(nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.SiLU(),
                nn.Linear(2 * width, width)) for _ in hidden)
            output_dim = width
        else:
            raise ValueError(f"unknown actor architecture {architecture!r}")
        self.mu = nn.Linear(output_dim, act_dim)
        with torch.no_grad():                       # small final init: near-zero targets at start
            self.mu.weight.mul_(0.01)
            self.mu.bias.zero_()
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.architecture == "mlp":
            features = self.trunk(obs)
        else:
            physical, task = obs[:, :-self.task_dim], obs[:, -self.task_dim:]
            features = torch.nn.functional.silu(self.feature_in(physical))
            embedding = self.task_encoder(task)
            for block, modulation in zip(self.blocks, self.film):
                gamma, beta = modulation(embedding).chunk(2, dim=-1)
                conditioned = features * (1.0 + 0.25 * torch.tanh(gamma)) + 0.25 * beta
                features = features + 0.5 * block(conditioned)
        return self.mu(features)


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden):
        super().__init__()
        self.net = nn.Sequential(_mlp([in_dim, *hidden]), nn.Linear(hidden[-1], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def logp_tanh(z: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """log pi(tanh(z)) for the squashed gaussian; z is the PRE-tanh sample."""
    std = log_std.exp()
    base = (-0.5 * ((z - mu) / std) ** 2 - log_std - 0.5 * LOG2PI).sum(-1)
    return base - torch.log(1.0 - torch.tanh(z) ** 2 + 1e-6).sum(-1)


def entropy_tanh(mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """One-sample estimate of H[tanh(Z)] = H[Z] + E[log|dtanh/dz|]."""
    z = mu + log_std.exp() * torch.randn_like(mu)
    base = (log_std + 0.5 * (1.0 + LOG2PI)).sum(-1)
    return base + torch.log(1.0 - torch.tanh(z) ** 2 + 1e-6).sum(-1)


def clip_actor_critic_gradients(actor: nn.Module, critic: nn.Module,
                                max_norm: float = 1.0
                                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip disjoint policy/value gradients without cross-network scaling."""
    actor_norm = nn.utils.clip_grad_norm_(actor.parameters(), max_norm)
    critic_norm = nn.utils.clip_grad_norm_(critic.parameters(), max_norm)
    return actor_norm, critic_norm


def compute_gae(rewards: torch.Tensor, dones: torch.Tensor,
                values: torch.Tensor, last_value: torch.Tensor,
                gamma: float = GAMMA, lam: float = LAM) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimates for a time-major rollout.

    Time-limit terminal values are folded into ``rewards`` before this helper is
    called. Consequently both true terminations and truncations stop the recursive
    GAE tail; only the truncation reward contains the terminal value bootstrap.
    Keeping this pure makes the most failure-prone PPO bookkeeping hand-testable.
    """
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_value)
    for t in reversed(range(rewards.shape[0])):
        nonterminal = 1.0 - dones[t]
        next_value = last_value if t == rewards.shape[0] - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    return adv, adv + values


_ENV_STATE_TENSORS = (
    "_cmd", "_timer", "_t", "_air", "_prev_a", "_prev_xy",
    "_progress_ema", "_duty_ema", "_foot_duty_ema",
    "_hop_peak_z", "_hop_airborne",
    "_prev_dist", "_prev_dealt", "_vel_ema", "_combat_time",
    "_prev_dist_b", "_prev_dealt_b", "_vel_ema_b", "_combat_time_b",
    "_qpos0", "_attack_leg", "_attack_active", "_attack_override_leg",
    "_attack_override_active", "_attack_timer", "_attack_phase_step",
    "_prev_extension", "_prev_support",
    "_pose_command", "_height_command", "_goal", "_heading_command",
    "_velocity_command", "_route_index", "_task_t", "_constraint_age",
    "_cycle_contact_sum", "_cycle_contact_steps",
    "_prev_goal_distance",
    "_lidar", "_lidar_previous", "_ladder_prev_distance",
    "_constraint_duals", "_constraint_error_square",
    "_competence_duals", "_competence_error_square",
    "_ladder_prev_rod", "_ladder_prev_taken",
    "actions_a", "actions_b",
)


def duty_stagnation_tripwire_enabled(geometry: str, rung: int | None) -> bool:
    """Whether full-time stance is evidence of a failed locomotion policy."""
    return geometry != "ladder_locomotion"


def capture_env_state(env) -> dict:
    """Capture the physical and episodic state needed for exact continuation."""
    if hasattr(env, "envs"):
        return {"ensemble": [capture_env_state(member) for member in env.envs]}
    tensors = {
        "qpos": env.qpos.detach().cpu().clone(),
        "qvel": env.qvel.detach().cpu().clone(),
        "qacc_warmstart": env.qacc_warmstart.detach().cpu().clone(),
        "sim_time": env.sim_time.detach().cpu().clone(),
    }
    for name in _ENV_STATE_TENSORS:
        if hasattr(env, name):
            tensors[name] = getattr(env, name).detach().cpu().clone()
    return {"tensors": tensors, "generator": env._gen.get_state().cpu()}


def restore_env_state(env, state: dict) -> None:
    """Restore ``capture_env_state`` and refresh all derived MuJoCo fields."""
    if "ensemble" in state:
        for member, member_state in zip(env.envs, state["ensemble"]):
            restore_env_state(member, member_state)
        return
    tensors = state["tensors"]
    for name in ("qpos", "qvel", "sim_time"):
        dst = getattr(env, name)
        dst.copy_(tensors[name].to(device=dst.device, dtype=dst.dtype))
    for name in _ENV_STATE_TENSORS:
        if name in tensors and hasattr(env, name):
            dst = getattr(env, name)
            source = tensors[name].to(device=dst.device, dtype=dst.dtype)
            if (source.shape != dst.shape
                    and name in ("_constraint_duals", "_constraint_error_square")
                    and source.ndim == dst.ndim == 1):
                # Constraint sets may gain a new independently learned physical
                # contract. Preserve existing multipliers by name/order and
                # initialize only the newly introduced constraints at zero.
                dst.zero_()
                common = min(source.numel(), dst.numel())
                dst[:common].copy_(source[:common])
            else:
                dst.copy_(source)
    # ``torch.load(..., map_location="cuda")`` also maps serialized RNG byte
    # tensors to CUDA, but Generator.set_state() requires a CPU ByteTensor even
    # when the generator itself is CUDA-backed.
    env._gen.set_state(state["generator"].cpu())
    with wp.ScopedDevice(env._wp_device):
        mjwp.forward(env._wm, env._wd)
    if hasattr(env, "_refresh_outputs"):
        env._refresh_outputs()
    # Forward recomputes derived fields. Restore solver warm-start last so the
    # next transition matches an uninterrupted run as closely as the backend permits.
    env.qacc_warmstart.copy_(tensors["qacc_warmstart"].to(
        device=env.qacc_warmstart.device, dtype=env.qacc_warmstart.dtype))


def capture_runtime_state(env) -> dict:
    out = {"env": capture_env_state(env), "torch_rng": torch.get_rng_state()}
    if torch.cuda.is_available():
        out["cuda_rng"] = torch.cuda.get_rng_state_all()
    return out


def restore_runtime_state(env, state: dict | None) -> None:
    if not state:
        return
    restore_env_state(env, state["env"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if torch.cuda.is_available() and "cuda_rng" in state:
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])


def checkpoint_contract(env, args) -> dict:
    """Semantic identity required before policy/optimizer state may be reused."""
    contract = {
        "geometry": args.geometry,
        "model_hash": env.model_hash,
        "action_semantics": getattr(
            env, "action_semantics", "pd_target@50hz:lowpass+torque_speed_v1"),
        "observation_semantics": getattr(
            env, "observation_semantics",
            f"actor{env.obs_dim}+priv{env.priv_dim}:v1"),
        "reward_semantics": getattr(
            env, "reward_semantics", f"{args.geometry}:velocity_command:v1"),
    }
    architecture = getattr(args, "architecture", "mlp")
    if architecture != "mlp":
        contract.update(actor_architecture=architecture,
                        actor_task_dim=int(getattr(env, "architecture_task_dim", 0)))
    return contract


def validate_training_args(args, env, hidden: tuple[int, ...]) -> None:
    batch = int(args.envs) * int(args.horizon)
    if min(args.envs, args.horizon, args.minibatches, args.epochs, args.steps,
           getattr(args, "eval_envs", 1), getattr(args, "eval_steps", 1),
           getattr(args, "diagnostic_eval_seeds", 1),
           getattr(args, "checkpoint_replay_steps", 1),
           getattr(args, "early_patience", 1)) <= 0:
        raise ValueError(
            "training, evaluation, and diagnostic sizes must all be positive")
    if batch % args.minibatches:
        raise ValueError(
            f"rollout batch {batch} is not divisible by {args.minibatches} minibatches; "
            "the current slicing would silently discard samples")
    if getattr(args, "target_kl", 0.0) < 0.0:
        raise ValueError("--target-kl must be non-negative (zero disables the controller)")
    if getattr(args, "kl_stop_multiplier", 1.5) <= 1.0:
        raise ValueError("--kl-stop-multiplier must be greater than one")
    if args.preflight != "off":
        from preflight import preflight_check
        preflight_check(
            steps=args.steps, batch=args.envs, minibatches=1, unroll=args.horizon,
            episode_length=args.episode_length, discounting=GAMMA,
            control_dt=env._dt, obs_dim=env.obs_dim, hidden0=hidden[0],
            from_scratch=not bool(args.resume or args.init_policy), mode=args.preflight,
            tag=Path(args.tag).name, run_dir=Path(args.tag).parent,
            resolved=vars(args))


# ---------------------------------------------------------------------- io
def save_ckpt(path: Path, step: int, actor, critic, obs_norm, priv_norm, opt, args,
              *, contract: dict, runtime: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "actor": actor.state_dict(), "critic": critic.state_dict(),
                "obs_norm": obs_norm.state_dict(), "priv_norm": priv_norm.state_dict(),
                "opt": opt.state_dict(), "args": vars(args), "contract": contract,
                "runtime": runtime}, path)


def load_ckpt(path, actor, critic, obs_norm, priv_norm, opt, device, *,
              expected_contract: dict, allow_legacy: bool = False,
              allow_reward_migration: bool = False) -> tuple[int, dict | None]:
    ck = torch.load(path, map_location=device, weights_only=False)
    got = ck.get("contract")
    if got is None and not allow_legacy:
        raise ValueError(
            f"checkpoint {path} has no model/action/observation contract; "
            "use --allow-legacy-resume only for a deliberate diagnostic load")
    mismatch = ({k: (got.get(k), want) for k, want in expected_contract.items()
                 if got.get(k) != want} if got is not None else {})
    reward_migration = bool(
        allow_reward_migration and set(mismatch) == {"reward_semantics"})
    if mismatch and not reward_migration:
        raise ValueError(f"checkpoint {path} is incompatible: {mismatch}")
    actor.load_state_dict(ck["actor"])
    obs_norm.load_state_dict(ck["obs_norm"])
    priv_norm.load_state_dict(ck["priv_norm"])
    if reward_migration:
        # The behavior and physical state remain useful, but a critic trained on
        # the old reward and its Adam moments are stale. Retain only the old
        # conservative learning rate so the first migrated actor update cannot
        # jump back to the CLI default.
        old_groups = (ck.get("opt") or {}).get("param_groups", ())
        if opt is not None:
            for current, previous in zip(opt.param_groups, old_groups):
                current["lr"] = float(previous.get("lr", current["lr"]))
        print(f"reward migration {mismatch['reward_semantics'][0]} -> "
              f"{mismatch['reward_semantics'][1]}: actor/runtime preserved; "
              "critic and optimizer state reset", flush=True)
    else:
        critic.load_state_dict(ck["critic"])
        if opt is not None and ck.get("opt") is not None:
            opt.load_state_dict(ck["opt"])
    return int(ck["step"]), ck.get("runtime")


def load_policy(path, obs_dim: int, act_dim: int, device):
    """Load a deterministic actor and its observation normalizer."""
    ck = torch.load(path, map_location=device, weights_only=False)
    hidden = tuple(int(v) for v in ck.get("args", {}).get("hidden", "512,256,128").split(","))
    architecture = ck.get("args", {}).get("architecture", "mlp")
    task_dim = int(ck.get("args", {}).get("actor_task_dim", 0))
    actor = Actor(obs_dim, act_dim, hidden, architecture=architecture,
                  task_dim=task_dim).to(device)
    norm = RunningNorm(obs_dim).to(device)
    actor.load_state_dict(ck["actor"])
    norm.load_state_dict(ck["obs_norm"])
    actor.eval()
    norm.eval()

    @torch.no_grad()
    def policy(obs):
        return torch.tanh(actor(norm(obs)))

    return policy


def initialize_policy(path, actor, obs_norm, device) -> int:
    """Warm-start actor + actor normalization, leaving critic/optimizer fresh.

    A ladder transition changes the reward contract, so resuming the optimizer
    would incorrectly retain the previous value target and Adam moments.  Exact
    tensor shapes are required; cross-family transitions start a new policy.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    try:
        actor.load_state_dict(ck["actor"], strict=True)
        obs_norm.load_state_dict(ck["obs_norm"], strict=True)
    except RuntimeError as error:
        raise ValueError(
            f"initial policy {path} has a different observation/action architecture; "
            "warm starts are only valid inside one ladder family") from error
    return int(ck.get("step", 0))


def frozen_anchor_policy(path, obs_dim: int, act_dim: int, hidden,
                         architecture: str, task_dim: int, device):
    """Load the accepted pre-rung actor/norm as a no-gradient teacher."""
    ck = torch.load(path, map_location=device, weights_only=False)
    actor = Actor(obs_dim, act_dim, hidden, architecture=architecture,
                  task_dim=task_dim).to(device)
    norm = RunningNorm(obs_dim).to(device)
    try:
        actor.load_state_dict(ck["actor"], strict=True)
        norm.load_state_dict(ck["obs_norm"], strict=True)
    except RuntimeError as error:
        raise ValueError(f"anchor policy {path} is not architecture-compatible") from error
    actor.eval(); norm.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor, norm


def schedules(step: int, args) -> tuple[float, float, float]:
    """(ent_coef, alpha, imit_anneal) at env-step `step` — all linear."""
    p = min(step / max(args.steps, 1), 1.0)
    ent = ENT_START + (ENT_END - ENT_START) * p
    ap = min(p / max(args.alpha_frac, 1e-9), 1.0)
    alpha = args.alpha_start + (args.alpha_end - args.alpha_start) * ap
    imit = max(0.0, 1.0 - p / max(args.imit_anneal_frac, 1e-9))
    return ent, alpha, imit


def parse_early_gates(specs: list[str]) -> tuple[tuple[str, str, float], ...]:
    gates = []
    for spec in specs:
        try:
            metric, comparison, threshold = spec.split(",", 2)
            if comparison not in (">=", "<="):
                raise ValueError
            gates.append((metric, comparison, float(threshold)))
        except ValueError as error:
            raise ValueError(
                f"invalid --early-gate {spec!r}; expected metric,>=|<=,threshold") from error
    return tuple(gates)


def action_prior_weight(base: float, floor: float, progress: int, total: int,
                        constraint_pressure: float = 0.0,
                        competence_pressure: float = 1.0) -> float:
    """Anneal a behavior prior while automatically yielding to safety pressure.

    Purely time-based annealing can discard a useful scaffold even though its
    demonstrated competence has not yet transferred.  The normalized competence
    shortfall therefore supplies a second schedule: prior influence decays only
    when either the usual annealing is incomplete or the target has actually been
    learned.  Safety pressure remains the denominator, so an unsafe teacher still
    yields automatically.  This is a dimensionless arbitration between measured
    contracts rather than another rung-specific reward coefficient.
    """
    time_schedule = max(
        float(floor), 1.0 - (float(progress) / max(total, 1)) / 0.60)
    pressure = max(float(competence_pressure), 1.0)
    competence_schedule = 1.0 - 1.0 / pressure
    schedule = max(time_schedule, competence_schedule)
    return (float(base) * schedule
            / (1.0 + max(float(constraint_pressure), 0.0)))


def prior_competence_pressure(target: float, observed: float) -> float:
    """Dimensionless pressure to recover a missing demonstrated competence."""
    target = max(float(target), 1.0e-9)
    observed = max(float(observed), 0.0)
    shortfall = max(target - observed, 0.0)
    return 1.0 + shortfall / max(observed, 0.10 * target)


def early_gates_pass(gates: tuple[tuple[str, str, float], ...], metrics: dict) -> bool:
    for metric, comparison, threshold in gates:
        if metric not in metrics:
            return False
        value = float(metrics[metric])
        if comparison == ">=" and value < threshold:
            return False
        if comparison == "<=" and value > threshold:
            return False
    return bool(gates)


def gate_diagnostics(gates: tuple[tuple[str, str, float], ...], metrics: dict) -> dict:
    """Report signed, scale-free headroom for every acceptance contract.

    Positive margin means pass, negative means fail.  Scaling by the contract
    value lets a dashboard compare unlike units (fall probability, metres per
    second, duty factor) without inventing another reward weight.
    """
    checks = []
    for metric, comparison, threshold in gates:
        raw = metrics.get(metric)
        if raw is None:
            checks.append({
                "metric": metric, "comparison": comparison, "threshold": threshold,
                "value": None, "pass": False, "margin": None,
                "relative_margin": None,
            })
            continue
        value = float(raw)
        margin = value - threshold if comparison == ">=" else threshold - value
        scale = max(abs(float(threshold)), 1.0e-9)
        checks.append({
            "metric": metric, "comparison": comparison, "threshold": threshold,
            "value": value, "pass": margin >= 0.0, "margin": margin,
            "relative_margin": margin / scale,
        })
    worst = min(
        (row for row in checks if row["relative_margin"] is not None),
        key=lambda row: row["relative_margin"], default=None)
    return {
        "all_pass": bool(checks) and all(row["pass"] for row in checks),
        "worst_metric": worst["metric"] if worst else None,
        "worst_relative_margin": worst["relative_margin"] if worst else None,
        "checks": checks,
    }


def evaluation_trends(previous: dict | None, current: dict, step: int) -> dict:
    """Per-million-sample slopes for high-value metrics between evaluations."""
    if not previous:
        return {}
    delta_steps = int(step) - int(previous.get("step", step))
    if delta_steps <= 0:
        return {}
    scale = 1_000_000.0 / delta_steps
    keys = (
        "reward", "xprogress", "lateral", "align", "catrate", "fallrate",
        "ladder_step_clock", "ladder_swing_clearance",
    )
    return {
        f"{key}_per_million": (float(current[key]) - float(previous[key])) * scale
        for key in keys if key in current and key in previous
    }


def incremental_eval_interval(current_step: int, target_step: int, evals: int,
                              rollout_steps: int) -> int:
    """Space evaluations across the work remaining in this invocation.

    ``--steps`` is an absolute checkpoint target.  Using that lifetime total to
    schedule a resumed run makes feedback progressively sparser: a 2M-step
    continuation from step 12M would otherwise wait the entire increment for a
    single evaluation.  Round to a rollout boundary so the requested number of
    deterministic evaluations covers the *new* experience instead.
    """
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    remaining = max(int(target_step) - int(current_step), 1)
    desired = math.ceil(remaining / max(int(evals), 1))
    return max(rollout_steps, math.ceil(desired / rollout_steps) * rollout_steps)


def kl_epoch_should_stop(observed_kl: float, target_kl: float,
                         multiplier: float = 1.5) -> bool:
    """Stop extra PPO epochs once the whole rollout leaves its trust region."""
    return (target_kl > 0.0 and math.isfinite(observed_kl)
            and observed_kl > target_kl * multiplier)


def adaptive_ppo_learning_rate(current: float, ceiling: float,
                               observed_kl: float, target_kl: float) -> float:
    """Scale the next PPO update from measured KL, within conservative bounds."""
    current, ceiling = float(current), float(ceiling)
    if target_kl <= 0.0 or not math.isfinite(observed_kl):
        return current
    ratio = max(float(observed_kl), 0.0) / target_kl
    if ratio > 2.0:
        candidate = 0.5 * current
    elif ratio > 1.25:
        candidate = current / 1.5
    elif ratio < 0.5:
        candidate = 1.1 * current
    else:
        candidate = current
    return min(ceiling, max(0.05 * ceiling, candidate))


@torch.no_grad()
def policy_epoch_trust(actor, observations: torch.Tensor,
                       sampled_pre_tanh: torch.Tensor,
                       old_logp: torch.Tensor) -> dict:
    """Lightweight whole-rollout KL used by the online epoch controller."""
    mean = actor(observations)
    new_logp = logp_tanh(sampled_pre_tanh, mean, actor.log_std)
    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)
    return {
        "approx_kl": float(((ratio - 1.0) - log_ratio).mean()),
        "clip_fraction": float(((ratio - 1.0).abs() > CLIP).float().mean()),
    }


@torch.no_grad()
def evaluate(env, actor, obs_norm, alpha, imit, steps: int, *, reset_seed: int) -> dict:
    """Fixed-scenario pass with optional sampled actions and replay fingerprint."""
    return evaluate_policy(
        env, actor, obs_norm, alpha, imit, steps, reset_seed=reset_seed)


@torch.no_grad()
def evaluate_policy(env, actor, obs_norm, alpha, imit, steps: int, *,
                    reset_seed: int, stochastic: bool = False,
                    action_seed: int | None = None,
                    fingerprint: bool = False) -> dict:
    env._gen.manual_seed(reset_seed)
    obs = env.reset()
    tel = EvalTelemetry(env.device)
    action_generator = torch.Generator(device=env.device)
    action_generator.manual_seed(
        int(action_seed if action_seed is not None else reset_seed + 1_000_003))
    digest = hashlib.sha256() if fingerprint else None
    for _ in range(steps):
        mean = actor(obs_norm(obs))
        if stochastic:
            noise = torch.randn(mean.shape, generator=action_generator,
                                dtype=mean.dtype, device=mean.device)
            pre_tanh = mean + actor.log_std.exp() * noise
        else:
            pre_tanh = mean
        a = torch.tanh(pre_tanh)
        obs, rew, done, info = env.step(a, alpha=alpha, imit_anneal=imit)
        tel.add(rew, info)
        if digest is not None:
            for value in (a, rew[:, None], done[:, None]):
                quantized = torch.round(value.detach() * 1.0e6).to(
                    torch.int32).cpu().contiguous()
                digest.update(quantized.numpy().tobytes())
    result = tel.result()
    if digest is not None:
        result["trajectory_sha256"] = digest.hexdigest()
    return result


# ---------------------------------------------------------------------- train
def train(args) -> dict:
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    EnvClass = GEOMETRIES[args.geometry]
    env_kwargs = dict(seed=args.seed, device=device, episode_length=args.episode_length)
    eval_kwargs = dict(seed=args.seed + 1000, device=device,
                       episode_length=args.episode_length)
    if args.geometry in ("ladder_locomotion", "ladder_combat"):
        if args.rung is None:
            raise ValueError(f"--geometry {args.geometry} requires --rung")
        env_kwargs["rung"] = eval_kwargs["rung"] = args.rung
    env = EnvClass(args.envs, **env_kwargs)
    eval_env = EnvClass(args.eval_envs, **eval_kwargs)
    dev = env.device
    if args.action_prior_json:
        if not hasattr(env, "configure_action_prior"):
            raise ValueError("--action-prior-json requires an environment prior loader")
        env.configure_action_prior(args.action_prior_json)
        eval_env.configure_action_prior(args.action_prior_json)
        print(f"action-prior artifact={args.action_prior_json}", flush=True)
    hidden = tuple(int(h) for h in args.hidden.split(","))
    validate_training_args(args, env, hidden)
    args.actor_task_dim = int(getattr(env, "architecture_task_dim", 0))
    actor = Actor(env.obs_dim, env.act_dim, hidden, architecture=args.architecture,
                  task_dim=args.actor_task_dim).to(dev)
    critic = Critic(env.obs_dim + env.priv_dim, hidden).to(dev)
    obs_norm = RunningNorm(env.obs_dim).to(dev)
    priv_norm = RunningNorm(env.priv_dim).to(dev)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr)
    opponent_path = getattr(args, "opponent", None)
    if opponent_path:
        if not hasattr(env, "set_opponent"):
            raise ValueError("--opponent is only valid for a two-policy environment")
        opponent = load_policy(opponent_path, env.obs_dim, env.act_dim, dev)
        env.set_opponent(opponent)
        eval_env.set_opponent(opponent)
    global_step = 0
    contract = checkpoint_contract(env, args)
    if args.init_policy and args.resume:
        raise ValueError("--init-policy and --resume are mutually exclusive")
    if args.init_policy:
        source_step = initialize_policy(args.init_policy, actor, obs_norm, dev)
        print(f"initialized actor from {args.init_policy} (source step {source_step}); "
              "critic and optimizer are fresh", flush=True)
    if args.resume:
        global_step, runtime = load_ckpt(
            args.resume, actor, critic, obs_norm, priv_norm, opt, dev,
            expected_contract=contract, allow_legacy=args.allow_legacy_resume,
            allow_reward_migration=args.allow_reward_migration)
        restore_runtime_state(env, runtime)
        print(f"resumed {args.resume} at step {global_step}", flush=True)
    anchor_actor = anchor_norm = None
    anchor_indices: tuple[int, ...] = ()
    if args.anchor_policy:
        anchor_actor, anchor_norm = frozen_anchor_policy(
            args.anchor_policy, env.obs_dim, env.act_dim, hidden,
            args.architecture, args.actor_task_dim, dev)
        if args.anchor_task_indices:
            anchor_indices = tuple(int(value) for value in args.anchor_task_indices.split(","))
            if any(value < 0 or value >= args.actor_task_dim for value in anchor_indices):
                raise ValueError("--anchor-task-indices contains an invalid task channel")
        print(f"retention teacher={args.anchor_policy} task_indices={anchor_indices or 'current'} "
              f"weight={args.distill_weight}", flush=True)
    transfer_policy = None
    transfer_obs_dim = int(args.transfer_obs_dim or env.obs_dim)
    if args.transfer_policy:
        if not 0 < transfer_obs_dim <= env.obs_dim:
            raise ValueError("--transfer-obs-dim must be in [1, env.obs_dim]")
        transfer_policy = load_policy(
            args.transfer_policy, transfer_obs_dim, env.act_dim, dev)
        print(f"behavior transfer teacher={args.transfer_policy} "
              f"obs_prefix={transfer_obs_dim}", flush=True)
    ckpt_path = Path(f"{args.tag}.pt")
    rollout_steps = args.horizon * args.envs
    eval_interval = incremental_eval_interval(
        global_step, args.steps, args.evals, rollout_steps)
    next_eval = min(args.steps, global_step + eval_interval)
    print(f"train_mesh_warp: geometry={args.geometry} device={dev} envs={args.envs} "
          f"horizon={args.horizon} steps={args.steps} hidden={hidden} "
          f"imitation={'ON' if env.gait_loaded else 'off'} ckpt={ckpt_path}", flush=True)

    T, N = args.horizon, args.envs
    obs = env.observe()
    priv = env.privileged()
    b_obs = torch.zeros((T, N, env.obs_dim), device=dev)       # normalized (as acted on)
    b_priv = torch.zeros((T, N, env.priv_dim), device=dev)     # normalized (critic input)
    b_raw_obs = torch.zeros_like(b_obs)                        # raw, for norm updates
    b_raw_priv = torch.zeros_like(b_priv)
    b_z = torch.zeros((T, N, env.act_dim), device=dev)         # pre-tanh samples
    b_logp = torch.zeros((T, N), device=dev)
    b_rew = torch.zeros((T, N), device=dev)
    b_done = torch.zeros((T, N), device=dev)
    b_val = torch.zeros((T, N), device=dev)
    start_step = global_step
    t_start = time.time()
    provenance = training_provenance(args, env, contract, dev)
    metrics_path = Path(f"{args.tag}.metrics.jsonl")
    diagnostics_path = Path(f"{args.tag}.diagnostics.json")
    stats = {
        "schema_version": 3,
        "run": provenance,
        "updates": [],
        "evals": [],
        "diagnostics": [],
        "ckpt": str(ckpt_path),
        "metrics_jsonl": str(metrics_path),
        "diagnostics_latest": str(diagnostics_path),
    }
    append_jsonl(metrics_path, {
        "event": "run_start", "run": provenance, "start_step": start_step,
        "target_step": args.steps,
    })
    early_gates = parse_early_gates(args.early_gate)
    early_streak = 0
    action_prior_base_weight = args.action_prior_weight * float(
        getattr(env, "action_prior_scale", 1.0))
    action_prior_floor = float(getattr(env, "action_prior_floor_fraction", 0.0))
    constraint_names = tuple(getattr(env, "adaptive_constraint_names", ()))
    competence_names = tuple(getattr(env, "adaptive_competence_names", ()))
    prior_competence_metric = getattr(env, "action_prior_competence_metric", None)
    prior_competence_target = next((threshold for metric, comparison, threshold in early_gates
                                    if metric == prior_competence_metric
                                    and comparison == ">="), None)
    prior_competence_ema = prior_competence_target
    current_action_prior_weight = action_prior_base_weight
    previous_eval_norm = normalization_snapshot(obs_norm)
    while global_step < args.steps:
        update_started = time.perf_counter()
        diagnostic_update = (global_step + T * N >= next_eval
                             or global_step + T * N >= args.steps)
        obs_norm_before = normalization_snapshot(obs_norm)
        ent_coef, alpha, imit = schedules(global_step, args)
        constraint_pressure = (float(getattr(
            env, "action_prior_suppression_pressure", env.constraint_duals.max()))
            if constraint_names else 0.0)
        competence_pressure = (prior_competence_pressure(
            prior_competence_target, prior_competence_ema)
            if prior_competence_target is not None and prior_competence_ema is not None
            else 1.0)
        current_action_prior_weight = action_prior_weight(
            action_prior_base_weight, action_prior_floor, global_step,
            args.steps, constraint_pressure, competence_pressure)
        with torch.no_grad():
            rollout_telemetry = EvalTelemetry(dev) if diagnostic_update else None
            constraint_sums = torch.zeros(len(constraint_names), device=dev)
            competence_constraint_sums = torch.zeros(len(competence_names), device=dev)
            competence_sum = torch.zeros((), device=dev)
            for t in range(T):
                obs_n, priv_n = obs_norm(obs), priv_norm(priv)
                mu = actor(obs_n)
                z = mu + actor.log_std.exp() * torch.randn_like(mu)
                nobs, rew, done, info = env.step(torch.tanh(z), alpha=alpha, imit_anneal=imit)
                if rollout_telemetry is not None:
                    rollout_telemetry.add(rew, info)
                trunc = info["truncated"]
                # time-limit bootstrap: V(terminal obs) folded into the reward
                tv = critic(torch.cat([obs_norm(info["terminal_obs"]),
                                       priv_norm(info["terminal_priv"])], -1))
                b_raw_obs[t], b_raw_priv[t] = obs, priv
                b_obs[t], b_priv[t], b_z[t] = obs_n, priv_n, z
                b_logp[t] = logp_tanh(z, mu, actor.log_std)
                b_val[t] = critic(torch.cat([obs_n, priv_n], -1))
                b_rew[t] = rew + GAMMA * tv * trunc
                b_done[t] = done
                for constraint_index, name in enumerate(constraint_names):
                    constraint_sums[constraint_index].add_(info[name].mean())
                for competence_index, name in enumerate(competence_names):
                    competence_constraint_sums[competence_index].add_(info[name].mean())
                if prior_competence_metric is not None:
                    competence_sum.add_(info[prior_competence_metric].mean())
                obs, priv = nobs, info["priv"]
            last_val = critic(torch.cat([obs_norm(obs), priv_norm(priv)], -1))
            adv, ret = compute_gae(b_rew, b_done, b_val, last_val)
            if constraint_names:
                constraint_observed = constraint_sums / float(T)
                env.update_constraint_duals(constraint_observed)
            else:
                constraint_observed = constraint_sums
            if competence_names:
                competence_observed = competence_constraint_sums / float(T)
                env.update_competence_duals(competence_observed)
            else:
                competence_observed = competence_constraint_sums
            if prior_competence_metric is not None:
                observed_competence = float(competence_sum / float(T))
                prior_competence_ema = (observed_competence if prior_competence_ema is None
                                        else 0.9 * prior_competence_ema
                                        + 0.1 * observed_competence)
        training_rollout_metrics = (rollout_telemetry.result()
                                    if rollout_telemetry is not None else {})
        rollout_finished = time.perf_counter()
        obs_norm.update(b_raw_obs.reshape(-1, env.obs_dim))
        priv_norm.update(b_raw_priv.reshape(-1, env.priv_dim))

        B = T * N
        f_obs = b_obs.reshape(B, -1)
        f_raw_obs = b_raw_obs.reshape(B, -1)
        f_cin = torch.cat([f_obs, b_priv.reshape(B, -1)], -1)
        f_z, f_logp = b_z.reshape(B, -1), b_logp.reshape(B)
        f_adv_raw = adv.reshape(B)
        f_adv = (f_adv_raw - f_adv_raw.mean()) / (f_adv_raw.std() + 1e-8)
        f_ret = ret.reshape(B)
        actor_before = parameter_snapshot(actor)
        critic_before = parameter_snapshot(critic)
        with torch.no_grad():
            ret_var = f_ret.var(unbiased=False)
            explained_variance = torch.where(
                ret_var > 1.0e-8,
                1.0 - (f_ret - b_val.reshape(B)).var(unbiased=False) / ret_var,
                torch.zeros_like(ret_var))
            rollout_diagnostics = {
                "reward_mean": float(b_rew.mean()),
                "termination_rate": float(b_done.mean()),
                "advantage_mean": float(f_adv_raw.mean()),
                "advantage_std": float(f_adv_raw.std(unbiased=False)),
                "return_mean": float(f_ret.mean()),
                "return_std": float(f_ret.std(unbiased=False)),
                "explained_variance_before_update": float(explained_variance),
            }
            critic_before_update = critic_calibration(b_val.reshape(B), f_ret)
            next_values = torch.cat((b_val[1:], last_val.unsqueeze(0)), dim=0)
            td_error = (b_rew + GAMMA * next_values * (1.0 - b_done) - b_val).reshape(B)
            observation_normalization = normalization_diagnostics(
                obs_norm_before, obs_norm, f_raw_obs)
        mb = B // args.minibatches
        pi_l = v_l = ent_l = distill_l = action_prior_l = 0.0
        approx_kl_l = clip_fraction_l = gradient_norm_l = 0.0
        actor_gradient_norm_l = critic_gradient_norm_l = 0.0
        actor_gradient_norms: list[float] = []
        critic_gradient_norms: list[float] = []
        epoch_trust_region: list[dict] = []
        objective_gradients: dict = {}
        epochs_completed = 0
        kl_early_stop = False
        learning_rate_used = float(opt.param_groups[0]["lr"])
        prior_axis_rmse_t: dict[str, torch.Tensor] = {}
        prior_leg_rmse_t: dict[str, torch.Tensor] = {}
        for epoch_index in range(args.epochs):
            perm = torch.randperm(B, device=dev)
            for i in range(args.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                mu = actor(f_obs[idx])
                logp = logp_tanh(f_z[idx], mu, actor.log_std)
                ratio = torch.exp(logp - f_logp[idx])
                a_mb = f_adv[idx]
                pg = -torch.min(ratio * a_mb,
                                ratio.clamp(1.0 - CLIP, 1.0 + CLIP) * a_mb).mean()
                vloss = 0.5 * ((critic(f_cin[idx]) - f_ret[idx]) ** 2).mean()
                ent = entropy_tanh(mu, actor.log_std).mean()
                distill = torch.zeros((), device=dev)
                if anchor_actor is not None and args.distill_weight > 0.0:
                    anchor_raw = f_raw_obs[idx].clone()
                    if anchor_indices:
                        rows = torch.arange(len(idx), device=dev)
                        choices = torch.as_tensor(anchor_indices, device=dev)[
                            rows % len(anchor_indices)]
                        anchor_raw[:, -args.actor_task_dim:] = 0.0
                        anchor_raw[rows, anchor_raw.shape[1] - args.actor_task_dim + choices] = 1.0
                    with torch.no_grad():
                        teacher_mu = anchor_actor(anchor_norm(anchor_raw))
                    student_mu = actor(obs_norm(anchor_raw))
                    distill = ((student_mu - teacher_mu) ** 2).mean()
                action_prior = torch.zeros((), device=dev)
                if current_action_prior_weight > 0.0 and hasattr(env, "policy_mean_prior"):
                    prior_base = None
                    if anchor_actor is not None and anchor_indices:
                        transfer_raw = f_raw_obs[idx].clone()
                        transfer_raw[:, -args.actor_task_dim:] = 0.0
                        transfer_raw[:, transfer_raw.shape[1] - args.actor_task_dim
                                     + max(anchor_indices)] = 1.0
                        if getattr(env, "rung", None) == 7:
                            transfer_raw[:, 47:50] = 0.0
                        with torch.no_grad():
                            prior_base = anchor_actor(anchor_norm(transfer_raw))
                    transfer_action = None
                    if transfer_policy is not None:
                        with torch.no_grad():
                            transfer_action = transfer_policy(
                                f_raw_obs[idx, :transfer_obs_dim])
                    prior = env.policy_mean_prior(
                        f_raw_obs[idx], prior_base,
                        transfer_action=transfer_action)
                    if prior is not None:
                        prior_target, prior_mask = prior
                        action_prior = (((mu - prior_target) * prior_mask) ** 2).sum() \
                            / prior_mask.sum().clamp_min(1.0)
                        with torch.no_grad():
                            error = mu - prior_target
                            active_prior = prior_mask.abs() > 0.0
                            for axis_name, axis in (("yaw", 0), ("pitch", 1), ("lift", 2)):
                                selected = active_prior[:, axis::3]
                                squared = error[:, axis::3].square()
                                prior_axis_rmse_t[axis_name] = torch.sqrt(
                                    (squared * selected).sum()
                                    / selected.sum().clamp_min(1)).detach()
                            for leg_index, leg_name in enumerate(("fl", "fr", "rl", "rr")):
                                selected = active_prior[:, leg_index * 3:(leg_index + 1) * 3]
                                squared = error[:, leg_index * 3:(leg_index + 1) * 3].square()
                                prior_leg_rmse_t[leg_name] = torch.sqrt(
                                    (squared * selected).sum()
                                    / selected.sum().clamp_min(1)).detach()
                loss = (pg + 0.5 * vloss - ent_coef * ent
                        + args.distill_weight * distill
                        + current_action_prior_weight * action_prior)
                opt.zero_grad()
                if diagnostic_update and i == args.minibatches - 1:
                    objective_gradients = objective_gradient_diagnostics({
                        "ppo_policy": pg,
                        "entropy": -ent_coef * ent,
                        "retention_distillation": args.distill_weight * distill,
                        "action_prior": current_action_prior_weight * action_prior,
                    }, actor.parameters())
                loss.backward()
                # Actor and critic have disjoint parameters and fundamentally
                # different loss scales.  Joint clipping lets a large raw-return
                # value loss suppress the policy gradient by the critic's clip
                # factor.  Bound each optimizer subspace independently so reward
                # scale cannot silently freeze policy learning.
                actor_gradient_norm, critic_gradient_norm = \
                    clip_actor_critic_gradients(actor, critic)
                opt.step()
                with torch.no_grad():
                    log_ratio = logp - f_logp[idx]
                    approx_kl_l = float(((ratio - 1.0) - log_ratio).mean())
                    clip_fraction_l = float(((ratio - 1.0).abs() > CLIP).float().mean())
                    actor_gradient_norm_l = float(actor_gradient_norm)
                    critic_gradient_norm_l = float(critic_gradient_norm)
                    actor_gradient_norms.append(actor_gradient_norm_l)
                    critic_gradient_norms.append(critic_gradient_norm_l)
                    gradient_norm_l = math.hypot(
                        actor_gradient_norm_l, critic_gradient_norm_l)
                pi_l, v_l, ent_l = float(pg.detach()), float(vloss.detach()), float(ent.detach())
                distill_l = float(distill.detach())
                action_prior_l = float(action_prior.detach())
            if diagnostic_update:
                epoch_record = policy_trust_region_diagnostics(
                    actor, f_obs, f_z, f_logp, logp_tanh, CLIP)
                epoch_record["epoch"] = epoch_index + 1
                epoch_trust_region.append(epoch_record)
            else:
                epoch_record = policy_epoch_trust(actor, f_obs, f_z, f_logp)
            epochs_completed = epoch_index + 1
            if kl_epoch_should_stop(
                    epoch_record["approx_kl"], args.target_kl,
                    args.kl_stop_multiplier):
                kl_early_stop = True
                break
        optimization_finished = time.perf_counter()
        trust_region = policy_trust_region_diagnostics(
            actor, f_obs, f_z, f_logp, logp_tanh, CLIP)
        learning_rate_next = adaptive_ppo_learning_rate(
            learning_rate_used, args.lr, trust_region["approx_kl"], args.target_kl)
        for parameter_group in opt.param_groups:
            parameter_group["lr"] = learning_rate_next
        actor_update = parameter_update_diagnostics(
            actor, actor_before, include_layers=diagnostic_update)
        critic_update = parameter_update_diagnostics(
            critic, critic_before, include_layers=diagnostic_update)
        optimizer_state = optimizer_diagnostics(opt) if diagnostic_update else {}
        clipping = gradient_clip_diagnostics(
            actor_gradient_norms, critic_gradient_norms)
        with torch.no_grad():
            critic_after_update = critic_calibration(critic(f_cin), f_ret)
        global_step += T * N
        actor_std = actor.log_std.detach().exp()
        prior_axis_rmse = {key: float(value) for key, value in prior_axis_rmse_t.items()}
        prior_leg_rmse = {key: float(value) for key, value in prior_leg_rmse_t.items()}
        update_record = {
            "step": global_step,
            "pi_loss": pi_l,
            "v_loss": v_l,
            "entropy": ent_l,
            "ent_coef": ent_coef,
            "alpha": alpha,
            "imit_anneal": imit,
            "distill_loss": distill_l,
            "weighted_distill_loss": args.distill_weight * distill_l,
            "action_prior_loss": action_prior_l,
            "action_prior_weight": current_action_prior_weight,
            "weighted_action_prior_loss": current_action_prior_weight * action_prior_l,
            "approx_kl_last_minibatch": approx_kl_l,
            "clip_fraction_last_minibatch": clip_fraction_l,
            "gradient_norm_before_clip": gradient_norm_l,
            "actor_gradient_norm_before_clip": actor_gradient_norm_l,
            "critic_gradient_norm_before_clip": critic_gradient_norm_l,
            "actor_clip_fraction": clipping["actor"]["clipped_fraction"],
            "critic_clip_fraction": clipping["critic"]["clipped_fraction"],
            "full_rollout_approx_kl": trust_region["approx_kl"],
            "full_rollout_clip_fraction": trust_region["clip_fraction"],
            "effective_sample_fraction": trust_region["effective_sample_fraction"],
            "actor_relative_update": actor_update["relative_update"],
            "critic_relative_update": critic_update["relative_update"],
            "critic_bias_before_update": critic_before_update["bias"],
            "critic_bias_after_update": critic_after_update["bias"],
            "critic_normalized_rmse_after_update": critic_after_update["normalized_rmse"],
            "learning_rate": learning_rate_used,
            "learning_rate_next": learning_rate_next,
            "ppo_epochs_completed": epochs_completed,
            "kl_early_stop": kl_early_stop,
            "actor_std": actor_std.cpu().tolist(),
            "actor_std_by_axis": {
                "yaw": float(actor_std[0::3].mean()),
                "pitch": float(actor_std[1::3].mean()),
                "lift": float(actor_std[2::3].mean()),
            },
            "action_prior_rmse_by_axis": prior_axis_rmse,
            "action_prior_rmse_by_leg": prior_leg_rmse,
            "constraint_duals": (env.constraint_duals.detach().cpu().tolist()
                                 if constraint_names else []),
            "competence_duals": (env.competence_duals.detach().cpu().tolist()
                                 if competence_names else []),
            **rollout_diagnostics,
        }
        stats["updates"].append(update_record)

        update_diagnostics = {
            "schema_version": 1,
            "step": global_step,
            "parameter_updates": {"actor": actor_update, "critic": critic_update},
            "optimizer": optimizer_state,
            "gradient_clipping": clipping,
            "trust_region": trust_region,
            "trust_region_by_epoch": epoch_trust_region,
            "kl_controller": {
                "target_kl": args.target_kl,
                "stop_multiplier": args.kl_stop_multiplier,
                "early_stop": kl_early_stop,
                "epochs_requested": args.epochs,
                "epochs_completed": epochs_completed,
                "learning_rate_used": learning_rate_used,
                "learning_rate_next": learning_rate_next,
            },
            "objective_gradients": objective_gradients,
            "adaptive_contracts": {
                "dual_max": float(getattr(env, "adaptive_dual_max", 10.0)),
                "constraints": [
                    {
                        "name": name,
                        "observed": float(constraint_observed[index]),
                        "target": float(getattr(
                            env, "adaptive_constraint_limits", ())[index]),
                        "comparison": "<=",
                        "dual": float(env.constraint_duals[index]),
                    }
                    for index, name in enumerate(constraint_names)
                ],
                "competence": [
                    {
                        "name": name,
                        "observed": float(competence_observed[index]),
                        "target": float(getattr(
                            env, "adaptive_competence_targets", ())[index]),
                        "comparison": ">=",
                        "dual": float(env.competence_duals[index]),
                    }
                    for index, name in enumerate(competence_names)
                ],
            },
            "policy_distribution": {
                "log_std": tensor_stats(actor.log_std),
                "std": tensor_stats(actor.log_std.exp()),
            },
            "losses": {
                "policy": pi_l,
                "value": v_l,
                "entropy": ent_l,
                "entropy_coefficient": ent_coef,
                "retention_distillation": distill_l,
                "retention_distillation_weight": args.distill_weight,
                "action_prior": action_prior_l,
                "action_prior_weight": current_action_prior_weight,
            },
            "critic": {"before_update": critic_before_update,
                       "after_update": critic_after_update,
                       "td_error": tensor_stats(td_error)},
            "observation_normalization": observation_normalization,
            "training_rollout": training_rollout_metrics,
            "integrity": {
                "nonfinite_count": int((~torch.isfinite(f_raw_obs)).sum()
                                       + (~torch.isfinite(f_ret)).sum()
                                       + actor_update["nonfinite_parameter_count"]
                                       + critic_update["nonfinite_parameter_count"]),
                "rollout_samples_expected": B,
                "rollout_samples_used_per_epoch": mb * args.minibatches,
                "rollout_samples_dropped": B - mb * args.minibatches,
            },
            "timing": {
                "rollout_seconds": rollout_finished - update_started,
                "optimization_seconds": optimization_finished - rollout_finished,
                "total_update_seconds": optimization_finished - update_started,
            },
        }
        if dev.type == "cuda":
            update_diagnostics["hardware"] = {
                "memory_allocated_bytes": int(torch.cuda.memory_allocated(dev)),
                "memory_reserved_bytes": int(torch.cuda.memory_reserved(dev)),
                "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
            }
        if diagnostic_update:
            stats["diagnostics"].append(update_diagnostics)

        if global_step >= next_eval or global_step >= args.steps:
            evaluation_started = time.perf_counter()
            if constraint_names:
                eval_env.constraint_duals.copy_(env.constraint_duals)
            if competence_names:
                eval_env.competence_duals.copy_(env.competence_duals)
            metric_keys = tuple(dict.fromkeys((
                "reward", "track", "verr", "align", "speed", "progress",
                "xprogress", "lateral", "duty", "up", "fallrate", "catrate",
                "ladder_step_clock", "ladder_swing_clearance",
                "attack_selected_hit", "attack_wrong_hit", "attack_support",
                "attack_kick_speed", "ladder_goal_hit", "ladder_goal_progress",
                *(metric for metric, _, _ in early_gates),
            )))
            eval_seed_count = max(1, int(getattr(args, "diagnostic_eval_seeds", 3)))
            eval_results = []
            eval_seeds = []
            for seed_index in range(eval_seed_count):
                eval_seed = args.seed + 1000 + 10_007 * seed_index
                eval_seeds.append(eval_seed)
                eval_results.append(evaluate_policy(
                    eval_env, actor, obs_norm, alpha, imit, args.eval_steps,
                    reset_seed=eval_seed))
            m = eval_results[0]
            seed_summary = multi_seed_summary(eval_results, metric_keys)
            seed_summary["seeds"] = eval_seeds

            stochastic_metrics = evaluate_policy(
                eval_env, actor, obs_norm, alpha, imit, args.eval_steps,
                reset_seed=eval_seeds[0], stochastic=True,
                action_seed=args.seed + 2_000_003)
            deterministic_stochastic_gap = scalar_metric_gap(
                m, stochastic_metrics, metric_keys)

            frozen_normalizer = normalizer_from_snapshot(
                previous_eval_norm, env.obs_dim, dev)
            frozen_norm_metrics = evaluate_policy(
                eval_env, actor, frozen_normalizer, alpha, imit, args.eval_steps,
                reset_seed=eval_seeds[0])
            frozen_live_norm_gap = scalar_metric_gap(
                frozen_norm_metrics, m, metric_keys)
            previous_eval_norm = normalization_snapshot(obs_norm)

            train_eval_gap = scalar_metric_gap(
                update_diagnostics.get("training_rollout", {}), m, metric_keys)
            elapsed = time.time() - t_start
            gate_report = gate_diagnostics(early_gates, m)
            trend = evaluation_trends(
                stats["evals"][-1] if stats["evals"] else None, m, global_step)
            env_steps_per_second = (global_step - start_step) / max(elapsed, 1.0e-9)
            print(f"METRIC step={global_step} reward={m['reward']:.3f} track={m['track']:.3f} "
                  f"verr={m['verr']:.3f} align={m['align']:.3f} speed={m['speed']:.3f} "
                  f"progress={m['progress']:.3f} duty={m['duty']:.3f} air={m['air']:.3f} "
                  f"diagsync={m['diagsync']:.3f} alpha={alpha:.2f} entcoef={ent_coef:.4f} "
                  f"catrate={m['catrate']:.6f} xprog={m.get('xprogress', 0.0):.3f} "
                  f"fallrate={m.get('fallrate', 0.0):.5f} "
                  f"lat={m.get('lateral', 0.0):.3f} "
                  f"fwfrac={m.get('forward_speed_fraction', 0.0):.3f} "
                  f"latfwd={m.get('lateral_forward_ratio', 0.0):.2f} "
                  f"xp10={m.get('xprogress_p10', 0.0):.3f} "
                  f"xp90={m.get('xprogress_p90', 0.0):.3f} "
                  f"progema={m.get('progress_ema', 0.0):.3f} "
                  f"catprog={m.get('cat_progress', 0.0):.3f} "
                  f"catduty={m.get('cat_duty', 0.0):.3f} "
                  f"fduty={m.get('foot_duty_ema', 0.0):.3f} "
                  f"catfduty={m.get('cat_foot_duty', 0.0):.3f} "
                  f"hpeak={m.get('hop_peak', 0.0):.3f} "
                  f"hland={m.get('hop_stable_landing', 0.0):.3f} "
                  f"catslip={m.get('cat_slip', 0.0):.6f} "
                  f"catsupp={m.get('cat_support', 0.0):.3f} "
                  f"catbody={m.get('cat_body', 0.0):.3f} "
                  f"atkhit={m.get('attack_selected_hit', 0.0):.3f} "
                  f"atkwrong={m.get('attack_wrong_hit', 0.0):.3f} "
                  f"atksupp={m.get('attack_support', 0.0):.3f} "
                  f"kickspeed={m.get('attack_kick_speed', 0.0):.3f} "
                  f"pose={m.get('ladder_pose_score', 0.0):.3f} "
                  f"heightctl={m.get('ladder_height_score', 0.0):.3f} "
                  f"yawctl={m.get('ladder_yaw_score', 0.0):.3f} "
                  f"headctl={m.get('ladder_heading_score', 0.0):.3f} "
                  f"stopctl={m.get('ladder_stop_score', 0.0):.3f} "
                  f"moveprog={m.get('ladder_move_progress', 0.0):.3f} "
                  f"stepclk={m.get('ladder_step_clock', 0.0):.3f} "
                  f"swingclr={m.get('ladder_swing_clearance', 0.0):.3f} "
                  f"stepact={m.get('ladder_step_action_score', 0.0):.3f} "
                  f"safeprog={m.get('ladder_safe_progress', 0.0):.3f} "
                  f"stanceslip={m.get('ladder_stance_slip_ratio', 0.0):.3f} "
                  f"dual={(float(env.constraint_duals[0]) if constraint_names else 0.0):.3f} "
                  f"cdual={(float(env.competence_duals[0]) if competence_names else 0.0):.3f} "
                  f"aploss={action_prior_l:.3f} apw={current_action_prior_weight:.2f} "
                  f"goalhit={m.get('ladder_goal_hit', 0.0):.3f} "
                  f"goalprog={m.get('ladder_goal_progress', 0.0):.3f} "
                  f"clear={m.get('ladder_obstacle_clearance', 0.0):.3f} "
                  f"approach={m.get('ladder_approach', 0.0):.3f} "
                  f"rodhit={m.get('ladder_rod_hit', 0.0):.3f} "
                  f"taskrew={m.get('ladder_task_reward', 0.0):.3f} "
                  f"prior={m.get('motion_prior', 0.0):.3f} "
                  f"kl={update_record['approx_kl_last_minibatch']:.5f} "
                  f"clipfrac={update_record['clip_fraction_last_minibatch']:.3f} "
                  f"fullkl={update_record['full_rollout_approx_kl']:.5f} "
                  f"fullclip={update_record['full_rollout_clip_fraction']:.3f} "
                  f"ess={update_record['effective_sample_fraction']:.3f} "
                  f"aupd={update_record['actor_relative_update']:.6f} "
                  f"cupd={update_record['critic_relative_update']:.6f} "
                  f"epochs={update_record['ppo_epochs_completed']} "
                  f"klstop={int(update_record['kl_early_stop'])} "
                  f"lrnext={update_record['learning_rate_next']:.2e} "
                  f"ev={update_record['explained_variance_before_update']:.3f} "
                  f"gnorm={update_record['gradient_norm_before_clip']:.2f} "
                  f"agnorm={update_record['actor_gradient_norm_before_clip']:.2f} "
                  f"cgnorm={update_record['critic_gradient_norm_before_clip']:.2f} "
                  f"advstd={update_record['advantage_std']:.2f} "
                  f"worstgate={gate_report['worst_metric'] or 'none'} "
                  f"sps={env_steps_per_second:.0f} ({elapsed:.0f}s)", flush=True)
            replay_steps = min(
                args.eval_steps, max(1, int(getattr(args, "checkpoint_replay_steps", 32))))
            replay_seed = args.seed + 9_000_001
            replay_before = evaluate_policy(
                eval_env, actor, obs_norm, alpha, imit, replay_steps,
                reset_seed=replay_seed, fingerprint=True)
            checkpoint_started = time.perf_counter()
            save_ckpt(ckpt_path, global_step, actor, critic, obs_norm, priv_norm, opt, args,
                      contract=contract, runtime=capture_runtime_state(env))
            checkpoint_hash = sha256_file(ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location=dev, weights_only=False)
            replay_actor = Actor(
                env.obs_dim, env.act_dim, hidden, architecture=args.architecture,
                task_dim=args.actor_task_dim).to(dev)
            replay_norm = RunningNorm(env.obs_dim).to(dev)
            replay_actor.load_state_dict(checkpoint["actor"])
            replay_norm.load_state_dict(checkpoint["obs_norm"])
            replay_actor.eval(); replay_norm.eval()
            replay_after = evaluate_policy(
                eval_env, replay_actor, replay_norm, alpha, imit, replay_steps,
                reset_seed=replay_seed, fingerprint=True)
            replay_differences = {
                key: abs(float(replay_after[key]) - float(replay_before[key]))
                for key in metric_keys
                if isinstance(replay_before.get(key), (int, float))
                and isinstance(replay_after.get(key), (int, float))
            }
            checkpoint_replay = {
                "steps": replay_steps,
                "seed": replay_seed,
                "before_sha256": replay_before.get("trajectory_sha256"),
                "after_sha256": replay_after.get("trajectory_sha256"),
                "fingerprint_match": (replay_before.get("trajectory_sha256")
                                      == replay_after.get("trajectory_sha256")),
                "max_abs_metric_difference": max(replay_differences.values(), default=0.0),
                "metric_differences": replay_differences,
            }
            checkpoint_replay["pass"] = (
                checkpoint_replay["fingerprint_match"]
                or checkpoint_replay["max_abs_metric_difference"] <= 1.0e-6)
            del checkpoint, replay_actor, replay_norm, frozen_normalizer

            update_diagnostics.update({
                "train_eval_gap": train_eval_gap,
                "multi_seed_evaluation": seed_summary,
                "deterministic_stochastic_gap": deterministic_stochastic_gap,
                "frozen_live_normalization_gap": frozen_live_norm_gap,
                "checkpoint_replay": checkpoint_replay,
            })
            update_diagnostics["timing"].update(
                evaluation_seconds=time.perf_counter() - evaluation_started,
                checkpoint_seconds=time.perf_counter() - checkpoint_started,
            )
            update_diagnostics["alerts"] = diagnostic_alerts(
                update_diagnostics, gate_report, m)
            eval_record = {
                "step": global_step,
                **m,
                "wall_seconds": elapsed,
                "env_steps_per_second": env_steps_per_second,
                "learner": update_record,
                "gates": gate_report,
                "trend": trend,
                "checkpoint_sha256": checkpoint_hash,
                "diagnostics": update_diagnostics,
            }
            stats["evals"].append(eval_record)
            write_json_atomic(diagnostics_path, {
                "schema_version": 1,
                "run_id": provenance["run_id"],
                "step": global_step,
                "checkpoint_sha256": checkpoint_hash,
                "evaluation": m,
                "diagnostics": update_diagnostics,
            })
            append_jsonl(metrics_path, {
                "event": "evaluation",
                "run_id": provenance["run_id"],
                **eval_record,
            })
            for alert in update_diagnostics["alerts"]:
                print(f"ALERT severity={alert['severity']} code={alert['code']} "
                      f"message={alert['message']}", flush=True)
            next_eval = min(args.steps, next_eval + eval_interval)
            if early_gates_pass(early_gates, m):
                early_streak += 1
            else:
                early_streak = 0
            if early_streak >= args.early_patience:
                stats["early_stop"] = {
                    "step": global_step, "consecutive_passes": early_streak}
                print(f"EARLY_STOP consecutive_gate_passes={early_streak} "
                      f"step={global_step}", flush=True)
                break
            if (duty_stagnation_tripwire_enabled(args.geometry, args.rung)
                    and args.steps >= 1_000_000 and m["duty"] > 0.98
                    and global_step > 0.5 * args.steps):
                append_jsonl(metrics_path, {
                    "event": "tripwire", "run_id": provenance["run_id"],
                    "step": global_step, "reason": "duty_stagnation",
                    "duty": m["duty"],
                })
                print("TRIPWIRE duty stagnation", flush=True)
                sys.exit(3)

    save_ckpt(ckpt_path, global_step, actor, critic, obs_norm, priv_norm, opt, args,
              contract=contract, runtime=capture_runtime_state(env))
    stats_path = Path(f"{args.tag}.stats.json")
    finished = time.time()
    stats["run"].update(
        finished_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=finished - t_start,
        env_steps_per_second=(global_step - start_step) / max(finished - t_start, 1.0e-9),
        final_step=global_step,
        checkpoint_sha256=sha256_file(ckpt_path),
    )
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    stats["stats"] = str(stats_path)
    append_jsonl(metrics_path, {
        "event": "run_complete", "run_id": provenance["run_id"],
        "step": global_step, "run": stats["run"], "stats": str(stats_path),
    })
    print(f"DONE step={global_step} ckpt={ckpt_path}", flush=True)
    return stats


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", choices=tuple(GEOMETRIES), default="mesh",
                    help=("mesh/walker = commanded locomotion; combat = symmetric fight; "
                          "leg_attack = runtime-selectable FL/FR/RL/RR kick"))
    ap.add_argument("--rung", type=int, default=None,
                    help="task number for a ladder_locomotion/ladder_combat environment")
    ap.add_argument("--steps", type=int, default=20_000_000, help="total env steps")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--episode-length", type=int, default=800)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tag", default="mesh_warp")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-policy", default=None,
                    help="warm-start actor+observation norm only; reset critic and optimizer")
    ap.add_argument("--anchor-policy", default=None,
                    help="accepted pre-rung policy used as a frozen anti-forgetting teacher")
    ap.add_argument("--anchor-task-indices", default="",
                    help="comma-separated task one-hot indices replayed for distillation")
    ap.add_argument("--distill-weight", type=float, default=0.0,
                    help="behavior-distillation weight for old task commands")
    ap.add_argument("--action-prior-weight", type=float, default=0.5,
                    help="behavioral-prior weight for environments that expose one")
    ap.add_argument("--action-prior-json", default=None,
                    help="versioned searched behavior-prior artifact loaded by the environment")
    ap.add_argument("--transfer-policy", default=None,
                    help="frozen legacy policy used as a behavioral transfer teacher")
    ap.add_argument("--transfer-obs-dim", type=int, default=None,
                    help="leading observation dimensions consumed by --transfer-policy")
    ap.add_argument("--opponent", default=None,
                    help="frozen Torch checkpoint for the combat B policy")
    ap.add_argument("--allow-legacy-resume", action="store_true",
                    help="diagnostic only: load a checkpoint with no semantic contract")
    ap.add_argument("--allow-reward-migration", action="store_true",
                    help=("preserve actor/runtime across a reward-only contract change; "
                          "reset critic and optimizer moments"))
    ap.add_argument("--preflight", choices=("strict", "warn", "off"), default="strict",
                    help="derived training-config gate; long launches must use strict")
    ap.add_argument("--evals", type=int, default=20)
    ap.add_argument("--eval-envs", type=int, default=32)
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--diagnostic-eval-seeds", type=int, default=3,
                    help="held-out deterministic seeds summarized at each evaluation")
    ap.add_argument("--checkpoint-replay-steps", type=int, default=32,
                    help="fixed rollout steps compared immediately across save/reload")
    ap.add_argument("--early-gate", action="append", default=[],
                    help="repeatable metric,>=|<=,threshold early-stop condition")
    ap.add_argument("--early-patience", type=int, default=2,
                    help="consecutive deterministic evaluations required to stop early")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--target-kl", type=float, default=0.02,
                    help="whole-rollout PPO KL target; zero disables adaptive control")
    ap.add_argument("--kl-stop-multiplier", type=float, default=1.5,
                    help="stop remaining PPO epochs above this multiple of target KL")
    ap.add_argument("--hidden", default="512,256,128")
    ap.add_argument("--architecture", choices=("mlp", "task_film"), default="mlp",
                    help="task_film = task-conditioned residual actor for multi-skill policies")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    ap.add_argument("--alpha-start", type=float, default=0.0)
    ap.add_argument("--alpha-end", type=float, default=1.0)
    ap.add_argument("--alpha-frac", type=float, default=0.6,
                    help="fraction of --steps over which alpha ramps")
    ap.add_argument("--imit-anneal-frac", type=float, default=0.7,
                    help="imitation weight 1 -> 0 over this fraction of --steps")
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(build_args())
