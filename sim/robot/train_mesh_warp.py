# SPDX-License-Identifier: MIT
"""Torch PPO over the canonical MuJoCo-Warp environments.

PPO with GAE(lambda=0.95, gamma=0.99), clip 0.2, minibatch SGD, running obs
normalization (saved in the checkpoint), and an ASYMMETRIC critic: the actor
sees the 50-obs policy input, the critic sees obs + the env's privileged tensor
(contact/force proxies, qfrc_actuator, true root vel, loop slides). The mlp
actor is a tapering --hidden tanh-gaussian (squashed, with the log-det
correction) with small final-layer init. The FiLM/recurrent families instead
build len(hidden) CONSTANT-WIDTH residual blocks at width hidden[0] — say this
honestly with --width/--blocks; every checkpoint and stats file records the
resolved shapes and true parameter counts. Schedules, all linear in total env
steps:

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
import shutil
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
from ladder_warp_env import (LadderCombatWarpEnv, LadderLocomotionWarpEnv,
                             UniversalCommandWarpEnv,
                             UniversalControlWarpEnv)  # noqa: E402
from codesign_warp_env import DesignEnsembleWarpEnv  # noqa: E402
from predictive_control import (  # noqa: E402
    MorphologyTokenEncoder, RecurrentTrajectoryDecoder,
    TemporalTransformerTrajectoryDecoder,
    TRAJECTORY_RAW_DIM, guided_action_sequence, stabilized_trajectory_target,
    trajectory_calibration_metrics, trajectory_prediction_loss)
from training_diagnostics import (  # noqa: E402
    checkpoint_replay_comparison,
    checkpoint_replay_tolerances,
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
    "universal_control": UniversalControlWarpEnv,
    "universal_command": UniversalCommandWarpEnv,
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
    """Fingerprint the executable Python source even on pods without ``.git``.

    Recursive: nested packages (warplayer/, arena/, ...) implement imported
    physics and must move this hash; a flat glob once let materially different
    runs share identical provenance.
    """
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
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


OPTIONAL_ENV_CAPABILITIES = (
    "policy_action_mask", "policy_mean_prior", "configure_action_prior",
    "adaptive_constraint_names", "adaptive_competence_names",
    "action_prior_competence_metric", "morphology_tokens",
    "trajectory_state", "interaction_target", "set_opponent",
    "external_clock_observation_indices", "gait_loaded",
)


def report_optional_env_capabilities(env) -> dict:
    """Say at startup which optional capabilities are ABSENT.

    The trainer probes ~30 optional env attributes and falls back to plausible
    defaults; without this line, partially disabled features are
    indistinguishable from deliberate configuration.
    """
    present = {name: hasattr(env, name) for name in OPTIONAL_ENV_CAPABILITIES}
    absent = sorted(name for name, available in present.items() if not available)
    print("env capabilities: "
          + (f"absent={absent}" if absent else "all optional present"),
          flush=True)
    return present


def training_provenance(args, env, contract: dict, device: torch.device) -> dict:
    """Capture enough immutable context to reproduce or compare an invocation."""
    input_names = (
        "resume", "init_policy", "anchor_policy", "transfer_policy",
        "action_prior_json", "opponent", "design_bank_json",
        "eval_design_bank_json",
    )
    list_input_names = ("opponent_pool", "replay_artifact")
    hashes = {}
    cache: dict[str, str | None] = {}

    def record(name: str, value) -> None:
        # replay artifacts arrive as "path,pressure" specs; hash the file part
        path_part = str(value).split(",", 1)[0]
        key = str(Path(path_part).resolve())
        if key not in cache:
            cache[key] = sha256_file(path_part)
        hashes[name] = {"path": str(value), "sha256": cache[key]}

    for name in input_names:
        value = getattr(args, name, None)
        if value:
            record(name, value)
    for name in list_input_names:
        for index, value in enumerate(getattr(args, name, None) or ()):
            record(f"{name}[{index}]", value)
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
    def update(self, x: torch.Tensor, *, frozen_suffix: int = 0):
        """Update continuous features while optionally freezing categorical tail.

        Task-FiLM observations end in a one-hot task identity. Running statistics
        are appropriate for physical sensors, but make a never-before-seen task
        channel explode to the normalization clip and then drift throughout the
        first training run. The task tail is therefore frozen once loaded; its
        categorical meaning is handled by the actor's task encoder.
        """
        frozen_suffix = int(frozen_suffix)
        if not 0 <= frozen_suffix <= x.shape[-1]:
            raise ValueError("frozen_suffix must be within the feature dimension")
        mutable = x.shape[-1] - frozen_suffix
        bm = x[:, :mutable].mean(0)
        bv = x[:, :mutable].var(0, unbiased=False)
        bc = float(x.shape[0])
        delta = bm - self.mean[:mutable]
        tot = self.count + bc
        self.mean[:mutable].add_(delta * (bc / tot))
        self.var[:mutable].copy_((self.var[:mutable] * self.count + bv * bc
                                  + delta ** 2 * self.count * bc / tot) / tot)
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
                 architecture: str = "mlp", task_dim: int = 0,
                 prediction_decoder: str = "recurrent"):
        super().__init__()
        self.architecture = architecture
        self.task_dim = int(task_dim)
        if architecture == "mlp":
            self.trunk = _mlp([obs_dim, *hidden])
            output_dim = hidden[-1]
        elif architecture in ("task_film", "task_film_gru", "predictive_token_gru"):
            if not 0 < self.task_dim < obs_dim:
                raise ValueError(f"{architecture} requires 0 < task_dim < obs_dim")
            width = int(hidden[0])
            embed = min(128, width)
            self.feature_in = nn.Linear(obs_dim - self.task_dim, width)
            self.task_encoder = nn.Sequential(
                nn.Linear(self.task_dim, embed), nn.SiLU(), nn.Linear(embed, embed), nn.SiLU())
            if architecture in ("task_film_gru", "predictive_token_gru"):
                self.recurrent_film = nn.Linear(embed, 2 * width)
                self.gru = nn.GRUCell(width, width)
            if architecture == "predictive_token_gru":
                self.morphology_encoder = MorphologyTokenEncoder(width)
                self.morphology_film = nn.Linear(width, 2 * width)
                if prediction_decoder == "recurrent":
                    self.trajectory_decoder = RecurrentTrajectoryDecoder(
                        width, act_dim, width)
                elif prediction_decoder == "transformer":
                    self.trajectory_decoder = TemporalTransformerTrajectoryDecoder(
                        width, act_dim, width)
                else:
                    raise ValueError(f"unknown prediction decoder {prediction_decoder!r}")
                self.prediction_decoder = prediction_decoder
                self.register_buffer("prediction_error_ema", torch.tensor(10.0))
                self.register_buffer("prediction_updates", torch.tensor(0, dtype=torch.long))
                self.register_buffer("prediction_calibration_ema", torch.tensor(10.0))
                self.register_buffer("prediction_calibration_updates",
                                     torch.tensor(0, dtype=torch.long))
                self.register_buffer("prediction_best_calibration",
                                     torch.tensor(float("inf")))
                self.register_buffer("prediction_degraded_streak",
                                     torch.tensor(0, dtype=torch.long))
                self.register_buffer("prediction_frozen", torch.tensor(False))
                self.prediction_freeze_tolerance = 0.15
                self.prediction_freeze_patience = 3
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

    @property
    def is_recurrent(self) -> bool:
        return self.architecture in ("task_film_gru", "predictive_token_gru")

    @property
    def is_predictive(self) -> bool:
        return self.architecture == "predictive_token_gru"

    @property
    def recurrent_state_dim(self) -> int:
        return int(self.feature_in.out_features) if self.is_recurrent else 0

    def initial_state(self, batch: int, *, device=None, dtype=None) -> torch.Tensor | None:
        if not self.is_recurrent:
            return None
        reference = self.mu.weight
        return torch.zeros(
            (int(batch), self.recurrent_state_dim),
            device=reference.device if device is None else device,
            dtype=reference.dtype if dtype is None else dtype)

    def encode_morphology(self, numeric: torch.Tensor, token_types: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        if not self.is_predictive:
            raise ValueError("morphology encoding requires predictive_token_gru")
        return self.morphology_encoder(numeric, token_types, mask)

    def _task_features(self, obs: torch.Tensor, state: torch.Tensor | None = None,
                       morphology: torch.Tensor | None = None
                       ) -> tuple[torch.Tensor, torch.Tensor | None]:
        physical, task = obs[:, :-self.task_dim], obs[:, -self.task_dim:]
        features = torch.nn.functional.silu(self.feature_in(physical))
        embedding = self.task_encoder(task)
        next_state = None
        if self.is_recurrent:
            if state is None:
                state = self.initial_state(len(obs), device=obs.device, dtype=obs.dtype)
            gamma, beta = self.recurrent_film(embedding).chunk(2, dim=-1)
            recurrent_input = (features * (1.0 + 0.25 * torch.tanh(gamma))
                               + 0.25 * beta)
            if self.is_predictive:
                if morphology is None:
                    raise ValueError("predictive_token_gru requires a morphology embedding")
                morph_gamma, morph_beta = self.morphology_film(morphology).chunk(2, dim=-1)
                recurrent_input = (recurrent_input
                                   * (1.0 + 0.25 * torch.tanh(morph_gamma))
                                   + 0.25 * morph_beta)
            next_state = self.gru(recurrent_input, state)
            features = next_state
        for block, modulation in zip(self.blocks, self.film):
            gamma, beta = modulation(embedding).chunk(2, dim=-1)
            conditioned = features * (1.0 + 0.25 * torch.tanh(gamma)) + 0.25 * beta
            features = features + 0.5 * block(conditioned)
        return features, next_state

    def step(self, obs: torch.Tensor, state: torch.Tensor | None = None,
             morphology: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.architecture == "mlp":
            return self.mu(self.trunk(obs)), None
        features, next_state = self._task_features(obs, state, morphology)
        return self.mu(features), next_state

    def sequence(self, obs: torch.Tensor, initial_state: torch.Tensor | None = None,
                 reset_before: torch.Tensor | None = None,
                 morphology: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run a time-major sequence while resetting memory only at real resets."""
        if obs.ndim != 3:
            raise ValueError("recurrent sequence observations must be [time, batch, feature]")
        if not self.is_recurrent:
            flat = self.forward(obs.reshape(-1, obs.shape[-1]))
            return flat.reshape(*obs.shape[:2], -1), None
        state = (self.initial_state(obs.shape[1], device=obs.device, dtype=obs.dtype)
                 if initial_state is None else initial_state)
        if state.shape != (obs.shape[1], self.recurrent_state_dim):
            raise ValueError("initial recurrent state has the wrong shape")
        if reset_before is None:
            reset_before = torch.zeros(obs.shape[:2], dtype=torch.bool, device=obs.device)
        if reset_before.shape != obs.shape[:2]:
            raise ValueError("reset_before must match [time, batch]")
        means = []
        for time_index in range(obs.shape[0]):
            state = state * (~reset_before[time_index].bool()).to(obs.dtype).unsqueeze(-1)
            mean, state = self.step(obs[time_index], state, morphology)
            means.append(mean)
        return torch.stack(means), state

    def sequence_with_states(self, obs: torch.Tensor,
                             initial_state: torch.Tensor | None = None,
                             reset_before: torch.Tensor | None = None,
                             morphology: torch.Tensor | None = None
                             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """As :meth:`sequence`, retaining each post-observation GRU state."""
        if not self.is_recurrent or obs.ndim != 3:
            raise ValueError("sequence_with_states requires recurrent [time,batch,feature]")
        state = (self.initial_state(obs.shape[1], device=obs.device, dtype=obs.dtype)
                 if initial_state is None else initial_state)
        reset_before = (torch.zeros(obs.shape[:2], dtype=torch.bool, device=obs.device)
                        if reset_before is None else reset_before)
        means, states = [], []
        for time_index in range(obs.shape[0]):
            state = state * (~reset_before[time_index].bool()).to(obs.dtype).unsqueeze(-1)
            mean, state = self.step(obs[time_index], state, morphology)
            means.append(mean); states.append(state)
        return torch.stack(means), torch.stack(states), state

    def predict_trajectory(self, state: torch.Tensor, morphology: torch.Tensor,
                           future_actions: torch.Tensor) -> torch.Tensor:
        if not self.is_predictive:
            raise ValueError("trajectory decoding requires predictive_token_gru")
        return self.trajectory_decoder(state, morphology, future_actions)

    @torch.no_grad()
    def observe_prediction_error(self, value: torch.Tensor) -> None:
        if not self.is_predictive or not torch.isfinite(value):
            return
        rate = 1.0 if int(self.prediction_updates) == 0 else 0.05
        self.prediction_error_ema.lerp_(value.detach(), rate)
        self.prediction_updates.add_(1)

    @torch.no_grad()
    def observe_prediction_calibration(self, value: torch.Tensor) -> None:
        """Update authority only from the rollout anchor excluded from training.

        The same held-out signal drives the degradation freeze: once the
        calibration EMA has risen a tolerance fraction above its own best for
        `prediction_freeze_patience` consecutive observations, decoder training
        stops.  Calibration keeps being observed while frozen, so training
        resumes automatically if the forecast recovers.  A late-run predictor
        can therefore not keep optimizing its training objective while its
        out-of-sample forecast is getting worse.
        """
        if not self.is_predictive or not torch.isfinite(value):
            return
        rate = 1.0 if int(self.prediction_calibration_updates) == 0 else 0.10
        self.prediction_calibration_ema.lerp_(value.detach(), rate)
        self.prediction_calibration_updates.add_(1)
        if (self.prediction_freeze_tolerance <= 0.0
                or int(self.prediction_calibration_updates) < 10):
            return  # same warmup as authority: too few held-out forecasts
        ema = float(self.prediction_calibration_ema)
        if ema < float(self.prediction_best_calibration):
            self.prediction_best_calibration.fill_(ema)
        best = float(self.prediction_best_calibration)
        if ema > best * (1.0 + self.prediction_freeze_tolerance):
            self.prediction_degraded_streak.add_(1)
            if int(self.prediction_degraded_streak) >= self.prediction_freeze_patience:
                self.prediction_frozen.fill_(True)
        else:
            self.prediction_degraded_streak.zero_()
            if (bool(self.prediction_frozen)
                    and ema <= best * (1.0 + 0.5 * self.prediction_freeze_tolerance)):
                self.prediction_frozen.fill_(False)

    @property
    def prediction_training_enabled(self) -> bool:
        """False while held-out calibration degradation has frozen the decoder."""
        return self.is_predictive and not bool(self.prediction_frozen)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Tolerate checkpoints that predate the calibration-freeze buffers."""
        if self.is_predictive:
            freeze_buffers = ("prediction_best_calibration",
                              "prediction_degraded_streak", "prediction_frozen")
            missing = [name for name in freeze_buffers if name not in state_dict]
            if missing:
                state_dict = dict(state_dict)
                for name in missing:
                    state_dict[name] = getattr(self, name).detach().clone()
        return super().load_state_dict(state_dict, strict=strict)

    @property
    def guidance_confidence(self) -> float:
        """Continuous self-tuning authority from dimensionless held-out error."""
        if not self.is_predictive or int(self.prediction_calibration_updates) < 10:
            return 0.0
        return float(torch.exp(
            -4.0 * self.prediction_calibration_ema.clamp(min=0.0, max=5.0)))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.architecture == "mlp":
            return self.mu(self.trunk(obs))
        return self.step(obs, None)[0]


@torch.no_grad()
def inherit_task_conditioning(actor: Actor, normalizer: RunningNorm,
                              source_index: int, target_index: int) -> float:
    """Make a new task channel reproduce its predecessor at initialization.

    Only the target column of the first task-encoder layer is changed. Existing
    task behavior is untouched, and the target remains independently trainable.
    The adjustment is solved in the actor's actual normalized/clipped input
    space, so it also repairs legacy checkpoints whose unseen one-hot channels
    normalize to the clip ceiling.
    """
    if actor.architecture not in ("task_film", "task_film_gru", "predictive_token_gru"):
        raise ValueError("task inheritance requires a task_film actor")
    task_dim = actor.task_dim
    if not (0 <= source_index < task_dim and 0 <= target_index < task_dim):
        raise ValueError("task inheritance index is outside the task dimension")
    if source_index == target_index:
        return 0.0
    mean = normalizer.mean[-task_dim:]
    std = torch.sqrt(normalizer.var[-task_dim:] + 1.0e-8)
    source = torch.zeros(task_dim, device=mean.device, dtype=mean.dtype)
    target = torch.zeros_like(source)
    source[source_index] = 1.0
    target[target_index] = 1.0
    source_z = ((source - mean) / std).clamp(-normalizer.clip, normalizer.clip)
    target_z = ((target - mean) / std).clamp(-normalizer.clip, normalizer.clip)
    layer = actor.task_encoder[0]
    weight = layer.weight
    desired_delta = weight @ (source_z - target_z)
    leverage = target_z[target_index] - source_z[target_index]
    if float(leverage.abs()) < 1.0e-6:
        raise ValueError("target task channel has no distinct normalized leverage")
    weight[:, target_index].add_(desired_delta / leverage)
    residual = (weight @ target_z - weight @ source_z).abs().max()
    return float(residual)


def parse_index_spec(value) -> tuple[int, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return tuple(int(item) for item in value.split(",") if item.strip())
    return tuple(int(item) for item in value)


def policy_observation(raw_obs: torch.Tensor, clock_indices=(), excluded_indices=()
                       ) -> torch.Tensor:
    """Mask explicit scaffolds while retaining physical state and history."""
    indices = tuple(dict.fromkeys(
        (*parse_index_spec(clock_indices), *parse_index_spec(excluded_indices))))
    if not indices:
        return raw_obs
    if min(indices) < 0 or max(indices) >= raw_obs.shape[-1]:
        raise ValueError("policy clock index is outside the observation")
    transformed = raw_obs.clone()
    transformed[..., list(indices)] = 0.0
    return transformed


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden):
        super().__init__()
        self.net = nn.Sequential(_mlp([in_dim, *hidden]), nn.Linear(hidden[-1], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def logp_tanh(z: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor,
              action_mask: torch.Tensor | None = None) -> torch.Tensor:
    """log pi(tanh(z)) for the squashed gaussian; z is the PRE-tanh sample."""
    std = log_std.exp()
    terms = (-0.5 * ((z - mu) / std) ** 2 - log_std - 0.5 * LOG2PI
             - torch.log(1.0 - torch.tanh(z) ** 2 + 1e-6))
    if action_mask is not None:
        terms = terms * action_mask
    return terms.sum(-1)


def entropy_tanh(mu: torch.Tensor, log_std: torch.Tensor,
                 action_mask: torch.Tensor | None = None) -> torch.Tensor:
    """One-sample estimate of H[tanh(Z)] = H[Z] + E[log|dtanh/dz|]."""
    z = mu + log_std.exp() * torch.randn_like(mu)
    terms = (log_std + 0.5 * (1.0 + LOG2PI)
             + torch.log(1.0 - torch.tanh(z) ** 2 + 1e-6))
    if action_mask is not None:
        terms = terms * action_mask
    return terms.sum(-1)


def partition_policy_and_predictor_parameters(
        actor: nn.Module) -> tuple[list, list]:
    """Split actor parameters into the policy subspace and the decoder subspace.

    The trajectory decoder receives gradient only from the auxiliary prediction
    loss, so it forms its own optimizer/clipping subspace: a large decoder (the
    temporal Transformer is ~2.5x the recurrent one) must not shrink the policy
    gradient through a shared clip, nor inherit the adaptive PPO learning rate.
    """
    if not getattr(actor, "is_predictive", False):
        return list(actor.parameters()), []
    predictor = list(actor.trajectory_decoder.parameters())
    predictor_ids = {id(parameter) for parameter in predictor}
    policy = [parameter for parameter in actor.parameters()
              if id(parameter) not in predictor_ids]
    return policy, predictor


def clip_actor_critic_gradients(actor: nn.Module, critic: nn.Module,
                                max_norm: float = 1.0
                                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clip disjoint policy/value/predictor gradients without cross scaling."""
    policy, predictor = partition_policy_and_predictor_parameters(actor)
    actor_norm = nn.utils.clip_grad_norm_(policy, max_norm)
    critic_norm = nn.utils.clip_grad_norm_(critic.parameters(), max_norm)
    predictor_norm = (nn.utils.clip_grad_norm_(predictor, max_norm)
                      if predictor else torch.zeros(()))
    return actor_norm, critic_norm, predictor_norm


def scale_invariant_value_loss(prediction: torch.Tensor,
                               target: torch.Tensor,
                               target_scale: torch.Tensor) -> torch.Tensor:
    """Value MSE in units of one rollout return standard deviation.

    The critic optimum is unchanged, but raw reward units can no longer make
    every critic minibatch hit the same global gradient clip.  Persistent
    clipping turns a safety bound into implicit gradient normalization and hides
    genuine spikes; scale-aware loss restores clipping's intended meaning.
    """
    scale = target_scale.detach().clamp_min(1.0)
    return 0.5 * (((prediction - target) / scale) ** 2).mean()


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
    "_prev_dist", "_prev_dealt", "_vel_ema", "_combat_time",
    "_prev_dist_b", "_prev_dealt_b", "_vel_ema_b", "_combat_time_b",
    "_qpos0", "_attack_leg", "_attack_active", "_attack_override_leg",
    "_attack_override_active", "_attack_timer", "_attack_phase_step",
    "_prev_extension", "_prev_support",
    "_pose_command", "_height_command", "_goal", "_heading_command",
    "_velocity_command", "_route_index", "_task_t", "_constraint_age",
    "_disturbance_impulse", "_disturbance_fire",
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
    # Ladder environments own stricter cycle-level duty contracts themselves;
    # the generic tripwire would falsely kill static skills and combat tasks.
    return geometry not in ("ladder_locomotion", "universal_control",
                            "universal_command")


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


def capture_runtime_state(env, *, schedule_progress: float | None = None,
                          policy_state: torch.Tensor | None = None) -> dict:
    out = {"env": capture_env_state(env), "torch_rng": torch.get_rng_state()}
    if torch.cuda.is_available():
        out["cuda_rng"] = torch.cuda.get_rng_state_all()
    if schedule_progress is not None:
        out["schedule_progress"] = min(max(float(schedule_progress), 0.0), 1.0)
    if policy_state is not None:
        out["policy_state"] = policy_state.detach().cpu().clone()
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
        conditioning = ("command_film_v2"
                        if getattr(env, "task_conditioning_kind", "rung_id")
                        == "command" else "frozen_norm+previous_exact_v1")
        contract.update(actor_architecture=architecture,
                        actor_task_dim=int(getattr(env, "architecture_task_dim", 0)),
                        task_conditioning_semantics=conditioning)
    if architecture in ("task_film_gru", "predictive_token_gru"):
        contract.update(
            recurrent_semantics="gru_state_reset_on_physical_done_v1",
            recurrent_clock_indices=parse_index_spec(
                getattr(args, "recurrent_clock_indices", "")))
    if architecture == "predictive_token_gru":
        contract.update(
            morphology_semantics="typed_model_body_joint_actuator_tokens_v1",
            predictive_semantics=(
                "local_world_interaction_trajectory+heldout_calibrated_guidance_v2"),
            prediction_decoder=str(getattr(args, "prediction_decoder", "recurrent")),
            predictive_horizon=int(getattr(args, "prediction_horizon", 32)))
        morphology_indices = parse_index_spec(
            getattr(args, "morphology_observation_indices", ""))
        if morphology_indices:
            contract["morphology_observation_indices"] = morphology_indices
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
    if (getattr(args, "architecture", "mlp") in ("task_film_gru", "predictive_token_gru")
            and int(args.envs) % int(args.minibatches)):
        raise ValueError(
            f"recurrent PPO requires {args.envs} environments to be divisible by "
            f"{args.minibatches} minibatches so complete sequences are retained")
    if getattr(args, "architecture", "mlp") == "predictive_token_gru":
        for name in ("morphology_tokens", "morphology_token_types",
                     "morphology_token_mask", "trajectory_state",
                     "interaction_target"):
            if not hasattr(env, name):
                raise ValueError(f"predictive_token_gru requires env.{name}")
        if not 1 <= int(args.prediction_horizon) < int(args.horizon):
            raise ValueError("--prediction-horizon must be in [1, horizon-1]")
        if int(args.guidance_horizon) < 1:
            raise ValueError("--guidance-horizon must be positive")
        if int(args.prediction_freeze_patience) < 1:
            raise ValueError("--prediction-freeze-patience must be positive")
        if args.prediction_lr is not None and float(args.prediction_lr) <= 0.0:
            raise ValueError("--prediction-lr must be positive")
    if getattr(args, "plateau_slack", 0.0) > 0.0:
        if int(getattr(args, "plateau_min_evals", 2)) < 2:
            raise ValueError("--plateau-min-evals must be at least 2")
        if int(getattr(args, "plateau_patience", 1)) < 1:
            raise ValueError("--plateau-patience must be positive")
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
def resolved_architecture(actor: nn.Module, critic: nn.Module) -> dict:
    """True constructed shapes and parameter counts — never trust a flag string.

    The FiLM branch reinterprets --hidden as a block count at constant width,
    so any size comparison must come from this record, which is written into
    every checkpoint and stats file.
    """
    def summary(module: nn.Module) -> dict:
        return {
            "parameters": int(sum(p.numel() for p in module.parameters())),
            "shapes": {name: list(parameter.shape)
                       for name, parameter in module.named_parameters()},
        }

    record = {"actor": summary(actor), "critic": summary(critic)}
    if getattr(actor, "is_predictive", False):
        record["trajectory_decoder"] = summary(actor.trajectory_decoder)
        record["morphology_encoder"] = summary(actor.morphology_encoder)
    return record


def save_ckpt(path: Path, step: int, actor, critic, obs_norm, priv_norm, opt, args,
              *, contract: dict, runtime: dict | None = None, prediction_opt=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "actor": actor.state_dict(), "critic": critic.state_dict(),
                "obs_norm": obs_norm.state_dict(), "priv_norm": priv_norm.state_dict(),
                "opt": opt.state_dict(), "args": vars(args), "contract": contract,
                "prediction_opt": (prediction_opt.state_dict()
                                   if prediction_opt is not None else None),
                "resolved_architecture": resolved_architecture(actor, critic),
                "runtime": runtime}, path)


def load_ckpt(path, actor, critic, obs_norm, priv_norm, opt, device, *,
              expected_contract: dict, allow_legacy: bool = False,
              allow_reward_migration: bool = False,
              prediction_opt=None) -> tuple[int, dict | None]:
    ck = torch.load(path, map_location=device, weights_only=True)
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
              "critic, optimizer, and reward-semantic competence state reset",
              flush=True)
    else:
        critic.load_state_dict(ck["critic"])
        if opt is not None and ck.get("opt") is not None:
            try:
                opt.load_state_dict(ck["opt"])
            except ValueError as error:
                raise ValueError(
                    f"checkpoint {path} optimizer state does not match the "
                    "current parameter partition: the trajectory decoder now "
                    "trains under its own optimizer. Restart this experimental "
                    "predictive run, or warm-start weights via --init-policy"
                ) from error
        if prediction_opt is not None:
            if ck.get("prediction_opt") is not None:
                prediction_opt.load_state_dict(ck["prediction_opt"])
            else:
                print("checkpoint predates the separate predictor optimizer; "
                      "decoder Adam moments start fresh", flush=True)
    runtime = dict(ck.get("runtime") or {})
    if reward_migration:
        # A competence multiplier is part of the reward interpretation, not the
        # physical trajectory. Carrying a dual that meant "match gait phase"
        # into an objective where it means "use every foot" would smuggle the
        # retired specification through an otherwise clean migration. Physical
        # state, RNG, safety duals, and annealing progress remain intact.
        def clear_competence(state: dict) -> None:
            for member in state.get("ensemble", ()):
                clear_competence(member)
            tensors = state.get("tensors", {})
            tensors.pop("_competence_duals", None)
            tensors.pop("_competence_error_square", None)

        env_state = runtime.get("env")
        if isinstance(env_state, dict):
            clear_competence(env_state)
    # Before schedule_progress was persisted explicitly, every run annealed
    # against the target stored in its own checkpoint.  Recover that completed
    # fraction so extending a resumed run cannot make entropy/imitation younger.
    saved_steps = (ck.get("args") or {}).get("steps")
    if saved_steps:
        previous_progress = min(int(ck["step"]) / max(int(saved_steps), 1), 1.0)
        runtime["schedule_progress"] = max(
            float(runtime.get("schedule_progress", 0.0)), previous_progress)
    return int(ck["step"]), runtime


def _load_actor_normalizer_compatible(ck: dict, actor: Actor,
                                      normalizer: RunningNorm) -> bool:
    """Load exactly, or widen an older task-FiLM policy into a universal actor."""
    try:
        actor.load_state_dict(ck["actor"], strict=True)
        normalizer.load_state_dict(ck["obs_norm"], strict=True)
        return False
    except RuntimeError as exact_error:
        args = ck.get("args", {})
        if (actor.architecture not in ("task_film", "task_film_gru")
                or args.get("architecture") != actor.architecture):
            raise exact_error
        old_task_dim = int(args.get("actor_task_dim", 0))
        if not 0 < old_task_dim <= actor.task_dim:
            raise exact_error
        old_state = ck["actor"]
        widened = actor.state_dict()
        permitted = {"feature_in.weight", "task_encoder.0.weight",
                     "mu.weight", "mu.bias", "log_std"}
        for name, old_value in old_state.items():
            if name not in widened:
                raise exact_error
            target = widened[name]
            if target.shape == old_value.shape:
                target.copy_(old_value)
                continue
            if name not in permitted or target.ndim != old_value.ndim:
                raise exact_error
            if name in ("mu.bias", "log_std"):
                if target.shape[0] < old_value.shape[0]:
                    raise exact_error
                target[:old_value.shape[0]].copy_(old_value)
            elif name == "mu.weight":
                if target.shape[1] != old_value.shape[1] \
                        or target.shape[0] < old_value.shape[0]:
                    raise exact_error
                target[:old_value.shape[0]].copy_(old_value)
            else:
                if target.shape[0] != old_value.shape[0] \
                        or target.shape[1] < old_value.shape[1]:
                    raise exact_error
                target[:, :old_value.shape[1]].copy_(old_value)
        actor.load_state_dict(widened)

        old_norm = ck["obs_norm"]
        old_dim = int(old_norm["mean"].numel())
        new_dim = int(normalizer.mean.numel())
        old_physical = old_dim - old_task_dim
        new_physical = new_dim - actor.task_dim
        if old_physical > new_physical or old_task_dim > actor.task_dim:
            raise exact_error
        normalizer.mean.zero_(); normalizer.var.fill_(1.0)
        normalizer.mean[:old_physical].copy_(old_norm["mean"][:old_physical])
        normalizer.var[:old_physical].copy_(old_norm["var"][:old_physical])
        normalizer.mean[new_physical:new_physical + old_task_dim].copy_(
            old_norm["mean"][old_physical:])
        normalizer.var[new_physical:new_physical + old_task_dim].copy_(
            old_norm["var"][old_physical:])
        normalizer.count.copy_(old_norm["count"])
        return True


def expected_load_semantics(env) -> dict:
    """The input-interpretation contract a checkpoint must share to act here.

    Only observation/conditioning semantics are enforced: how a policy's
    inputs are MEANT is not evaluable behavior. Action/reward semantics stay
    unenforced on load so re-proof runs can deliberately evaluate an accepted
    checkpoint under changed physics.
    """
    return {
        "observation_semantics": getattr(env, "observation_semantics", None),
        "task_conditioning_semantics": (
            "command_film_v2"
            if getattr(env, "task_conditioning_kind", "rung_id") == "command"
            else "frozen_norm+previous_exact_v1"),
    }


def require_compatible_checkpoint(ck: dict, path, expected: dict | None,
                                  *, role: str) -> None:
    """Refuse to reinterpret a checkpoint trained under different semantics.

    Shape equality is NOT identity: the v1 task-ID and v2 command contracts
    share every tensor shape by design, so a shape-only load would silently
    misread the conditioning channels. Legacy checkpoints without a stored
    contract pass (nothing to compare); declared mismatches are errors.
    """
    if not expected:
        return
    got = ck.get("contract") or {}
    mismatch = {
        key: {"checkpoint": got.get(key), "environment": want}
        for key, want in expected.items()
        if want is not None and got.get(key) is not None and got.get(key) != want
    }
    if mismatch:
        raise ValueError(
            f"{role} checkpoint {path} was trained under different input "
            f"semantics: {mismatch}. Matching tensor shapes do not make "
            "contracts interchangeable.")


def load_policy(path, obs_dim: int, act_dim: int, device,
                task_dim: int | None = None, morphology_source=None,
                expected_semantics: dict | None = None):
    """Load a deterministic actor and its observation normalizer."""
    ck = torch.load(path, map_location=device, weights_only=True)
    require_compatible_checkpoint(ck, path, expected_semantics, role="policy")
    hidden = tuple(int(v) for v in ck.get("args", {}).get("hidden", "512,256,128").split(","))
    architecture = ck.get("args", {}).get("architecture", "mlp")
    prediction_decoder = ck.get("args", {}).get("prediction_decoder", "recurrent")
    task_dim = (int(ck.get("args", {}).get("actor_task_dim", 0))
                if task_dim is None else int(task_dim))
    actor = Actor(obs_dim, act_dim, hidden, architecture=architecture,
                  task_dim=task_dim,
                  prediction_decoder=prediction_decoder).to(device)
    actor.guidance_horizon = int(ck.get("args", {}).get("guidance_horizon", 16))
    actor.guidance_steps = int(ck.get("args", {}).get("guidance_steps", 2))
    actor.guidance_interval = int(ck.get("args", {}).get("guidance_interval", 4))
    norm = RunningNorm(obs_dim).to(device)
    _load_actor_normalizer_compatible(ck, actor, norm)
    actor.eval()
    norm.eval()
    clock_indices = parse_index_spec(
        ck.get("args", {}).get("recurrent_clock_indices", ""))
    excluded_indices = parse_index_spec(
        ck.get("args", {}).get("morphology_observation_indices", ""))

    class LoadedPolicy:
        def __init__(self):
            self.state = None
            self.morphology = None
            if morphology_source is not None:
                self.bind_morphology(morphology_source)

        @torch.no_grad()
        def bind_morphology(self, source) -> None:
            if not actor.is_predictive:
                return
            self.morphology = actor.encode_morphology(
                source.morphology_tokens, source.morphology_token_types,
                source.morphology_token_mask)

        @property
        def is_recurrent(self) -> bool:
            return actor.is_recurrent

        def preprocess(self, obs: torch.Tensor) -> torch.Tensor:
            return policy_observation(obs, clock_indices, excluded_indices)

        @torch.no_grad()
        def __call__(self, obs: torch.Tensor) -> torch.Tensor:
            actor_obs = self.preprocess(obs)
            if actor.is_recurrent:
                if self.state is None or self.state.shape[0] != len(obs):
                    self.state = actor.initial_state(len(obs))
                morphology = self.morphology
                if actor.is_predictive:
                    if morphology is None:
                        raise ValueError("predictive policy must be bound to an environment morphology")
                    if len(morphology) != len(obs):
                        morphology = morphology[:1].expand(len(obs), -1)
                mean, self.state = actor.step(norm(actor_obs), self.state, morphology)
            else:
                mean = actor(norm(actor_obs))
            return torch.tanh(mean)

        @torch.no_grad()
        def reset(self, mask: torch.Tensor | None = None) -> None:
            if not actor.is_recurrent:
                return
            if mask is None:
                self.state = None
            elif self.state is not None:
                self.state.mul_(~mask.bool().unsqueeze(-1))

    return LoadedPolicy()


def initialize_policy(path, actor, obs_norm, device,
                      expected_semantics: dict | None = None) -> int:
    """Warm-start actor + actor normalization, leaving critic/optimizer fresh.

    A ladder transition changes the reward contract, so resuming the optimizer
    would incorrectly retain the previous value target and Adam moments.  Exact
    tensor shapes are required; cross-family transitions start a new policy.
    """
    ck = torch.load(path, map_location=device, weights_only=True)
    require_compatible_checkpoint(ck, path, expected_semantics, role="init policy")
    try:
        widened = _load_actor_normalizer_compatible(ck, actor, obs_norm)
    except RuntimeError as error:
        raise ValueError(
            f"initial policy {path} has a different observation/action architecture; "
            "warm starts are only valid inside one ladder family") from error
    if widened:
        print("widened prior task-FiLM policy into universal controller contract",
              flush=True)
    return int(ck.get("step", 0))


def frozen_anchor_policy(path, obs_dim: int, act_dim: int, hidden,
                         architecture: str, task_dim: int, device,
                         prediction_decoder: str = "recurrent",
                         expected_semantics: dict | None = None):
    """Load the accepted pre-rung actor/norm as a no-gradient teacher."""
    ck = torch.load(path, map_location=device, weights_only=True)
    require_compatible_checkpoint(ck, path, expected_semantics, role="anchor")
    actor = Actor(obs_dim, act_dim, hidden, architecture=architecture,
                  task_dim=task_dim,
                  prediction_decoder=prediction_decoder).to(device)
    norm = RunningNorm(obs_dim).to(device)
    try:
        _load_actor_normalizer_compatible(ck, actor, norm)
    except RuntimeError as error:
        raise ValueError(f"anchor policy {path} is not architecture-compatible") from error
    actor.eval(); norm.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor, norm


def schedule_progress(step: int, args, progress_floor: float = 0.0) -> float:
    """Monotonic annealing progress across extensions of a resumed run."""
    return min(max(float(progress_floor), step / max(args.steps, 1)), 1.0)


def schedules(step: int, args, progress_floor: float = 0.0) -> tuple[float, float, float]:
    """(ent_coef, alpha, imit_anneal) at env-step `step` — all linear."""
    p = schedule_progress(step, args, progress_floor)
    ent = ENT_START + (ENT_END - ENT_START) * p
    # A plateau-intervention retry may reinject exploration.  The boost scales
    # the scheduled coefficient but never exceeds the from-scratch start value:
    # this is a partial rewind of the anneal, not a new exploration regime.
    ent = min(ENT_START, ent * max(float(getattr(args, "entropy_boost", 1.0)), 1.0))
    # Acquisition runs keep an exploration floor: annealing to ENT_END before
    # the skill exists starves the stochastic discovery the task requires.
    ent = min(ENT_START, max(ent, float(getattr(args, "entropy_floor", ENT_END))))
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
                        competence_pressure: float = 1.0,
                        progress_fraction: float | None = None) -> float:
    """Anneal a behavior prior while automatically yielding to safety pressure.

    Purely time-based annealing can discard a useful scaffold even though its
    demonstrated competence has not yet transferred.  The normalized competence
    shortfall therefore supplies a second schedule: prior influence decays only
    when either the usual annealing is incomplete or the target has actually been
    learned.  Safety pressure remains the denominator, so an unsafe teacher still
    yields automatically.  This is a dimensionless arbitration between measured
    contracts rather than another rung-specific reward coefficient.
    """
    if progress_fraction is None:
        progress_fraction = float(progress) / max(total, 1)
    time_schedule = max(float(floor), 1.0 - float(progress_fraction) / 0.60)
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


def load_replay_artifacts(specs: list[str], obs_dim: int, act_dim: int,
                          expected_observation_semantics: str | None = None) -> list[dict]:
    """Load real old-task state/action samples with an adaptive pressure.

    Each CLI value is ``PATH[,PRESSURE]``.  Keeping the pressure outside the
    tensor artifact lets the ladder raise or relax replay without rebuilding
    expensive physics rollouts.
    """
    bank = []
    for spec in specs:
        path_text, separator, pressure_text = spec.rpartition(",")
        if separator:
            try:
                pressure = float(pressure_text)
                path = Path(path_text)
            except ValueError:
                pressure, path = 1.0, Path(spec)
        else:
            pressure, path = 1.0, Path(spec)
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        stamped = artifact.get("observation_semantics")
        if (expected_observation_semantics is not None and stamped is not None
                and stamped != expected_observation_semantics):
            # Matching widths are not matching meanings: a v1 task-ID replay
            # distilled into a v2 command actor would teach the wrong channels.
            raise ValueError(
                f"replay artifact {path} was collected under "
                f"{stamped!r}, not {expected_observation_semantics!r}")
        observations = artifact.get("observations")
        actions = artifact.get("actions")
        if not isinstance(observations, torch.Tensor) or not isinstance(actions, torch.Tensor):
            raise ValueError(f"replay artifact {path} lacks observation/action tensors")
        if observations.ndim not in (2, 3) or observations.shape[-1] != obs_dim:
            raise ValueError(
                f"replay artifact {path} observation shape {tuple(observations.shape)} "
                f"does not match (*,{obs_dim}) or (time,batch,{obs_dim})")
        expected_action_shape = (*observations.shape[:-1], act_dim)
        if actions.shape != expected_action_shape:
            raise ValueError(
                f"replay artifact {path} action shape {tuple(actions.shape)} "
                f"does not match {expected_action_shape}")
        dones = artifact.get("dones")
        if observations.ndim == 3:
            if dones is None:
                dones = torch.zeros(observations.shape[:2], dtype=torch.bool)
            if not isinstance(dones, torch.Tensor) or dones.shape != observations.shape[:2]:
                raise ValueError(
                    f"replay artifact {path} dones must match {observations.shape[:2]}")
        bank.append({
            "path": str(path),
            "rung": int(artifact.get("rung", -1)),
            "pressure": min(max(float(pressure), 0.1), 10.0),
            "observations": observations.float().contiguous(),
            "actions": actions.float().contiguous(),
            "dones": None if dones is None else dones.bool().contiguous(),
        })
    return bank


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


def gate_margin_projection(history: list[tuple[int, float]], total_steps: int,
                           *, slack: float, window: int) -> dict:
    """Extrapolate the worst gate margin and judge whether this attempt is doomed.

    A least-squares line over the last `window` (step, worst_relative_margin)
    evaluations projects the step at which the margin would cross zero.  The
    attempt earns a strike when it is still failing and the projected crossing
    lies beyond `slack` times the remaining budget — including a flat or
    negative slope, which never crosses.  One strike is never a verdict: RL
    margins are noisy and nonmonotonic, so the caller requires consecutive
    strikes before aborting.  A passing margin never strikes; the early-gate
    stop owns that case.
    """
    result = {"strike": False, "slope_per_step": None, "projected_crossing_step": None,
              "observations": len(history), "window": int(window)}
    if len(history) < max(2, int(window)):
        return result
    recent = history[-int(window):]
    last_step, last_margin = recent[-1]
    if last_margin >= 0.0:
        return result
    steps = [float(step) for step, _ in recent]
    margins = [float(margin) for _, margin in recent]
    mean_step = sum(steps) / len(steps)
    mean_margin = sum(margins) / len(margins)
    variance = sum((step - mean_step) ** 2 for step in steps)
    if variance <= 0.0:
        return result
    slope = sum((step - mean_step) * (margin - mean_margin)
                for step, margin in zip(steps, margins)) / variance
    result["slope_per_step"] = slope
    remaining = max(float(total_steps) - last_step, 0.0)
    if slope <= 0.0:
        result["strike"] = True
        return result
    crossing = last_step + (0.0 - last_margin) / slope
    result["projected_crossing_step"] = crossing
    result["strike"] = crossing > last_step + slack * remaining
    return result


def robust_gate_diagnostics(gates: tuple[tuple[str, str, float], ...],
                            seed_summary: dict) -> dict:
    """Apply each contract to its adverse value over all diagnostic seeds."""
    adverse = {}
    summaries = seed_summary.get("metrics", {})
    for metric, comparison, _ in gates:
        values = summaries.get(metric, {}).get("values", [])
        finite = [float(value) for value in values
                  if isinstance(value, (int, float)) and math.isfinite(float(value))]
        if finite:
            adverse[metric] = min(finite) if comparison == ">=" else max(finite)
    report = gate_diagnostics(gates, adverse)
    report["adverse_metrics"] = adverse
    return report


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
        "ladder_foot_activity", "ladder_step_clock", "ladder_swing_clearance",
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
                       old_logp: torch.Tensor,
                       action_mask: torch.Tensor | None = None,
                       mean_override: torch.Tensor | None = None) -> dict:
    """Lightweight whole-rollout KL used by the online epoch controller."""
    mean = actor(observations) if mean_override is None else mean_override
    new_logp = logp_tanh(sampled_pre_tanh, mean, actor.log_std, action_mask)
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
                    fingerprint: bool = False,
                    guidance_enabled: bool = True) -> dict:
    env._gen.manual_seed(reset_seed)
    obs = env.reset()
    tel = EvalTelemetry(env.device)
    action_generator = torch.Generator(device=env.device)
    action_generator.manual_seed(
        int(action_seed if action_seed is not None else reset_seed + 1_000_003))
    digest = hashlib.sha256() if fingerprint else None
    policy_state = actor.initial_state(env.nworld) if actor.is_recurrent else None
    morphology = (actor.encode_morphology(
        env.morphology_tokens, env.morphology_token_types,
        env.morphology_token_mask) if actor.is_predictive else None)
    guidance_cost_sums: dict[str, float] = {}
    guidance_calls = 0
    guidance_realized_sums: dict[str, float] = {}
    clock_indices = getattr(actor, "policy_clock_indices", ())
    excluded_indices = getattr(actor, "policy_excluded_indices", ())
    # Measure held-out forecast quality on the evaluation environment itself.
    # When the eval env carries designs the training env never compiled, this
    # is predictor calibration on unseen morphology tokens — the quantity the
    # decoder ablation is judged on.
    calibration_horizon = 0
    if actor.is_predictive and hasattr(env, "trajectory_state"):
        calibration_horizon = min(
            int(getattr(actor, "prediction_horizon", 32)), steps - 1)
    calibration_anchor = steps - calibration_horizon
    anchor_policy_state = anchor_snapshot = None
    future_actions: list[torch.Tensor] = []
    future_snapshots: list[torch.Tensor] = []
    future_dones: list[torch.Tensor] = []
    for _ in range(steps):
        actor_obs = policy_observation(obs, clock_indices, excluded_indices)
        mean, policy_state = actor.step(obs_norm(actor_obs), policy_state, morphology)
        if stochastic:
            noise = torch.randn(mean.shape, generator=action_generator,
                                dtype=mean.dtype, device=mean.device)
            pre_tanh = mean + actor.log_std.exp() * noise
        else:
            pre_tanh = mean
        a = torch.tanh(pre_tanh)
        confidence = actor.guidance_confidence
        interval = max(int(getattr(actor, "guidance_interval", 4)), 1)
        guided_this_step = False
        if (guidance_enabled and not stochastic and actor.is_predictive and confidence > 0.0
                and _ % interval == 0):
            horizon = int(getattr(actor, "guidance_horizon", 16))
            candidate = mean.detach().unsqueeze(0).expand(horizon, -1, -1).clone()
            interaction_target = env.interaction_target(horizon)
            with torch.enable_grad():
                planned, costs = guided_action_sequence(
                    actor.trajectory_decoder, policy_state, morphology, candidate,
                    steps=int(getattr(actor, "guidance_steps", 2)),
                    interaction_target=interaction_target)
            a = (1.0 - confidence) * a + confidence * planned[0]
            guidance_calls += 1
            guided_this_step = True
            for name, value in costs.items():
                guidance_cost_sums[name] = guidance_cost_sums.get(name, 0.0) + float(value)
        if calibration_horizon > 0 and _ >= calibration_anchor:
            if _ == calibration_anchor:
                anchor_policy_state = policy_state.detach().clone()
                anchor_snapshot = env.trajectory_state()
            future_actions.append(a.detach().clone())
        obs, rew, done, info = env.step(a, alpha=alpha, imit_anneal=imit)
        if calibration_horizon > 0 and _ >= calibration_anchor:
            future_snapshots.append(env.trajectory_state())
            future_dones.append(done.detach().clone())
        if guided_this_step:
            for name in ("reward", "track", "progress", "attack_selected_hit",
                         "attack_wrong_hit", "attack_support", "ladder_goal_progress",
                         "ladder_approach", "ladder_rod_hit"):
                value = rew if name == "reward" else info.get(name)
                if isinstance(value, torch.Tensor):
                    guidance_realized_sums[name] = (
                        guidance_realized_sums.get(name, 0.0) + float(value.mean()))
        if policy_state is not None:
            policy_state.mul_((~done.bool()).to(policy_state.dtype).unsqueeze(-1))
        tel.add(rew, info)
        if digest is not None:
            for value in (a, rew[:, None], done[:, None]):
                quantized = torch.round(value.detach() * 1.0e6).to(
                    torch.int32).cpu().contiguous()
                digest.update(quantized.numpy().tobytes())
    result = tel.result()
    if calibration_horizon > 0 and future_snapshots:
        future = torch.stack(future_snapshots)
        target = stabilized_trajectory_target(
            anchor_snapshot.unsqueeze(0).expand_as(future), future)
        valid = (torch.stack(future_dones).cumsum(0) == 0)
        prediction = actor.predict_trajectory(
            anchor_policy_state, morphology, torch.stack(future_actions))
        result["eval_predictor_calibration"] = {
            name: float(value) for name, value in trajectory_calibration_metrics(
                prediction, target, valid).items()}
    if actor.is_predictive:
        result.update(
            predictor_error_ema=float(actor.prediction_error_ema),
            predictor_updates=int(actor.prediction_updates),
            predictor_calibration_ema=float(actor.prediction_calibration_ema),
            predictor_calibration_updates=int(actor.prediction_calibration_updates),
            predictor_frozen=bool(actor.prediction_frozen),
            guidance_confidence=actor.guidance_confidence,
            guidance_calls=guidance_calls,
            guidance_enabled=bool(guidance_enabled),
            **{f"guidance_{name}": value / max(guidance_calls, 1)
               for name, value in guidance_cost_sums.items()},
            **{f"guidance_realized_{name}": value / max(guidance_calls, 1)
               for name, value in guidance_realized_sums.items()})
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
    universal_geometries = ("universal_control", "universal_command")
    if getattr(args, "power_model", "off") != "off":
        if args.geometry in ("combat", "leg_attack", "ladder_combat"):
            raise ValueError(
                "--power-model is not yet implemented for the fused combat layer")
        env_kwargs["power_model"] = eval_kwargs["power_model"] = args.power_model
    if args.geometry in ("ladder_locomotion", "ladder_combat",
                         *universal_geometries):
        if args.rung is None:
            raise ValueError(f"--geometry {args.geometry} requires --rung")
        env_kwargs["rung"] = eval_kwargs["rung"] = args.rung
    if args.design_bank_json:
        if args.geometry not in universal_geometries or args.rung != 30:
            raise ValueError("--design-bank-json requires a universal geometry "
                             "at rung 30")
        designs = json.loads(Path(args.design_bank_json).read_text())
        env_kwargs["designs"] = eval_kwargs["designs"] = designs
    if args.eval_design_bank_json:
        if args.geometry not in universal_geometries or args.rung != 30:
            raise ValueError(
                "--eval-design-bank-json requires a universal geometry at rung 30")
        eval_kwargs["designs"] = json.loads(
            Path(args.eval_design_bank_json).read_text())
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
    if args.architecture != "mlp" and len(set(hidden)) > 1:
        # The FiLM/recurrent families build len(hidden) residual blocks at
        # width hidden[0]; a tapering spec silently described a network that
        # was never constructed. Refuse to let that stay silent.
        print(f"WARNING: --hidden {args.hidden} taper is IGNORED by the "
              f"{args.architecture} actor: it constructs {len(hidden)} "
              f"residual blocks at constant width {hidden[0]}. Use "
              f"--width {hidden[0]} --blocks {len(hidden)} to say this "
              "honestly.", flush=True)
    args.actor_task_dim = int(getattr(env, "architecture_task_dim", 0))
    args.recurrent_clock_indices = (
        ",".join(str(index) for index in getattr(
            env, "external_clock_observation_indices", ()))
        if args.architecture in ("task_film_gru", "predictive_token_gru") else "")
    args.morphology_observation_indices = (
        ",".join(str(index) for index in getattr(
            env, "morphology_parameter_observation_indices", ()))
        if args.architecture == "predictive_token_gru" else "")
    validate_training_args(args, env, hidden)
    actor = Actor(env.obs_dim, env.act_dim, hidden, architecture=args.architecture,
                  task_dim=args.actor_task_dim,
                  prediction_decoder=args.prediction_decoder).to(dev)
    actor.guidance_horizon = int(args.guidance_horizon)
    actor.guidance_steps = int(args.guidance_steps)
    actor.guidance_interval = int(args.guidance_interval)
    if actor.is_predictive:
        actor.prediction_horizon = int(args.prediction_horizon)
        actor.prediction_freeze_tolerance = float(args.prediction_freeze_tolerance)
        actor.prediction_freeze_patience = int(args.prediction_freeze_patience)
    actor.policy_clock_indices = parse_index_spec(args.recurrent_clock_indices)
    actor.policy_excluded_indices = parse_index_spec(
        args.morphology_observation_indices)
    critic = Critic(env.obs_dim + env.priv_dim, hidden).to(dev)
    obs_norm = RunningNorm(env.obs_dim).to(dev)
    priv_norm = RunningNorm(env.priv_dim).to(dev)
    policy_parameters, predictor_parameters = \
        partition_policy_and_predictor_parameters(actor)
    opt = torch.optim.Adam(policy_parameters + list(critic.parameters()), lr=args.lr)
    prediction_opt = None
    if predictor_parameters:
        prediction_lr = (float(args.prediction_lr)
                         if args.prediction_lr is not None else args.lr)
        prediction_opt = torch.optim.Adam(predictor_parameters, lr=prediction_lr)
        print(f"predictor optimizer: separate Adam lr={prediction_lr:g} "
              "(constant; the adaptive PPO schedule does not apply)", flush=True)
    opponent_paths = list(getattr(args, "opponent_pool", []) or [])
    if getattr(args, "opponent", None):
        opponent_paths.append(args.opponent)
    if opponent_paths:
        if not hasattr(env, "set_opponent"):
            raise ValueError("--opponent is only valid for a two-policy environment")
        def build_opponent(target_env):
            """Per-env opponent mixture: recurrent hidden state must never be
            shared between the training and evaluation environments, and the
            env resets it at episode boundaries through .reset(mask)."""
            policies = [load_policy(
                path, target_env.obs_dim, target_env.act_dim, dev,
                task_dim=getattr(target_env, "architecture_task_dim", 0),
                morphology_source=target_env,
                expected_semantics=expected_load_semantics(target_env))
                for path in opponent_paths]
            assignment = torch.arange(
                target_env.nworld, device=dev) % len(policies)

            class OpponentMixture:
                def __call__(self, obs):
                    actions = torch.stack(
                        [policy(obs) for policy in policies], dim=0)
                    rows = torch.arange(len(obs), device=obs.device)
                    return actions[assignment[:len(obs)], rows]

                def reset(self, mask=None):
                    for policy in policies:
                        reset = getattr(policy, "reset", None)
                        if callable(reset):
                            reset(mask)

            return OpponentMixture()

        env.set_opponent(build_opponent(env))
        eval_env.set_opponent(build_opponent(eval_env))
        print(f"opponent mixture={len(opponent_paths)} policies across "
              f"{env.nworld} worlds", flush=True)
    global_step = 0
    schedule_progress_floor = 0.0
    runtime: dict = {}
    contract = checkpoint_contract(env, args)
    if args.init_policy and args.resume:
        raise ValueError("--init-policy and --resume are mutually exclusive")
    if args.init_policy:
        source_step = initialize_policy(
            args.init_policy, actor, obs_norm, dev,
            expected_semantics=expected_load_semantics(env))
        if (args.architecture in ("task_film", "task_film_gru", "predictive_token_gru")
                and args.rung is not None
                and 1 < args.rung <= args.actor_task_dim
                and getattr(env, "task_conditioning_kind", "rung_id") != "command"):
            # Command-conditioned environments have no per-rung channels to
            # inherit: the shared command manifold is the transfer mechanism.
            inheritance_error = inherit_task_conditioning(
                actor, obs_norm, args.rung - 2, args.rung - 1)
            print(f"inherited task conditioning {args.rung - 1}->{args.rung} "
                  f"max_pre_activation_error={inheritance_error:.3g}", flush=True)
        print(f"initialized actor from {args.init_policy} (source step {source_step}); "
              "critic and optimizer are fresh", flush=True)
    if args.resume:
        global_step, runtime = load_ckpt(
            args.resume, actor, critic, obs_norm, priv_norm, opt, dev,
            expected_contract=contract, allow_legacy=args.allow_legacy_resume,
            allow_reward_migration=args.allow_reward_migration,
            prediction_opt=prediction_opt)
        restore_runtime_state(env, runtime)
        schedule_progress_floor = float(runtime.get("schedule_progress", 0.0))
        if prediction_opt is not None:
            # The decoder rate is a constant CLI contract; never let a resumed
            # param-group value silently override the requested one.
            for parameter_group in prediction_opt.param_groups:
                parameter_group["lr"] = prediction_opt.defaults["lr"]
        if args.learning_rate_restart:
            for parameter_group in opt.param_groups:
                parameter_group["lr"] = args.lr
            print(f"learning-rate restart: policy/value Adam reset to ceiling "
                  f"{args.lr:g}", flush=True)
        print(f"resumed {args.resume} at step {global_step}", flush=True)
    anchor_actor = anchor_norm = None
    anchor_indices: tuple[int, ...] = ()
    if args.anchor_policy:
        anchor_actor, anchor_norm = frozen_anchor_policy(
            args.anchor_policy, env.obs_dim, env.act_dim, hidden,
            args.architecture, args.actor_task_dim, dev,
            args.prediction_decoder,
            expected_semantics=expected_load_semantics(env))
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
        # No semantics guard here by design: the transfer teacher is an
        # explicitly legacy policy reading a declared observation PREFIX.
        transfer_policy = load_policy(
            args.transfer_policy, transfer_obs_dim, env.act_dim, dev)
        if getattr(transfer_policy, "is_recurrent", False):
            raise ValueError(
                "--transfer-policy must be a stateless (non-recurrent) teacher: "
                "the action-prior path evaluates it on shuffled minibatch rows, "
                "which would feed a GRU hidden state from unrelated trajectories")
        print(f"behavior transfer teacher={args.transfer_policy} "
              f"obs_prefix={transfer_obs_dim}", flush=True)
    replay_bank = load_replay_artifacts(
        args.replay_artifact, env.obs_dim, env.act_dim,
        expected_observation_semantics=getattr(
            env, "observation_semantics", None))
    if replay_bank:
        description = ", ".join(
            f"rung {item['rung']} x{item['pressure']:.2f}" for item in replay_bank)
        print(f"real retention replay={description}", flush=True)
    replay_weights = torch.as_tensor(
        [item["pressure"] for item in replay_bank], dtype=torch.float32)
    ckpt_path = Path(f"{args.tag}.pt")
    rollout_steps = args.horizon * args.envs
    eval_interval = incremental_eval_interval(
        global_step, args.steps, args.evals, rollout_steps)
    next_eval = min(args.steps, global_step + eval_interval)
    architecture_record = resolved_architecture(actor, critic)
    environment_capabilities = report_optional_env_capabilities(env)
    print(f"train_mesh_warp: geometry={args.geometry} device={dev} envs={args.envs} "
          f"horizon={args.horizon} steps={args.steps} hidden={hidden} "
          f"actor_params={architecture_record['actor']['parameters']} "
          f"critic_params={architecture_record['critic']['parameters']} "
          f"imitation={'ON' if env.gait_loaded else 'off'} ckpt={ckpt_path}", flush=True)

    T, N = args.horizon, args.envs
    action_mask = getattr(env, "policy_action_mask", None)
    if action_mask is not None:
        action_mask = action_mask.to(device=dev, dtype=torch.float32)
    obs = env.observe()
    priv = env.privileged()
    clock_indices = actor.policy_clock_indices
    excluded_indices = actor.policy_excluded_indices
    actor_state = actor.initial_state(N)
    morph_numeric = getattr(env, "morphology_tokens", None)
    morph_types = getattr(env, "morphology_token_types", None)
    morph_mask = getattr(env, "morphology_token_mask", None)
    saved_policy_state = runtime.get("policy_state")
    if actor_state is not None and isinstance(saved_policy_state, torch.Tensor):
        if saved_policy_state.shape != actor_state.shape:
            raise ValueError("checkpoint recurrent policy state has the wrong shape")
        actor_state.copy_(saved_policy_state.to(device=dev, dtype=actor_state.dtype))
    b_obs = torch.zeros((T, N, env.obs_dim), device=dev)       # normalized (as acted on)
    b_priv = torch.zeros((T, N, env.priv_dim), device=dev)     # normalized (critic input)
    b_raw_obs = torch.zeros_like(b_obs)                        # raw, for norm updates
    b_raw_priv = torch.zeros_like(b_priv)
    b_z = torch.zeros((T, N, env.act_dim), device=dev)         # pre-tanh samples
    b_logp = torch.zeros((T, N), device=dev)
    b_rew = torch.zeros((T, N), device=dev)
    b_done = torch.zeros((T, N), device=dev)
    b_val = torch.zeros((T, N), device=dev)
    b_traj_state = (torch.zeros((T + 1, N, TRAJECTORY_RAW_DIM), device=dev)
                    if actor.is_predictive else None)
    start_step = global_step
    t_start = time.time()
    provenance = training_provenance(args, env, contract, dev)
    metrics_path = Path(f"{args.tag}.metrics.jsonl")
    diagnostics_path = Path(f"{args.tag}.diagnostics.json")
    stats = {
        "schema_version": 3,
        "run": provenance,
        "resolved_architecture": architecture_record,
        "power_model": getattr(env, "power_model_record",
                               {"model": getattr(args, "power_model", "off")}),
        "env_capabilities": environment_capabilities,
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
    plateau_streak = 0
    margin_history: list[tuple[int, float]] = []
    best_candidate_margin = -math.inf
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
    previous_eval_norm = normalization_snapshot(obs_norm)
    while global_step < args.steps:
        update_started = time.perf_counter()
        diagnostic_update = (global_step + T * N >= next_eval
                             or global_step + T * N >= args.steps)
        obs_norm_before = normalization_snapshot(obs_norm)
        schedule_progress_floor = schedule_progress(
            global_step, args, schedule_progress_floor)
        ent_coef, alpha, imit = schedules(
            global_step, args, schedule_progress_floor)
        constraint_pressure = (float(getattr(
            env, "action_prior_suppression_pressure", env.constraint_duals.max()))
            if constraint_names else 0.0)
        competence_pressure = (prior_competence_pressure(
            prior_competence_target, prior_competence_ema)
            if prior_competence_target is not None and prior_competence_ema is not None
            else 1.0)
        current_action_prior_weight = action_prior_weight(
            action_prior_base_weight, action_prior_floor, global_step,
            args.steps, constraint_pressure, competence_pressure,
            progress_fraction=schedule_progress_floor)
        rollout_initial_state = (None if actor_state is None
                                 else actor_state.detach().clone())
        with torch.no_grad():
            rollout_morphology = (actor.encode_morphology(
                morph_numeric, morph_types, morph_mask) if actor.is_predictive else None)
            if b_traj_state is not None:
                b_traj_state[0].copy_(env.trajectory_state())
            rollout_telemetry = EvalTelemetry(dev) if diagnostic_update else None
            constraint_sums = torch.zeros(len(constraint_names), device=dev)
            competence_constraint_sums = torch.zeros(len(competence_names), device=dev)
            competence_sum = torch.zeros((), device=dev)
            for t in range(T):
                actor_raw_obs = policy_observation(obs, clock_indices, excluded_indices)
                obs_n, priv_n = obs_norm(actor_raw_obs), priv_norm(priv)
                mu, next_actor_state = actor.step(
                    obs_n, actor_state, rollout_morphology)
                z = mu + actor.log_std.exp() * torch.randn_like(mu)
                if action_mask is not None:
                    z = torch.where(action_mask.bool(), z, torch.zeros_like(z))
                nobs, rew, done, info = env.step(torch.tanh(z), alpha=alpha, imit_anneal=imit)
                if rollout_telemetry is not None:
                    rollout_telemetry.add(rew, info)
                trunc = info["truncated"]
                # time-limit bootstrap: V(terminal obs) folded into the reward
                terminal_actor_obs = policy_observation(
                    info["terminal_obs"], clock_indices, excluded_indices)
                tv = critic(torch.cat([obs_norm(terminal_actor_obs),
                                       priv_norm(info["terminal_priv"])], -1))
                b_raw_obs[t], b_raw_priv[t] = actor_raw_obs, priv
                b_obs[t], b_priv[t], b_z[t] = obs_n, priv_n, z
                b_logp[t] = logp_tanh(z, mu, actor.log_std, action_mask)
                b_val[t] = critic(torch.cat([obs_n, priv_n], -1))
                b_rew[t] = rew + GAMMA * tv * trunc
                b_done[t] = done
                if next_actor_state is not None:
                    actor_state = next_actor_state * (~done.bool()).to(
                        next_actor_state.dtype).unsqueeze(-1)
                for constraint_index, name in enumerate(constraint_names):
                    constraint_sums[constraint_index].add_(info[name].mean())
                for competence_index, name in enumerate(competence_names):
                    competence_constraint_sums[competence_index].add_(info[name].mean())
                if prior_competence_metric is not None:
                    competence_sum.add_(info[prior_competence_metric].mean())
                obs, priv = nobs, info["priv"]
                if b_traj_state is not None:
                    b_traj_state[t + 1].copy_(env.trajectory_state())
            last_actor_obs = policy_observation(obs, clock_indices, excluded_indices)
            last_val = critic(torch.cat([obs_norm(last_actor_obs), priv_norm(priv)], -1))
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
        obs_norm.update(
            b_raw_obs.reshape(-1, env.obs_dim),
            frozen_suffix=(args.actor_task_dim
                           if args.architecture in (
                               "task_film", "task_film_gru", "predictive_token_gru") else 0))
        priv_norm.update(b_raw_priv.reshape(-1, env.priv_dim))

        B = T * N
        b_reset_before = torch.zeros((T, N), dtype=torch.bool, device=dev)
        b_reset_before[1:] = b_done[:-1].bool()
        f_obs = b_obs.reshape(B, -1)
        f_raw_obs = b_raw_obs.reshape(B, -1)
        f_cin = torch.cat([f_obs, b_priv.reshape(B, -1)], -1)
        f_z, f_logp = b_z.reshape(B, -1), b_logp.reshape(B)
        f_adv_raw = adv.reshape(B)
        f_adv = (f_adv_raw - f_adv_raw.mean()) / (f_adv_raw.std() + 1e-8)
        f_ret = ret.reshape(B)
        critic_target_scale = f_ret.std(unbiased=False).detach().clamp_min(1.0)
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
                "critic_target_scale": float(critic_target_scale),
                "explained_variance_before_update": float(explained_variance),
            }
            critic_before_update = critic_calibration(b_val.reshape(B), f_ret)
            next_values = torch.cat((b_val[1:], last_val.unsqueeze(0)), dim=0)
            td_error = (b_rew + GAMMA * next_values * (1.0 - b_done) - b_val).reshape(B)
            observation_normalization = normalization_diagnostics(
                obs_norm_before, obs_norm, f_raw_obs)
        mb = B // args.minibatches
        pi_l = v_l = ent_l = distill_l = action_prior_l = prediction_loss_l = 0.0
        prediction_parts_l: dict[str, float] = {}
        approx_kl_l = clip_fraction_l = gradient_norm_l = 0.0
        actor_gradient_norm_l = critic_gradient_norm_l = 0.0
        predictor_gradient_norm_l = 0.0
        actor_gradient_norms: list[float] = []
        critic_gradient_norms: list[float] = []
        epoch_trust_region: list[dict] = []
        objective_gradients: dict = {}
        epochs_completed = 0
        kl_early_stop = False
        learning_rate_used = float(opt.param_groups[0]["lr"])
        prior_axis_rmse_t: dict[str, torch.Tensor] = {}
        prior_leg_rmse_t: dict[str, torch.Tensor] = {}

        def current_rollout_mean() -> torch.Tensor | None:
            if not actor.is_recurrent:
                return None
            with torch.no_grad():
                context = (actor.encode_morphology(
                    morph_numeric, morph_types, morph_mask)
                           if actor.is_predictive else None)
                return actor.sequence(
                    b_obs, rollout_initial_state, b_reset_before,
                    context)[0].reshape(B, -1)

        for epoch_index in range(args.epochs):
            perm = torch.randperm(N if actor.is_recurrent else B, device=dev)
            for i in range(args.minibatches):
                if actor.is_recurrent:
                    env_mb = N // args.minibatches
                    env_idx = perm[i * env_mb:(i + 1) * env_mb]
                    idx = (torch.arange(T, device=dev)[:, None] * N
                           + env_idx[None, :]).reshape(-1)
                    minibatch_morphology = (actor.encode_morphology(
                        morph_numeric[env_idx], morph_types[env_idx], morph_mask[env_idx])
                        if actor.is_predictive else None)
                    if actor.is_predictive:
                        mu_sequence, state_sequence, _ = actor.sequence_with_states(
                            b_obs[:, env_idx], rollout_initial_state[env_idx],
                            b_reset_before[:, env_idx], minibatch_morphology)
                    else:
                        mu_sequence, _ = actor.sequence(
                            b_obs[:, env_idx], rollout_initial_state[env_idx],
                            b_reset_before[:, env_idx])
                        state_sequence = None
                    mu = mu_sequence.reshape(-1, env.act_dim)
                else:
                    idx = perm[i * mb:(i + 1) * mb]
                    mu = actor(f_obs[idx])
                logp = logp_tanh(f_z[idx], mu, actor.log_std, action_mask)
                ratio = torch.exp(logp - f_logp[idx])
                a_mb = f_adv[idx]
                pg = -torch.min(ratio * a_mb,
                                ratio.clamp(1.0 - CLIP, 1.0 + CLIP) * a_mb).mean()
                vloss = scale_invariant_value_loss(
                    critic(f_cin[idx]), f_ret[idx], critic_target_scale)
                ent = entropy_tanh(mu, actor.log_std, action_mask).mean()
                prediction_loss = torch.zeros((), device=dev)
                prediction_parts: dict[str, torch.Tensor] = {}
                # The physics targets do not change across PPO epochs. One
                # decoder pass per rollout is enough; repeating it on every PPO
                # epoch multiplies recurrent work without adding experience.
                if (actor.is_predictive and args.prediction_loss_weight > 0.0
                        and epoch_index == 0
                        and actor.prediction_training_enabled):
                    horizon = int(args.prediction_horizon)
                    anchor_count = min(int(args.prediction_anchors), T - horizon)
                    anchor_times = torch.randperm(
                        T - horizon, device=dev)[:anchor_count].sort().values
                    losses = []
                    part_accumulator: dict[str, list[torch.Tensor]] = {}
                    for anchor_time in anchor_times.tolist():
                        future_actions = torch.tanh(
                            b_z[anchor_time:anchor_time + horizon, env_idx])
                        prediction = actor.predict_trajectory(
                            state_sequence[anchor_time], minibatch_morphology,
                            future_actions)
                        anchor_state = b_traj_state[anchor_time, env_idx]
                        future_state = b_traj_state[
                            anchor_time + 1:anchor_time + horizon + 1, env_idx]
                        target = stabilized_trajectory_target(
                            anchor_state.unsqueeze(0).expand_as(future_state), future_state)
                        valid = (b_done[
                            anchor_time:anchor_time + horizon, env_idx].cumsum(0) == 0)
                        local_loss, local_parts = trajectory_prediction_loss(
                            prediction, target, valid)
                        losses.append(local_loss)
                        for name, value in local_parts.items():
                            part_accumulator.setdefault(name, []).append(value)
                    prediction_loss = torch.stack(losses).mean()
                    prediction_parts = {
                        name: torch.stack(values).mean()
                        for name, values in part_accumulator.items()}
                distill = torch.zeros((), device=dev)
                if replay_bank and args.distill_weight > 0.0:
                    replay_item = replay_bank[int(torch.multinomial(
                        replay_weights, 1).item())]
                    replay_observations = replay_item["observations"]
                    if actor.is_recurrent and replay_observations.ndim == 3:
                        replay_count = min(env_mb, replay_observations.shape[1])
                        replay_index = torch.randint(
                            replay_observations.shape[1], (replay_count,))
                        replay_raw_sequence = replay_observations[:, replay_index].to(dev)
                        teacher_sequence = replay_item["actions"][:, replay_index].to(dev)
                        replay_reset = torch.zeros(
                            replay_raw_sequence.shape[:2], dtype=torch.bool, device=dev)
                        replay_done = replay_item.get("dones")
                        if replay_done is not None:
                            replay_reset[1:] = replay_done[:-1, replay_index].to(dev)
                        replay_normalized = obs_norm(
                            replay_raw_sequence.reshape(-1, env.obs_dim)).reshape_as(
                                replay_raw_sequence)
                        replay_morphology = (actor.encode_morphology(
                            morph_numeric[:replay_count], morph_types[:replay_count],
                            morph_mask[:replay_count]) if actor.is_predictive else None)
                        student_sequence, _ = actor.sequence(
                            replay_normalized, None, replay_reset, replay_morphology)
                        replay_raw = replay_raw_sequence.reshape(-1, env.obs_dim)
                        teacher_action = teacher_sequence.reshape(-1, env.act_dim)
                        student_action = torch.tanh(
                            student_sequence.reshape(-1, env.act_dim))
                    else:
                        # Collected replay is time-major [time, env, feature].
                        # Feed-forward actors consume independent rows, so
                        # flatten both sample axes before drawing a minibatch.
                        # Indexing only the time axis leaves a 3-D tensor and
                        # makes FiLM split the env axis as if it were features.
                        replay_observations = replay_observations.reshape(
                            -1, env.obs_dim)
                        replay_actions = replay_item["actions"].reshape(
                            -1, env.act_dim)
                        replay_count = min(len(idx), len(replay_observations))
                        replay_index = torch.randint(
                            len(replay_observations), (replay_count,))
                        replay_raw = replay_observations[replay_index].to(dev)
                        teacher_action = replay_actions[replay_index].to(dev)
                        if actor.is_predictive:
                            replay_morphology = actor.encode_morphology(
                                morph_numeric[:replay_count], morph_types[:replay_count],
                                morph_mask[:replay_count])
                            student_action = torch.tanh(actor.step(
                                obs_norm(replay_raw), None, replay_morphology)[0])
                        else:
                            student_action = torch.tanh(actor(obs_norm(replay_raw)))
                    if env.obs_dim == 256 and env.act_dim == 14:
                        replay_mask = replay_raw[:, 211:225]
                    else:
                        replay_mask = torch.ones_like(student_action)
                    distill = (((student_action - teacher_action) * replay_mask) ** 2).sum() \
                        / replay_mask.sum().clamp_min(1.0)
                elif anchor_actor is not None and args.distill_weight > 0.0:
                    anchor_raw = (b_raw_obs[:, env_idx].clone()
                                  if actor.is_recurrent else f_raw_obs[idx].clone())
                    if anchor_indices:
                        batch_count = anchor_raw.shape[-2] if actor.is_recurrent else len(idx)
                        rows = torch.arange(batch_count, device=dev)
                        choices = torch.as_tensor(anchor_indices, device=dev)[
                            rows % len(anchor_indices)]
                        anchor_raw[..., -args.actor_task_dim:] = 0.0
                        if actor.is_recurrent:
                            anchor_raw[:, rows,
                                       anchor_raw.shape[-1] - args.actor_task_dim
                                       + choices] = 1.0
                        else:
                            anchor_raw[rows, anchor_raw.shape[1]
                                       - args.actor_task_dim + choices] = 1.0
                    if actor.is_recurrent:
                        normalized_anchor = obs_norm(
                            anchor_raw.reshape(-1, env.obs_dim)).reshape_as(anchor_raw)
                        student_anchor_morphology = (actor.encode_morphology(
                            morph_numeric[env_idx], morph_types[env_idx],
                            morph_mask[env_idx]) if actor.is_predictive else None)
                        with torch.no_grad():
                            teacher_anchor_morphology = (anchor_actor.encode_morphology(
                                morph_numeric[env_idx], morph_types[env_idx],
                                morph_mask[env_idx]) if anchor_actor.is_predictive else None)
                            teacher_mu, _ = anchor_actor.sequence(
                                anchor_norm(anchor_raw.reshape(-1, env.obs_dim)).reshape_as(
                                    anchor_raw), None, b_reset_before[:, env_idx],
                                teacher_anchor_morphology)
                        student_mu, _ = actor.sequence(
                            normalized_anchor, None, b_reset_before[:, env_idx],
                            student_anchor_morphology)
                    else:
                        with torch.no_grad():
                            teacher_mu = anchor_actor(anchor_norm(anchor_raw))
                        student_mu = actor(obs_norm(anchor_raw))
                    distill = ((student_mu - teacher_mu) ** 2).mean()
                action_prior = torch.zeros((), device=dev)
                if current_action_prior_weight > 0.0 and hasattr(env, "policy_mean_prior"):
                    prior_base = None
                    if anchor_actor is not None and anchor_indices:
                        transfer_raw = f_raw_obs[idx].clone()
                        if getattr(env, "task_conditioning_kind",
                                   "rung_id") == "command":
                            # Commands are the conditioning: the anchor reads
                            # the same manifold, so nothing may be rewritten.
                            pass
                        else:
                            transfer_raw[:, -args.actor_task_dim:] = 0.0
                            transfer_raw[:, transfer_raw.shape[1] - args.actor_task_dim
                                         + max(anchor_indices)] = 1.0
                            if getattr(env, "rung", None) == 7:
                                transfer_raw[:, 47:50] = 0.0
                        with torch.no_grad():
                            if anchor_actor.is_predictive:
                                flat_env_indices = (idx % N).long()
                                anchor_morphology = anchor_actor.encode_morphology(
                                    morph_numeric[flat_env_indices],
                                    morph_types[flat_env_indices],
                                    morph_mask[flat_env_indices])
                                prior_base = anchor_actor.step(
                                    anchor_norm(transfer_raw), None,
                                    anchor_morphology)[0]
                            else:
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
                        + args.prediction_loss_weight * prediction_loss
                        + args.distill_weight * distill
                        + current_action_prior_weight * action_prior)
                opt.zero_grad()
                if prediction_opt is not None:
                    prediction_opt.zero_grad()
                if diagnostic_update and i == args.minibatches - 1:
                    objective_gradients = objective_gradient_diagnostics({
                        "ppo_policy": pg,
                        "entropy": -ent_coef * ent,
                        "trajectory_prediction": (
                            args.prediction_loss_weight * prediction_loss),
                        "retention_distillation": args.distill_weight * distill,
                        "action_prior": current_action_prior_weight * action_prior,
                    }, actor.parameters())
                loss.backward()
                # Actor and critic have disjoint parameters and fundamentally
                # different loss scales.  Joint clipping lets a large raw-return
                # value loss suppress the policy gradient by the critic's clip
                # factor.  Bound each optimizer subspace independently so reward
                # scale cannot silently freeze policy learning.
                actor_gradient_norm, critic_gradient_norm, predictor_gradient_norm = \
                    clip_actor_critic_gradients(actor, critic)
                opt.step()
                if actor.is_predictive and prediction_parts:
                    # Step the decoder only on minibatches that computed its
                    # loss; a zero-gradient Adam step would still decay the
                    # decoder along stale momentum.
                    if prediction_opt is not None:
                        prediction_opt.step()
                    actor.observe_prediction_error(prediction_loss)
                with torch.no_grad():
                    log_ratio = logp - f_logp[idx]
                    approx_kl_l = float(((ratio - 1.0) - log_ratio).mean())
                    clip_fraction_l = float(((ratio - 1.0).abs() > CLIP).float().mean())
                    actor_gradient_norm_l = float(actor_gradient_norm)
                    critic_gradient_norm_l = float(critic_gradient_norm)
                    if prediction_parts:
                        predictor_gradient_norm_l = float(predictor_gradient_norm)
                    actor_gradient_norms.append(actor_gradient_norm_l)
                    critic_gradient_norms.append(critic_gradient_norm_l)
                    gradient_norm_l = math.hypot(
                        actor_gradient_norm_l, critic_gradient_norm_l)
                pi_l, v_l, ent_l = float(pg.detach()), float(vloss.detach()), float(ent.detach())
                distill_l = float(distill.detach())
                action_prior_l = float(action_prior.detach())
                if prediction_parts:
                    prediction_loss_l = float(prediction_loss.detach())
                    prediction_parts_l = {
                        name: float(value.detach())
                        for name, value in prediction_parts.items()}
            if diagnostic_update:
                recurrent_mean = current_rollout_mean()
                epoch_record = policy_trust_region_diagnostics(
                    actor, f_obs, f_z, f_logp,
                    lambda z, mean, std: logp_tanh(
                        z, mean, std, action_mask), CLIP,
                    mean_override=recurrent_mean)
                epoch_record["epoch"] = epoch_index + 1
                epoch_trust_region.append(epoch_record)
            else:
                recurrent_mean = current_rollout_mean()
                epoch_record = policy_epoch_trust(
                    actor, f_obs, f_z, f_logp, action_mask,
                    mean_override=recurrent_mean)
            epochs_completed = epoch_index + 1
            if kl_epoch_should_stop(
                    epoch_record["approx_kl"], args.target_kl,
                    args.kl_stop_multiplier):
                kl_early_stop = True
                break
        prediction_calibration_l: dict[str, float] = {}
        if actor.is_predictive and b_traj_state is not None:
            # Training anchors are sampled from [0, T-horizon); the final valid
            # anchor T-horizon is deliberately excluded and used only here.
            # Authority therefore follows a normalized out-of-sample forecast,
            # not the decoder objective it just optimized.
            calibration_horizon = min(int(args.prediction_horizon), T - 1)
            calibration_anchor = T - calibration_horizon
            with torch.no_grad():
                calibration_morphology = actor.encode_morphology(
                    morph_numeric, morph_types, morph_mask)
                _, calibration_states, _ = actor.sequence_with_states(
                    b_obs, rollout_initial_state, b_reset_before,
                    calibration_morphology)
                calibration_prediction = actor.predict_trajectory(
                    calibration_states[calibration_anchor], calibration_morphology,
                    torch.tanh(b_z[calibration_anchor:T]))
                calibration_anchor_state = b_traj_state[calibration_anchor]
                calibration_future_state = b_traj_state[calibration_anchor + 1:T + 1]
                calibration_target = stabilized_trajectory_target(
                    calibration_anchor_state.unsqueeze(0).expand_as(
                        calibration_future_state), calibration_future_state)
                calibration_valid = (b_done[calibration_anchor:T].cumsum(0) == 0)
                calibration = trajectory_calibration_metrics(
                    calibration_prediction, calibration_target, calibration_valid)
                frozen_before = bool(actor.prediction_frozen)
                actor.observe_prediction_calibration(calibration["overall"])
                if bool(actor.prediction_frozen) != frozen_before:
                    transition = "frozen" if bool(actor.prediction_frozen) else "unfrozen"
                    print(f"predictor {transition} at step {global_step + T * N}: "
                          f"held-out calibration ema="
                          f"{float(actor.prediction_calibration_ema):.4f} "
                          f"best={float(actor.prediction_best_calibration):.4f}",
                          flush=True)
                prediction_calibration_l = {
                    name: float(value) for name, value in calibration.items()}
        optimization_finished = time.perf_counter()
        recurrent_mean = current_rollout_mean()
        trust_region = policy_trust_region_diagnostics(
            actor, f_obs, f_z, f_logp,
            lambda z, mean, std: logp_tanh(z, mean, std, action_mask), CLIP,
            mean_override=recurrent_mean)
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
            "trajectory_prediction_loss": prediction_loss_l,
            "weighted_trajectory_prediction_loss": (
                args.prediction_loss_weight * prediction_loss_l),
            "trajectory_prediction_components": prediction_parts_l,
            "trajectory_prediction_calibration": prediction_calibration_l,
            "trajectory_prediction_frozen": (bool(actor.prediction_frozen)
                                             if actor.is_predictive else False),
            "trajectory_prediction_best_calibration": (
                float(actor.prediction_best_calibration)
                if actor.is_predictive else None),
            "trajectory_prediction_degraded_streak": (
                int(actor.prediction_degraded_streak)
                if actor.is_predictive else 0),
            "predictor_gradient_norm_before_clip": predictor_gradient_norm_l,
            "predictor_learning_rate": (
                float(prediction_opt.param_groups[0]["lr"])
                if prediction_opt is not None else None),
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
                "reset_phase_randomization_probability": float(getattr(
                    env, "reset_phase_randomization_probability",
                    torch.zeros((), device=dev))),
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
                "trajectory_prediction": prediction_loss_l,
                "trajectory_prediction_weight": args.prediction_loss_weight,
                "trajectory_prediction_components": prediction_parts_l,
                "predictor_error_ema": (float(actor.prediction_error_ema)
                                        if actor.is_predictive else None),
                "predictor_calibration": prediction_calibration_l,
                "predictor_calibration_ema": (
                    float(actor.prediction_calibration_ema)
                    if actor.is_predictive else None),
                "predictor_calibration_updates": (
                    int(actor.prediction_calibration_updates)
                    if actor.is_predictive else 0),
                "predictor_frozen": (bool(actor.prediction_frozen)
                                     if actor.is_predictive else False),
                "predictor_best_calibration": (
                    float(actor.prediction_best_calibration)
                    if actor.is_predictive else None),
                "guidance_confidence": actor.guidance_confidence,
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
                "ladder_foot_activity", "ladder_step_clock", "ladder_swing_clearance",
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

            guidance_ab: dict = {}
            if actor.is_predictive:
                unguided_metrics = evaluate_policy(
                    eval_env, actor, obs_norm, alpha, imit, args.eval_steps,
                    reset_seed=eval_seeds[0], guidance_enabled=False)
                comparison_keys = tuple(dict.fromkeys((
                    *metric_keys, "attack_selected_hit", "attack_wrong_hit",
                    "attack_support", "ladder_approach", "ladder_rod_hit",
                    "ladder_goal_progress")))
                deltas = {
                    key: float(m[key]) - float(unguided_metrics[key])
                    for key in comparison_keys
                    if isinstance(m.get(key), (int, float))
                    and isinstance(unguided_metrics.get(key), (int, float))
                }
                guidance_ab = {
                    "seed": eval_seeds[0],
                    "steps": args.eval_steps,
                    "guided": {key: m[key] for key in deltas},
                    "unguided": {key: unguided_metrics[key] for key in deltas},
                    "guided_minus_unguided": deltas,
                }

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
            robust_gate_report = robust_gate_diagnostics(early_gates, seed_summary)
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
                  f"footact={m.get('ladder_foot_activity', 0.0):.3f} "
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
                  f"gconf={m.get('guidance_confidence', 0.0):.3f} "
                  f"gdelta={m.get('guidance_action_delta_rms', 0.0):.6f} "
                  f"gtaskdrop={(m.get('guidance_before_task', 0.0) - m.get('guidance_after_task', 0.0)):.4f} "
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
                      contract=contract, prediction_opt=prediction_opt,
                      runtime=capture_runtime_state(
                          env, schedule_progress=schedule_progress_floor,
                          policy_state=actor_state))
            checkpoint_hash = sha256_file(ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location=dev, weights_only=True)
            replay_actor = Actor(
                env.obs_dim, env.act_dim, hidden, architecture=args.architecture,
                task_dim=args.actor_task_dim,
                prediction_decoder=args.prediction_decoder).to(dev)
            replay_actor.policy_clock_indices = actor.policy_clock_indices
            replay_actor.policy_excluded_indices = actor.policy_excluded_indices
            replay_norm = RunningNorm(env.obs_dim).to(dev)
            replay_actor.load_state_dict(checkpoint["actor"])
            replay_norm.load_state_dict(checkpoint["obs_norm"])
            replay_actor.eval(); replay_norm.eval()
            replay_after = evaluate_policy(
                eval_env, replay_actor, replay_norm, alpha, imit, replay_steps,
                reset_seed=replay_seed, fingerprint=True)
            replay_atol, replay_rtol = checkpoint_replay_tolerances(args.geometry)
            replay_comparison = checkpoint_replay_comparison(
                replay_before, replay_after, metric_keys,
                atol=replay_atol, rtol=replay_rtol)
            checkpoint_replay = {
                "steps": replay_steps,
                "seed": replay_seed,
                "before_sha256": replay_before.get("trajectory_sha256"),
                "after_sha256": replay_after.get("trajectory_sha256"),
                "fingerprint_match": (replay_before.get("trajectory_sha256")
                                      == replay_after.get("trajectory_sha256")),
                **replay_comparison,
            }
            checkpoint_replay["pass"] = (
                checkpoint_replay["fingerprint_match"]
                or replay_comparison["pass"])
            candidate_margin = robust_gate_report.get("worst_relative_margin")
            if candidate_margin is not None and candidate_margin > best_candidate_margin:
                best_candidate_margin = float(candidate_margin)
                archive = Path(f"{args.tag}.candidates")
                archive.mkdir(parents=True, exist_ok=True)
                candidate_path = archive / f"step_{global_step:012d}.pt"
                shutil.copy2(ckpt_path, candidate_path)
                write_json_atomic(candidate_path.with_suffix(".json"), {
                    "schema_version": 1,
                    "step": global_step,
                    "source_checkpoint": str(ckpt_path),
                    "checkpoint_sha256": checkpoint_hash,
                    "robust_gates": robust_gate_report,
                    "multi_seed_evaluation": seed_summary,
                    "checkpoint_replay_pass": checkpoint_replay["pass"],
                })
                print(f"CANDIDATE_ARCHIVE step={global_step} "
                      f"robust_margin={best_candidate_margin:.4f} "
                      f"path={candidate_path}", flush=True)
            del checkpoint, replay_actor, replay_norm, frozen_normalizer

            update_diagnostics.update({
                "train_eval_gap": train_eval_gap,
                "multi_seed_evaluation": seed_summary,
                "robust_gates": robust_gate_report,
                "deterministic_stochastic_gap": deterministic_stochastic_gap,
                "frozen_live_normalization_gap": frozen_live_norm_gap,
                "guidance_ab": guidance_ab,
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
            # Abort an attempt whose worst gate margin is failing and, on its
            # observed trend, cannot cross zero within slack times the
            # remaining budget.  The ladder retries the same checkpoint with
            # changed dynamics instead of buying more identical steps.
            if early_gates and args.plateau_slack > 0.0:
                worst_margin = (robust_gate_report or gate_report).get(
                    "worst_relative_margin")
                if worst_margin is not None and math.isfinite(float(worst_margin)):
                    margin_history.append((global_step, float(worst_margin)))
                    projection = gate_margin_projection(
                        margin_history, args.steps, slack=args.plateau_slack,
                        window=args.plateau_min_evals)
                    plateau_streak = (plateau_streak + 1 if projection["strike"]
                                      else 0)
                    if plateau_streak >= args.plateau_patience:
                        stats["plateau_abort"] = {
                            "step": global_step,
                            "consecutive_strikes": plateau_streak,
                            "worst_relative_margin": float(worst_margin),
                            "slope_per_step": projection["slope_per_step"],
                            "projected_crossing_step":
                                projection["projected_crossing_step"],
                            "slack": float(args.plateau_slack),
                        }
                        append_jsonl(metrics_path, {
                            "event": "plateau_abort",
                            "run_id": provenance["run_id"],
                            **stats["plateau_abort"],
                        })
                        print(f"PLATEAU_ABORT step={global_step} "
                              f"worst_margin={float(worst_margin):.4f} "
                              f"strikes={plateau_streak}", flush=True)
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
              contract=contract, prediction_opt=prediction_opt,
              runtime=capture_runtime_state(
                  env, schedule_progress=schedule_progress_floor,
                  policy_state=actor_state))
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
    ap.add_argument("--replay-artifact", action="append", default=[],
                    metavar="PATH[,PRESSURE]",
                    help=("real accepted-policy state/action replay; repeat per old task, "
                          "optionally with adaptive pressure"))
    ap.add_argument("--action-prior-weight", type=float, default=0.5,
                    help="behavioral-prior weight for environments that expose one")
    ap.add_argument("--action-prior-json", default=None,
                    help="versioned searched behavior-prior artifact loaded by the environment")
    ap.add_argument("--design-bank-json", default=None,
                    help="co-design adaptation bank for universal-control rung 30")
    ap.add_argument("--eval-design-bank-json", default=None,
                    help="held-out design bank compiled only into the evaluation "
                         "environment (universal-control rung 30); predictor "
                         "calibration measured there covers morphologies the "
                         "policy and decoder never trained on")
    ap.add_argument("--transfer-policy", default=None,
                    help="frozen legacy policy used as a behavioral transfer teacher")
    ap.add_argument("--transfer-obs-dim", type=int, default=None,
                    help="leading observation dimensions consumed by --transfer-policy")
    ap.add_argument("--opponent", default=None,
                    help="frozen Torch checkpoint for the combat B policy")
    ap.add_argument("--opponent-pool", action="append", default=[],
                    help="repeat to mix frozen PFSP opponents across parallel worlds")
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
    ap.add_argument("--plateau-slack", type=float, default=2.0,
                    help="abort the run early when the worst early-gate margin "
                         "is failing and its projected zero crossing exceeds "
                         "this multiple of the remaining step budget; <= 0 "
                         "disables the plateau abort")
    ap.add_argument("--plateau-min-evals", type=int, default=4,
                    help="evaluations fitted by the margin projection before a "
                         "plateau strike is possible")
    ap.add_argument("--plateau-patience", type=int, default=3,
                    help="consecutive plateau strikes before aborting")
    ap.add_argument("--entropy-floor", type=float, default=ENT_END,
                    help="minimum entropy coefficient regardless of anneal "
                         "progress; raised by the ladder for from-scratch "
                         "acquisition rungs")
    ap.add_argument("--entropy-boost", type=float, default=1.0,
                    help="multiplier on the scheduled entropy coefficient, "
                         "capped at the from-scratch start value; used by "
                         "plateau-intervention retries")
    ap.add_argument("--learning-rate-restart", action="store_true",
                    help="on resume, reset the policy/value Adam learning rate "
                         "to the --lr ceiling instead of the adapted value in "
                         "the checkpoint (plateau-intervention warm restart)")
    ap.add_argument("--early-patience", type=int, default=2,
                    help="consecutive deterministic evaluations required to stop early")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--target-kl", type=float, default=0.02,
                    help="whole-rollout PPO KL target; zero disables adaptive control")
    ap.add_argument("--kl-stop-multiplier", type=float, default=1.5,
                    help="stop remaining PPO epochs above this multiple of target KL")
    ap.add_argument("--hidden", default="512,256,128",
                    help="layer sizes for the mlp actor; for FiLM/recurrent "
                         "actors only the count and first value matter (see "
                         "--width/--blocks for the honest parameters)")
    ap.add_argument("--width", type=int, default=None,
                    help="residual-stream width for FiLM/recurrent actors; "
                         "overrides --hidden as width repeated --blocks times")
    ap.add_argument("--blocks", type=int, default=None,
                    help="residual block count for FiLM/recurrent actors "
                         "(default 3 when --width is given)")
    ap.add_argument("--architecture", choices=(
        "mlp", "task_film", "task_film_gru", "predictive_token_gru"),
                    default="mlp",
                    help=("task_film = task-conditioned residual actor; task_film_gru "
                          "adds learned memory and receives no explicit gait clock; "
                          "predictive_token_gru adds morphology tokens and future prediction"))
    ap.add_argument("--prediction-horizon", type=int, default=32,
                    help="self-supervised future horizon in control frames (32 = 0.64 s)")
    ap.add_argument("--prediction-decoder", choices=("recurrent", "transformer"),
                    default="recurrent",
                    help=("future-physics decoder: sequential GRU baseline or causal "
                          "non-autoregressive temporal Transformer"))
    ap.add_argument("--prediction-loss-weight", type=float, default=0.25,
                    help="auxiliary locally stabilized trajectory prediction loss")
    ap.add_argument("--prediction-lr", type=float, default=None,
                    help="constant Adam learning rate for the trajectory decoder "
                         "(default: --lr); the decoder always trains under its "
                         "own optimizer and never follows the adaptive PPO "
                         "learning-rate schedule")
    ap.add_argument("--prediction-freeze-tolerance", type=float, default=0.15,
                    help="freeze decoder training once the held-out calibration "
                         "EMA rises this fraction above its own best; training "
                         "resumes when it recovers to half the tolerance. "
                         "<= 0 disables the freeze")
    ap.add_argument("--prediction-freeze-patience", type=int, default=3,
                    help="consecutive degraded held-out calibration observations "
                         "before the decoder freezes")
    ap.add_argument("--prediction-anchors", type=int, default=4,
                    help="random rollout anchors per environment used by the decoder loss")
    ap.add_argument("--guidance-horizon", type=int, default=16,
                    help="candidate future action frames optimized through the decoder")
    ap.add_argument("--guidance-steps", type=int, default=2,
                    help="prediction-gradient refinements per guided decision")
    ap.add_argument("--guidance-interval", type=int, default=4,
                    help="run planning every N control frames and reuse its first-action bias")
    ap.add_argument("--power-model", choices=("off", "shared_bus"),
                    default="off",
                    help="shared_bus applies the robot-wide electrical budget "
                         "(bus current sum, voltage droop, current limit) with "
                         "conservative per-world randomized supply parameters; "
                         "changes action semantics to +shared_bus_v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    ap.add_argument("--alpha-start", type=float, default=0.0)
    ap.add_argument("--alpha-end", type=float, default=1.0)
    ap.add_argument("--alpha-frac", type=float, default=0.6,
                    help="fraction of --steps over which alpha ramps")
    ap.add_argument("--imit-anneal-frac", type=float, default=0.7,
                    help="imitation weight 1 -> 0 over this fraction of --steps")
    args = ap.parse_args(argv)
    if args.width is not None:
        if args.width < 1 or (args.blocks is not None and args.blocks < 1):
            ap.error("--width and --blocks must be positive")
        args.hidden = ",".join([str(int(args.width))] * int(args.blocks or 3))
    elif args.blocks is not None:
        ap.error("--blocks requires --width")
    return args


if __name__ == "__main__":
    train(build_args())
