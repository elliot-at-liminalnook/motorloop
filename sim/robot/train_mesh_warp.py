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
+step; resumable via --resume). TRIPWIRE: duty > 0.98 past half the run means
the creep optimum won — exit code 3.

--geometry {mesh,walker,combat} selects the slider-crank locomotor, the
12-servo hardware-contract walker, or the symmetric two-robot fight model.
All share one Torch learner and MuJoCo-Warp physics path.

Rollout, GAE, and PPO updates all live on the env's device; physics is CUDA-
graph-captured by the env when a GPU is present. The same code runs (slowly)
on CPU:

  .venv-warp/bin/python sim/robot/train_mesh_warp.py --geometry walker \
      --steps 200000 --envs 64 --horizon 64 --tag /tmp/walkerwarp_smoke --evals 4
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import mujoco_warp as mjwp  # noqa: E402

from mesh_warp_env import EvalTelemetry, MeshWarpEnv  # noqa: E402
from walker_warp_env import WalkerWarpEnv  # noqa: E402
from combat_warp_env import CombatWarpEnv  # noqa: E402
from codesign_warp_env import DesignEnsembleWarpEnv  # noqa: E402

# --geometry selects the batched env; both expose the SAME interface (obs_dim,
# priv_dim, act_dim=12, step/observe/privileged/reset, gait_loaded) and each
# defaults its own reference-gait path, so nothing else in the trainer changes.
GEOMETRIES = {
    "mesh": MeshWarpEnv,
    "walker": WalkerWarpEnv,
    "combat": CombatWarpEnv,
    "universal": DesignEnsembleWarpEnv,
}

GAMMA, LAM, CLIP = 0.99, 0.95, 0.2
ENT_START, ENT_END = 3e-2, 5e-3
LOG2PI = math.log(2.0 * math.pi)


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


def _mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.SiLU()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden):
        super().__init__()
        self.trunk = _mlp([obs_dim, *hidden])
        self.mu = nn.Linear(hidden[-1], act_dim)
        with torch.no_grad():                       # small final init: near-zero targets at start
            self.mu.weight.mul_(0.01)
            self.mu.bias.zero_()
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mu(self.trunk(obs))


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
    "actions_a", "actions_b",
)


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
            dst.copy_(tensors[name].to(device=dst.device, dtype=dst.dtype))
    env._gen.set_state(state["generator"])
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
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and "cuda_rng" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def checkpoint_contract(env, args) -> dict:
    """Semantic identity required before policy/optimizer state may be reused."""
    return {
        "geometry": args.geometry,
        "model_hash": env.model_hash,
        "action_semantics": getattr(
            env, "action_semantics", "pd_target@50hz:lowpass+torque_speed_v1"),
        "observation_semantics": f"actor{env.obs_dim}+priv{env.priv_dim}:v1",
        "reward_semantics": getattr(
            env, "reward_semantics", f"{args.geometry}:velocity_command:v1"),
    }


def validate_training_args(args, env, hidden: tuple[int, ...]) -> None:
    batch = int(args.envs) * int(args.horizon)
    if min(args.envs, args.horizon, args.minibatches, args.epochs, args.steps) <= 0:
        raise ValueError("steps/envs/horizon/minibatches/epochs must all be positive")
    if batch % args.minibatches:
        raise ValueError(
            f"rollout batch {batch} is not divisible by {args.minibatches} minibatches; "
            "the current slicing would silently discard samples")
    if args.preflight != "off":
        from preflight import preflight_check
        preflight_check(
            steps=args.steps, batch=args.envs, minibatches=1, unroll=args.horizon,
            episode_length=args.episode_length, discounting=GAMMA,
            control_dt=env._dt, obs_dim=env.obs_dim, hidden0=hidden[0],
            from_scratch=not bool(args.resume), mode=args.preflight,
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
              expected_contract: dict, allow_legacy: bool = False) -> tuple[int, dict | None]:
    ck = torch.load(path, map_location=device, weights_only=False)
    got = ck.get("contract")
    if got is None and not allow_legacy:
        raise ValueError(
            f"checkpoint {path} has no model/action/observation contract; "
            "use --allow-legacy-resume only for a deliberate diagnostic load")
    if got is not None:
        mismatch = {k: (got.get(k), want) for k, want in expected_contract.items()
                    if got.get(k) != want}
        if mismatch:
            raise ValueError(f"checkpoint {path} is incompatible: {mismatch}")
    actor.load_state_dict(ck["actor"])
    critic.load_state_dict(ck["critic"])
    obs_norm.load_state_dict(ck["obs_norm"])
    priv_norm.load_state_dict(ck["priv_norm"])
    if opt is not None and ck.get("opt") is not None:
        opt.load_state_dict(ck["opt"])
    return int(ck["step"]), ck.get("runtime")


def load_policy(path, obs_dim: int, act_dim: int, device):
    """Load a deterministic actor and its observation normalizer."""
    ck = torch.load(path, map_location=device, weights_only=False)
    hidden = tuple(int(v) for v in ck.get("args", {}).get("hidden", "512,256,128").split(","))
    actor = Actor(obs_dim, act_dim, hidden).to(device)
    norm = RunningNorm(obs_dim).to(device)
    actor.load_state_dict(ck["actor"])
    norm.load_state_dict(ck["obs_norm"])
    actor.eval()
    norm.eval()

    @torch.no_grad()
    def policy(obs):
        return torch.tanh(actor(norm(obs)))

    return policy


def schedules(step: int, args) -> tuple[float, float, float]:
    """(ent_coef, alpha, imit_anneal) at env-step `step` — all linear."""
    p = min(step / max(args.steps, 1), 1.0)
    ent = ENT_START + (ENT_END - ENT_START) * p
    ap = min(p / max(args.alpha_frac, 1e-9), 1.0)
    alpha = args.alpha_start + (args.alpha_end - args.alpha_start) * ap
    imit = max(0.0, 1.0 - p / max(args.imit_anneal_frac, 1e-9))
    return ent, alpha, imit


@torch.no_grad()
def evaluate(env, actor, obs_norm, alpha, imit, steps: int, *, reset_seed: int) -> dict:
    """Deterministic fixed-scenario pass; returns ``EvalTelemetry.result()``."""
    env._gen.manual_seed(reset_seed)
    obs = env.reset()
    tel = EvalTelemetry(env.device)
    for _ in range(steps):
        a = torch.tanh(actor(obs_norm(obs)))
        obs, rew, done, info = env.step(a, alpha=alpha, imit_anneal=imit)
        tel.add(rew, info)
    return tel.result()


# ---------------------------------------------------------------------- train
def train(args) -> dict:
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    EnvClass = GEOMETRIES[args.geometry]
    env = EnvClass(args.envs, seed=args.seed, device=device,
                   episode_length=args.episode_length)
    eval_env = EnvClass(args.eval_envs, seed=args.seed + 1000, device=device,
                        episode_length=args.episode_length)
    dev = env.device
    hidden = tuple(int(h) for h in args.hidden.split(","))
    validate_training_args(args, env, hidden)
    actor = Actor(env.obs_dim, env.act_dim, hidden).to(dev)
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
    if args.resume:
        global_step, runtime = load_ckpt(
            args.resume, actor, critic, obs_norm, priv_norm, opt, dev,
            expected_contract=contract, allow_legacy=args.allow_legacy_resume)
        restore_runtime_state(env, runtime)
        print(f"resumed {args.resume} at step {global_step}", flush=True)
    ckpt_path = Path(f"{args.tag}.pt")
    eval_interval = max(1, args.steps // max(args.evals, 1))
    next_eval = (global_step // eval_interval + 1) * eval_interval
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
    stats = {"updates": [], "evals": [], "ckpt": str(ckpt_path)}
    t_start = time.time()

    while global_step < args.steps:
        ent_coef, alpha, imit = schedules(global_step, args)
        with torch.no_grad():
            for t in range(T):
                obs_n, priv_n = obs_norm(obs), priv_norm(priv)
                mu = actor(obs_n)
                z = mu + actor.log_std.exp() * torch.randn_like(mu)
                nobs, rew, done, info = env.step(torch.tanh(z), alpha=alpha, imit_anneal=imit)
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
                obs, priv = nobs, info["priv"]
            last_val = critic(torch.cat([obs_norm(obs), priv_norm(priv)], -1))
            adv, ret = compute_gae(b_rew, b_done, b_val, last_val)
        obs_norm.update(b_raw_obs.reshape(-1, env.obs_dim))
        priv_norm.update(b_raw_priv.reshape(-1, env.priv_dim))

        B = T * N
        f_obs = b_obs.reshape(B, -1)
        f_cin = torch.cat([f_obs, b_priv.reshape(B, -1)], -1)
        f_z, f_logp = b_z.reshape(B, -1), b_logp.reshape(B)
        f_adv = adv.reshape(B)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
        f_ret = ret.reshape(B)
        mb = B // args.minibatches
        pi_l = v_l = ent_l = 0.0
        for _ in range(args.epochs):
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
                loss = pg + 0.5 * vloss - ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0)
                opt.step()
                pi_l, v_l, ent_l = float(pg.detach()), float(vloss.detach()), float(ent.detach())
        global_step += T * N
        stats["updates"].append({"step": global_step, "pi_loss": pi_l, "v_loss": v_l,
                                 "entropy": ent_l, "ent_coef": ent_coef, "alpha": alpha,
                                 "imit_anneal": imit})

        if global_step >= next_eval or global_step >= args.steps:
            m = evaluate(eval_env, actor, obs_norm, alpha, imit, args.eval_steps,
                         reset_seed=args.seed + 1000)
            elapsed = time.time() - t_start
            print(f"METRIC step={global_step} reward={m['reward']:.3f} track={m['track']:.3f} "
                  f"verr={m['verr']:.3f} align={m['align']:.3f} speed={m['speed']:.3f} "
                  f"progress={m['progress']:.3f} duty={m['duty']:.3f} air={m['air']:.3f} "
                  f"diagsync={m['diagsync']:.3f} alpha={alpha:.2f} entcoef={ent_coef:.4f} "
                  f"catrate={m['catrate']:.3f} xprog={m.get('xprogress', 0.0):.3f} "
                  f"lat={m.get('lateral', 0.0):.3f} "
                  f"progema={m.get('progress_ema', 0.0):.3f} "
                  f"catprog={m.get('cat_progress', 0.0):.3f} "
                  f"catduty={m.get('cat_duty', 0.0):.3f} "
                  f"fduty={m.get('foot_duty_ema', 0.0):.3f} "
                  f"catfduty={m.get('cat_foot_duty', 0.0):.3f} "
                  f"hpeak={m.get('hop_peak', 0.0):.3f} "
                  f"hland={m.get('hop_stable_landing', 0.0):.3f} "
                  f"catslip={m.get('cat_slip', 0.0):.3f} "
                  f"catsupp={m.get('cat_support', 0.0):.3f} "
                  f"catbody={m.get('cat_body', 0.0):.3f} "
                  f"prior={m.get('motion_prior', 0.0):.3f} ({elapsed:.0f}s)", flush=True)
            save_ckpt(ckpt_path, global_step, actor, critic, obs_norm, priv_norm, opt, args,
                      contract=contract, runtime=capture_runtime_state(env))
            stats["evals"].append({"step": global_step, **m})
            next_eval += eval_interval
            if args.steps >= 1_000_000 and m["duty"] > 0.98 and global_step > 0.5 * args.steps:
                print("TRIPWIRE duty stagnation", flush=True)
                sys.exit(3)

    save_ckpt(ckpt_path, global_step, actor, critic, obs_norm, priv_norm, opt, args,
              contract=contract, runtime=capture_runtime_state(env))
    print(f"DONE step={global_step} ckpt={ckpt_path}", flush=True)
    return stats


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", choices=tuple(GEOMETRIES), default="mesh",
                    help="mesh/walker = commanded locomotion; combat = symmetric fight")
    ap.add_argument("--steps", type=int, default=20_000_000, help="total env steps")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--episode-length", type=int, default=800)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tag", default="mesh_warp")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--opponent", default=None,
                    help="frozen Torch checkpoint for the combat B policy")
    ap.add_argument("--allow-legacy-resume", action="store_true",
                    help="diagnostic only: load a checkpoint with no semantic contract")
    ap.add_argument("--preflight", choices=("strict", "warn", "off"), default="strict",
                    help="derived training-config gate; long launches must use strict")
    ap.add_argument("--evals", type=int, default=20)
    ap.add_argument("--eval-envs", type=int, default=32)
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--hidden", default="512,256,128")
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
