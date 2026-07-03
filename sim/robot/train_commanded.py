# SPDX-License-Identifier: MIT
"""Train the command-conditioned locomotor (brax PPO over CommandedEnv) — a steerable
robot: it follows a remote directional command while balancing autonomously.

  python train_commanded.py [--steps 8000000 --envs 4096]
  python train_commanded.py --tiny
Streams `METRIC`/CSV with the velocity-tracking score; checkpoints every eval (resumable).
"""

from __future__ import annotations

import argparse, json, os, pickle, sys, time
from pathlib import Path
import jax, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from commanded_env import (  # noqa: E402
    CMD_CONTROL_MODE,
    CMD_REWARD_MODE,
    CMD_TRAIN_MODE,
    FALL_Z,
    MIN_UP_Z,
    OBS_PRIOR_STRENGTH,
    OBS_ROUTE_CONTEXT,
    TRACK_SIGMA,
    VMAX,
    _build,
)
from brax.training.agents.ppo import train as ppo  # noqa: E402
import ppo_nets as ppo_networks  # noqa: E402  shared (512,256,128) + small-final-init factory (audit item 5)

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out")); OUT.mkdir(parents=True, exist_ok=True)


def METRIC(**kw): print("METRIC " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def _infer_policy_obs(policy) -> int | None:
    pp = policy.get("params", policy) if isinstance(policy, dict) else {}
    try:
        return int(pp["hidden_0"]["kernel"].shape[0])
    except Exception:
        return None


def warm_start(path, obs_dim):
    """Load a PPO params tuple and pad observation-facing leaves to this env's obs size.

    The original universal locomotor is 38-D. Commanded locomotion is the same 38-D
    block plus 2 command inputs. Padding the normalizer and first policy/value kernels
    preserves the existing gait while initializing command sensitivity to zero.
    """
    try:
        parts = list(pickle.load(open(path, "rb")))
        if len(parts) < 2:
            return tuple(parts)
        norm, nets = parts[0], parts[1:]
        old_obs = _infer_policy_obs(nets[0])
        if old_obs is None:
            print(f"WARM-START: could not infer obs size from {path}; using checkpoint unchanged", flush=True)
            return tuple(parts)
        pad = obs_dim - old_obs
        if pad < 0:
            raise ValueError(f"checkpoint obs {old_obs} is wider than env obs {obs_dim}")
        if pad == 0:
            print(f"WARM-START ok: obs {old_obs}->{obs_dim} unchanged", flush=True)
            return tuple(parts)
        c = getattr(norm, "count", None)
        cval = 1.0
        if c is not None and hasattr(c, "hi") and hasattr(c, "lo"):
            cval = float(jnp.asarray(c.hi)) * (2.0 ** 32) + float(jnp.asarray(c.lo))
        nkw = {}
        for fn in ("mean", "std", "summed_variance"):
            v = getattr(norm, fn, None)
            if hasattr(v, "ndim") and v.ndim >= 1 and v.shape[0] == old_obs:
                fill = (jnp.zeros(pad, dtype=v.dtype) if fn == "mean" else
                        jnp.ones(pad, dtype=v.dtype) if fn == "std" else
                        jnp.full((pad,), max(cval, 1.0), dtype=v.dtype))
                nkw[fn] = jnp.concatenate([v, fill])
        if nkw:
            norm = norm.replace(**nkw)

        def pad_leaf(x):
            if hasattr(x, "ndim") and x.ndim >= 1 and x.shape[0] == old_obs:
                return jnp.concatenate([x, jnp.zeros((pad,) + x.shape[1:], dtype=x.dtype)], axis=0)
            return x
        nets = [jax.tree_util.tree_map(pad_leaf, n) for n in nets]
        print(f"WARM-START ok: obs {old_obs}->{obs_dim} (+{pad} command dims)", flush=True)
        return tuple([norm] + nets)
    except Exception as e:
        print(f"WARM-START failed ({type(e).__name__}: {e}) -> training from scratch", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8_000_000)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--unroll", type=int, default=10)
    ap.add_argument("--updates", type=int, default=4)
    ap.add_argument("--evals", type=int, default=0)
    ap.add_argument("--episode-length", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=1e-2)
    ap.add_argument("--tag", default="cmd")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--preflight", choices=["strict", "warn", "off"],
                    default=os.environ.get("CMD_PREFLIGHT", "strict"),
                    help="T2 config sanity gate (strict refuses <200-iteration from-scratch runs "
                         "and sub-stride credit horizons); --tiny smoke runs auto-downgrade to warn")
    # T6 stagnation tripwire: the 12M-step run that moved 0.18 m should have
    # self-terminated at ~2M. If mean speed is still under the floor once this
    # fraction of the budget is spent, stop the run and stop billing.
    ap.add_argument("--stagnation-speed-floor", type=float, default=0.03,
                    help="mean |v| (m/s) the policy must exceed by --stagnation-frac of "
                         "the step budget, else the run aborts (0 = tripwire off)")
    ap.add_argument("--stagnation-frac", type=float, default=0.3,
                    help="fraction of --steps at which the stagnation floor is enforced")
    args = ap.parse_args()
    if args.tiny:
        args.steps, args.envs, args.batch, args.minibatches, args.unroll, args.evals = 16000, 256, 256, 8, 5, 2
        if args.preflight == "strict":
            args.preflight = "warn"          # a smoke run is not a training run
    n_eval = args.evals or max(6, args.steps // 1_000_000)

    Env = _build(); env = Env()
    print(f"commanded env: obs={env.observation_size} (incl. 2-D command) act={env.action_size}", flush=True)
    restore = warm_start(args.resume, env.observation_size) if (args.resume and os.path.exists(args.resume)) else None

    meta = dict(tag=args.tag, steps=args.steps, envs=args.envs, batch=args.batch,
                minibatches=args.minibatches, unroll=args.unroll, updates=args.updates,
                evals=n_eval, episode_length=args.episode_length, seed=args.seed,
                v_max=VMAX, track_sigma=TRACK_SIGMA, fall_z=FALL_Z, min_up_z=MIN_UP_Z,
                cmd_train_mode=CMD_TRAIN_MODE, cmd_reward_mode=CMD_REWARD_MODE,
                cmd_control_mode=CMD_CONTROL_MODE,
                obs_prior_strength=OBS_PRIOR_STRENGTH,
                obs_route_context=OBS_ROUTE_CONTEXT,
                resume=os.path.basename(args.resume) if args.resume else None)
    (OUT / f"{args.tag}_train_meta.json").write_text(json.dumps(meta, indent=2))
    t0 = time.time(); csv = OUT / f"{args.tag}_metrics.csv"
    csv.write_text("step,reward,reward_mean,track,track_mean,verr,verr_mean,align,align_mean,"
                   "speed,speed_mean,progress,progress_mean,up,up_mean,height,height_mean,sec\n")
    fj = OUT / f"{args.tag}_train.jsonl"; fj.write_text("")
    tm = {"c": None}
    last = {"r": float("nan"), "reward_mean": float("nan"), "track": 0.0, "track_mean": 0.0,
            "verr": 0.0, "verr_mean": 0.0, "align": 0.0, "align_mean": 0.0, "step": 0}
    g = lambda m, k: float(m.get(f"eval/episode_{k}", 0.0))
    def prog(s, m):
        if tm["c"] is None: tm["c"] = time.time() - t0
        r = g(m, "reward"); tr = g(m, "track"); ve = g(m, "verr")
        al = g(m, "align"); sp = g(m, "speed"); pr = g(m, "progress"); up = g(m, "up"); ht = g(m, "height")
        ep = float(args.episode_length)
        rec = dict(step=int(s), sec=round(time.time() - t0, 0), reward=round(r, 3),
                   reward_mean=round(r / ep, 5), track=round(tr, 4), track_mean=round(tr / ep, 5),
                   verr=round(ve, 4), verr_mean=round(ve / ep, 5),
                   align=round(al, 4), align_mean=round(al / ep, 5),
                   speed=round(sp, 4), speed_mean=round(sp / ep, 5),
                   progress=round(pr, 4), progress_mean=round(pr / ep, 5),
                   up=round(up, 4), up_mean=round(up / ep, 5),
                   height=round(ht, 4), height_mean=round(ht / ep, 5), tag=args.tag)
        with open(csv, "a") as f:
            f.write(f"{rec['step']},{r:.3f},{r/ep:.6f},{tr:.4f},{tr/ep:.6f},"
                    f"{ve:.4f},{ve/ep:.6f},{al:.4f},{al/ep:.6f},{sp:.4f},{sp/ep:.6f},"
                    f"{pr:.4f},{pr/ep:.6f},{up:.4f},{up/ep:.6f},{ht:.4f},{ht/ep:.6f},{rec['sec']:.0f}\n")
        with open(fj, "a") as f:
            f.write(json.dumps(rec) + "\n")
        last.update(r=r, reward_mean=r / ep, track=tr, track_mean=tr / ep,
                    verr=ve, verr_mean=ve / ep, align=al, align_mean=al / ep, step=int(s))
        print(f"  [{args.tag}] step {int(s):>9,} reward/step {r/ep:6.3f} "
              f"track/step {tr/ep:.3f} verr/step {ve/ep:.3f} "
              f"progress/step {pr/ep:+.3f} align/step {al/ep:+.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        # T6 stagnation tripwire — behavioral metric, not reward: reward can climb
        # while the robot farms standing terms; mean speed cannot be farmed motionless.
        if (args.stagnation_speed_floor > 0 and s >= args.stagnation_frac * args.steps
                and (sp / ep) < args.stagnation_speed_floor):
            print(f"  [{args.tag}] TRIPWIRE-STAGNATION: mean speed {sp/ep:.4f} m/s < "
                  f"{args.stagnation_speed_floor} at {int(s):,}/{args.steps:,} steps — "
                  f"aborting the run (the 0.18m/12M-step lesson).", flush=True)
            os._exit(3)
    import ckpt_meta
    from gen_robot_mjcf import build_mjcf as _bm, load_spec as _ls
    _model_hash = ckpt_meta.current_model_hash(_bm(_ls(Path(__file__).resolve().parent / "robot.toml")))
    def _write_meta(path):
        ckpt_meta.write_meta(path, action_semantics=ckpt_meta.COMMANDED_PD_SEMANTICS,
                             obs_size=env.observation_size, model_hash=_model_hash,
                             behavior={k: last.get(k) for k in ("track_mean", "verr_mean", "step")},
                             extra=dict(tag=args.tag, control_mode=os.environ.get("CMD_CONTROL_MODE", "pd")))
    def ck(*a):
        try:
            with open(OUT / f"{args.tag}_ckpt.pkl", "wb") as f:
                pickle.dump(a[-1], f)
            _write_meta(OUT / f"{args.tag}_ckpt.pkl")
        except Exception as e:
            print(f"  [ck] save {args.tag}_ckpt.pkl failed: {e}", flush=True)

    if getattr(args, "preflight", "strict") != "off":
        from preflight import preflight_check
        preflight_check(steps=args.steps, batch=args.batch, minibatches=args.minibatches,
                        unroll=args.unroll, episode_length=args.episode_length,
                        discounting=0.99, control_dt=0.02, obs_dim=env.observation_size,
                        hidden0=512, from_scratch=(args.resume is None),
                        mode=getattr(args, "preflight", "strict"), tag=args.tag,
                        run_dir=OUT, resolved=vars(args))
    make_inf, params, _ = ppo.train(
        environment=env, num_timesteps=args.steps, num_evals=n_eval, episode_length=args.episode_length,
        num_envs=args.envs, batch_size=args.batch, num_minibatches=args.minibatches,
        unroll_length=args.unroll, num_updates_per_batch=args.updates, learning_rate=args.lr,
        entropy_cost=args.entropy, discounting=0.99, reward_scaling=0.1, normalize_observations=True,
        # γ=0.99: 2 s credit horizon at 50 Hz — 0.97 gave 0.66 s, shorter than one stride
        # (audit item 3); a swing phase couldn't see the reward of the step it set up.
        network_factory=ppo_networks.make_ppo_networks,
        seed=args.seed, progress_fn=prog, policy_params_fn=ck, restore_params=restore)
    pickle.dump(params, open(OUT / f"{args.tag}.pkl", "wb"))
    _write_meta(OUT / f"{args.tag}.pkl")
    METRIC(stage="cmd_train", train_s=f"{time.time()-t0:.1f}", env_steps=last["step"],
           final_track_mean=f"{last['track_mean']:.3f}", final_verr_mean=f"{last['verr_mean']:.3f}",
           final_align_mean=f"{last['align_mean']:.3f}")
    print(f"TRAINED: command-conditioned locomotor artifact {args.tag}.pkl. "
          f"Final eval per-step track={last['track_mean']:.3f} (1.0 is perfect), "
          f"velocity error={last['verr_mean']:.3f}, alignment={last['align_mean']:+.3f}. "
          f"Run eval_commanded.py --tag {args.tag} to validate deployment.", flush=True)


if __name__ == "__main__":
    main()
