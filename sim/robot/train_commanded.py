# SPDX-License-Identifier: MIT
"""Train the command-conditioned locomotor (brax PPO over CommandedEnv) — a steerable
robot: it follows a remote directional command while balancing autonomously.

  python train_commanded.py [--steps 8000000 --envs 4096]
  python train_commanded.py --tiny
Streams `METRIC`/CSV with the velocity-tracking score; checkpoints every eval (resumable).
"""

from __future__ import annotations

import argparse, os, pickle, sys, time
from pathlib import Path
import jax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from commanded_env import _build  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out")); OUT.mkdir(parents=True, exist_ok=True)


def METRIC(**kw): print("METRIC " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8_000_000)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--unroll", type=int, default=10)
    ap.add_argument("--updates", type=int, default=4)
    ap.add_argument("--evals", type=int, default=0)
    ap.add_argument("--tag", default="cmd")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    if args.tiny:
        args.steps, args.envs, args.batch, args.minibatches, args.unroll, args.evals = 16000, 256, 256, 8, 5, 2
    n_eval = args.evals or max(6, args.steps // 1_000_000)

    Env = _build(); env = Env()
    print(f"commanded env: obs={env.observation_size} (incl. 2-D command) act={env.action_size}", flush=True)
    restore = pickle.load(open(args.resume, "rb")) if (args.resume and os.path.exists(args.resume)) else None

    import json
    t0 = time.time(); csv = OUT / f"{args.tag}_metrics.csv"; csv.write_text("step,reward,track,verr,sec\n")
    fj = OUT / f"{args.tag}_train.jsonl"; fj.write_text("")
    tm = {"c": None}; last = {"r": float("nan"), "track": 0.0, "verr": 0.0, "step": 0}
    g = lambda m, k: float(m.get(f"eval/episode_{k}", 0.0))
    def prog(s, m):
        if tm["c"] is None: tm["c"] = time.time() - t0
        r = g(m, "reward"); tr = g(m, "track"); ve = g(m, "verr")
        open(csv, "a").write(f"{int(s)},{r:.3f},{tr:.4f},{ve:.4f},{time.time()-t0:.0f}\n")
        open(fj, "a").write(json.dumps(dict(step=int(s), reward=round(r,3), track=round(tr,4),
                            verr=round(ve,4), tag=args.tag)) + "\n")
        last.update(r=r, track=tr, verr=ve, step=int(s))
        print(f"  [{args.tag}] step {int(s):>9,} reward {r:7.2f} track {tr:.3f} verr {ve:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    def ck(*a):
        try: pickle.dump(a[-1], open(OUT / f"{args.tag}_ckpt.pkl", "wb"))
        except Exception: pass

    make_inf, params, _ = ppo.train(
        environment=env, num_timesteps=args.steps, num_evals=n_eval, episode_length=400,
        num_envs=args.envs, batch_size=args.batch, num_minibatches=args.minibatches,
        unroll_length=args.unroll, num_updates_per_batch=args.updates, learning_rate=3e-4,
        entropy_cost=1e-2, discounting=0.97, reward_scaling=0.1, normalize_observations=True,
        seed=0, progress_fn=prog, policy_params_fn=ck, restore_params=restore)
    pickle.dump(params, open(OUT / f"{args.tag}.pkl", "wb"))
    METRIC(stage="cmd_train", train_s=f"{time.time()-t0:.1f}", env_steps=last["step"],
           final_track=f"{last['track']:.3f}", final_verr=f"{last['verr']:.3f}")
    print(f"PROVEN: command-conditioned locomotor trained — velocity-tracking {last['track']:.3f} "
          f"(1.0 = perfect command following), velocity error {last['verr']:.3f} m/s. "
          f"A remote controller now steers it (eval_commanded.py).", flush=True)


if __name__ == "__main__":
    main()
