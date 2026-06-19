# SPDX-License-Identifier: MIT
"""Evaluate + record the combat-dodge quadruped (notes/rl-combat-dodge-checklist §6).

Loads the torch policy BEFORE the mujoco env (heap-order fix), runs CombatDodgeEnv,
and reports the combat metrics: how it ended (strike / fall / survived), steps
survived, standoff distance, and vulnerable-exposure (a foot in the strike band
while in the bite radius). Records qpos/qvel + the mocap pursuer position each step
for torch-free rendering.
  python eval_combat.py [--model ...] [--random] [--difficulty 0.6] [--weapon spinner] [--video --tag combat]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

from stable_baselines3 import PPO  # noqa: E402  (torch first)
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "sim" / "build" / "rl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--motor", default="db42s03")
    ap.add_argument("--weapon", default="spinner")
    ap.add_argument("--difficulty", type=float, default=0.6)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--tag", default="combat")
    args = ap.parse_args()

    use_model = args.model and not args.random
    model = PPO.load(args.model) if use_model else None
    norm = None
    if use_model:
        import pickle
        vp = Path(args.model).with_name(Path(args.model).name
                                        .replace("ppo_", "vecnorm_").replace(".zip", ".pkl"))
        if vp.is_file():
            with open(vp, "rb") as f:
                vn = pickle.load(f)
            norm = (vn.obs_rms.mean, np.sqrt(vn.obs_rms.var + vn.epsilon), vn.clip_obs)

    from combat_env import CombatDodgeEnv   # lazy: mujoco AFTER the torch model
    env = CombatDodgeEnv(motor=args.motor, weapon=args.weapon, difficulty=args.difficulty)
    returns, steps_l, ends, expose, standoff = [], [], [], [], []
    qpos_rec, qvel_rec, mocap_rec = [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        done = trunc = False
        ret = 0.0; steps = 0; vuln = 0; dsum = 0.0; end = "survived"
        while not (done or trunc) and steps < args.max_steps:
            if use_model:
                o = obs if norm is None else np.clip((obs - norm[0]) / norm[1],
                                                     -norm[2], norm[2])
                a = model.predict(o, deterministic=True)[0]
            else:
                a = env.action_space.sample()
            obs, r, done, trunc, info = env.step(a)
            ret += r; steps += 1
            dsum += info["d_com"]
            vuln += int(info["in_reach"] and info["min_foot_clear"] < 0)
            if done:
                end = "strike" if info["strike"] else "fall"
            if args.video and ep == 0:
                qpos_rec.append(env.data.qpos.copy())
                qvel_rec.append(env.data.qvel.copy())
                mocap_rec.append(env.data.mocap_pos.copy())
        returns.append(ret); steps_l.append(steps); ends.append(end)
        expose.append(vuln / max(steps, 1)); standoff.append(dsum / max(steps, 1))
    n = len(ends)
    print(f"{args.tag} ({args.weapon}): return={np.mean(returns):.1f} | "
          f"steps survived={np.mean(steps_l):.0f} | "
          f"ended by strike={ends.count('strike')}/{n} fall={ends.count('fall')}/{n} "
          f"survived={ends.count('survived')}/{n} | "
          f"vulnerable={100*np.mean(expose):.0f}% time | standoff={np.mean(standoff):.2f} m")
    if args.video and qpos_rec:
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / f"{args.tag}_traj.npz"
        np.savez(p, qpos=np.array(qpos_rec), qvel=np.array(qvel_rec),
                 mocap=np.array(mocap_rec), robot="combat")
        print(f"recorded {len(qpos_rec)} steps -> {p}")


if __name__ == "__main__":
    main()
