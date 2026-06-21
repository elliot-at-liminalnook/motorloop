# SPDX-License-Identifier: MIT
"""Deploy test: drive the trained policy with a LIVE remote-style command sequence and
record whether the robot actually follows it. Cycles the command through a square
(+x → +y → −x → −y → hold), overwriting `state.info["cmd"]` each step (exactly what a
remote controller would do), and logs commanded vs achieved base velocity + the xy path.
Saves `<out>/cmd_eval.npz` for `make_command_figure.py`.

  python eval_commanded.py [--ckpt out/cmd_ckpt.pkl --hold 120]
"""

from __future__ import annotations

import argparse, os, pickle, sys
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from commanded_env import _build, VMAX  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.acme import running_statistics  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT / "cmd_ckpt.pkl"))
    ap.add_argument("--hold", type=int, default=120, help="control steps per command leg")
    args = ap.parse_args()

    Env = _build(); env = Env()
    params = pickle.load(open(args.ckpt, "rb"))
    net = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                         preprocess_observations_fn=running_statistics.normalize)
    inf = ppo_networks.make_inference_fn(net)(params, deterministic=True)

    # the "remote controller" program: a square of directions + a hold
    legs = [(VMAX, 0.0), (0.0, VMAX), (-VMAX, 0.0), (0.0, -VMAX), (0.0, 0.0)]
    cmds = np.array([c for c in legs for _ in range(args.hold)])

    key = jax.random.PRNGKey(0)
    st = env.reset_with_command(key, cmds[0])
    step = jax.jit(env.step)
    rec = []
    for i in range(len(cmds)):
        st.info["cmd"] = jnp.asarray(cmds[i])                 # remote controller overwrites cmd
        # rebuild obs tail with the new command so the policy sees it
        st = st.replace(obs=st.obs.at[-2:].set(jnp.asarray(cmds[i])))
        key, k = jax.random.split(key)
        act, _ = inf(st.obs, k)
        st = step(st, act)
        dx = st.pipeline_state
        rec.append([i, float(cmds[i][0]), float(cmds[i][1]),
                    float(dx.qvel[0]), float(dx.qvel[1]),
                    float(dx.qpos[0]), float(dx.qpos[1]), float(dx.qpos[2])])
        if float(dx.qpos[2]) < 0.18:
            print(f"fell at step {i} (lost balance)"); break
    R = np.array(rec)
    np.savez(OUT / "cmd_eval.npz", t=R[:, 0], cmd_vx=R[:, 1], cmd_vy=R[:, 2],
             vx=R[:, 3], vy=R[:, 4], x=R[:, 5], y=R[:, 6], z=R[:, 7], hold=args.hold)
    # how well did achieved velocity align with the command (on moving legs)?
    mv = (np.abs(R[:, 1]) + np.abs(R[:, 2])) > 1e-6
    if mv.sum():
        cmd = R[mv, 1:3]; ach = R[mv, 3:5]
        cos = np.sum(cmd * ach, 1) / (np.linalg.norm(cmd, axis=1) * np.linalg.norm(ach, axis=1) + 1e-6)
        print(f"[deploy] mean direction alignment (cos) achieved-vs-commanded = {np.nanmean(cos):+.2f} "
              f"(1=perfect); mean speed achieved {np.linalg.norm(ach,axis=1).mean():.2f} m/s "
              f"(commanded {VMAX}); survived {len(R)}/{len(cmds)} steps")
    print(f"saved {OUT/'cmd_eval.npz'} -> make_command_figure.py")


if __name__ == "__main__":
    main()
