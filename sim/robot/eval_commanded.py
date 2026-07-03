# SPDX-License-Identifier: MIT
"""Deploy test: drive the trained policy with a LIVE remote-style command sequence and
record whether the robot actually follows it. Cycles the command through a square
(+x → +y → −x → −y → hold), overwriting `state.info["cmd"]` each step (exactly what a
remote controller would do), and logs commanded vs achieved base velocity + the xy path.
Saves `<out>/cmd_eval.npz` for `make_command_figure.py`.

  python eval_commanded.py [--ckpt out/cmd_ckpt.pkl --hold 120]
"""

from __future__ import annotations

import argparse, json, os, pickle, sys
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from commanded_env import FALL_Z, MIN_UP_Z, TRACK_SIGMA, VMAX, _build  # noqa: E402
import ppo_nets as ppo_networks  # shared (512,256,128) + small-final-init factory (audit item 5)  # noqa: E402
from brax.training.acme import running_statistics  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))


def resolve_ckpt(tag, ckpt):
    if ckpt:
        return Path(ckpt)
    for name in (f"{tag}_ckpt.pkl", f"{tag}.pkl", "cmd_ckpt.pkl"):
        p = OUT / name
        if p.exists():
            return p
    return OUT / f"{tag}_ckpt.pkl"


def command_program(mode, hold, speed, seed=0):
    speed = float(speed)
    if mode == "forward":
        legs = [(speed, 0.0)]
    elif mode == "backward":
        legs = [(-abs(speed), 0.0)]
    elif mode == "left":
        legs = [(0.0, abs(speed))]
    elif mode == "right":
        legs = [(0.0, -abs(speed))]
    elif mode == "sweep":
        legs = [(0.0, 0.0), (0.25 * speed, 0.0), (0.5 * speed, 0.0),
                (0.75 * speed, 0.0), (speed, 0.0), (0.0, 0.0)]
    elif mode == "random":
        rng = np.random.default_rng(seed)
        headings = rng.uniform(-np.pi, np.pi, size=12)
        mags = rng.uniform(0.25 * abs(speed), abs(speed), size=12)
        legs = [(float(m * np.cos(a)), float(m * np.sin(a))) for a, m in zip(headings, mags)]
        legs.insert(0, (0.0, 0.0))
        legs.append((0.0, 0.0))
    else:
        s = abs(speed)
        legs = [(s, 0.0), (0.0, s), (-s, 0.0), (0.0, -s), (0.0, 0.0)]
    return np.array([c for c in legs for _ in range(hold)], dtype=np.float32)


def summarize(R, total_steps, ckpt, tag, mode, hold, speed, fall_reason):
    if R.size == 0:
        return dict(tag=tag, ckpt=str(ckpt), mode=mode, hold=hold, command_speed=speed,
                    samples=0, total_steps=total_steps, survived_full=False, fall_reason=fall_reason)
    cmd = R[:, 1:3]; ach = R[:, 3:5]
    cmd_norm = np.linalg.norm(cmd, axis=1)
    ach_norm = np.linalg.norm(ach, axis=1)
    err = np.linalg.norm(ach - cmd, axis=1)
    mv = cmd_norm > 1e-6
    cos = np.sum(cmd[mv] * ach[mv], axis=1) / (cmd_norm[mv] * ach_norm[mv] + 1e-6) if mv.any() else np.array([])
    return dict(tag=tag, ckpt=str(ckpt), mode=mode, hold=hold, command_speed=float(speed),
                samples=int(len(R)), total_steps=int(total_steps), survived_full=bool(len(R) == total_steps),
                fall_reason=fall_reason, mean_speed=float(ach_norm[mv].mean()) if mv.any() else 0.0,
                max_speed=float(ach_norm.max()) if len(ach_norm) else 0.0,
                mean_cmd_speed=float(cmd_norm[mv].mean()) if mv.any() else 0.0,
                mean_vector_error=float(err[mv].mean()) if mv.any() else 0.0,
                rmse_vector_error=float(np.sqrt(np.mean(err[mv] ** 2))) if mv.any() else 0.0,
                mean_track=float(np.exp(-(err[mv] ** 2) / TRACK_SIGMA).mean()) if mv.any() else 0.0,
                mean_alignment=float(np.nanmean(cos)) if cos.size else 0.0,
                x_delta=float(R[-1, 5] - R[0, 5]), y_delta=float(R[-1, 6] - R[0, 6]),
                z_start=float(R[0, 7]), z_end=float(R[-1, 7]), z_min=float(R[:, 7].min()),
                up_min=float(R[:, 8].min()), last_step=int(R[-1, 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cmd")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out-npz", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--hold", type=int, default=120, help="control steps per command leg")
    ap.add_argument("--mode", choices=["square", "forward", "backward", "left", "right", "sweep", "random"], default="square")
    ap.add_argument("--speed", type=float, default=VMAX)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fall-z", type=float, default=FALL_Z)
    ap.add_argument("--min-up-z", type=float, default=MIN_UP_Z)
    args = ap.parse_args()

    Env = _build(); env = Env()
    ckpt = resolve_ckpt(args.tag, args.ckpt)
    params = pickle.load(open(ckpt, "rb"))
    net = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                         preprocess_observations_fn=running_statistics.normalize)
    inf = ppo_networks.make_inference_fn(net)(params, deterministic=True)

    # the "remote controller" program: a square of directions + a hold
    cmds = command_program(args.mode, args.hold, args.speed, args.seed)

    key = jax.random.PRNGKey(args.seed)
    st = env.reset_with_command(key, cmds[0])
    step = jax.jit(env.step)
    rec = []; fall_reason = ""
    for i in range(len(cmds)):
        # remote controller overwrites cmd; keep this functional so JIT state pytrees stay explicit
        st = st.replace(info={**st.info, "cmd": jnp.asarray(cmds[i]), "remote": jnp.array(True)})
        # rebuild obs tail with the new command so the policy sees it
        st = st.replace(obs=st.obs.at[-2:].set(jnp.asarray(cmds[i])))
        key, k = jax.random.split(key)
        act, _ = inf(st.obs, k)
        st = step(st, act)
        dx = st.pipeline_state
        up = 1.0 - 2.0 * (float(dx.qpos[4]) ** 2 + float(dx.qpos[5]) ** 2)
        rec.append([i, float(cmds[i][0]), float(cmds[i][1]),
                    float(dx.qvel[0]), float(dx.qvel[1]),
                    float(dx.qpos[0]), float(dx.qpos[1]), float(dx.qpos[2]), up])
        if float(dx.qpos[2]) < args.fall_z:
            fall_reason = f"height<{args.fall_z:.3f}"
            print(f"fell at step {i} ({fall_reason})"); break
        if up < args.min_up_z:
            fall_reason = f"up<{args.min_up_z:.3f}"
            print(f"fell at step {i} ({fall_reason})"); break
    R = np.array(rec)
    npz = Path(args.out_npz) if args.out_npz else OUT / f"{args.tag}_eval.npz"
    js = Path(args.out_json) if args.out_json else OUT / f"{args.tag}_eval_summary.json"
    np.savez(npz, t=R[:, 0], cmd_vx=R[:, 1], cmd_vy=R[:, 2],
             vx=R[:, 3], vy=R[:, 4], x=R[:, 5], y=R[:, 6], z=R[:, 7], up=R[:, 8],
             hold=args.hold, ckpt=str(ckpt), tag=args.tag, mode=args.mode)
    summary = summarize(R, len(cmds), ckpt, args.tag, args.mode, args.hold, args.speed, fall_reason)
    js.write_text(json.dumps(summary, indent=2))
    # how well did achieved velocity align with the command (on moving legs)?
    print(f"[deploy] align={summary.get('mean_alignment', 0.0):+.2f} "
          f"track={summary.get('mean_track', 0.0):.2f} "
          f"speed={summary.get('mean_speed', 0.0):.2f}/{summary.get('mean_cmd_speed', 0.0):.2f} m/s "
          f"err={summary.get('mean_vector_error', 0.0):.2f} "
          f"survived={summary['samples']}/{summary['total_steps']} ckpt={ckpt.name}")
    print("METRIC " + " ".join(f"{k}={v}" for k, v in dict(
        stage="cmd_eval", tag=args.tag, survived=int(summary["survived_full"]),
        samples=summary["samples"], total=summary["total_steps"],
        align=f"{summary.get('mean_alignment', 0.0):.3f}",
        track=f"{summary.get('mean_track', 0.0):.3f}",
        err=f"{summary.get('mean_vector_error', 0.0):.3f}").items()))
    print(f"saved {npz} and {js} -> make_command_figure.py --tag {args.tag}")


if __name__ == "__main__":
    main()
