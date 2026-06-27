# SPDX-License-Identifier: MIT
"""Render a no-GL top-down fight rollout from fighter checkpoints.

This is the headless fallback for pods that can run MJX but cannot create an
EGL/OSMesa MuJoCo rendering context. It rolls out the same `AdversarialEnv` as
`render_fight_video.py`, then draws A/B torso trails plus A striking-body
positions in a compact top-down video/GIF.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import train_adversarial as T


def _rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return rgba[:, :, :3].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", default="")
    ap.add_argument("--opponent", choices=["passive", "frozen"], default="frozen")
    ap.add_argument("--out", default="fight_trace.mp4")
    ap.add_argument("--steps", type=int, default=220)
    ap.add_argument("--sep", type=float, default=0.5)
    ap.add_argument("--sep-lo", type=float, default=None)
    ap.add_argument("--sep-hi", type=float, default=None)
    ap.add_argument("--azimuth", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode-index", type=int, default=-1,
                    help="if >=0, split PRNGKey(seed) like the benchmark and render this episode key")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--engage-obs", action="store_true")
    ap.add_argument("--lean-contacts", action="store_true")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    inf_a = T.load_opponent(args.a)
    inf_b = None
    if args.opponent == "frozen":
        if not args.b:
            raise SystemExit("--opponent frozen requires --b")
        inf_b = T.load_opponent(args.b)
    env = T.AdversarialEnv(
        frame_skip=5,
        striker=True,
        sep=args.sep,
        sep_lo=args.sep if args.sep_lo is None else args.sep_lo,
        sep_hi=args.sep if args.sep_hi is None else args.sep_hi,
        azimuth=args.azimuth,
        self_collision=not args.lean_contacts,
        opponent=args.opponent,
        opp_infer=inf_b,
        engage_obs=args.engage_obs,
    )
    step = jax.jit(env.step)
    key = jax.random.PRNGKey(args.seed)
    if args.episode_index >= 0:
        key = jax.random.split(key, args.episode_index + 1)[args.episode_index]
    state = env.reset(key)

    a_xy, b_xy, strike_xy, dealt, taken = [], [], [], [], []
    for _ in range(args.steps):
        key, sk = jax.random.split(key)
        act, _ = inf_a(state.obs, sk)
        state = step(state, act)
        ps = state.pipeline_state
        a_xy.append(np.asarray(ps.xpos[env._At][:2]))
        b_xy.append(np.asarray(ps.xpos[env._Bt][:2]))
        strike_xy.append(np.asarray(ps.xpos[env._Astrike][:, :2]))
        dealt.append(float(state.metrics.get("dealt", 0.0)))
        taken.append(float(state.metrics.get("taken", 0.0)))

    a_xy = np.asarray(a_xy)
    b_xy = np.asarray(b_xy)
    strike_xy = np.asarray(strike_xy)
    all_xy = np.concatenate([a_xy, b_xy, strike_xy.reshape(-1, 2)], axis=0)
    lo = np.nanmin(all_xy, axis=0)
    hi = np.nanmax(all_xy, axis=0)
    ctr = 0.5 * (lo + hi)
    span = float(max(np.max(hi - lo), 1.0))
    lim = span * 0.58

    frames = []
    dpi = 100
    figsize = (args.w / dpi, args.h / dpi)
    for i in range(args.steps):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor("#111318")
        ax.set_facecolor("#171a20")
        ax.plot(a_xy[: i + 1, 0], a_xy[: i + 1, 1], color="#29b6f6", lw=2.0, label="A trail")
        ax.plot(b_xy[: i + 1, 0], b_xy[: i + 1, 1], color="#ffb74d", lw=2.0, label="B trail")
        ax.scatter(strike_xy[i, :, 0], strike_xy[i, :, 1], s=18, color="#ab47bc", alpha=0.8, label="A weapons")
        ax.scatter(a_xy[i, 0], a_xy[i, 1], s=90, color="#00acc1", edgecolor="white", linewidth=0.8)
        ax.scatter(b_xy[i, 0], b_xy[i, 1], s=90, color="#f9a825", edgecolor="white", linewidth=0.8)
        ax.set_xlim(ctr[0] - lim, ctr[0] + lim)
        ax.set_ylim(ctr[1] - lim, ctr[1] + lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="white", alpha=0.12, linewidth=0.6)
        ax.tick_params(colors="#cfd8dc", labelsize=7)
        title = args.label or Path(args.a).stem
        ax.set_title(
            f"{title}  step {i + 1}/{args.steps}  "
            f"cum dealt={sum(dealt[:i+1]):.3f} taken={sum(taken[:i+1]):.3f}",
            color="white",
            fontsize=9,
        )
        ax.legend(loc="upper right", fontsize=7, framealpha=0.25)
        fig.tight_layout(pad=0.5)
        frames.append(_rgb(fig))
        plt.close(fig)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(out, frames, fps=args.fps, codec="libx264", quality=8)
        print(f"wrote {out} ({len(frames)} top-down frames @ {args.fps}fps)")
    except Exception as e:
        gif = out.with_suffix(".gif")
        imageio.mimsave(gif, frames[::2], duration=2.0 / args.fps)
        print(f"mp4 failed ({e}); wrote {gif}")


if __name__ == "__main__":
    os.environ.pop("MUJOCO_GL", None)
    main()
