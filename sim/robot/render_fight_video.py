# SPDX-License-Identifier: MIT
"""Render a 1v1 'fight' VIDEO from two trained self-play checkpoints.

A (we drive its actions) vs B (frozen opponent via the env's opp_infer hook). Rolls out the real
MJX AdversarialEnv (obs/ctrl/striker/contact exactly as trained), collects qpos, then renders each
frame on a CPU MuJoCo model and encodes an mp4.

  MUJOCO_GL=egl python render_fight_video.py --a spr6_best.pkl --b spr3_best.pkl --out fight.mp4
"""
import argparse, os
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import mujoco
import imageio

import train_adversarial as T
from gen_robot_mjcf import build_match

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default="fight.mp4")
    ap.add_argument("--steps", type=int, default=220)
    ap.add_argument("--sep", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--label", default="", help="overlay text (e.g. the milestone tag); reads {label}.json sidecar for win/ratio")
    args = ap.parse_args()

    infA = T.load_opponent(args.a)
    infB = T.load_opponent(args.b)

    env = T.AdversarialEnv(frame_skip=5, striker=True, sep=args.sep, azimuth=0.3,
                           opponent="frozen", opp_infer=infB)
    step = jax.jit(env.step)
    key = jax.random.PRNGKey(args.seed)
    state = env.reset_with(key, jnp.full(3, 0.5))          # nominal body = matches the CPU render model

    qpos_log, cum_dealt, cum_taken = [], 0.0, 0.0
    for t in range(args.steps):
        key, k = jax.random.split(key)
        act, _ = infA(state.obs, k)
        state = step(state, act)
        qpos_log.append(np.asarray(state.pipeline_state.qpos))
        cum_dealt += float(state.metrics.get("dealt", 0.0))
        cum_taken += float(state.metrics.get("taken", 0.0))
    print(f"rollout: {len(qpos_log)} steps  cum dealt={cum_dealt:.3f} taken={cum_taken:.3f} "
          f"ratio={cum_dealt/max(cum_taken,1e-6):.2f}")

    m = mujoco.MjModel.from_xml_string(build_match(T.SPEC, T.SPEC, sep=args.sep, self_collision=True,
                                                   striker=True, striker_b=True))  # B armed too (opp=frozen)
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, args.h, args.w)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.25]; cam.distance = 2.2; cam.elevation = -18; cam.azimuth = 120

    frames = []
    for q in qpos_log:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=cam)
        frames.append(renderer.render())

    # optional text overlay (evolution label): step + held-out metrics from the sidecar + live ratio
    if args.label:
        txt = args.label
        try:
            import json, re
            mstep = re.search(r"ms_(\d+)", args.label)
            sc = Path(args.a).with_suffix(".json")
            extra = ""
            if sc.exists():
                s = json.loads(sc.read_text())
                extra = f"  win {s.get('win','?')} ratio {s.get('ratio','?')} surv {s.get('survival','?')}"
            stepM = f"step {int(mstep.group(1))/1e6:.1f}M" if mstep else args.label
            txt = f"{stepM}{extra}   (live dmg-ratio {cum_dealt/max(cum_taken,1e-6):.2f})"
        except Exception:
            pass
        try:
            from PIL import Image, ImageDraw
            for i, fr in enumerate(frames):
                im = Image.fromarray(fr); dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
                dr.text((4, 4), txt, fill=(255, 255, 255))
                frames[i] = np.asarray(im)
        except Exception as e:
            print(f"label skipped ({e})")

    out = args.out if os.path.isabs(args.out) else str(HERE / args.out)
    try:
        imageio.mimsave(out, frames, fps=args.fps, codec="libx264", quality=8)
        print(f"wrote {out} ({len(frames)} frames @ {args.fps}fps)")
    except Exception as e:                                  # mp4 backend hiccup -> GIF fallback
        gif = os.path.splitext(out)[0] + ".gif"
        imageio.mimsave(gif, frames[::2], duration=2.0 / args.fps)
        print(f"mp4 failed ({e}); wrote {gif} instead")


if __name__ == "__main__":
    main()
