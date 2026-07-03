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
    ap.add_argument("--b", default="")
    ap.add_argument("--out", default="fight.mp4")
    ap.add_argument("--steps", type=int, default=220)
    ap.add_argument("--sep", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--label", default="", help="overlay text (e.g. the milestone tag); reads {label}.json sidecar for win/ratio")
    # asymmetric / lidar checkpoints: B has no lidar sensors, so a lidar policy cannot
    # drive a frozen B. --passive-b renders the trained policy (A) against a passive B,
    # building an env whose obs matches the checkpoint (lidar/her dims). Forwarded
    # train-only flags are ignored (parse_known_args).
    ap.add_argument("--passive-b", action="store_true",
                    help="render A vs a passive B (required for lidar/asymmetric checkpoints)")
    ap.add_argument("--lidar-obs", action="store_true")
    ap.add_argument("--lidar-n-rays", type=int, default=128)
    ap.add_argument("--lidar-n-vertical", type=int, default=16)
    ap.add_argument("--lidar-max-range", type=float, default=2.0)
    ap.add_argument("--lidar-frame-stack", type=int, default=3)
    ap.add_argument("--lidar-latency-steps", type=int, default=0)
    ap.add_argument("--hierarchical", action="store_true")
    ap.add_argument("--gate-threshold", type=float, default=0.3)
    ap.add_argument("--engage-obs", action="store_true")
    ap.add_argument("--contact-obs", action="store_true")
    ap.add_argument("--her-coefficient", type=float, default=0.0)
    args, _ignored = ap.parse_known_args()

    # Build an env whose observation matches the trained checkpoint. For a lidar
    # checkpoint, B cannot be driven by the policy (no B-side lidar) -> passive B.
    env_kw = dict(frame_skip=5, striker=True, sep=args.sep, azimuth=0.3,
                  engage_obs=args.engage_obs, contact_obs=args.contact_obs,
                  hierarchical=args.hierarchical, gate_threshold=args.gate_threshold,
                  her_coefficient=args.her_coefficient)
    if args.lidar_obs:
        env_kw.update(lidar=True, lidar_n_rays=args.lidar_n_rays,
                      lidar_n_vertical=args.lidar_n_vertical, lidar_max_range=args.lidar_max_range,
                      lidar_frame_stack=args.lidar_frame_stack,
                      lidar_latency_steps=args.lidar_latency_steps,
                      lidar_noise_sigma=0.0, lidar_dropout_rate=0.0)  # deterministic render
    infB = None
    if args.passive_b or not args.b:
        env = T.AdversarialEnv(opponent="passive", **env_kw)
    else:
        # non-lidar self-play: B is a flat frozen snapshot
        infB = T.load_opponent(args.b)
        env = T.AdversarialEnv(opponent="frozen", opp_infer=infB, **env_kw)
    # Build A's inference matching the env's obs structure (dict for lidar) so the
    # asymmetric normalizer + dict obs are consumed correctly.
    infA = T.load_policy(args.a, env.observation_size, env.action_size)
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

    # Match the env's qpos layout: B has striker DOFs only when armed (frozen opponent).
    m = mujoco.MjModel.from_xml_string(build_match(T.SPEC, T.SPEC, sep=args.sep, self_collision=True,
                                                   striker=True, striker_b=infB is not None))
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
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    try:
        imageio.mimsave(out, frames, fps=args.fps, codec="libx264", quality=8)
        print(f"wrote {out} ({len(frames)} frames @ {args.fps}fps)")
    except Exception as e:                                  # mp4 backend hiccup -> GIF fallback
        gif = os.path.splitext(out)[0] + ".gif"
        imageio.mimsave(gif, frames[::2], duration=2.0 / args.fps)
        print(f"mp4 failed ({e}); wrote {gif} instead")


if __name__ == "__main__":
    main()
