# SPDX-License-Identifier: MIT
"""L-R1 — MJX (JAX) vs mujoco_warp per-step throughput on the project scenes.

One harness, two engines, three scenes; measures steady-state env-steps/s AFTER
warmup (compile excluded) and prints a single parseable RESULT line. Runs on CPU
(warp falls back to its LLVM backend, jax to its CPU backend) so it can be smoke-
tested locally at small --nenv, and unchanged on an A100 pod at large --nenv.

  # local smoke (each engine in its own venv):
  .venv-warp/bin/python sim/robot/bench_warp_vs_mjx.py --scene fight --engine warp --nenv 8 --steps 200
  .venv-sim/bin/python  sim/robot/bench_warp_vs_mjx.py --scene fight --engine mjx  --nenv 8 --steps 200

  # A100 pod:
  .venv-warp/bin/python sim/robot/bench_warp_vs_mjx.py --scene fight --engine warp --nenv 4096 --steps 1000
  .venv-sim/bin/python  sim/robot/bench_warp_vs_mjx.py --scene fight --engine mjx  --nenv 4096 --steps 1000

Engines are imported lazily so the script runs in either venv:
  mjx  -> .venv-sim  (mujoco 3.9.0 + jax 0.6.2, the pinned training stack)
  warp -> .venv-warp (mujoco 3.10.0 + mujoco-warp 3.10.0.1 + warp-lang 1.14.0)

Scenes:
  single -> sim/robot/model.xml            (one paramquad)
  fight  -> build_match(spec, spec, sep=1.2, striker=True, striker_b=True)
  mesh   -> sim/robot/mesh_robot.xml       (equality connects + worm frictionloss)

All envs start from qpos0 with small deterministic qvel noise (quat-safe; keeps
worlds decorrelated without touching free-joint quaternions). Zero ctrl.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

QVEL_NOISE = 0.05  # rad/s scale, deterministic per env


def build_scene_xml(scene: str) -> tuple[str, str]:
    """Returns (xml_string_or_path, kind) with kind in {'path','string'}."""
    if scene == "single":
        return str(HERE / "model.xml"), "path"
    if scene == "mesh":
        return str(HERE / "mesh_robot.xml"), "path"
    if scene == "fight":
        from gen_robot_mjcf import build_match, load_spec  # noqa: PLC0415
        spec = load_spec(HERE / "robot.toml")
        return build_match(spec, spec, sep=1.2, striker=True, striker_b=True), "string"
    raise ValueError(scene)


def load_model(scene: str):
    import mujoco  # noqa: PLC0415
    src, kind = build_scene_xml(scene)
    if kind == "path":
        # read the file and load via from_xml_string: V.6 guardrail — envs/tools build
        # from spec or explicit strings so a stale on-disk artifact can't sneak in.
        src, kind = open(src).read(), "string"
    return mujoco.MjModel.from_xml_string(src)


def qvel_noise(nenv: int, nv: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-QVEL_NOISE, QVEL_NOISE, size=(nenv, nv))


def bench_mjx(mjm, nenv: int, steps: int, warmup: int, device: str | None):
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from mujoco import mjx  # noqa: PLC0415

    dev = jax.devices(device)[0] if device else jax.devices()[0]
    with jax.default_device(dev):
        mx = mjx.put_model(mjm)
        d0 = mjx.make_data(mx)
        vel = jnp.asarray(qvel_noise(nenv, mjm.nv))
        dxs = jax.vmap(lambda v: mjx.forward(mx, d0.replace(qvel=v)))(vel)

        @jax.jit
        def roll(d, n):
            return jax.lax.fori_loop(0, n, lambda i, dd: jax.vmap(lambda x: mjx.step(mx, x))(dd), d)

        t0 = time.time()
        dxs = roll(dxs, warmup)                 # jit compile + warm
        jax.block_until_ready(dxs.qpos)
        compile_s = time.time() - t0
        t0 = time.time()
        dxs = roll(dxs, steps)
        jax.block_until_ready(dxs.qpos)
        wall = time.time() - t0
        assert not bool(jnp.isnan(dxs.qpos).any()), "NaN qpos after benchmark rollout"
    return wall, compile_s, str(dev)


def bench_warp(mjm, nenv: int, steps: int, warmup: int, device: str | None,
               nconmax: int | None, njmax: int | None):
    import mujoco  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415
    import mujoco_warp as mjwp  # noqa: PLC0415

    wp.init()
    dev = wp.get_device(device) if device else wp.get_device()
    with wp.ScopedDevice(dev):
        mjd0 = mujoco.MjData(mjm)
        mujoco.mj_resetData(mjm, mjd0)
        mujoco.mj_forward(mjm, mjd0)
        m = mjwp.put_model(mjm)
        d = mjwp.put_data(mjm, mjd0, nworld=nenv, nconmax=nconmax, njmax=njmax)
        d.qvel.assign(qvel_noise(nenv, mjm.nv).astype(np.float32))

        t0 = time.time()
        graph = None
        if dev.is_cuda:                          # capture once, launch many (pod path)
            mjwp.step(m, d)                      # load modules before capture
            wp.synchronize()
            with wp.ScopedCapture() as cap:
                mjwp.step(m, d)
            graph = cap.graph
        step = (lambda: wp.capture_launch(graph)) if graph else (lambda: mjwp.step(m, d))
        for _ in range(warmup):                  # first calls compile kernels (LLVM/NVRTC)
            step()
        wp.synchronize()
        compile_s = time.time() - t0

        t0 = time.time()
        for _ in range(steps):
            step()
        wp.synchronize()
        wall = time.time() - t0
        assert not np.isnan(d.qpos.numpy()).any(), "NaN qpos after benchmark rollout"
    return wall, compile_s, str(dev)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", choices=("single", "fight", "mesh"), required=True)
    ap.add_argument("--engine", choices=("mjx", "warp"), required=True)
    ap.add_argument("--nenv", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", default=None,
                    help="engine device: mjx e.g. 'cpu'/'gpu', warp e.g. 'cpu'/'cuda:0' (default: engine's default)")
    ap.add_argument("--nconmax", type=int, default=None, help="warp only: contact-pool override")
    ap.add_argument("--njmax", type=int, default=None, help="warp only: constraint-row override")
    args = ap.parse_args()

    mjm = load_model(args.scene)
    if args.engine == "mjx":
        wall, compile_s, dev = bench_mjx(mjm, args.nenv, args.steps, args.warmup, args.device)
    else:
        wall, compile_s, dev = bench_warp(mjm, args.nenv, args.steps, args.warmup, args.device,
                                          args.nconmax, args.njmax)

    sps = args.nenv * args.steps / wall
    print(f"RESULT bench=warp_vs_mjx scene={args.scene} engine={args.engine} nenv={args.nenv} "
          f"steps={args.steps} warmup={args.warmup} device={dev!r} env_steps_per_s={sps:.1f} "
          f"wall_s={wall:.3f} warmup_s={compile_s:.2f}", flush=True)


if __name__ == "__main__":
    main()
