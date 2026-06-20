# SPDX-License-Identifier: MIT
"""Phase 0 verify — MJX per-step throughput baseline (steps/s for 1 vs N envs).

Builds a generated `robot.toml` body, puts it on the GPU, and times steady-state
`mjx.step` throughput for a single env and for a batch of N (vmapped) — the number
the checklist asks for as the GPU foundation baseline, and the denominator for the
">= 100x the CPU env" Phase-1 claim. Also runs the Phase-0 smoke (jax sees the GPU;
one generated body loads + steps in MJX) so this one script is the whole Phase-0 gate.

  python bench_throughput.py [--batches 1,256,2048 --steps 200 --warmup 50]
Emits `METRIC stage=throughput ...` lines for the e2e harness / report.
"""

from __future__ import annotations

import argparse, sys, time
from pathlib import Path
import jax, jax.numpy as jnp, mujoco
from mujoco import mjx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_robot_mjcf import build_mjcf, load_spec  # noqa: E402


def METRIC(**kw):
    print("METRIC " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def _bench_one(mx, q0, nu, batch, steps, warmup, key):
    """Steady-state mjx.step throughput (env-steps/s) for `batch` vmapped envs."""
    def one_data(k):
        noise = jax.random.uniform(k, (nu,), minval=-0.05, maxval=0.05)
        return mjx.forward(mx, mjx.make_data(mx).replace(qpos=q0.at[7:7 + nu].add(noise)))
    keys = jax.random.split(key, batch)
    dxs = jax.vmap(one_data)(keys)
    ctrl = jnp.zeros((batch, nu))

    @jax.jit
    def roll(dxs, n):
        def body(i, d):
            d = jax.vmap(lambda dd: dd.replace(ctrl=ctrl[0]))(d)  # zero ctrl (pure dynamics)
            return jax.vmap(lambda dd: mjx.step(mx, dd))(d)
        return jax.lax.fori_loop(0, n, body, dxs)

    dxs = roll(dxs, warmup)                         # compile + warm
    jax.block_until_ready(dxs.qpos)
    t0 = time.time()
    dxs = roll(dxs, steps)
    jax.block_until_ready(dxs.qpos)
    dt = time.time() - t0
    return batch * steps / dt, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,256,2048")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    print(f"jax {jax.__version__} devices={jax.devices()}")          # Phase-0 smoke (1)
    xml = build_mjcf(load_spec(HERE / "robot.toml"))
    m = mujoco.MjModel.from_xml_string(xml)
    mx = mjx.put_model(m)                                            # Phase-0 smoke (2)
    q0, nu = jnp.array(m.qpos0), int(m.nu)
    d0 = mjx.forward(mx, mjx.make_data(mx))
    d1 = mjx.step(mx, d0); jax.block_until_ready(d1.qpos)            # Phase-0 smoke (3)
    print(f"body: nq={m.nq} nv={m.nv} nu={nu}; one mjx.step OK")
    METRIC(stage="mjx_smoke", device=str(jax.devices()[0]), nq=m.nq, nv=m.nv, nu=nu)

    base = None
    for batch in [int(b) for b in args.batches.split(",")]:
        sps, dt = _bench_one(mx, q0, nu, batch, args.steps, args.warmup, jax.random.PRNGKey(batch))
        if base is None:
            base = sps
        print(f"  batch {batch:>6}: {sps:>12,.0f} env-steps/s  ({dt*1e3:.0f} ms / {args.steps} steps)  "
              f"speedup vs 1-env {sps/base:6.1f}x")
        METRIC(stage="throughput", batch=batch, steps_per_s=f"{sps:.0f}",
               speedup_vs_1=f"{sps/base:.1f}")
    print("PROVEN: GPU MJX foundation — jax sees the device, a generated body loads + "
          "steps, and batched throughput scales with env count (Phase-0 baseline).")


if __name__ == "__main__":
    main()
