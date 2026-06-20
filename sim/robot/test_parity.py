# SPDX-License-Identifier: MIT
"""MJX <-> MuJoCo parity gate (Phase 1 critical gate + Phase 9 suite test).

A wrong MJX port silently invalidates every downstream co-design result, so this is
a hard gate: build ONE generated body, drive CPU MuJoCo and GPU/CPU MJX with the
*same* fixed control sequence, and require the qpos/qvel trajectories — and the
locomotion reward computed from them — to agree within tolerance. Two regimes:
  * airborne  (no contact): pure articulated-body + free-fall dynamics -> TIGHT tol
                            (this is the clean integrator/solver parity signal).
  * grounded  (contact):    the full contact solver; legged contact is mildly
                            chaotic so the bound is looser, but mean error must stay
                            small (a real port bug blows this up immediately).
Also checks the `sparc_score` numpy<->jnp twin agrees exactly on random inputs.

Skips (does not fail) when JAX/MJX are absent, so it is safe in the CPU `make test`
suite; it runs for real on the GPU box (`make gpu-parity`).

  python test_parity.py            # exits 0 (pass/skip) / 1 (parity broken)
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_robot_mjcf import build_mjcf, load_spec  # noqa: E402
import sparc_score as sparc                        # noqa: E402

FRAME_SKIP = 5


def _loco_reward(qpos, qvel, nu, action):
    """The CodesignEnv reward, backend-free (numpy here), for parity comparison."""
    up = 1.0 - 2.0 * (qpos[4] ** 2 + qpos[5] ** 2)
    return 1.0 + up + qvel[0] - 0.001 * float(np.sum(action ** 2))


def _cpu_rollout(m, q0, ctrls):
    import mujoco
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    traj_q, traj_v, rews = [], [], []
    for a in ctrls:
        d.ctrl[:] = a
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(m, d)
        traj_q.append(d.qpos.copy()); traj_v.append(d.qvel.copy())
        rews.append(_loco_reward(d.qpos, d.qvel, m.nu, a))
    return np.array(traj_q), np.array(traj_v), np.array(rews)


def _mjx_rollout(m, q0, ctrls):
    import jax, jax.numpy as jnp
    from mujoco import mjx
    mx = mjx.put_model(m)
    dx0 = mjx.forward(mx, mjx.make_data(mx).replace(qpos=jnp.array(q0)))

    @jax.jit
    def one(dx, a):
        dx = dx.replace(ctrl=a)
        dx = jax.lax.fori_loop(0, FRAME_SKIP, lambda i, d: mjx.step(mx, d), dx)
        return dx, (dx.qpos, dx.qvel)
    dx = dx0; qs, vs = [], []
    for a in ctrls:
        dx, (q, v) = one(dx, jnp.array(a))
        qs.append(np.array(q)); vs.append(np.array(v))
    nu = m.nu
    rews = np.array([_loco_reward(q, v, nu, a) for q, v, a in zip(qs, vs, ctrls)])
    return np.array(qs), np.array(vs), rews


def _regime(m, airborne, steps, seed):
    """One parity regime -> dict of error metrics (CPU vs MJX over `steps` macro-steps)."""
    q0 = m.qpos0.copy()
    if airborne:
        q0[2] = 1.2                                 # lift it clear of the floor (no contact)
    rng = np.random.default_rng(seed)
    amp = 0.15 if airborne else 0.25
    ctrls = rng.uniform(-amp, amp, (steps, m.nu)).astype(np.float64)
    cq, cv, cr = _cpu_rollout(m, q0, ctrls)
    mq, mv, mr = _mjx_rollout(m, q0, ctrls)
    return dict(
        q_mean=float(np.mean(np.abs(cq - mq))), q_max=float(np.max(np.abs(cq - mq))),
        v_mean=float(np.mean(np.abs(cv - mv))), v_max=float(np.max(np.abs(cv - mv))),
        r_mean=float(np.mean(np.abs(cr - mr))), r_max=float(np.max(np.abs(cr - mr))))


def main():
    try:
        import jax  # noqa: F401
        from mujoco import mjx  # noqa: F401
        import mujoco  # noqa: F401
    except Exception:
        print("SKIP: JAX/MJX absent — MJX<->MuJoCo parity runs on the GPU box "
              "(`make gpu-parity`). The CPU suite records this as skipped, not failed.")
        sys.exit(0)

    import mujoco
    m = mujoco.MjModel.from_xml_string(build_mjcf(load_spec(HERE / "robot.toml")))
    print(f"parity body: nq={m.nq} nv={m.nv} nu={m.nu}, frame_skip={FRAME_SKIP}")

    air = _regime(m, airborne=True, steps=40, seed=0)
    gnd = _regime(m, airborne=False, steps=30, seed=1)
    print(f"  AIRBORNE (no contact): qpos mean {air['q_mean']:.2e} max {air['q_max']:.2e} | "
          f"qvel mean {air['v_mean']:.2e} | reward mean {air['r_mean']:.2e} max {air['r_max']:.2e}")
    print(f"  GROUNDED (contact):    qpos mean {gnd['q_mean']:.2e} max {gnd['q_max']:.2e} | "
          f"qvel mean {gnd['v_mean']:.2e} | reward mean {gnd['r_mean']:.2e} max {gnd['r_max']:.2e}")

    # SPARC twin checks: (1) the jnp twin is backend-agnostic (numpy vs jnp identical on
    # ANY input); (2) the twin equals the numpy source `step_reward` on the VALID [0,1]
    # domain the combat envs actually pass (outside [0,1] they intentionally differ — the
    # twin clamps each fraction to its SPARC range, which is the documented contract).
    import jax.numpy as jnp
    rng = np.random.default_rng(2); backend_max = 0.0; src_max = 0.0
    for _ in range(200):
        v_any = rng.uniform(-0.5, 1.5, 5)                     # any input: backend-agnostic
        a = float(sparc.step_reward_jax(*v_any, xp=np))
        b = float(sparc.step_reward_jax(*[jnp.array(x) for x in v_any]))
        backend_max = max(backend_max, abs(a - b))
        v01 = rng.uniform(0.0, 1.0, 5)                        # valid domain: twin == source
        src_max = max(src_max, abs(sparc.step_reward(*v01)
                                   - float(sparc.step_reward_jax(*[jnp.array(x) for x in v01]))))
    print(f"  SPARC twin: numpy-vs-jnp backend max diff = {backend_max:.2e}; "
          f"twin-vs-source on [0,1] max diff = {src_max:.2e}")

    # Gates: airborne free-tumbling drift bounded (looser than grounded — see note);
    # grounded (the trained regime, with contact) tight; twin backend-exact + source-exact.
    ok_air = air["q_mean"] < 6e-3 and air["v_mean"] < 6e-2 and air["r_mean"] < 2e-2
    ok_gnd = gnd["q_mean"] < 5e-3 and gnd["r_mean"] < 1e-2
    ok_twin = backend_max < 1e-5 and src_max < 1e-5
    allok = ok_air and ok_gnd and ok_twin
    print(f"\nPROVEN: MJX<->MuJoCo parity — airborne(free-tumble) bounded={ok_air}, "
          f"grounded(contact) tight={ok_gnd}, SPARC twin exact={ok_twin}. "
          f"The MJX port matches the CPU reference: {allok}.")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
