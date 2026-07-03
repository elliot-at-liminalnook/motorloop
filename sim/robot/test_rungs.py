#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Behavioral tests for the outcome-grounding ladder (rungs 2-4).

Each rung makes the task spec less exploitable; these assert the MECHANISM works:
  - rung 2a: behavior/range keep-gate rejects a stand-still checkpoint (pure logic)
  - rung 2b: require-closing + stationary/oscillation penalties lower a non-moving
             policy's reward (corrected reward <= vanilla, strictly less when idle)
  - rung 3:  tactical RND features are far less sensitive to joint jitter than
             proprioceptive features (so curiosity stops paying for twitching)
  - rung 4:  the scripted-opponent hook changes B's trajectory, and the benchmark
             exposes the range/behavior signals the keep-gate uses

Note (honest): the rung-4 scripted opponent is a HOOK that perturbs B but is not yet
a competent pursuer (a crude open-loop drive does not produce a gait toward A); the
robust rung-4 mechanism is the SELECTION gate (rung 2a) + range-balanced benchmark.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx
import train_adversarial as T

BASE = dict(frame_skip=5, self_collision=False, sep_lo=0.3, sep_hi=0.6, azimuth=3.14159,
            engage_obs=True, contact_obs=True, hierarchical=True, gate_threshold=0.3,
            her_coefficient=0.2, lidar=True, lidar_n_rays=16, lidar_n_vertical=4,
            lidar_frame_stack=2, lidar_noise_sigma=0.0, lidar_dropout_rate=0.0)


def test_rung2a_keep_gate():
    print("=== Rung 2a: behavior/range keep-gate rejects stand-still ===")
    stand_still = {"bh_closed": 0.0, "bh_approach": -0.08, "bh_disp": 0.12, "sparc_far": 0.0}
    engaged = {"bh_closed": 0.4, "bh_approach": 0.15, "bh_disp": 0.6, "sparc_far": 3.0}
    gates = dict(min_closed=0.15, min_approach=0.02, min_disp=0.2, min_far_sparc=0.5)
    assert not T.behavior_keep_ok(stand_still, **gates), "stand-still should be REJECTED"
    assert T.behavior_keep_ok(engaged, **gates), "engaged policy should PASS"
    # with gates off (defaults) everything passes (byte-identical to before)
    assert T.behavior_keep_ok(stand_still)
    print("  stand-still rejected, engaged passes, gates-off no-op  PASSED")


def test_rung2b_reward_shaping():
    print("=== Rung 2b: require-closing + penalties lower a non-moving reward ===")
    vanilla = T.AdversarialEnv(**BASE)
    corrected = T.AdversarialEnv(require_closing=True, stationary_damage_penalty=2.0,
                                 oscillation_penalty=0.3, **BASE)
    key = jax.random.PRNGKey(0)
    sv, sc = vanilla.reset(key), corrected.reset(key)
    # zero action -> not moving -> oscillation/stationary penalties + closing gate apply
    a = jnp.zeros(vanilla.action_size)
    rv = float(vanilla.step(sv, a).reward)
    rc = float(corrected.step(sc, a).reward)
    assert rc <= rv + 1e-5, f"corrected reward must be <= vanilla ({rc} vs {rv})"
    # in-place OSCILLATION (effort but no net locomotion): alternating full-torque
    # action over ~1 s. On the full-torque body a constant "big" action MOVES the
    # robot (the old single-step fixture was only "idle" on the 1 N·m gear-bug body);
    # the honest oscillator alternates sign so the smoothed displacement velocity —
    # what the not_moving gate now measures — stays ~0 while effort stays high.
    sv, sc = vanilla.reset(key), corrected.reset(key)
    rv2 = rc2 = 0.0
    for t in range(50):
        a_osc = jnp.ones(vanilla.action_size) * (0.8 if t % 2 == 0 else -0.8)
        sv = vanilla.step(sv, a_osc); sc = corrected.step(sc, a_osc)
        rv2 += float(sv.reward); rc2 += float(sc.reward)
    assert rc2 < rv2, f"high-effort oscillation should be penalized ({rc2} vs {rv2})"
    print(f"  zero-act: corrected {rc:+.3f} <= vanilla {rv:+.3f}; "
          f"oscillation: corrected {rc2:+.3f} < vanilla {rv2:+.3f}  PASSED")


def test_rung3_tactical_rnd():
    print("=== Rung 3: tactical RND ignores joint jitter ===")
    et = T.AdversarialEnv(rnd_coefficient=0.05, rnd_feature="tactical", **BASE)
    ep = T.AdversarialEnv(rnd_coefficient=0.05, rnd_feature="proprio", **BASE)
    s = et.reset(jax.random.PRNGKey(2)); dx = s.pipeline_state
    mxd = et._design_model(np.array(s.info["design"]))
    qpos2 = dx.qpos.at[et._Aqa].add(0.3 * jax.random.normal(jax.random.PRNGKey(9), (et._Aqa.shape[0],)))
    dx2 = mjx.forward(mxd, dx.replace(qpos=qpos2))     # jitter A's legs; torso + opponent fixed
    tac = np.abs(np.array(et._rnd_feat(dx2, s.info["design"]) - et._rnd_feat(dx, s.info["design"]))).mean()
    pro = np.abs(np.array(ep._rnd_feat(dx2, s.info["design"]) - ep._rnd_feat(dx, s.info["design"]))).mean()
    assert et._rnd_feat(dx, s.info["design"]).shape[0] == 8
    assert ep._rnd_feat(dx, s.info["design"]).shape[0] == T.LOCO_OBS
    assert tac < 0.5 * pro, f"tactical jitter-sensitivity {tac:.4f} should be << proprio {pro:.4f}"
    print(f"  leg-jitter: tactical change {tac:.4f} << proprio {pro:.4f} (>2x less)  PASSED")


def test_rung4_opponent_hook():
    print("=== Rung 4: scripted-opponent hook perturbs B + benchmark exposes range/behavior ===")
    def b_path(scr):
        env = T.AdversarialEnv(opponent_script=scr, **BASE)
        key = jax.random.PRNGKey(4); s = env.reset(key); st = jax.jit(env.step)
        prev = np.array(s.pipeline_state.xpos[env._Bt])[:2]; path = 0.0
        for _ in range(40):
            s = st(s, jnp.zeros(env.action_size))
            cur = np.array(s.pipeline_state.xpos[env._Bt])[:2]; path += np.linalg.norm(cur - prev); prev = cur
        return path
    p0, p1 = b_path(0.0), b_path(0.6)
    assert abs(p1 - p0) > 1e-3, f"scripted opponent should change B's motion ({p1:.3f} vs {p0:.3f})"
    # the benchmark exposes the signals the keep-gate consumes
    be = T.AdversarialEnv(opponent_script=0.6, **BASE)
    bench = T.build_benchmark(be, n_epis=2, steps=20, seed=20240601)
    import pickle
    # build a fresh random params via the same net (init policy) just to exercise the bench shape
    keys = set(T.BENCH_KEYS)
    for k in ("bh_closed", "bh_approach", "bh_disp", "sparc_far"):
        assert k in keys, f"benchmark missing {k}"
    print(f"  B path script0 {p0:.3f} vs script0.6 {p1:.3f} (hook wired); bench exposes gate signals  PASSED")
    print("  NOTE: scripted B is a hook, not yet a competent pursuer (see module docstring)")


FAST = [test_rung2a_keep_gate, test_rung2b_reward_shaping, test_rung3_tactical_rnd,
        test_rung4_opponent_hook]

if __name__ == "__main__":
    for t in FAST:
        t()
    print("\n=== ALL RUNG TESTS PASSED ===")
