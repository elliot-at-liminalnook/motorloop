# SPDX-License-Identifier: MIT
"""M4 gate: the training-loop integration demo runs rollout->update cycles on
the fused path with the (nworld, ...) buffers consumed zero-copy.

Checks the PLUMBING (the M4 deliverable), not RL quality: shapes, finiteness,
parameters actually updating, and — on CPU — that obs_numpy()/reward_numpy()
really alias the warp buffers (zero-copy contract for the dlpack story)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))      # sim/robot: warplayer + gen_robot_mjcf + constants

from warplayer.fused import FightLayer  # noqa: E402
from warplayer.m4_train_demo import LinearPolicy, run  # noqa: E402

wp.init()


def test_rollout_update_cycle_runs():
    history, pol, lay = run(nworld=8, horizon=6, iters=3, lidar=False,
                            seed=0, verbose=False)
    assert len(history) == 3
    assert np.isfinite(history).all()
    assert np.abs(pol.W).max() > 0.0, "REINFORCE update never moved the parameters"
    assert lay.obs.shape == (8, lay.obs_dim)
    assert lay.reward.shape == (8,)


def test_zero_copy_buffer_aliasing_cpu():
    """On CPU, wp.array.numpy() must alias the kernel-written buffer — the
    learner reads what the fused graph wrote with no copy (dlpack contract)."""
    if wp.get_device().is_cuda:
        pytest.skip("aliasing check is the CPU zero-copy contract")
    lay = FightLayer(nworld=4, mode="fused", lidar=False, seed=1)
    view = lay.obs_numpy()
    before = view[0, 0].copy()
    lay.set_actions(np.full((4, lay.idx.nuA), 0.4))
    lay.step_fused()                        # kernels write the SAME memory
    after = view[0, 0]
    assert view.base is not None or view.flags["OWNDATA"] is False
    assert before != after, "obs view did not observe the kernel write (copy, not alias)"


def test_policy_consumes_lidar_obs_dim():
    history, pol, lay = run(nworld=4, horizon=3, iters=1, lidar=True,
                            seed=2, verbose=False)
    from constants import LOCO_OBS
    assert lay.obs_dim == LOCO_OBS + 144
    assert pol.W.shape == (lay.obs_dim, lay.idx.nuA)
    assert np.isfinite(history).all()


def test_update_direction_sanity():
    """A hand-built case: positive-advantage world pushes W toward its own
    obs·xi outer product (sign check of the REINFORCE estimator)."""
    pol = LinearPolicy(obs_dim=2, act_dim=1, sigma=0.5, scale=1.0, lr=1.0, seed=0)
    obs = np.array([[1.0, 0.0], [0.0, 0.0]])          # world 0 informative, world 1 blank
    xi = np.array([[1.0], [0.0]])
    ret = np.array([1.0, 0.0])                        # world 0 did better
    pol.update([obs], [xi], ret)
    assert pol.W[0, 0] > 0.0                          # reinforce the action taken there
    assert pol.W[1, 0] == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
