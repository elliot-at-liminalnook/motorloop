# SPDX-License-Identifier: MIT
"""M4 — minimal training-loop integration over the fused path (§10c M4).

Demonstrates the PLUMBING the fused layer exists for: rollout -> update cycles
where the learner consumes the (nworld, obs_dim) obs and (nworld,) reward
buffers that the fused kernels write, with NO device->host copy of the
simulation state itself. On CPU `wp.array.numpy()` aliases the warp buffer
(zero-copy view); on CUDA the same buffers go to torch via
`torch.from_dlpack(layer.obs)` — the learner framework is interchangeable,
which is the point (".. gradients happen wherever").

The policy is a deliberately tiny linear-Gaussian REINFORCE learner in pure
numpy (no torch/optax in .venv-warp, and RL quality is explicitly NOT the
goal): a = clip(obs @ W * scale + sigma * xi). Actions are the ONE remaining
host->device write per control step (the policy lives on host here); in
production the policy output would already be a device buffer written through
dlpack, closing the loop entirely on-device inside the captured graph's
input buffers.

  .venv-warp/bin/python sim/robot/warplayer/m4_train_demo.py \
      --nworld 16 --horizon 20 --iters 5

Prints one RESULT line per update with the mean return — the demo passes if
the cycle runs (finite rewards, changing parameters), not if the number goes
up impressively.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402


class LinearPolicy:
    """a = clip(obs @ W * scale + sigma * xi, -1, 1); REINFORCE with mean baseline."""

    def __init__(self, obs_dim: int, act_dim: int, sigma: float = 0.2,
                 scale: float = 0.1, lr: float = 3e-3, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.W = np.zeros((obs_dim, act_dim), dtype=np.float64)
        self.sigma, self.scale, self.lr = sigma, scale, lr

    def act(self, obs: np.ndarray):
        mean = obs @ self.W * self.scale
        xi = self.rng.normal(size=mean.shape)
        return np.clip(mean + self.sigma * xi, -1.0, 1.0), xi

    def update(self, obs_traj, xi_traj, returns):
        """REINFORCE: dJ/dW ∝ E[(R - b) * sum_t obs_t^T xi_t / sigma]."""
        adv = returns - returns.mean()
        g = np.zeros_like(self.W)
        for obs_t, xi_t in zip(obs_traj, xi_traj):
            g += np.einsum("wi,wj->ij", obs_t * adv[:, None], xi_t)
        g *= self.scale / (self.sigma * len(obs_traj) * len(adv))
        self.W += self.lr * g
        return float(np.abs(self.lr * g).max())


def run(nworld=16, horizon=20, iters=5, lidar=False, seed=0, device=None, verbose=True):
    from warplayer.fused import FightLayer

    dev = wp.get_device(device) if device else wp.get_device()
    with wp.ScopedDevice(dev):
        lay = FightLayer(nworld=nworld, mode="fused", lidar=lidar, seed=seed)
        if dev.is_cuda:
            lay.capture()                    # rollout runs as one graph per control step
        pol = LinearPolicy(lay.obs_dim, lay.idx.nuA, seed=seed)
        history = []
        for it in range(iters):
            lay.reset(seed=seed + it)
            obs_traj, xi_traj = [], []
            ret = np.zeros(nworld)
            t0 = time.time()
            for _ in range(horizon):
                obs = lay.obs_numpy()        # ZERO-COPY view of the fused kernel's buffer (CPU)
                a, xi = pol.act(obs)
                obs_traj.append(obs.copy())  # policy-side rollout storage (learner memory)
                xi_traj.append(xi)
                lay.set_actions(a)
                lay.step_fused()
                ret += lay.reward_numpy()    # zero-copy consumption of the reward buffer
            dW = pol.update(obs_traj, xi_traj, ret / horizon)
            wall = time.time() - t0
            history.append(ret.mean() / horizon)
            if verbose:
                print(f"RESULT bench=m4_train iter={it} nworld={nworld} horizon={horizon} "
                      f"lidar={int(lidar)} obs_dim={lay.obs_dim} mean_reward={history[-1]:.4f} "
                      f"max_dW={dW:.2e} wall_s={wall:.3f}", flush=True)
            assert np.isfinite(ret).all(), "non-finite return"
        return history, pol, lay


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nworld", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--lidar", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    wp.init()
    run(args.nworld, args.horizon, args.iters, args.lidar, args.seed, args.device)


if __name__ == "__main__":
    main()
