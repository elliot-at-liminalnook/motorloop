#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Deterministic MuJoCo-Warp walker trajectory fingerprint."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import walker_warp_env as W

N_STEPS = 50
COMMAND = (0.4, 0.0, 0.0)


def fingerprint(env_name="commanded_pd"):
    if env_name != "commanded_pd":
        raise ValueError(env_name)
    old_cat = W.CAT_ON
    W.CAT_ON = False
    try:
        env = W.WalkerWarpEnv(1, seed=0, device="cpu", episode_length=None)
        env._reset_noise.zero_(); env.reset()
        reward_sum = 0.0
        for _ in range(N_STEPS):
            env._cmd.copy_(torch.tensor([COMMAND])); env._timer.zero_()
            _, reward, _, _ = env.step(torch.zeros((1, env.act_dim)))
            reward_sum += float(reward)
        qpos = env.qpos[0].double().numpy()
        return {"qpos_sum": round(float(qpos.sum()), 6),
                "qpos_abs_sum": round(float(np.abs(qpos).sum()), 6),
                "final_z": round(float(qpos[2]), 6),
                "reward_sum": round(reward_sum, 6)}
    finally:
        W.CAT_ON = old_cat


if __name__ == "__main__":
    print(json.dumps(fingerprint(), indent=2, sort_keys=True))
