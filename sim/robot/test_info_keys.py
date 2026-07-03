#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""V.5: every `state.info` key must be REGISTERED with a lifetime (info_keys.py).

An unregistered key is per-env state nobody has decided an owner for — the
exact shape of the audit's bank-swap caveat (per-env RND predictor + Adam
survived episode boundaries by accident of brax semantics, not by contract).
New mechanism => new key => this test fails until its lifetime is declared.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax

from info_keys import COMMANDED_INFO_KEYS, FIGHTER_INFO_KEYS

BASE = dict(frame_skip=5, self_collision=False, sep_lo=0.3, sep_hi=0.6)


def _assert_registered(observed, registry, label):
    unregistered = sorted(set(observed) - set(registry))
    assert not unregistered, (
        f"{label}: info keys with NO declared lifetime: {unregistered} — "
        f"register them in info_keys.py as episodic or persistent")


def test_fighter_info_keys_registered_everything_on():
    import train_adversarial as T
    env = T.AdversarialEnv(engage_obs=True, contact_obs=True, her_coefficient=0.2,
                           rnd_coefficient=0.05, opponent="frozen",
                           gait_airtime_w=1.0, gait_slip_w=0.1, loco_drill_frac=0.25,
                           ko_weight=10.0, reality_gap=True, n_worlds=4, **BASE)
    s = env.reset(jax.random.PRNGKey(0))
    _assert_registered(s.info.keys(), FIGHTER_INFO_KEYS, "fighter(everything-on)")


def test_fighter_info_keys_registered_lidar():
    import train_adversarial as T
    env = T.AdversarialEnv(her_coefficient=0.2, lidar=True, lidar_n_rays=8,
                           lidar_n_vertical=2, lidar_frame_stack=2,
                           lidar_latency_steps=1, lidar_noise_sigma=0.0,
                           lidar_dropout_rate=0.0, **BASE)
    s = env.reset(jax.random.PRNGKey(1))
    _assert_registered(s.info.keys(), FIGHTER_INFO_KEYS, "fighter(lidar)")


def test_commanded_info_keys_registered():
    import commanded_env as C
    env = C._build()()
    s = env.reset(jax.random.PRNGKey(2))
    _assert_registered(s.info.keys(), COMMANDED_INFO_KEYS, "commanded")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
