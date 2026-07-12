# SPDX-License-Identifier: MIT
"""Curriculum selection and combat reward configuration contracts."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_adversarial import AdversarialEnv, behavior_keep_ok


def test_behavior_gate_rejects_stationary_checkpoint():
    gates = dict(min_closed=0.2, min_approach=0.01, min_disp=0.1, min_far_sparc=0.0)
    assert not behavior_keep_ok({}, **gates)
    assert behavior_keep_ok(dict(bh_closed=0.3, bh_approach=0.02,
                                 bh_disp=0.2, sparc_far=1.0), **gates)


def test_reward_knobs_reach_fused_kernel_configuration():
    env = AdversarialEnv(nworld=1, device="cpu", approach_weight=3.0,
                         upright_weight=0.7, trade_weight=2.0)
    assert env.layer.cfg.approach_w == 3.0
    assert env.layer.cfg.upright_w == 0.7
    assert env.layer.cfg.trade_w == 2.0
