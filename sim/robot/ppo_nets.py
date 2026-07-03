# SPDX-License-Identifier: MIT
"""Shared PPO network factory — drop-in for brax's `ppo_networks` module.

Audit item 5 (notes/training-uplift-audit.md): brax's defaults were never
overridden anywhere in this repo, so every run trained a (32,32,32,32) policy —
on the lidar stack that meant 470-dim observations feeding a 32-wide first layer
(a 15:1 bottleneck), ~10x smaller than anything published for 12-DOF locomotion.

This module exists so there is exactly ONE place that decides network
architecture. Trainers AND every checkpoint-reconstruction site (eval/render/
frozen-opponent loaders) import it as `import ppo_nets as ppo_networks`; a site
that constructs networks directly from brax will silently disagree with the
checkpoints the trainers now produce.

Two changes vs brax defaults:
  * policy/value hidden layers (512, 256, 128)
  * policy FINAL-layer kernels scaled 0.01x at init: the policy starts as
    "stand still with healthy exploration std" instead of saturated bang-bang
    (loc ~ 0 -> tanh-normal centered on the stand pose in PD mode; the scale
    logits also start ~0 -> std ~ softplus(0) ~ 0.69, real exploration).
    Andrychowicz et al. (2021) rank both choices among the highest-impact PPO
    decisions. Done by init surgery on the last MLP layer because brax's
    tanh_normal path emits [loc, scale] from one MLP with a single kernel_init.

Old (pre-2026-07) checkpoints were saved from 4x32 networks and will fail to
load against these shapes. That is intentional: they were trained on the
gear-bug body and are retired wholesale.
"""
from __future__ import annotations

import dataclasses

from brax.training import types
from brax.training.agents.ppo import networks as _brax

POLICY_HIDDEN = (512, 256, 128)
VALUE_HIDDEN = (512, 256, 128)
FINAL_LAYER_SCALE = 0.01


def _scale_final_layer(policy_network, scale: float):
    """Wrap a FeedForwardNetwork so init() returns params with the last MLP
    layer's kernel scaled. Params look like {'params': {'hidden_0': ...}}."""
    orig_init = policy_network.init

    def init(key):
        params = orig_init(key)
        inner = params["params"]
        last = max((k for k in inner if k.startswith("hidden_")),
                   key=lambda s: int(s.rsplit("_", 1)[1]))
        inner[last] = dict(inner[last], kernel=inner[last]["kernel"] * scale)
        return params

    return dataclasses.replace(policy_network, init=init)


def make_ppo_networks(observation_size, action_size,
                      preprocess_observations_fn=types.identity_observation_preprocessor,
                      **kwargs):
    """brax-compatible signature; repo-standard architecture defaults.

    Any explicit kwarg (policy_obs_key='state'/value_obs_key='value_state' for
    the asymmetric lidar nets, hidden sizes for experiments) passes through.
    """
    kwargs.setdefault("policy_hidden_layer_sizes", POLICY_HIDDEN)
    kwargs.setdefault("value_hidden_layer_sizes", VALUE_HIDDEN)
    net = _brax.make_ppo_networks(
        observation_size, action_size,
        preprocess_observations_fn=preprocess_observations_fn, **kwargs)
    return net.replace(
        policy_network=_scale_final_layer(net.policy_network, FINAL_LAYER_SCALE))


def __getattr__(name):
    # Everything else (make_inference_fn, PPONetworks, ...) is brax's, unchanged.
    return getattr(_brax, name)
