# SPDX-License-Identifier: MIT
"""V.4: named-slice schema for the fighter observation — the layout contract,
written down once.

The env ASSEMBLES the obs (train_adversarial._lidar_obs / _obs); this module
DESCRIBES it: an ordered name -> slice map computed from the same env config.
test_obs_schema.py pins the two against each other (schema total ==
env.observation_size, her_goal slice == info["her_goal"] on a real reset), so
any assembly change that isn't mirrored here breaks loud in CI instead of
silently shifting what HER relabels or a renderer slices.

Slice order (must mirror the assembly exactly):
  actor  (lidar):  loco | lidar | hist | prev_act | her_goal
  critic (lidar):  loco | lidar | priv | hist | prev_act | contacts | her_goal
  flat (non-lidar, actor==critic, contacts deliberately absent — they are
  privileged and the flat obs feeds the actor): loco | priv | hist | prev_act | her_goal

her_goal is LAST wherever present — that is her_goal.relabel_goal_arrays'
last-``GOAL_DIM``-columns contract, imported from there (single owner).
"""
from __future__ import annotations

from collections import OrderedDict

from constants import LOCO_OBS
from her_goal import GOAL_DIM


def _build(parts):
    """[(name, size), ...] with size>0 -> OrderedDict[name, slice]."""
    out, start = OrderedDict(), 0
    for name, size in parts:
        if size > 0:
            out[name] = slice(start, start + size)
            start += size
    return out


def _hist_parts(env):
    if getattr(env, "_hist_len", 0) <= 0:
        return []
    return [("hist", env._hist_len * 2 * env._n_hinge),
            ("prev_act", env.action_size)]


def _her_part(env):
    return [("her_goal", GOAL_DIM)] if env._her_coeff > 0 else []


def _priv_size(env):
    return 6 + (8 if env._engage_obs else 0) + (8 if env._contact_obs else 0)


def _flat_slices(env):
    return _build([("loco", LOCO_OBS), ("priv", _priv_size(env))]
                  + _hist_parts(env) + _her_part(env))


def actor_slices(env):
    """Ordered name->slice map for what the POLICY consumes."""
    if not env._lidar:
        return _flat_slices(env)
    return _build([("loco", LOCO_OBS), ("lidar", env._lidar_scan_dim)]
                  + _hist_parts(env) + _her_part(env))


def critic_slices(env):
    """Ordered name->slice map for what the VALUE net consumes (privileged)."""
    if not env._lidar:
        return _flat_slices(env)
    return _build([("loco", LOCO_OBS), ("lidar", env._lidar_scan_dim),
                   ("priv", _priv_size(env))]
                  + _hist_parts(env)
                  + [("contacts", int(env._Afeet_gids.shape[0]))]
                  + _her_part(env))


def total(slices):
    """Total width described by a slice map."""
    return next(reversed(slices.values())).stop if slices else 0
