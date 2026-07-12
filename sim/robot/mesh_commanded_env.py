# SPDX-License-Identifier: MIT
"""Compatibility entry point for the MuJoCo-Warp mesh locomotion environment."""

from mesh_locomotion_spec import *  # noqa: F401,F403 - compatibility re-export


def _build():
    from mesh_warp_env import MeshWarpEnv

    class MeshCommandedWarpEnv(MeshWarpEnv):
        def __init__(self, nworld: int = 1, **kwargs):
            super().__init__(nworld=nworld, **kwargs)

        @property
        def observation_size(self):
            return self.obs_dim

        @property
        def action_size(self):
            return self.act_dim

        @property
        def backend(self):
            return "mujoco_warp"

    return MeshCommandedWarpEnv
