# SPDX-License-Identifier: MIT
"""Compatibility aliases for the converted MuJoCo-Warp design environments.

New code should import :mod:`direct_warp_env` or :mod:`codesign_warp_env`.
"""

from direct_warp_env import DirectWarpEnv
from codesign_warp_env import DesignEnsembleWarpEnv

class CodesignEnv(DirectWarpEnv):
    def __init__(self, xml: str, frame_skip: int = 5, design=None,
                 nworld: int = 1, **kwargs):
        super().__init__(xml, nworld=nworld, frame_skip=frame_skip,
                         design=design, **kwargs)


class UniversalEnv(DesignEnsembleWarpEnv):
    def __init__(self, xml: str | None = None, frame_skip: int = 5,
                 fixed_design=None, nworld: int = 1, **kwargs):
        del xml, frame_skip
        designs = [fixed_design] if fixed_design is not None else kwargs.pop("designs", None)
        if designs is not None:
            kwargs["designs"] = designs
        super().__init__(nworld=nworld, **kwargs)
