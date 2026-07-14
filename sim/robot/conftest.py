# SPDX-License-Identifier: MIT
"""Shared pytest fixtures for the robot verification tiers."""

from __future__ import annotations

import os

import pytest


# A CUDA-capable host makes CUDA Warp's process-wide default.  The non-GPU
# verification tier is our deterministic CPU oracle, so let its launcher pin
# the default before pytest imports any test modules (many of which call
# ``wp.init()`` during collection).
_warp_test_device = os.environ.get("ROBOT_WARP_DEVICE")
if _warp_test_device:
    import warp as wp

    wp.init()
    wp.set_device(_warp_test_device)


@pytest.fixture(scope="session")
def gpu_device():
    """Return the required CUDA device without allowing a false-green full gate."""
    import torch

    if not torch.cuda.is_available():
        message = "CUDA is required for tests marked gpu"
        if os.environ.get("ROBOT_REQUIRE_GPU") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return "cuda"
