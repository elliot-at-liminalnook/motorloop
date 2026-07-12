# SPDX-License-Identifier: MIT
"""Shared pytest fixtures for the robot verification tiers."""

from __future__ import annotations

import os

import pytest


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
