# SPDX-License-Identifier: MIT
"""Shared fixtures for the simulation verification suite.

- Loads sim/config/params.toml once and prints the assumption banner before
  any test output (provenance convention, see notes/architecture.md).
- Builds the C++ bench module (cmake/ninja, idempotent) and makes it
  importable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "sim" / "scripts"
TESTS_DIR = Path(__file__).resolve().parent
BUILD_DIR = PROJECT_ROOT / "sim" / "build" / "cpp"

for path in (SCRIPTS_DIR, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import sim_params  # noqa: E402  (needs SCRIPTS_DIR on sys.path)


def pytest_configure(config):
    params = sim_params.load()
    print()
    print(params.banner_text())
    # Name the active platform set (platform-abstraction stage 3.2): the
    # registered BOMs and the default. A scenario selects one via
    # bench_factory.platform(); cfg["platform"] then carries the active name.
    try:
        from bench_factory import PLATFORMS, DEFAULT_PLATFORM
        print(f"  Platforms: {', '.join(PLATFORMS)}")
        print(f"  Default platform (active unless a scenario overrides): "
              f"{DEFAULT_PLATFORM}")
        print("=" * 78)
    except ImportError:
        pass  # bench_factory optional (pre-stage-3.2 checkout): banner extras only, tests unaffected


@pytest.fixture(scope="session")
def params() -> sim_params.SimParams:
    return sim_params.load()


@pytest.fixture(scope="session")
def bldcsim():
    if os.environ.get("BLDCSIM_BENCH_PREBUILT") != "1":
        subprocess.run(
            ["bash", str(SCRIPTS_DIR / "build_bench.sh")],
            check=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            # build against THIS interpreter: a bench built for the system
            # python is unimportable from a venv with a different ABI
            env={**os.environ, "PYTHON": sys.executable},
        )
    elif not any(BUILD_DIR.glob("bldcsim*.so")):
        pytest.fail(
            "BLDCSIM_BENCH_PREBUILT=1 but the bldcsim extension is missing; "
            "run `make bench` before starting parallel workers"
        )
    if str(BUILD_DIR) not in sys.path:
        sys.path.insert(0, str(BUILD_DIR))
    import bldcsim as module

    return module


@pytest.fixture(scope="session")
def one_phase_reference():
    """The importable Python reference runner (executable spec)."""
    import run_one_phase_reference as module

    return module
