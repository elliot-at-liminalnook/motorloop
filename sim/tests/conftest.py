"""Shared fixtures for the simulation verification suite.

- Loads sim/config/params.toml once and prints the assumption banner before
  any test output (provenance convention, see notes/architecture.md).
- Builds the C++ bench module (cmake/ninja, idempotent) and makes it
  importable.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "sim" / "scripts"
BUILD_DIR = PROJECT_ROOT / "sim" / "build" / "cpp"

for path in (SCRIPTS_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import sim_params  # noqa: E402  (needs SCRIPTS_DIR on sys.path)


def pytest_configure(config):
    params = sim_params.load()
    print()
    print(params.banner_text())


@pytest.fixture(scope="session")
def params() -> sim_params.SimParams:
    return sim_params.load()


@pytest.fixture(scope="session")
def bldcsim():
    subprocess.run(
        ["bash", str(SCRIPTS_DIR / "build_bench.sh")],
        check=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
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
