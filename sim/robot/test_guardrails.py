#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Source guardrails for the MuJoCo-Warp-only robot runtime."""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _python_files():
    return sorted(HERE.rglob("*.py"))


def test_retired_backend_imports_are_absent():
    # Tokens are assembled so this scanner does not flag itself.
    tokens = ("import " + "ja" + "x", "from " + "bra" + "x",
              "from mujoco import " + "mj" + "x", "import mujoco." + "mj" + "x",
              "from " + "py" + "drake", "import " + "py" + "drake")
    violations = []
    for path in _python_files():
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if any(token in line for token in tokens):
                violations.append(f"{path.relative_to(ROOT)}:{line_no}: {line.strip()}")
    assert not violations, "retired backend imports remain:\n" + "\n".join(violations)


def test_physics_models_are_not_loaded_from_stale_paths():
    needle = "MjModel." + "from_xml_" + "path("
    violations = [str(path.relative_to(ROOT)) for path in _python_files()
                  if needle in path.read_text()]
    assert not violations, f"stale-path model loads remain: {violations}"


def test_torque_divisors_do_not_trust_forcerange_alone():
    pattern = re.compile(r"\b_?tmax\s*=(?!=).*(actuator_)?forcerange")
    violations = []
    for path in _python_files():
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line) and "gear" not in line:
                violations.append(f"{path.relative_to(ROOT)}:{line_no}")
    assert not violations, f"forcerange-derived torque limits remain: {violations}"
