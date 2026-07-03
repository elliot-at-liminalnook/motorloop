#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""V.6 source-scan guardrails over sim/robot/*.py — executable versions of the
three contracts that past incidents showed drift silently:

  1. PPO network construction goes through ppo_nets.py ONLY (the shared factory
     contract) — no other file may import brax's ppo networks module directly.
  2. No NEW torque-max (`tmax` / `_tmax`) assignment sourced from
     actuator_forcerange without gear on the same statement. Delivered motor
     torque = gear x ctrl; forcerange only CLAMPS — dividing PD torque by
     forcerange is the historical 8%-of-design-torque bug (see
     commanded_env.py / validate_body.py comments). Three pre-fix CPG search
     scripts are grandfathered; the list must only ever shrink.
  3. No env file loads physics via mujoco `from_xml_path` (e.g. a stale
     model.xml on disk) — envs must build the model from the spec
     (robot.toml -> build_mjcf -> from_xml_string) so training and validation
     always see the same body.

Pure source scans: no jax, no mujoco, runs in milliseconds.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SELF = Path(__file__).name


def _scan_files():
    """All top-level sim/robot python files except this guardrail file itself
    (which necessarily contains the forbidden patterns as string literals)."""
    return sorted(p for p in HERE.glob("*.py") if p.name != SELF)


# ---------------------------------------------------------------------------
# 1. shared PPO factory contract
# ---------------------------------------------------------------------------
# built by concatenation so the needle never appears verbatim in any scanner
_PPO_NEEDLE = "from brax.training.agents.ppo " + "import networks"
_PPO_ALLOWED = {"ppo_nets.py"}


def test_ppo_networks_import_only_in_ppo_nets():
    violators = []
    for p in _scan_files():
        if p.name in _PPO_ALLOWED:
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if _PPO_NEEDLE in line:
                violators.append(f"{p.name}:{i}: {line.strip()}")
    assert not violators, (
        "direct brax PPO networks import outside ppo_nets.py — actor/critic nets "
        "must come from the shared factory (ppo_nets.py) so checkpoints stay "
        "loadable everywhere. Violators:\n  " + "\n  ".join(violators)
    )


# ---------------------------------------------------------------------------
# 2. no NEW forcerange-derived tmax (torque divisor) without gear
# ---------------------------------------------------------------------------
# Matches `tmax =` / `_tmax =` assignments (incl. `self._tmax =`) but NOT
# longer names like `_cpg_tmax =`, and not `==` comparisons.
_TMAX_RE = re.compile(r"\b_?tmax\s*=(?!=)")

# Pre-2026-07 CPG search scripts written before the gear fix; their forcerange
# reads are numerically identical on the current body (gear == forcerange) and
# they are frozen experiment drivers. Do NOT add to this list — new code must
# derive tmax from actuator_gear (see commanded_env.py).
_TMAX_GRANDFATHERED = {
    "search_cpg_gait.py",
    "search_cpg_gait_mjx.py",
    "search_cpg_route_mjx.py",
}


def _flags_tmax_from_forcerange(line: str) -> bool:
    """True if `line` assigns tmax/_tmax from forcerange without gear on the
    same statement."""
    m = _TMAX_RE.search(line)
    if not m:
        return False
    rhs = line[m.end():]
    return ("forcerange" in rhs) and ("gear" not in line)


def test_tmax_regex_catches_the_historical_bug():
    """Plant test: the exact pre-fix commanded_env line MUST be flagged."""
    planted = "self._tmax = jnp.array(m.actuator_forcerange[:m.nu, 1])"
    assert _flags_tmax_from_forcerange(planted), (
        "guardrail regex failed to catch the known-bad line: " + planted
    )
    # and the fixed line (gear on the same statement) must pass
    fixed = ("self._tmax = jnp.array(np.where(gear > 0, gear, "
             "m.actuator_forcerange[:m.nu, 1]))")
    assert not _flags_tmax_from_forcerange(fixed), (
        "guardrail regex wrongly flags the gear-based fallback: " + fixed
    )


def test_no_new_forcerange_tmax_divisor():
    violators = []
    for p in _scan_files():
        if p.name in _TMAX_GRANDFATHERED:
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if _flags_tmax_from_forcerange(line):
                violators.append(f"{p.name}:{i}: {line.strip()}")
    assert not violators, (
        "NEW tmax assignment sourced from actuator_forcerange without gear — "
        "delivered torque = gear x ctrl; forcerange only clamps (the 8%-torque "
        "bug class). Derive tmax from actuator_gear instead. Violators:\n  "
        + "\n  ".join(violators)
    )


# ---------------------------------------------------------------------------
# 3. envs must build from spec, not from a model.xml path on disk
# ---------------------------------------------------------------------------
_XML_NEEDLE = "from_xml" + "_path"
# validate_body.py + tests + render/eval scripts may legitimately point mujoco
# at explicit files; env/training modules must not. (As of 2026-07 the tree has
# ZERO from_xml_path hits anywhere — these exclusions are prescriptive scope,
# not workarounds for existing hits.)
_XML_EXCLUDED_NAMES = {"validate_body.py"}
_XML_EXCLUDED_PREFIXES = ("test_", "render", "eval")


def test_envs_build_from_spec_not_xml_path():
    violators = []
    for p in _scan_files():
        if p.name in _XML_EXCLUDED_NAMES or p.name.startswith(_XML_EXCLUDED_PREFIXES):
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if _XML_NEEDLE in line:
                violators.append(f"{p.name}:{i}: {line.strip()}")
    assert not violators, (
        "env/training file loads physics with from_xml_path — envs must build "
        "the model from the spec (robot.toml -> build_mjcf -> from_xml_string) "
        "so a stale model.xml can never diverge from training. Violators:\n  "
        + "\n  ".join(violators)
    )


if __name__ == "__main__":
    test_ppo_networks_import_only_in_ppo_nets()
    test_tmax_regex_catches_the_historical_bug()
    test_no_new_forcerange_tmax_divisor()
    test_envs_build_from_spec_not_xml_path()
    print("guardrails: all source-scan contracts hold")
