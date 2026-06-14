"""Formal verification regression tier (formal-checklist stage 11).

Gates on the formal proof results: every safety-category property must be
PROVEN (unbounded) with its non-vacuity covers reachable, and nothing may be
FAILED or INCOMPLETE (vacuous). The single DOCUMENTED gap (circle_limit's
nonlinear magnitude bound, intractable for the open SMT stack) is allowed and
checked to match the manifest.

This reads the cached results from the last `formal/run_formal.py` run (fast);
CI regenerates them first. Skips gracefully if the formal toolchain or results
are absent, mirroring the optional omc/ngspice tiers.
"""

from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "formal" / "work" / "results.json"
MANIFEST = ROOT / "formal" / "manifest.toml"
OSS = Path.home() / "oss-cad-suite" / "bin"

_have_tools = shutil.which("sby") is not None or (OSS / "sby").exists()

pytestmark = pytest.mark.skipif(
    not (_have_tools and RESULTS.exists()),
    reason="formal toolchain or results.json absent "
           "(run formal/run_formal.py)")


def _results():
    return {r["id"]: r for r in json.loads(RESULTS.read_text())}


def _manifest():
    return {c["id"]: c for c in tomllib.loads(MANIFEST.read_text())["check"]}


def test_no_failed_or_vacuous_proofs():
    """No proof is FAILED, and no safety proof is INCOMPLETE (vacuous)."""
    bad = [f"{r['id']}: {r['status']}"
           for r in _results().values()
           if r["status"].startswith(("FAILED", "INCOMPLETE"))]
    assert not bad, f"failed/vacuous proofs: {bad}"


def test_every_manifest_check_has_a_result():
    """No claimed property is missing a result (the proof surface is run)."""
    res = _results()
    missing = [cid for cid in _manifest() if cid not in res]
    assert not missing, f"manifest checks with no result: {missing}"


def test_safety_properties_proven_with_covers():
    """Every safety property is PROVEN with its non-vacuity covers reachable
    (or an explicitly DOCUMENTED gap matching the manifest)."""
    manifest = _manifest()
    failures = []
    for r in _results().values():
        if r["category"] != "safety":
            continue
        expected = manifest[r["id"]]["expected"]
        if expected == "DOCUMENTED":
            if r["status"] != "DOCUMENTED":
                failures.append(f"{r['id']}: expected DOCUMENTED, "
                                f"got {r['status']}")
            continue
        if r["status"] != "PROVEN":
            failures.append(f"{r['id']}: {r['status']} (expected PROVEN)")
        elif r.get("covers") and r["covers"] != "REACHED":
            failures.append(f"{r['id']}: non-vacuity cover {r['covers']}")
    assert not failures, f"safety proof issues: {failures}"


def test_crown_jewels_proven():
    """The headline plant-independent guarantees are unbounded-proven."""
    res = _results()
    for cid in ("pwm_generator", "pwm_deadtime",
                "controller_top_composition"):
        assert cid in res, f"{cid} not run"
        assert res[cid]["status"] == "PROVEN", (
            f"{cid}: {res[cid]['status']}")
