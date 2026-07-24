# SPDX-License-Identifier: MIT
"""Fresh-process CUDA repeatability gate for every MuJoCo-Warp geometry."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MESH_CAT", "0")

import torch

from train_mesh_warp import build_args, train


# GPU contact assembly/solving uses parallel reductions, so ground-contact runs
# are numerically repeatable rather than bitwise repeatable. These bounds include
# margin over repeated fresh-process L40S trials; mesh remained near-bitwise,
# walker/universal occasionally selected a different low-order contact path, and
# combat needs the widest bounds. Keys, shapes, steps, and semantic checkpoint
# contracts are always exact.
METRIC_TOLERANCES = {
    "mesh": {"updates": (2e-3, 2e-5), "evals": (1e-5, 5e-5)},
    "ground": {"updates": (1e-2, 5e-2), "evals": (1e-3, 3e-3)},
    "combat": {"updates": (1e-2, 1e-1), "evals": (2e-3, 5e-3)},
}
TENSOR_TOLERANCES = {
    "mesh": {
        "actor": (2e-4, 2e-3), "critic": (2e-4, 2e-3),
        "obs_norm": (1e-4, 2e-3), "priv_norm": (1e-4, 2e-3),
    },
    "ground": {
        "actor": (1e-3, 2e-2), "critic": (1e-3, 2e-2),
        "obs_norm": (2e-2, 2e-2), "priv_norm": (2e-2, 2e-2),
    },
    "combat": {
        "actor": (3e-3, 5e-2), "critic": (5e-3, 8e-2),
        "obs_norm": (1e-1, 1e-1), "priv_norm": (1e-1, 1e-1),
    },
}

# These fields identify a particular serialized artifact.  Two fresh runs write
# different checkpoint containers (and therefore different hashes) even when
# the policy tensors and all numeric behavior are repeatable.  Keep requiring
# the same schema, but do not treat provenance identifiers as physics metrics.
NON_REPEATABILITY_METRIC_KEYS = frozenset({
    # These are derived by the checkpoint replay's own tolerance check.  Their
    # pass/fail verdict is compared, but the fraction of tolerance consumed may
    # differ between two independently rounded GPU trajectories.
    "max_tolerance_ratio",
    "metric_tolerance_ratios",
    # Host scheduling and allocator state are operational diagnostics, not
    # seeded policy behavior.
    "env_steps_per_second",
    "hardware",
    # The same update record is already compared through first["updates"] with
    # the update tolerance.  Do not compare its duplicate inside an evaluation
    # with the usually tighter evaluation tolerance.
    "learner",
    "solver_iterations",
    # The evaluation record already exposes behavior, learner updates, gates,
    # and trends at the top level.  This nested bundle duplicates them together
    # with timing, allocator, solver-count, and tolerance-consumption details
    # whose repeatability has a different contract.
    "diagnostics",
})


def _non_repeatability_metric_key(key: object) -> bool:
    """Return true for provenance/performance fields outside seeded behavior."""
    return (isinstance(key, str)
            and (key.endswith("_sha256")
                 or key.endswith("_seconds")
                 or key in NON_REPEATABILITY_METRIC_KEYS))


def _tolerance_group(geometry: str) -> str:
    if geometry in ("combat", "leg_attack", "ladder_combat"):
        return "combat"
    if geometry == "mesh":
        return "mesh"
    return "ground"


def _args(root: Path, geometry: str, tag: str):
    envs, horizon, updates = 16, 16, 2
    argv = [
        "--geometry", geometry,
        "--device", "cuda",
        "--steps", str(envs * horizon * updates),
        "--envs", str(envs),
        "--horizon", str(horizon),
        "--episode-length", "64",
        "--hidden", "32,32",
        "--epochs", "2",
        "--minibatches", "4",
        "--evals", "1",
        "--eval-envs", "4",
        "--eval-steps", "16",
        "--seed", "20260709",
        "--preflight", "off",
        "--tag", str(root / tag),
    ]
    if geometry == "ladder_locomotion":
        argv += ["--rung", "23"]
    elif geometry == "ladder_combat":
        argv += ["--rung", "26"]
    return build_args(argv)


def _metric_drift(first, second, prefix="") -> tuple[float, float, str]:
    worst_abs = worst_rel = 0.0
    worst_path = prefix
    if isinstance(first, dict):
        assert first.keys() == second.keys(), f"metric keys differ at {prefix}"
        children = [
            _metric_drift(first[key], second[key], f"{prefix}.{key}")
            for key in first if not _non_repeatability_metric_key(key)
        ]
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second), f"metric lengths differ at {prefix}"
        children = [
            _metric_drift(a, b, f"{prefix}[{index}]")
            for index, (a, b) in enumerate(zip(first, second))
        ]
    elif isinstance(first, (int, float)) and isinstance(second, (int, float)):
        absolute = abs(float(first) - float(second))
        relative = absolute / max(abs(float(first)), abs(float(second)), 1e-12)
        return absolute, relative, prefix
    else:
        assert first == second, f"non-numeric metric differs at {prefix}: {first!r} != {second!r}"
        return 0.0, 0.0, prefix
    for absolute, relative, path in children:
        if absolute > worst_abs:
            worst_abs, worst_rel, worst_path = absolute, relative, path
    return worst_abs, worst_rel, worst_path


def _metric_failures(first, second, atol: float, rtol: float, prefix="") -> list[str]:
    if isinstance(first, dict):
        assert first.keys() == second.keys(), f"metric keys differ at {prefix}"
        return [failure for key in first
                if not _non_repeatability_metric_key(key)
                for failure in _metric_failures(
                    first[key], second[key], atol, rtol, f"{prefix}.{key}")]
    if isinstance(first, (list, tuple)):
        assert len(first) == len(second), f"metric lengths differ at {prefix}"
        return [failure for index, (left, right) in enumerate(zip(first, second))
                for failure in _metric_failures(
                    left, right, atol, rtol, f"{prefix}[{index}]")]
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        if math.isclose(float(first), float(second), abs_tol=atol, rel_tol=rtol):
            return []
        return [f"{prefix}: {first!r} != {second!r} (atol={atol:g}, rtol={rtol:g})"]
    assert first == second, f"non-numeric metric differs at {prefix}: {first!r} != {second!r}"
    return []


def _tensor_tolerance(geometry: str, section: str, key: str) -> tuple[float, float]:
    group = _tolerance_group(geometry)
    if key == "count":
        return 0.0, 0.0
    if group == "combat" and section == "obs_norm" and key == "var":
        return 1.5, 1.5e-1
    if group == "combat" and section == "priv_norm" and key == "var":
        return 1.0, 2e-1
    if group == "ground" and section in ("obs_norm", "priv_norm") and key == "var":
        return 0.25, 2e-2
    return TENSOR_TOLERANCES[group][section]


def _assert_identical(first: dict, second: dict, geometry: str, *, enforce: bool = True) -> None:
    update_abs, update_rel, update_path = _metric_drift(
        first["updates"], second["updates"], "updates")
    eval_abs, eval_rel, eval_path = _metric_drift(first["evals"], second["evals"], "evals")
    a = torch.load(first["ckpt"], map_location="cpu", weights_only=True)
    b = torch.load(second["ckpt"], map_location="cpu", weights_only=True)
    tensor_abs = tensor_rel = 0.0
    tensor_path = ""
    tensors_exact = True
    section_drifts = {}
    tensor_failures = []
    for section in ("actor", "critic", "obs_norm", "priv_norm"):
        assert a[section].keys() == b[section].keys()
        section_abs = section_rel = 0.0
        section_path = ""
        for key in a[section]:
            left, right = a[section][key], b[section][key]
            tensors_exact &= torch.equal(left, right)
            atol, rtol = _tensor_tolerance(geometry, section, key)
            if not torch.allclose(left, right, atol=atol, rtol=rtol):
                tensor_failures.append(
                    f"{section}.{key} exceeds atol={atol:g}, rtol={rtol:g}")
            if left.is_floating_point():
                absolute = float((left - right).abs().max())
                scale = max(float(left.abs().max()), float(right.abs().max()), 1e-12)
                relative = absolute / scale
                if absolute > tensor_abs:
                    tensor_abs, tensor_rel = absolute, relative
                    tensor_path = f"{section}.{key}"
                if absolute > section_abs:
                    section_abs, section_rel, section_path = absolute, relative, key
        section_drifts[section] = (section_abs, section_rel, section_path or "none")
    assert a["contract"] == b["contract"]
    print(
        f"DETERMINISM geometry={geometry} "
        f"update_abs={update_abs:.9g} update_rel={update_rel:.9g} path={update_path} "
        f"eval_abs={eval_abs:.9g} eval_rel={eval_rel:.9g} path={eval_path} "
        f"tensor_abs={tensor_abs:.9g} tensor_rel={tensor_rel:.9g} path={tensor_path or 'none'}",
        flush=True,
    )
    print(
        "DETERMINISM_TENSORS geometry=" + geometry + " " + " ".join(
            f"{section}_abs={absolute:.9g} {section}_rel={relative:.9g} "
            f"{section}_path={path}"
            for section, (absolute, relative, path) in section_drifts.items()
        ),
        flush=True,
    )
    if enforce:
        group = _tolerance_group(geometry)
        update_tol = METRIC_TOLERANCES[group]["updates"]
        eval_tol = METRIC_TOLERANCES[group]["evals"]
        failures = _metric_failures(
            first["updates"], second["updates"], *update_tol, "updates")
        failures += _metric_failures(first["evals"], second["evals"], *eval_tol, "evals")
        failures += tensor_failures
        assert not failures, geometry + " repeatability failures:\n" + "\n".join(failures[:20])


def _run_isolated(root: Path, geometry: str, tag: str) -> dict:
    """Run one training sample in a fresh process and return its serialized metrics."""
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker",
         "--geometry", geometry, "--root", str(root), "--tag", tag],
        check=True,
    )
    return json.loads((root / f"{tag}.json").read_text())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=("all", "walker", "mesh", "combat",
                                                "ladder_locomotion", "ladder_combat",
                                                "universal"),
                        default="all")
    parser.add_argument("--report-only", action="store_true",
                        help="print measured drift without applying acceptance bounds")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--tag", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA is unavailable")
    torch.use_deterministic_algorithms(True)
    if args.worker:
        if args.geometry == "all" or args.root is None or args.tag is None:
            parser.error("--worker requires one --geometry, --root, and --tag")
        result = train(_args(args.root, args.geometry, args.tag))
        (args.root / f"{args.tag}.json").write_text(json.dumps(result, sort_keys=True))
        return 0
    with tempfile.TemporaryDirectory(prefix="warp-gpu-canary-") as td:
        root = Path(td)
        geometries = ("walker", "mesh", "combat", "ladder_locomotion",
                      "ladder_combat", "universal") \
            if args.geometry == "all" else (args.geometry,)
        for geometry in geometries:
            first = _run_isolated(root, geometry, f"{geometry}_a")
            second = _run_isolated(root, geometry, f"{geometry}_b")
            _assert_identical(first, second, geometry, enforce=not args.report_only)
    if args.report_only:
        print("REPORT: measured same-seed drift for all selected Warp geometries")
    else:
        print("PASS: all Warp geometries satisfy the same-seed repeatability contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
