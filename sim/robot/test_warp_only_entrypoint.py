# SPDX-License-Identifier: MIT
"""Architecture contract for the canonical MuJoCo-Warp launch path."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_FILES = (
    ROOT / "scripts/run_pre_gpu_tests.sh",
    ROOT / "sim/robot/gpu_determinism_canary.py",
    ROOT / "sim/robot/setup_warp_pod.sh",
)
FORBIDDEN = ("jax", "mjx", "brax", "pydrake", "validate_gpu.sh", ".venv-sim")


def test_canonical_launch_path_is_warp_only():
    for path in ACTIVE_FILES:
        text = path.read_text().lower()
        found = [token for token in FORBIDDEN if token in text]
        assert not found, f"{path.relative_to(ROOT)} reintroduced inactive backends: {found}"


def test_entrypoint_runs_every_warp_geometry_and_full_robot_suite():
    runner = (ROOT / "scripts/run_pre_gpu_tests.sh").read_text()
    canary = (ROOT / "sim/robot/gpu_determinism_canary.py").read_text()
    assert "make test" in runner
    assert 'pytest sim/robot -q' in runner
    assert all(name in runner for name in (
        "walker_warp_env.py", "mesh_warp_env.py", "combat_warp_env.py",
        "ladder_warp_env.py"))
    assert all(f'"{name}"' in canary
               for name in ("walker", "mesh", "combat", "ladder_locomotion",
                            "ladder_combat", "universal"))


def test_runtime_scripts_do_not_reference_retired_environments():
    needles = ("JAX" + "_", "MJX" + "_PY", "mjx" + "-venv", ".venv" + "-sim")
    paths = [ROOT / "Makefile", *sorted((ROOT / "scripts").glob("*.sh")),
             *sorted((ROOT / "sim/robot").rglob("*.py"))]
    violations = []
    for path in paths:
        if path == Path(__file__):
            continue
        found = [needle for needle in needles if needle in path.read_text()]
        if found:
            violations.append(f"{path.relative_to(ROOT)}: {found}")
    assert not violations, "retired runtime environment references remain:\n" + "\n".join(violations)
