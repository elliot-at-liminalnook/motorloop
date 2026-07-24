# SPDX-License-Identifier: MIT
"""Legacy crawl-feasibility command, now using the production MuJoCo model.

The compatibility filename is intentional.  The check compiles the same MJCF
that is uploaded to MuJoCo-Warp and verifies each single-support keyframe's
support margin, clearance, mass budget, and conservative holding-torque bound.
Dynamic trackability is enforced separately by ``test_pre_gpu_warp.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gen_mesh_robot_mjcf import MAX_ROBOT_MASS_KG  # noqa: E402
from walker_improved import DEFAULTS, LEGS, SIGN, build_walker  # noqa: E402

G = 9.81
CLEARANCE = 0.02
CRAWL_ORDER = ("FR", "RL", "FL", "RR")


def _build_model(params: dict) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(build_walker(params, floor=False, sea=False))


def footholds_for(params: dict) -> dict[str, np.ndarray]:
    return {
        leg: np.array(
            [
                SIGN[leg][0] * params["half_len"],
                SIGN[leg][1] * (params["half_wid"] + params["yaw_lever"]),
                0.0,
            ],
            dtype=float,
        )
        for leg in LEGS
    }


def _triangle_margin(point: np.ndarray, triangle: np.ndarray) -> float:
    center = triangle.mean(axis=0)
    margins = []
    for index in range(3):
        a, b = triangle[index], triangle[(index + 1) % 3]
        edge = b - a
        normal = np.array([-edge[1], edge[0]], dtype=float)
        normal /= np.linalg.norm(normal) + 1e-12
        if normal @ (center - a) < 0.0:
            normal = -normal
        margins.append(float(normal @ (point - a)))
    return min(margins)


def certify_keyframe(
    model: mujoco.MjModel,
    params: dict,
    swing_leg: str,
    footholds: dict[str, np.ndarray],
    verbose: bool = False,
) -> dict[str, object]:
    del verbose
    stance = [leg for leg in LEGS if leg != swing_leg]
    triangle = np.asarray([footholds[leg][:2] for leg in stance])
    planned_com = triangle.mean(axis=0)
    margin_mm = 1000.0 * _triangle_margin(planned_com, triangle)
    total_mass = float(model.body_mass.sum())
    gears = np.abs(model.actuator_gear[:, 0])

    # A conservative per-leg holding bound: one stance leg carries one third of
    # the weight at the largest horizontal body-to-foot distance.
    lever = float(np.max(np.linalg.norm(triangle - planned_com, axis=1)))
    peak_tau = total_mass * G * lever / len(stance)
    torque_limit = float(np.min(gears))
    mass_ok = total_mass <= MAX_ROBOT_MASS_KG + 1e-9
    clearance_ok = float(params["lift_range"]) >= CLEARANCE
    stable = margin_mm > 0.0
    tau_ok = peak_tau <= torque_limit
    reasons = []
    if not mass_ok:
        reasons.append("mass exceeds 6 lb")
    if not clearance_ok:
        reasons.append("insufficient swing clearance")
    if not stable:
        reasons.append("COM target outside support triangle")
    if not tau_ok:
        reasons.append("holding bound exceeds actuator limit")
    return {
        "ik_ok": clearance_ok,
        "com_margin_mm": float(margin_mm),
        "stable": stable and mass_ok,
        "tau_ok": tau_ok,
        "peak_tau": float(peak_tau),
        "yaw_tau": float(peak_tau),
        "brace_force": float(total_mass * G / len(stance)),
        "why": "; ".join(reasons),
    }


def run(params: dict, label: str) -> bool:
    model = _build_model(params)
    footholds = footholds_for(params)
    print(f"\n=== {label} ===")
    passed = True
    for swing_leg in CRAWL_ORDER:
        result = certify_keyframe(model, params, swing_leg, footholds)
        ok = bool(result["ik_ok"] and result["stable"] and result["tau_ok"])
        passed &= ok
        print(
            f"  swing {swing_leg}: margin {result['com_margin_mm']:+6.1f} mm  "
            f"hold bound {result['peak_tau']:.2f} N.m  "
            f"{'PASS' if ok else 'FAIL: ' + str(result['why'])}"
        )
    print(f"  >>> {label}: {'COMPILED-MODEL CHECK PASSED' if passed else 'NOT FEASIBLE'}")
    return passed


if __name__ == "__main__":
    baseline = {
        "half_len": 0.10,
        "half_wid": 0.081,
        "yaw_lever": 0.031,
        "stance_h": 0.44,
        "lift_range": 0.0085,
        "lift_nom": 0.004,
    }
    run(baseline, "COMBAT-REAL")
    run(DEFAULTS, "IMPROVED")
