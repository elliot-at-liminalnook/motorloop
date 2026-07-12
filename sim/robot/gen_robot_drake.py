#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Legacy model-inspection entry point, now backed entirely by MuJoCo.

The filename is retained so old automation keeps working.  No Drake model is
built or imported: ``robot.toml`` is compiled through the production MJCF
generator and inspected with plain MuJoCo, the CPU oracle for MuJoCo-Warp.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "sim" / "tests"))

from gen_robot_mjcf import build_mjcf, load_spec  # noqa: E402
from motors import MOTORS, SERVOS  # noqa: E402


def joint_effort_limit_nm(spec: dict) -> float:
    """Re-derive peak hinge torque from the raw component profile."""
    actuator = spec["actuator"]
    motor_name = actuator["motor"]
    gear = float(actuator["gear"])
    if motor_name in SERVOS:
        servo = SERVOS[motor_name]
        voltage = float(actuator.get("voltage", max(servo.stall_torque_nm)))
        return float(servo.stall_torque_nm[voltage]) * gear
    motor = MOTORS[motor_name]
    return float(motor.kt) * float(actuator["peak_factor"]) * float(motor.rated_current_a) * gear


def build_model(spec: dict) -> mujoco.MjModel:
    """Compile the production robot body with plain MuJoCo."""
    return mujoco.MjModel.from_xml_string(build_mjcf(spec))


# Compatibility for callers that used the old function name.
build_plant = build_model


def hinge_effort_limits(model: mujoco.MjModel) -> dict[str, float]:
    limits: dict[str, float] = {}
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        limits[joint_name] = abs(float(model.actuator_gear[actuator_id, 0]))
    return limits


def summarize(spec: dict) -> str:
    model = build_model(spec)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    total_mass = float(model.body_mass.sum())
    configured_limit = joint_effort_limit_nm(spec)
    actuator_limits = hinge_effort_limits(model)
    if model.nu != 12:
        raise AssertionError(f"expected twelve actuators, compiled model has {model.nu}")
    # MJCF serializes the derived value with six significant digits.
    if any(abs(limit - configured_limit) > 2e-5 for limit in actuator_limits.values()):
        raise AssertionError("compiled actuator gear disagrees with the component-derived limit")

    lines = [
        f"{spec['meta']['name']} (MuJoCo {mujoco.__version__}, production MJCF)",
        f"  DoF: nq={model.nq} nv={model.nv}, actuators={model.nu}",
        f"  component-derived effort limit: {configured_limit:.4f} N.m",
    ]
    for name, limit in actuator_limits.items():
        lines.append(f"    {name:<12s} effort limit {limit:.4f} N.m")
    lines.append(f"  total compiled mass: {total_mass:.4f} kg")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(HERE / "robot.toml"))
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    spec = load_spec(args.config)
    if args.summary:
        print(summarize(spec))
        return
    model = build_model(spec)
    print(
        f"{spec['meta']['name']}: MuJoCo model compiled; "
        f"nq={model.nq} nv={model.nv} nu={model.nu}"
    )


if __name__ == "__main__":
    main()
