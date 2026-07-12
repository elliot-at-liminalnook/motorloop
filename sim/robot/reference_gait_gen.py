# SPDX-License-Identifier: MIT
"""Validate and export the canonical MuJoCo-Warp walker reference gait.

The checked-in reference is now the source artifact.  This compatibility entry
point validates its joint order, dimensions, ranges, and finite values against
the production MuJoCo model before exporting it.  The mandatory test gate then
proves that the same targets are dynamically trackable in MuJoCo-Warp.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from walker_improved import DEFAULTS, build_walker  # noqa: E402

CANONICAL = HERE / "reference_gait_walker.json"


def _validate(ref: dict, params: dict) -> np.ndarray:
    if any(not np.isclose(float(params[key]), float(DEFAULTS[key])) for key in DEFAULTS):
        raise ValueError("the checked reference is valid only for walker_improved.DEFAULTS")
    model = mujoco.MjModel.from_xml_string(build_walker(params, floor=True, sea=False))
    joint_ids = model.actuator_trnid[:, 0].astype(int)
    joint_order = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in joint_ids
    ]
    if ref.get("joint_order") != joint_order:
        raise ValueError(f"reference/model joint-order mismatch: {ref.get('joint_order')} != {joint_order}")

    targets = np.asarray(ref["qpos_targets"], dtype=float)
    velocities = np.asarray(ref["qvel_targets"], dtype=float)
    swing = np.asarray(ref["feet_swing"], dtype=bool)
    frames = int(ref["n"])
    if targets.shape != (frames, model.nu) or velocities.shape != targets.shape:
        raise ValueError("reference target arrays have the wrong shape")
    if swing.shape != (frames, 4) or np.any(swing.sum(axis=1) > 1):
        raise ValueError("reference must swing at most one foot per frame")
    if not np.isfinite(targets).all() or not np.isfinite(velocities).all():
        raise ValueError("reference contains non-finite values")

    lows = np.array([model.jnt_range[joint_id, 0] for joint_id in joint_ids])
    highs = np.array([model.jnt_range[joint_id, 1] for joint_id in joint_ids])
    # The historical optimizer left sub-micrometer negative residue on slide
    # joints whose lower limit is exactly zero.
    if np.any(targets < lows - 1e-6) or np.any(targets > highs + 1e-6):
        raise ValueError("reference target exceeds a compiled joint range")
    return targets


def generate(params: dict = DEFAULTS, out: str | Path | None = None):
    ref = json.loads(CANONICAL.read_text())
    targets = _validate(ref, params)
    destination = Path(out) if out is not None else CANONICAL
    if destination.resolve() != CANONICAL.resolve():
        destination.write_text(json.dumps(ref, indent=1) + "\n")
    return ref, targets


if __name__ == "__main__":
    reference, _ = generate()
    print(
        f"validated {CANONICAL.name}: {reference['n']} frames, "
        f"{reference['period_s']} s, {len(reference['joint_order'])} actuators"
    )
