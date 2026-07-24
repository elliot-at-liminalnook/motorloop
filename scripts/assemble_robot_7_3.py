#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Assemble a FULL 4-LEGGED ROBOT from the rigged 7-1-2026 leg and render a
labeled walking-trot + power-strike (stomp) animation.

Run with:
  blender --background --python scripts/assemble_robot_7_3.py

Builds on scripts/rig_foot_7_1.py (single-leg rig) and the AUTHORITATIVE
fourth-pass spec Test_Mesh_Leg_7-1-2026.physics.json:

  joint chain per leg:  hip_yaw (+/-45 deg placeholder, at the body mount,
  axis vertical, series-elastic belt w/ large rubber pulley)
    -> leg_swing (+/-25 deg, worm 20:1 self-locking, axis Z leg-local)
    -> knee_blade (-90..+10 deg, POWERED strike, toggle-press)
    -> toe_hinge / heel_pin (passive) + pushrod prismatic
       (slider-crank closure, 41.14 mm travel  --  CLOSED FORM reused below).

Mounting (per task spec): leg-local -Y = world down, swing axis (leg-local Z)
= body lateral (fore-aft stepping), yaw axis = world vertical at the
attachment.  Right-side legs (FR, RR) are mirrored across the sagittal plane;
because the whole mechanism is PLANAR in leg-local XY (every joint axis is
+/-Z, every pivot sits at z=0), the mirror is baked into the mesh vertex data
(leg-local z -> -z + normal flip) and the joint kinematics are unchanged.
Rear legs are additionally rotated 180 deg about vertical (arm points aft) so
the robot is fore/aft symmetric and the rear blades guard the rear.

All motion is KEYFRAMED KINEMATICS (no rigid-body dynamics), matching the
6-23 / 7-1 pipeline.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

LAYOUT_TEST = os.environ.get("ROBOT_ASM_LAYOUT_TEST", "") == "1"

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from bpy_extras.object_utils import world_to_camera_view

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Test_Mesh_Leg_7-1-2026.gltf"
LEG_JSON = ROOT / "Test_Mesh_Leg_7-1-2026.physics.json"
OUT_DIR = ROOT / "build" / "robot_assembly"
FRAMES_DIR = OUT_DIR / "frames"
LABELED_DIR = OUT_DIR / "frames_labeled"
PREFIX = "Robot_Assembly_7-3-2026"
BLEND_OUT = ROOT / f"{PREFIX}_physics.blend"
GLB_OUT = ROOT / f"{PREFIX}_physics.glb"
JSON_OUT = ROOT / f"{PREFIX}.physics.json"
GIF_OUT = ROOT / f"{PREFIX}_gait_anim.gif"
CONTACT_SHEET_OUT = ROOT / f"{PREFIX}_motion_contact_sheet.png"
PARTS_MAP_OUT = ROOT / f"{PREFIX}_parts_map.png"

# ---------------------------------------------------------------------------
# Timing: ~6 s @ 24 fps.  Walk = frames 1..108 (3 trot cycles of 1.5 s),
# finale = frames 109..144 (1.5 s: tuck, POWER STRIKE, hold).
# ---------------------------------------------------------------------------
FPS = 24
N_FRAMES = 144
WALK_END = 108
CYCLE = 36                 # 1.5 s trot cycle
RAMP = 12                  # 0.5 s stand -> walk amplitude ramp-in
TUCK_END = WALK_END + 8    # frames 109..116: ease to strike-ready pose
STRIKE_END = TUCK_END + 5  # frames 117..121: -10 -> -90 deg power strike

RES_X, RES_Y = 1100, 850
GIF_WIDTH = 640

# ---------------------------------------------------------------------------
# Leg-local geometry (verbatim from rig_foot_7_1.py / the fourth-pass json).
# ---------------------------------------------------------------------------
WORM_AXIS_POS = Vector((-0.2185, 0.0, 0.0))    # worm spins about leg-local +Y here
GEAR_PIVOT = Vector((-0.16269, 0.0, 0.0))      # leg_swing axle (axis leg-local Z)
KNEE_PIVOT = Vector((0.0, 0.0, 0.0))           # knee_blade pivot (axis leg-local Z)
HEEL_PIN_NEUTRAL = Vector((0.0, -0.025, 0.0))  # blade heel ear == pushrod clevis pin
BUSHING_CENTER = Vector((0.0, -0.2405, 0.0))   # distal-carrier bushing (prismatic origin)
TOE_PIN_LOCAL = Vector((0.0, 0.075, 0.0))      # toe hinge joining the two blade lengths
PUSHROD_TIP_NEUTRAL = Vector((0.0, -0.283, 0.0))
HIP_YAW_LOCAL = Vector((-0.219, 0.0, 0.0))     # hip_yaw origin (json: at the body mount)

ANKLE_MIN_DEG = -90.0      # blade fully struck out front (toggle-press stomp target)
ANKLE_MAX_DEG = 10.0
SWING_DEG = 25.0           # worm-driven leg-swing hard limit
YAW_DEG = 45.0             # hip yaw placeholder limit (json)
WORM_GEAR_RATIO = 20.0
YAW_BELT_RATIO = 6.0
# Hardware contract (2026-07-09): one ST3215-HS at every active joint.
SERVO_MODEL = "Waveshare ST3215-HS"
SERVO_STALL_TORQUE_NM = 20.0 * 0.0980665   # 20 kgf.cm @ 12 V
SERVO_FREE_SPEED_RAD_S = 106.0 * 2.0 * math.pi / 60.0
SERVO_MASS_KG = 0.068
SERVO_OUTPUT_INERTIA_EST = 2.7e-3          # not published; bench-measure

# Gait amplitudes (task spec): yaw sweep +/-25 within the +/-45 limit,
# pitch swing +/-15 within +/-25, blade partial extension -60..-10 deg.
GAIT_YAW_AMP = 25.0
GAIT_SWING_AMP = 15.0
GAIT_BLADE_MID = -35.0
GAIT_BLADE_AMP = 25.0
STRIKE_READY_DEG = -10.0

GROUPS = {
    "worm_input": ["Mesh_18", "Mesh_3"],
    "leg_swing_link": ["Mesh_19", "Mesh_17", "Mesh_39", "Mesh_40", "Mesh_41", "Mesh_42",
                       "Mesh_43", "Mesh_44", "Mesh_36", "Mesh_20", "Mesh_21", "Mesh_37",
                       "Mesh_9", "Mesh_10", "Mesh_11", "Mesh_46", "Mesh_47", "Mesh_45",
                       "Mesh_38", "Mesh_24", "Mesh_25"],
    "blade_upper": ["Mesh_35", "Mesh_1", "Mesh_34"],
    "blade_lower": ["Mesh_22", "Mesh_23", "Mesh_32", "Mesh_33"],
    "pushrod": ["Mesh_29", "Mesh_30", "Mesh_31", "Mesh_26", "Mesh_0", "Mesh_27", "Mesh_28"],
}
CHASSIS = ["Mesh_5", "Mesh_2", "Mesh_6", "Mesh_7", "Mesh_8", "Mesh_4", "Mesh_12", "Mesh_13",
           "Mesh_14", "Mesh_15", "Mesh_16"]
BODY_MASSES = {  # from Test_Mesh_Leg_7-1-2026.physics.json (placeholders for schema tests)
    "base_housing_and_axle": 0.5, "worm_input": 0.06, "leg_swing_link": 0.55,
    "blade_upper": 0.05, "blade_lower": 0.07, "pushrod": 0.08,
}

# ---------------------------------------------------------------------------
# CLOSED-FORM SLIDER-CRANK (reused verbatim from rig_foot_7_1.py):
#   crank: knee pin (0,0) -> toe hinge, r = 0.075
#   conrod: toe hinge -> heel pin, L = 0.100
#   slider: heel pin on the piston, constrained to the shin line x = 0
# ---------------------------------------------------------------------------
CRANK_R = TOE_PIN_LOCAL.y
CONROD_L = (TOE_PIN_LOCAL - HEEL_PIN_NEUTRAL).length


def toe_hinge_pos(phi_deg: float) -> Vector:
    phi = math.radians(phi_deg)
    return Vector((-CRANK_R * math.sin(phi), CRANK_R * math.cos(phi), 0.0))


def heel_pin_y(phi_deg: float) -> float:
    t = toe_hinge_pos(phi_deg)
    return t.y - math.sqrt(CONROD_L * CONROD_L - t.x * t.x)


def conrod_angle_deg(phi_deg: float) -> float:
    t = toe_hinge_pos(phi_deg)
    d = Vector((0.0, heel_pin_y(phi_deg), 0.0)) - t
    return math.degrees(math.atan2(d.x, -d.y))


def pushrod_drop_mm(phi_deg: float) -> float:
    return 1000.0 * (heel_pin_y(phi_deg) - HEEL_PIN_NEUTRAL.y)


# ---------------------------------------------------------------------------
# Legs.  Base orientation maps leg-local -> world:
#   front legs: R = Rx(90)            (local X -> +X fwd, local Y -> +Z up,
#                                      local Z -> -Y  => swing axis body-lateral)
#   rear legs:  R = Rz(180) @ Rx(90)  (arm points aft; swing axis still lateral)
# swing_sign_local: local swing angle that moves the foot toward world +X.
# yaw_sign: yaw command sign that sweeps the leg OUTBOARD (away from centerline).
# ---------------------------------------------------------------------------
R_FRONT = Matrix.Rotation(math.radians(90.0), 4, "X")
R_REAR = Matrix.Rotation(math.radians(180.0), 4, "Z") @ R_FRONT

LEG_SPECS = [
    # name, fore-aft sign, side sign (+Y = left), phase, mirrored, rear
    {"name": "FL", "fx": +1, "sy": +1, "phase": 0.0, "mirrored": False, "rear": False},
    {"name": "FR", "fx": +1, "sy": -1, "phase": math.pi, "mirrored": True, "rear": False},
    {"name": "RL", "fx": -1, "sy": +1, "phase": math.pi, "mirrored": False, "rear": True},
    {"name": "RR", "fx": -1, "sy": -1, "phase": 0.0, "mirrored": True, "rear": True},
]


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def import_leg_meshes() -> dict[str, bpy.types.Object]:
    """Import the leg gltf; return {base_name: obj} for the NEW meshes.

    Repeat imports suffix names (Mesh_0.001 ...) so we diff the scene and key
    by the base name.
    """
    before = {o.name for o in bpy.context.scene.objects}
    bpy.ops.import_scene.gltf(filepath=str(SRC))
    new = [o for o in bpy.context.scene.objects if o.name not in before]
    meshes = {o.name.split(".")[0]: o for o in new if o.type == "MESH"}
    extras = [o for o in new if o.type != "MESH"]
    for o in extras:  # be safe: the 7-1 gltf has 48 flat mesh nodes, nothing else
        bpy.data.objects.remove(o)
    if len(meshes) != 48:
        raise RuntimeError(f"expected 48 meshes per leg import, got {len(meshes)}")
    return meshes


def rebase_mesh_origin(obj: bpy.types.Object, pivot_local: Vector) -> None:
    """Rewrite vertex coords so the object's origin sits at the joint pivot and
    vertex coords are leg-local offsets from that pivot."""
    world_verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    obj.parent = None
    obj.matrix_world = Matrix.Translation(pivot_local)
    inv = obj.matrix_world.inverted()
    for v, w in zip(obj.data.vertices, world_verts):
        v.co = inv @ w


def parent_local(obj: bpy.types.Object, parent: bpy.types.Object,
                 loc: Vector | None = None) -> None:
    """Parent with identity inverse: obj.location is expressed in parent frame."""
    obj.parent = parent
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.location = Vector((0.0, 0.0, 0.0)) if loc is None else Vector(loc)


def add_empty(name: str, size: float = 0.03) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = "ARROWS"
    obj.empty_display_size = size
    bpy.context.collection.objects.link(obj)
    return obj


_MAT_CACHE: dict[str, bpy.types.Material] = {}


def get_material(name: str, color, alpha: float = 1.0, roughness: float = 0.5):
    if name in _MAT_CACHE:
        return _MAT_CACHE[name]
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, alpha)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        if alpha < 1.0:
            bsdf.inputs["Alpha"].default_value = alpha
    if alpha < 1.0:
        for attr, val in (("blend_method", "BLEND"), ("surface_render_method", "BLENDED")):
            try:
                setattr(m, attr, val)
            except (AttributeError, TypeError):
                pass
    _MAT_CACHE[name] = m
    return m


def make_housing_transparent(objs: dict[str, bpy.types.Object]) -> None:
    housing = objs["Mesh_5"]
    m = get_material("mat_housing_glass", (0.35, 0.12, 0.48), alpha=0.22, roughness=0.4)
    housing.data.materials.clear()
    housing.data.materials.append(m)
    housing.show_transparent = True


def mirror_leg_local_z(objs: dict[str, bpy.types.Object]) -> None:
    """Sagittal mirror for right-side legs: negate leg-local z + flip normals.

    Valid ONLY because every pivot sits at leg-local z=0 and every joint axis
    is +/-Z (planar mechanism): the mirrored leg runs the SAME joint values.
    Call AFTER rebase_mesh_origin (vertex coords are then leg-local offsets).
    """
    for obj in objs.values():
        for v in obj.data.vertices:
            v.co.z = -v.co.z
        obj.data.flip_normals()


# ---------------------------------------------------------------------------
# Leg build
# ---------------------------------------------------------------------------

def build_leg(spec: dict, mount_pos: Vector) -> dict:
    name = spec["name"]
    objs = import_leg_meshes()
    make_housing_transparent(objs)

    # rebase every mesh into the leg-local frame (chassis to the leg origin)
    pivots = {"worm_input": WORM_AXIS_POS, "leg_swing_link": GEAR_PIVOT,
              "blade_upper": KNEE_PIVOT, "blade_lower": TOE_PIN_LOCAL,
              "pushrod": HEEL_PIN_NEUTRAL}
    for group, names in GROUPS.items():
        for n in names:
            rebase_mesh_origin(objs[n], pivots[group])
            objs[n]["rigid_body_group"] = f"{name}_{group}"
    for n in CHASSIS:
        rebase_mesh_origin(objs[n], Vector((0.0, 0.0, 0.0)))
        objs[n]["rigid_body_group"] = f"{name}_base_housing_and_axle"

    if spec["mirrored"]:
        mirror_leg_local_z(objs)

    # hierarchy: mount (fixed) -> hip_yaw (animated about world Z) -> base
    # (leg-local frame) -> {chassis, j_worm, j_swing -> {j_knee, j_conrod, j_rod}}
    mount = add_empty(f"{name}_mount", 0.04)
    mount.location = mount_pos
    mount["attachment"] = "hip_yaw body-side frame (yaw_mount_plate in the leg json)"
    yaw = add_empty(f"{name}_hip_yaw", 0.05)
    parent_local(yaw, mount)
    yaw["joint_type"] = "revolute"
    yaw["axis_world"] = [0, 0, 1]
    yaw["limit_deg"] = [-YAW_DEG, YAW_DEG]
    base = add_empty(f"{name}_base", 0.06)
    parent_local(base, yaw)
    bpy.context.view_layer.update()
    rot = R_REAR if spec["rear"] else R_FRONT
    b_mat = Matrix.Translation(mount_pos) @ rot @ Matrix.Translation(-HIP_YAW_LOCAL)
    base.matrix_world = b_mat

    j_worm = add_empty(f"{name}_worm_spin", 0.02)
    parent_local(j_worm, base, WORM_AXIS_POS)
    j_worm["joint_type"] = "revolute"
    j_worm["axis_leg_local"] = [0, 1, 0]
    j_swing = add_empty(f"{name}_leg_swing", 0.03)
    parent_local(j_swing, base, GEAR_PIVOT)
    j_swing["joint_type"] = "revolute"
    j_swing["axis_leg_local"] = [0, 0, 1]
    j_swing["limit_deg"] = [-SWING_DEG, SWING_DEG]
    j_knee = add_empty(f"{name}_knee_blade", 0.02)
    parent_local(j_knee, j_swing, KNEE_PIVOT - GEAR_PIVOT)
    j_knee["joint_type"] = "revolute"
    j_knee["axis_leg_local"] = [0, 0, 1]
    j_knee["limit_deg"] = [ANKLE_MIN_DEG, ANKLE_MAX_DEG]
    j_conrod = add_empty(f"{name}_toe_hinge", 0.015)
    parent_local(j_conrod, j_swing, TOE_PIN_LOCAL - GEAR_PIVOT)
    j_conrod["joint_type"] = "revolute_passive"
    j_rod = add_empty(f"{name}_pushrod_follow", 0.015)
    parent_local(j_rod, j_swing, HEEL_PIN_NEUTRAL - GEAR_PIVOT)
    j_rod["joint_type"] = "prismatic_passive"

    for n in CHASSIS:
        parent_local(objs[n], base)
    for n in GROUPS["worm_input"]:
        parent_local(objs[n], j_worm)
    for n in GROUPS["leg_swing_link"]:
        parent_local(objs[n], j_swing)
    for n in GROUPS["blade_upper"]:
        parent_local(objs[n], j_knee)
    for n in GROUPS["blade_lower"]:
        parent_local(objs[n], j_conrod)
    for n in GROUPS["pushrod"]:
        parent_local(objs[n], j_rod)

    # knee-motor placeholder (gold ring above the knee pin, 7-1 convention)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.013, minor_radius=0.0022,
                                     major_segments=48, minor_segments=8)
    knee_motor = bpy.context.object
    knee_motor.name = f"{name}_actuator_knee_motor"
    knee_motor.data.materials.append(get_material("physics_knee_motor_gold", (1.0, 0.62, 0.0)))
    knee_motor["physics_role"] = "torque_motor_placeholder"
    knee_motor["drives_joint"] = f"{name}_knee_blade"
    knee_motor["motor_model"] = SERVO_MODEL
    knee_motor["max_torque_nm"] = SERVO_STALL_TORQUE_NM
    parent_local(knee_motor, j_swing, KNEE_PIVOT - GEAR_PIVOT + Vector((0.0, 0.0, 0.030)))
    knee_motor.rotation_euler = (0.0, 0.0, 0.0)

    # yaw drive placeholders: LARGE RUBBER PULLEY on the leg side (the
    # series-elastic element, yaws with the leg) + a small yaw motor cylinder
    # on the body side.  Both cosmetic, tagged.
    bpy.ops.mesh.primitive_cylinder_add(radius=0.050, depth=0.018, vertices=48)
    pulley = bpy.context.object
    pulley.name = f"{name}_yaw_rubber_pulley_placeholder"
    pulley.data.materials.append(get_material("mat_rubber_pulley", (0.09, 0.09, 0.10),
                                              roughness=0.9))
    pulley["physics_role"] = "series_elastic_belt_pulley_placeholder"
    pulley["note"] = "large rubber pulley = the compliant element of the yaw belt drive"
    parent_local(pulley, base, HIP_YAW_LOCAL + Vector((0.0, 0.073, 0.0)))
    pulley.rotation_euler = (math.pi / 2.0, 0.0, 0.0)  # cylinder axis -> leg-local Y (yaw axis)

    bpy.ops.mesh.primitive_cylinder_add(radius=0.014, depth=0.055, vertices=32)
    yaw_motor = bpy.context.object
    yaw_motor.name = f"{name}_yaw_motor_placeholder"
    yaw_motor.data.materials.append(get_material("mat_yaw_motor", (0.75, 0.30, 0.05)))
    yaw_motor["physics_role"] = "torque_motor_placeholder"
    yaw_motor["drives_joint"] = f"{name}_hip_yaw"
    yaw_motor["note"] = "body-side yaw motor; belt to the rubber pulley (not modeled)"
    parent_local(yaw_motor, mount, Vector((-spec["fx"] * 0.075, 0.0, 0.073)))

    return {"spec": spec, "mount": mount, "yaw": yaw, "base": base, "b_mat": b_mat.copy(),
            "j_worm": j_worm, "j_swing": j_swing, "j_knee": j_knee,
            "j_conrod": j_conrod, "j_rod": j_rod, "objs": objs,
            "swing_sign_local": -1.0 if spec["rear"] else 1.0,
            "yaw_sign": (1.0 if spec["sy"] * spec["fx"] > 0 else -1.0)}


def verify_leg_signs(legs: list[dict]) -> None:
    """Numeric self-check: +swing_sign_local * local swing must move the foot
    toward world +X (forward) for every leg."""
    bpy.context.view_layer.update()
    for leg in legs:
        j = leg["j_swing"]
        tip_local = PUSHROD_TIP_NEUTRAL - GEAR_PIVOT
        x0 = (j.matrix_world @ tip_local).x
        j.rotation_euler = (0.0, 0.0, leg["swing_sign_local"] * math.radians(5.0))
        bpy.context.view_layer.update()
        x1 = (j.matrix_world @ tip_local).x
        j.rotation_euler = (0.0, 0.0, 0.0)
        if x1 - x0 <= 0.0:
            raise RuntimeError(f"{leg['spec']['name']}: swing_sign_local is wrong "
                               f"(dx={x1 - x0:+.4f})")
    bpy.context.view_layer.update()


# ---------------------------------------------------------------------------
# Torso (PLACEHOLDER, sized from the measured leg)
# ---------------------------------------------------------------------------

def measure_leg(objs: dict[str, bpy.types.Object]) -> dict:
    """Leg-local measurements taken from the FIRST imported leg (post-rebase,
    pre-mount): housing yaw-sweep radius, housing width/height, foot reach."""
    def points(names):
        out = []
        for n in names:
            o = objs[n]
            pivot = o.matrix_world.translation  # rebased: T(pivot), no parent
            out.extend(pivot + v.co for v in o.data.vertices)
        return out

    housing = points(CHASSIS + GROUPS["worm_input"])
    r_sweep = max(math.hypot(p.x - HIP_YAW_LOCAL.x, p.z) for p in housing)
    z_ext = max(p.z for p in housing) - min(p.z for p in housing)
    y_max = max(p.y for p in housing)
    # deepest foot reach below the mount plane (leg-local y=0) during the WALK
    # envelope: pushrod tip swung about the gear pivot with the gait's
    # swing/blade phase coupling.  tip' = pivot + Rz(sig) @ (tip - pivot).
    arm_x = -GEAR_PIVOT.x  # tip sits at x=0, +0.16269 ahead of the swing axle
    depth = 0.0
    for k in range(721):
        th = 2.0 * math.pi * k / 720.0
        sig = math.radians(GAIT_SWING_AMP * math.sin(th))
        phi = GAIT_BLADE_MID + GAIT_BLADE_AMP * math.cos(th)
        tip_y = PUSHROD_TIP_NEUTRAL.y + (heel_pin_y(phi) - HEEL_PIN_NEUTRAL.y)
        y = arm_x * math.sin(sig) + tip_y * math.cos(sig)
        depth = max(depth, -y)
    stomp_depth = -(PUSHROD_TIP_NEUTRAL.y + heel_pin_y(-90.0) - HEEL_PIN_NEUTRAL.y)
    return {"r_sweep_housing": r_sweep, "housing_width": z_ext, "housing_y_max": y_max,
            "walk_foot_depth": depth, "stomp_foot_depth": stomp_depth}


def size_torso(meas: dict) -> dict:
    clear_fa = 0.064   # fore-aft clearance between the two swept housing circles
    clear_lat = 0.040  # lateral clearance on the width rule
    mount_sep = 2.0 * meas["r_sweep_housing"] + clear_fa
    torso_len = mount_sep + 0.10          # torso overhangs each mount by 50 mm
    torso_w = 2.0 * meas["housing_width"] + clear_lat
    torso_h = 0.10                        # placeholder
    mount_z = meas["walk_foot_depth"] + 0.002  # deepest walk foot pose skims 2 mm above floor
    pulley_top = meas["housing_y_max"] + 0.012 + 0.018  # pulley center offset + full depth
    bottom_z = mount_z + pulley_top + 0.006             # torso belly clears the pulleys
    return {"mount_sep": mount_sep, "len": torso_len, "width": torso_w, "height": torso_h,
            "mount_z": mount_z, "bottom_z": bottom_z, "center_z": bottom_z + torso_h / 2.0,
            "clear_fa": clear_fa, "clear_lat": clear_lat, "meas": meas}


def build_torso(dims: dict) -> dict:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, dims["center_z"]))
    torso = bpy.context.object
    torso.name = "torso_placeholder_box"
    torso.scale = (dims["len"], dims["width"], dims["height"])
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    torso.data.materials.append(get_material("mat_torso_placeholder", (0.30, 0.34, 0.42),
                                             roughness=0.7))
    torso["placeholder"] = True
    torso["note"] = ("PLACEHOLDER torso: box sized from the measured leg "
                     "(see physics.json torso.sizing_reasoning); mass TODO")

    bpy.ops.mesh.primitive_cylinder_add(radius=0.012, depth=0.18, vertices=24)
    striker = bpy.context.object
    striker.name = "striker_rod_placeholder"
    striker.rotation_euler = (0.0, math.pi / 2.0, 0.0)  # along +X out the front face
    striker.location = (dims["len"] / 2.0 + 0.055, 0.0, dims["center_z"])
    striker.data.materials.append(get_material("mat_striker", (0.80, 0.82, 0.86),
                                               roughness=0.25))
    striker["placeholder"] = True
    striker["cosmetic"] = True
    striker["note"] = "cosmetic striker-rod placeholder on the front face (pneumatic rod TBD)"

    bpy.ops.mesh.primitive_plane_add(size=60.0, location=(0.0, 0.0, 0.0))
    ground = bpy.context.object
    ground.name = "ground_plane_render_prop"
    ground.data.materials.append(get_material("mat_ground", (0.70, 0.70, 0.72),
                                              roughness=0.9))
    ground["cosmetic"] = True
    ground["note"] = "render prop only; not part of the robot"
    return {"torso": torso, "striker": striker, "ground": ground}


# ---------------------------------------------------------------------------
# Gait (keyframed kinematics -- explicitly NOT dynamics)
# ---------------------------------------------------------------------------

def gait_pose(frame: int, phase: float, yaw_sign: float) -> tuple[float, float, float]:
    """Return (yaw_deg_world, fwd_swing_deg, blade_deg) for a leg at `frame`.

    Walk: diagonal pairs in antiphase (`phase` 0 or pi).  Yaw and swing share
    the phase (the leg reaches forward and sweeps outboard together); the blade
    retracts (toward -10) while the foot travels forward (clearance) and
    extends (toward -60) during the stance half.  Finale: tuck to the
    strike-ready pose, then all four blades POWER-STRIKE to -90 together.
    """
    if frame <= WALK_END:
        th = 2.0 * math.pi * (frame - 1) / CYCLE + phase
        u = min(1.0, (frame - 1) / float(RAMP))
        e = u * u * (3.0 - 2.0 * u)
        yaw = yaw_sign * GAIT_YAW_AMP * e * math.sin(th)
        fwd = GAIT_SWING_AMP * e * math.sin(th)
        phi = GAIT_BLADE_MID + GAIT_BLADE_AMP * e * math.cos(th)
        return yaw, fwd, phi
    if frame <= TUCK_END:
        y0, f0, p0 = gait_pose(WALK_END, phase, yaw_sign)
        u = (frame - WALK_END) / float(TUCK_END - WALK_END)
        s = u * u * (3.0 - 2.0 * u)
        return (y0 * (1.0 - s), f0 * (1.0 - s), p0 + (STRIKE_READY_DEG - p0) * s)
    if frame <= STRIKE_END:
        u = (frame - TUCK_END) / float(STRIKE_END - TUCK_END)
        return 0.0, 0.0, STRIKE_READY_DEG + (ANKLE_MIN_DEG - STRIKE_READY_DEG) * u * u
    return 0.0, 0.0, ANKLE_MIN_DEG


def phase_name(frame: int) -> str:
    if frame <= RAMP:
        return "STAND -> WALK ramp-in"
    if frame <= WALK_END:
        return "WALK: kinematic trot (FL+RR / FR+RL antiphase)"
    if frame <= TUCK_END:
        return "PRE-STRIKE: tuck + wind blades to -10 deg"
    if frame <= STRIKE_END:
        return "POWER STRIKE: all 4 blades slam to -90 deg"
    return "HOLD: blades planted (worm self-locks the stance)"


def animate(legs: list[dict]) -> None:
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = N_FRAMES
    for leg in legs:
        spec = leg["spec"]
        for frame in range(1, N_FRAMES + 1):
            yaw, fwd, phi = gait_pose(frame, spec["phase"], leg["yaw_sign"])
            sigma = leg["swing_sign_local"] * fwd  # local leg_swing angle
            leg["yaw"].rotation_euler = (0.0, 0.0, math.radians(yaw))
            leg["j_swing"].rotation_euler = (0.0, 0.0, math.radians(sigma))
            leg["j_worm"].rotation_euler = (0.0, math.radians(-WORM_GEAR_RATIO * sigma), 0.0)
            leg["j_knee"].rotation_euler = (0.0, 0.0, math.radians(phi))
            leg["j_conrod"].location = toe_hinge_pos(phi) - GEAR_PIVOT
            leg["j_conrod"].rotation_euler = (0.0, 0.0, math.radians(conrod_angle_deg(phi)))
            leg["j_rod"].location = Vector((0.0, heel_pin_y(phi), 0.0)) - GEAR_PIVOT
            for obj in (leg["yaw"], leg["j_swing"], leg["j_worm"], leg["j_knee"],
                        leg["j_conrod"]):
                obj.keyframe_insert(data_path="rotation_euler", frame=frame)
            leg["j_conrod"].keyframe_insert(data_path="location", frame=frame)
            leg["j_rod"].keyframe_insert(data_path="location", frame=frame)
    # linear interpolation everywhere (dense per-frame keys anyway)
    for leg in legs:
        for key in ("yaw", "j_swing", "j_worm", "j_knee", "j_conrod", "j_rod"):
            ad = leg[key].animation_data
            if ad and ad.action:
                for fc in action_fcurves(ad.action):
                    for kp in fc.keyframe_points:
                        kp.interpolation = "LINEAR"


def action_fcurves(action):
    """Blender 5.x moved fcurves into layered actions (layers/strips/channelbags);
    fall back to the legacy flat collection on older builds."""
    if hasattr(action, "fcurves"):
        return list(action.fcurves)
    fcs = []
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                fcs.extend(bag.fcurves)
    return fcs


# ---------------------------------------------------------------------------
# Camera / lights / render
# ---------------------------------------------------------------------------

def setup_render() -> None:
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE"
    except TypeError:  # older 4.2-4.4 naming, just in case
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.fps = FPS
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"


def setup_camera_and_lights(ground: bpy.types.Object) -> None:
    scene = bpy.context.scene
    pts: list[Vector] = []
    meshes = [o for o in scene.objects if o.type == "MESH" and o is not ground]
    for frame in (1, 40, 70, 100, 120, 144):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        for o in meshes:
            pts.extend(o.matrix_world @ Vector(c) for c in o.bound_box)
    mn = Vector(tuple(min(p[i] for p in pts) for i in range(3)))
    mx = Vector(tuple(max(p[i] for p in pts) for i in range(3)))
    center = (mn + mx) * 0.5
    radius = max((p - center).length for p in pts)

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 50.0
    cam_data.sensor_width = 36.0
    cam_data.clip_end = 100.0
    half_h = math.atan(18.0 / cam_data.lens)
    half_v = math.atan(18.0 * RES_Y / RES_X / cam_data.lens)
    dist = radius / math.sin(min(half_h, half_v)) * 1.04
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    cam.location = center + Vector((1.25, 0.95, 0.52)).normalized() * dist
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam

    key = bpy.data.lights.new("key", type="SUN")
    key.energy = 3.0
    ko = bpy.data.objects.new("key", key)
    scene.collection.objects.link(ko)
    ko.rotation_euler = (Vector((-0.9, -0.55, -1.4))).to_track_quat("-Z", "Y").to_euler()
    fill = bpy.data.lights.new("fill", type="SUN")
    fill.energy = 1.1
    fo = bpy.data.objects.new("fill", fill)
    scene.collection.objects.link(fo)
    fo.rotation_euler = (Vector((0.7, 0.9, -0.9))).to_track_quat("-Z", "Y").to_euler()

    world = scene.world or bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.88, 0.88, 0.90, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    scene.frame_set(1)


def layout_frames() -> tuple[int, ...]:
    return (1, 55, 120)


def render_frames() -> None:
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    if LAYOUT_TEST:
        for frame in layout_frames():
            scene.frame_set(frame)
            scene.render.filepath = str(FRAMES_DIR / f"frame_{frame:04d}")
            bpy.ops.render.render(write_still=True)
        return
    scene.render.filepath = str(FRAMES_DIR / "frame_")
    bpy.ops.render.render(animation=True)


# ---------------------------------------------------------------------------
# Labels.  Parts map: ONE leg (FL) fully labeled + body aspects, leader-line
# style matching Test_Mesh_Leg_7-1-2026_parts_map.png.  Every animation frame
# gets the HUD + the four leg-ID tags.
# ---------------------------------------------------------------------------

LEG_ID_COLORS = {"FL": "#7fd0ff", "RR": "#7fd0ff", "FR": "#ffd27f", "RL": "#ffd27f"}
SHIN_MID_LOCAL = Vector((-0.0167, -0.15, 0.0063))

# body-aspect labels: pinned to the top-left corner (anchors are up at the torso)
LABEL_BODY = [
    ("[1] TORSO - placeholder box (mass TODO)",             "#8fa8c8", "torso"),
    ("[2] STRIKER ROD placeholder (cosmetic)",              "#c8ccd4", "striker"),
    ("[3] HIP-YAW MOUNTS x4 (axis vertical, +/-45 deg)",    "#ff8f5c", "mounts"),
]
# FL-leg part labels: auto-laid-out into left/right columns by projected anchor
LABEL_PARTS = [
    ("[4] YAW MOTOR + RUBBER PULLEY (SEA belt)",            "#e06840", "pulley"),
    ("[5] WORM input (pitch motor, 20:1 self-locking)",     "#c65cff", "worm"),
    ("[6] SECTOR GEAR - swings the leg",                    "#ff5cd0", "gear"),
    ("[7] DRIVE FRAME (red) - swings w/ gear",              "#ff3b30", "frame"),
    ("[8] GUIDE RAILS (twin)",                              "#b8c0c8", "rails"),
    ("[9] KNEE CARRIER + KNEE MOTOR (gold)",                "#ffb020", "carrier"),
    ("[10] SHIN RAIL",                                      "#9aa4ad", "shin"),
    ("[11] BLADE UPPER (knee crank, r 75 mm)",              "#cf86ff", "blade_u"),
    ("[12] BLADE LOWER (conrod, L 100 mm)",                 "#e070ff", "blade_l"),
    ("[13] TOE PIN hinge (joins blade lengths)",            "#ffd21e", "toe"),
    ("[14] HEEL PIN (blade ear -> piston)",                 "#ffa040", "heel"),
    ("[15] PUSHROD - piston (slider-crank closure)",        "#e8a020", "rod"),
    ("[16] DISTAL CARRIER + bushing",                       "#b070e0", "distal"),
]


def parts_map_anchors(legs: list[dict], dims: dict) -> dict:
    """World-space anchor points at frame 1 (neutral: yaw=0, swing=0, blade=-35)."""
    fl = next(l for l in legs if l["spec"]["name"] == "FL")
    b = fl["b_mat"]
    phi = GAIT_BLADE_MID
    toe = toe_hinge_pos(phi)
    heel = Vector((0.0, heel_pin_y(phi), 0.0))
    local = {
        "pulley": HIP_YAW_LOCAL + Vector((0.0, 0.073, 0.0)),
        "worm": WORM_AXIS_POS + Vector((0.0, 0.020, 0.0)),
        "gear": GEAR_PIVOT + Vector((0.0455, 0.0, 0.0)),
        "frame": Vector((-0.163, 0.0266, 0.0)),
        "rails": Vector((-0.10, -0.0167, 0.0063)),
        "carrier": Vector((-0.020, 0.014, 0.016)),
        "shin": Vector((-0.0167, -0.15, 0.0063)),
        "blade_u": toe * 0.5,
        "blade_l": Vector(((toe.x + heel.x) * 0.5 + 0.012, (toe.y + heel.y) * 0.5, 0.008)),
        "toe": toe,
        "heel": heel,
        "rod": Vector((0.0, -0.262, 0.008)),
        "distal": Vector((-0.0083, -0.245, 0.010)),
    }
    anchors = {k: b @ v for k, v in local.items()}
    anchors["torso"] = Vector((0.02, 0.0, dims["center_z"] + dims["height"] / 2.0))
    anchors["striker"] = Vector((dims["len"] / 2.0 + 0.10, 0.0, dims["center_z"]))
    anchors["mounts"] = [Vector(l["mount"].location) for l in legs]  # 4 leader lines
    return anchors


def compute_tracks(legs: list[dict]) -> dict:
    """Per-frame HUD text + leg-ID tag pixel positions."""
    scene = bpy.context.scene
    cam = scene.camera

    def px(world: Vector) -> tuple[int, int]:
        co = world_to_camera_view(scene, cam, Vector(world))
        return (int(min(max(co.x, 0.0), 1.0) * RES_X),
                int((1.0 - min(max(co.y, 0.0), 1.0)) * RES_Y))

    frames = layout_frames() if LAYOUT_TEST else range(1, N_FRAMES + 1)
    tracks: dict = {}
    for frame in frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        tags = []
        for leg in legs:
            name = leg["spec"]["name"]
            w = leg["j_swing"].matrix_world @ (SHIN_MID_LOCAL - GEAR_PIVOT)
            tags.append((name, LEG_ID_COLORS[name], px(w)))
        yaw, fwd, phi = gait_pose(frame, 0.0, 1.0)  # FL values (phase 0, +yaw outboard)
        hud = (f"frame {frame:3d}/{N_FRAMES}  |  {phase_name(frame)}  |  FL: yaw {yaw:+5.1f} "
               f"deg  swing {fwd:+5.1f} deg  blade {phi:+5.1f} deg  pushrod "
               f"{pushrod_drop_mm(phi):+5.1f} mm  |  keyframed kinematics, NOT dynamics")
        tracks[frame] = {"tags": tags, "hud": hud}
    return {"tracks": tracks, "px": px}


def annotate_frames(legs: list[dict], dims: dict) -> None:
    if LABELED_DIR.exists():
        shutil.rmtree(LABELED_DIR)
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    data = compute_tracks(legs)
    tracks, px = data["tracks"], data["px"]
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    anchors = parts_map_anchors(legs, dims)

    def base_cmd(frame: int) -> list[str]:
        spec = tracks[frame]
        cmd = ["-strokewidth", "2"]
        for name, color, (ax, ay) in spec["tags"]:
            cmd += ["-fill", color, "-stroke", "none", "-draw", f"circle {ax},{ay} {ax + 4},{ay}"]
        cmd += ["-strokewidth", "1", "-pointsize", "16"]
        for name, color, (ax, ay) in spec["tags"]:
            cmd += ["-stroke", "none", "-undercolor", "#000000bb", "-fill", color,
                    "-annotate", f"+{ax + 8}+{ay + 5}", f" {name} "]
        cmd += ["-pointsize", "13", "-undercolor", "#000000cc", "-fill", "#cfe2ff",
                "-annotate", f"+14+{RES_Y - 12}", f" {spec['hud']} "]
        return cmd

    frames = layout_frames() if LAYOUT_TEST else range(1, N_FRAMES + 1)
    for frame in frames:
        src = FRAMES_DIR / f"frame_{frame:04d}.png"
        dst = LABELED_DIR / f"frame_{frame:04d}.png"
        subprocess.run(["magick", str(src), *base_cmd(frame), str(dst)],
                       check=True, capture_output=True)

    # ---- parts map: frame 1 + full label set (leader-line style of 7-1) ----
    # auto-layout: FL part labels go to the column on their anchor's side of the
    # frame, sorted top-to-bottom by anchor height so the leader lines fan
    # without crossing; body labels pin to the top-left corner.
    placed = []  # (text, color, tx, ty, key)
    for i, (text, color, key) in enumerate(LABEL_BODY):
        placed.append((text, color, 12, 26 + 32 * i, key))
    cols = {"L": [], "R": []}
    for text, color, key in LABEL_PARTS:
        ax, ay = px(anchors[key])
        cols["L" if ax < 480 else "R"].append((ay, text, color, key))
    for side, items in cols.items():
        items.sort()
        x0 = 12 if side == "L" else 748
        y0 = 190 if side == "L" else 150
        step = min(34, max(28, (RES_Y - 60 - y0) // max(len(items), 1)))
        for i, (_ay, text, color, key) in enumerate(items):
            placed.append((text, color, x0, y0 + step * i, key))

    cmd = ["magick", str(FRAMES_DIR / "frame_0001.png"), "-strokewidth", "2"]
    for text, color, tx, ty, key in placed:
        targets = anchors[key] if isinstance(anchors[key], list) else [anchors[key]]
        sx = tx - 4 if tx > RES_X / 2 else tx + 6
        for w in targets:
            ax, ay = px(w)
            cmd += ["-fill", "none", "-stroke", color + "aa",
                    "-draw", f"line {sx},{ty - 5} {ax},{ay}"]
            cmd += ["-fill", color, "-stroke", "none", "-draw", f"circle {ax},{ay} {ax + 4},{ay}"]
    cmd += ["-strokewidth", "1", "-pointsize", "14"]
    for text, color, tx, ty, _key in placed:
        cmd += ["-stroke", "none", "-undercolor", "#000000bb", "-fill", color,
                "-annotate", f"+{tx}+{ty}", f" {text} "]
    for name, color, (ax, ay) in tracks[1]["tags"]:
        cmd += ["-pointsize", "17", "-undercolor", "#000000bb", "-fill", color,
                "-annotate", f"+{ax + 8}+{ay + 5}", f" {name} "]
    cmd += ["-pointsize", "13", "-undercolor", "#000000cc", "-fill", "#cfe2ff",
            "-annotate", f"+14+{RES_Y - 12}",
            (" parts map: FL leg labeled; FR/RR mirrored across the sagittal plane; "
             "rear legs face aft  |  frame 1 (stand) ")]
    cmd += [str(PARTS_MAP_OUT)]
    subprocess.run(cmd, check=True, capture_output=True)


def make_gif() -> None:
    frames = sorted(LABELED_DIR.glob("frame_*.png"))
    if not frames:
        raise RuntimeError("no labeled frames")
    h = round(RES_Y * GIF_WIDTH / RES_X)
    try:  # Blender's bundled python ships without PIL -> ImageMagick fallback
        from PIL import Image
        images = [Image.open(p).resize((GIF_WIDTH, h), Image.LANCZOS)
                  .convert("P", palette=Image.Palette.ADAPTIVE) for p in frames]
        images[0].save(GIF_OUT, save_all=True, append_images=images[1:],
                       duration=int(1000 / FPS), loop=0, optimize=True)
    except ImportError:
        subprocess.run(["magick", "-delay", f"1x{FPS}", "-loop", "0", *map(str, frames),
                        "-resize", f"{GIF_WIDTH}x{h}", "-layers", "optimize",
                        str(GIF_OUT)], check=True)


def make_contact_sheet() -> None:
    # 4x3 spanning stand/ramp, trot, tuck, strike, hold
    picks = [1, 14, 26, 38, 50, 62, 74, 86, 98, 112, 119, 140]
    frames = [LABELED_DIR / f"frame_{f:04d}.png" for f in picks]
    subprocess.run(["montage", *map(str, frames), "-tile", "4x3",
                    "-geometry", "360x278+6+6", "-background", "#202024",
                    str(CONTACT_SHEET_OUT)], check=True)


# ---------------------------------------------------------------------------
# MJCF-conversion-ready metadata
# ---------------------------------------------------------------------------

def fit_pushrod_polycoef() -> dict:
    """Quartic fit s(phi) for MuJoCo <equality joint polycoef> (option B):
    pushrod slide value (m, along leg-local +Y, 0 at blade neutral) vs knee
    angle (rad)."""
    phis = np.linspace(math.radians(ANKLE_MIN_DEG), math.radians(ANKLE_MAX_DEG), 181)
    s = np.array([heel_pin_y(math.degrees(p)) - HEEL_PIN_NEUTRAL.y for p in phis])
    coef = np.polyfit(phis, s, 4)[::-1]  # ascending order c0..c4 (MuJoCo polycoef order)
    fit = sum(c * phis ** i for i, c in enumerate(coef))
    return {"polycoef_c0_to_c4": [round(float(c), 8) for c in coef],
            "max_fit_error_m": round(float(np.max(np.abs(fit - s))), 8),
            "domain_rad": [round(math.radians(ANKLE_MIN_DEG), 6),
                           round(math.radians(ANKLE_MAX_DEG), 6)]}


def qround(q: Quaternion) -> list[float]:
    return [round(v, 6) for v in (q.w, q.x, q.y, q.z)]


def vround(v) -> list[float]:
    return [round(float(x), 6) for x in v]


def leg_entry(leg: dict, dims: dict) -> dict:
    spec = leg["spec"]
    n = spec["name"]
    b = leg["b_mat"]
    rot = b.to_3x3()
    q = rot.to_quaternion()

    def world_pt(p_local: Vector) -> list[float]:
        return vround(b @ Vector(p_local))

    def world_ax(a_local) -> list[float]:
        return vround(rot @ Vector(a_local))

    joints = [
        {"name": f"{n}_hip_yaw", "type": "revolute", "parent": "torso",
         "child": f"{n}_base_housing_and_axle",
         "origin_leg_local": vround(HIP_YAW_LOCAL), "origin_world_spawn": world_pt(HIP_YAW_LOCAL),
         "axis_leg_local": [0, 1, 0], "axis_world_spawn": [0, 0, 1],
         "limit_deg": [-YAW_DEG, YAW_DEG],
         "note": "PLACEHOLDER ROM +/-45 (leg json); series-elastic belt drive"},
        {"name": f"{n}_leg_swing", "type": "revolute", "parent": f"{n}_base_housing_and_axle",
         "child": f"{n}_leg_swing_link",
         "origin_leg_local": vround(GEAR_PIVOT), "origin_world_spawn": world_pt(GEAR_PIVOT),
         "axis_leg_local": [0, 0, 1], "axis_world_spawn": world_ax((0, 0, 1)),
         "limit_deg": [-SWING_DEG, SWING_DEG],
         "note": "worm 20:1 on the sector gear; SELF-LOCKING (non-backdrivable)"},
        {"name": f"{n}_knee_blade", "type": "revolute", "parent": f"{n}_leg_swing_link",
         "child": f"{n}_blade_upper",
         "origin_leg_local": vround(KNEE_PIVOT), "origin_world_spawn": world_pt(KNEE_PIVOT),
         "axis_leg_local": [0, 0, 1], "axis_world_spawn": world_ax((0, 0, 1)),
         "limit_deg": [ANKLE_MIN_DEG, ANKLE_MAX_DEG],
         "note": "POWERED downward strike (toggle-press); -90 = blade horizontal, toe out front"},
        {"name": f"{n}_toe_hinge", "type": "revolute_passive", "parent": f"{n}_blade_upper",
         "child": f"{n}_blade_lower",
         "origin_leg_local": vround(TOE_PIN_LOCAL), "origin_world_spawn": world_pt(TOE_PIN_LOCAL),
         "axis_leg_local": [0, 0, 1], "axis_world_spawn": world_ax((0, 0, 1)),
         "note": "passive; part of the slider-crank loop"},
        {"name": f"{n}_heel_pin", "type": "revolute_passive", "parent": f"{n}_blade_lower",
         "child": f"{n}_pushrod",
         "origin_leg_local": vround(HEEL_PIN_NEUTRAL),
         "origin_world_spawn": world_pt(HEEL_PIN_NEUTRAL),
         "axis_leg_local": [0, 0, 1], "axis_world_spawn": world_ax((0, 0, 1)),
         "note": "passive; the LOOP-CLOSURE joint (see mjcf_conversion.loop_closure)"},
        {"name": f"{n}_pushrod_slide", "type": "prismatic_passive",
         "parent": f"{n}_leg_swing_link", "child": f"{n}_pushrod",
         "origin_leg_local": vround(BUSHING_CENTER),
         "origin_world_spawn": world_pt(BUSHING_CENTER),
         "axis_leg_local": [0, 1, 0], "axis_world_spawn": world_ax((0, 1, 0)),
         "travel_m": [-0.04114, -0.00029],
         "note": "piston colinear with the shin rail; 41.14 mm travel at blade -90"},
    ]
    actuators = [
        {"name": f"{n}_hip_yaw_motor", "type": "series_elastic_belt_drive",
         "drives_joint": f"{n}_hip_yaw", "motor_model": SERVO_MODEL,
         "motor_mass_kg": SERVO_MASS_KG, "max_torque_nm": SERVO_STALL_TORQUE_NM,
         "transmission": "belt + LARGE RUBBER PULLEY on the leg side (the compliant element)",
         "gear_ratio": YAW_BELT_RATIO, "max_joint_torque_nm": SERVO_STALL_TORQUE_NM * YAW_BELT_RATIO,
         "series_stiffness_nm_per_rad": "TODO bench-measure",
         "series_damping": "TODO rubber hysteresis",
         "note": "model motor->spring/damper->joint, NOT a rigid gear (leg json 2026-07-03)"},
        {"name": f"{n}_worm_pitch_motor", "type": "torque_motor",
         "drives_joint": f"{n}_leg_swing", "motor_model": SERVO_MODEL,
         "motor_mass_kg": SERVO_MASS_KG, "max_torque_nm": SERVO_STALL_TORQUE_NM,
         "gear_ratio": WORM_GEAR_RATIO,
         "max_joint_torque_nm": SERVO_STALL_TORQUE_NM * WORM_GEAR_RATIO,
         "note": "worm on ~20T sector gear; self-locking: holds stance unpowered"},
        {"name": f"{n}_knee_strike_motor", "type": "torque_motor",
         "drives_joint": f"{n}_knee_blade", "motor_model": SERVO_MODEL,
         "motor_mass_kg": SERVO_MASS_KG, "max_torque_nm": SERVO_STALL_TORQUE_NM,
         "gear_ratio": 1.0, "max_joint_torque_nm": SERVO_STALL_TORQUE_NM,
         "note": "POWERED strike, gravity-assisted; toggle-press: dh/dphi -> 0 near stowed "
                 "(huge pinch force), 74.6 mm/rad at full extension (~26 N tip force at stall)"},
    ]
    return {
        "mirrored": spec["mirrored"],
        "orientation": "rear (arm aft, Rz180@Rx90)" if spec["rear"] else "front (arm fwd, Rx90)",
        "gait_phase_deg": round(math.degrees(spec["phase"]), 1),
        "mount_transform": {
            "attachment_point_world": vround(leg["mount"].location),
            "quat_leg_local_to_world_wxyz": qround(q),
            "leg_frame_origin_world": world_pt(Vector((0.0, 0.0, 0.0))),
            "note": "world = attachment + R @ (p_leg_local - hip_yaw_local); quat is the PURE "
                    "ROTATION -- the sagittal mirror of FR/RR is baked into their mesh "
                    "vertex data (leg-local z -> -z), NOT into this transform",
        },
        "sign_conventions": {
            "swing_sign_local": leg["swing_sign_local"],
            "yaw_sign_outboard": leg["yaw_sign"],
            "note": "+swing_sign_local * leg_swing moves the foot toward world +X; "
                    "+yaw_sign * hip_yaw sweeps the leg outboard. Verified numerically at build.",
        },
        "masses_kg": dict(BODY_MASSES),
        "leg_total_mass_kg": round(sum(BODY_MASSES.values()), 3),
        "joints": joints,
        "actuators": actuators,
    }


def physics_json(legs: list[dict], dims: dict) -> dict:
    meas = dims["meas"]
    motor_j = SERVO_OUTPUT_INERTIA_EST
    return {
        "units": "meters_kilograms",
        "generated_by": "scripts/assemble_robot_7_3.py",
        "command": "blender --background --python scripts/assemble_robot_7_3.py",
        "source_leg": {"mesh": SRC.name, "physics": LEG_JSON.name,
                       "status": "fourth_pass_hip_yaw_CONFIRMED_at_body_mount"},
        "status": "first_pass_full_quadruped_assembly_placeholder_torso",
        "hardware_contract": {
            "max_robot_mass_lb": 6.0,
            "max_robot_mass_kg": 6.0 * 0.45359237,
            "motor_model": SERVO_MODEL,
            "motor_count": 12,
            "motor_mass_kg_each": SERVO_MASS_KG,
            "operating_voltage_v": 12.0,
            "stall_torque_nm_each": SERVO_STALL_TORQUE_NM,
            "no_load_speed_rad_s_each": SERVO_FREE_SPEED_RAD_S,
            "mass_application": "raw CAD masses below remain placeholders; "
                                "gen_mesh_robot_mjcf.py scales all non-servo masses "
                                "to the budget remaining after 12x motor mass",
        },
        "reference_pose": "spawn = all joints 0 (blade vertical), yaw/swing neutral; the "
                          "ANIMATION starts at blade -35 deg but all transforms below are "
                          "at the joint-zero reference pose",
        "torso": {
            "type": "box_PLACEHOLDER",
            "size_xyz_m": [round(dims["len"], 4), round(dims["width"], 4),
                           round(dims["height"], 4)],
            "center_world": [0.0, 0.0, round(dims["center_z"], 4)],
            "mass_kg": "TODO placeholder (structure not designed yet)",
            "mount_points_world": {l["spec"]["name"]: vround(l["mount"].location)
                                   for l in legs},
            "sizing_reasoning": {
                "housing_yaw_sweep_radius_m": round(meas["r_sweep_housing"], 4),
                "rule_length": "front/rear hip-yaw circles (housing swept +/-45) must not "
                               "overlap: mount_sep = 2*R_sweep + clearance "
                               f"= 2*{meas['r_sweep_housing']:.4f} + {dims['clear_fa']} "
                               f"= {dims['mount_sep']:.4f}; torso length = mount_sep + 0.10 "
                               "overhang",
                "housing_width_m": round(meas["housing_width"], 4),
                "rule_width": "~2x housing width + clearance = "
                              f"2*{meas['housing_width']:.4f} + {dims['clear_lat']} "
                              f"= {dims['width']:.4f}; mounts on the +/-width/2 edges",
                "mount_height_m": round(dims["mount_z"], 4),
                "rule_height": "deepest WALK foot pose (swing +/-15, blade to -60, closed-form "
                               f"slider-crank) reaches {meas['walk_foot_depth']:.4f} below the "
                               "mount plane; mounts placed so that pose skims 2 mm above the "
                               f"floor. The STOMP (blade -90) reaches {meas['stomp_foot_depth']:.4f} "
                               "-> blades bite ~3 mm into the floor plane at full strike "
                               "(intentional, reads as a strike; kinematic only)",
                "rule_belly": "torso bottom clears the rubber-pulley tops by 6 mm",
            },
        },
        "striker_placeholder": {
            "type": "cylinder_COSMETIC_PLACEHOLDER", "radius_m": 0.012, "length_m": 0.18,
            "pose": "front face center, along +X",
            "note": "stand-in for the pneumatic striker rod; no joint, no actuator entry",
        },
        "legs": {l["spec"]["name"]: leg_entry(l, dims) for l in legs},
        "animation": {
            "name": "walk_trot_then_power_strike", "fps": FPS, "frames": N_FRAMES,
            "kinematics_only": True,
            "walk": {"frames": [1, WALK_END], "cycle_s": CYCLE / FPS,
                     "diagonal_pairs_antiphase": ["FL+RR", "FR+RL"],
                     "yaw_sweep_deg": GAIT_YAW_AMP, "pitch_swing_deg": GAIT_SWING_AMP,
                     "blade_partial_deg": [GAIT_BLADE_MID - GAIT_BLADE_AMP,
                                           GAIT_BLADE_MID + GAIT_BLADE_AMP],
                     "pushrod": "closed-form slider-crank follower (reused from rig_foot_7_1)"},
            "finale": {"frames": [WALK_END + 1, N_FRAMES],
                       "tuck_frames": [WALK_END + 1, TUCK_END],
                       "strike_frames": [TUCK_END + 1, STRIKE_END],
                       "description": "all four blades POWER-STRIKE to -90 simultaneously "
                                      "(the toggle-press stomp), then hold"},
        },
        "mjcf_conversion": {
            "naming_convention": {
                "pattern": "{leg}_{joint} with legs FL/FR/RL/RR, matching "
                           "sim/robot/gen_robot_mjcf.py _leg_xml (n = prefix + leg name)",
                "mapping_to_gen_robot_mjcf": {
                    "{leg}_hip_yaw": "{leg}_abd (the hip x-axis dof slot)",
                    "{leg}_leg_swing": "{leg}_flex",
                    "{leg}_knee_blade": "{leg}_knee",
                    "{leg}_pushrod_slide": "{leg}_strike slot if the blade is wired as the "
                                           "weapon dof (see loop_closure option_b)",
                },
            },
            "loop_closure": {
                "problem": "toe_hinge + heel_pin + pushrod_slide form a closed 4-bar "
                           "(slider-crank) loop; MJCF kinematic trees cannot express it "
                           "directly",
                "option_a_equality_connect": {
                    "recipe": "build blade_lower as a child of blade_upper via {leg}_toe_hinge; "
                              "build pushrod as a child of leg_swing_link via {leg}_pushrod_slide "
                              "(slide, axis leg-local (0,1,0), range [-0.0412, 0]); then close "
                              "the loop with <equality><connect body1='{leg}_blade_lower' "
                              "body2='{leg}_pushrod' anchor='0 -0.100 0'/> -- the anchor is the "
                              "HEEL PIN in blade_lower local coords (0.100 below the toe hinge)",
                    "note": "connect constrains 3 translations; the loop is planar so one "
                            "component is redundant but consistent -- fine for MuJoCo's soft "
                            "solver. Start solref='0.002 1' solimp defaults.",
                },
                "option_b_closed_form": {
                    "recipe": "drop blade_lower+pushrod passive dofs entirely OR keep the slide "
                              "and enslave it: <equality><joint joint1='{leg}_pushrod_slide' "
                              "joint2='{leg}_knee_blade' polycoef='c0 c1 c2 c3 c4'/> using the "
                              "quartic fit below (slide value in m vs knee angle in rad, both "
                              "ABSOLUTE with qpos0 = 0 for the two joints, which is exactly "
                              "MuJoCo's qpos0-relative convention then); attach the "
                              "foot/strike geom to the pushrod tip",
                    "closed_form": "s(phi) = r*cos(phi) - sqrt(L^2 - r^2*sin(phi)^2) - (r - L); "
                                   "r=0.075, L=0.100; s(-pi/2) = -0.04114",
                    "quartic_fit": fit_pushrod_polycoef(),
                },
            },
            "suggested_defaults": {
                "armature_note": "reflected servo-output inertia = J_output * external_ratio^2; "
                                 f"J_output={motor_j} kg.m2 is an explicit estimate because "
                                 "Waveshare does not publish it",
                "per_axis_armature_kg_m2": {
                    "hip_yaw": round(motor_j * YAW_BELT_RATIO ** 2, 6),
                    "leg_swing": round(motor_j * WORM_GEAR_RATIO ** 2, 6),
                    "knee_blade": motor_j,
                },
                "hip_yaw_series_elasticity": "do NOT model as a rigid gear: either (a) add a "
                                             "rotor body + hinge behind {leg}_hip_yaw joined by "
                                             "a stiffness/damping equality (SEA), or (b) "
                                             "minimum-viable: position actuator whose kp = belt "
                                             "torsional stiffness (TODO bench) so yaw steering "
                                             "stays springy",
                "leg_swing_self_locking": "worm is non-backdrivable: high joint damping or "
                                          "brake-when-idle; holds stance at zero power",
            },
            "contact_guidance": {
                "colliding_geoms": "pushrod tip (foot), blade_lower (strike face), torso shell",
                "non_colliding": "housing, worm, sector gear, drive frame, rails, carriers, "
                                 "pulley placeholders: contype='0' conaffinity='0' (the "
                                 "gen_robot_mjcf cc_upper reduced-collision lean mode)",
                "note": "exclude intra-leg pairs (blade vs carrier slot overlaps in the real "
                        "mesh); keep foot-floor and blade-opponent pairs",
            },
        },
        "assembly_notes": [
            ("Mounting per task spec: leg-local -Y = world down, swing axis (leg-local Z) = "
             "body lateral (fore-aft stepping), yaw axis vertical at the attachment."),
            ("AMBIGUITY (arm direction): the spec fixes axes but not the sign of the leg's +X "
             "arm; front legs point the arm forward, rear legs are rotated 180 about vertical "
             "(arm aft) for fore/aft symmetry -- rear blades guard the rear."),
            ("AMBIGUITY (stride-via-yaw): with the prescribed mounting the yaw arm points "
             "fore-aft, so a symmetric yaw sweep at neutral moves the foot mostly LATERALLY; "
             "fore-aft stride in this animation comes from the +/-15 pitch swing while the "
             "+/-25 yaw sweep provides the design-intent workspace cone. TRUE stride-via-yaw "
             "would mount the arm laterally (outboard) instead -- flagged for the design pass."),
            ("AMBIGUITY (task brief said 'worm along X'): the worm SPIN AXIS is leg-local +Y "
             "(rig + json authoritative); the worm/housing assembly is OFFSET along -X from "
             "the knee. Interpreted as the latter."),
            ("MIRRORING: FR/RR are mirrored by negating leg-local z in the mesh data (+ normal "
             "flip). Valid because the mechanism is PLANAR (all axes +/-Z, all pivots at z=0): "
             "mirrored legs run identical joint values; only cosmetic lateral asymmetries flip."),
            ("Sign conventions (swing forward / yaw outboard) were verified NUMERICALLY at "
             "build time by perturbing each joint and checking the foot displacement."),
            "Masses are the 7-1 leg-json placeholders; torso mass TODO.",
            ("The animation is keyframed kinematics (frame_set + empties), NOT a physics sim; "
             "feet skim up to ~30 mm above the floor mid-gait since body height is fixed."),
        ],
    }


# ---------------------------------------------------------------------------
# Save / main
# ---------------------------------------------------------------------------

def save_outputs(meta: dict) -> None:
    JSON_OUT.write_text(json.dumps(meta, indent=2) + "\n")
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_OUT))
    try:
        bpy.ops.export_scene.gltf(filepath=str(GLB_OUT), export_format="GLB",
                                  export_animations=True, export_extras=True)
    except TypeError:
        bpy.ops.export_scene.gltf(filepath=str(GLB_OUT), export_format="GLB")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_scene()

    # first import doubles as the measurement pass: rebase in-place, measure,
    # THEN size the torso and mount all four legs.
    first = import_leg_meshes()
    pivots = {"worm_input": WORM_AXIS_POS, "leg_swing_link": GEAR_PIVOT,
              "blade_upper": KNEE_PIVOT, "blade_lower": TOE_PIN_LOCAL,
              "pushrod": HEEL_PIN_NEUTRAL}
    for group, names in GROUPS.items():
        for n in names:
            rebase_mesh_origin(first[n], pivots[group])
    for n in CHASSIS:
        rebase_mesh_origin(first[n], Vector((0.0, 0.0, 0.0)))
    bpy.context.view_layer.update()
    dims = size_torso(measure_leg(first))
    for o in first.values():  # the measurement copy is rebuilt properly per leg
        bpy.data.objects.remove(o)

    props = build_torso(dims)
    legs = []
    for spec in LEG_SPECS:
        mount = Vector((spec["fx"] * dims["mount_sep"] / 2.0,
                        spec["sy"] * dims["width"] / 2.0, dims["mount_z"]))
        legs.append(build_leg(spec, mount))
    verify_leg_signs(legs)
    animate(legs)
    setup_render()
    setup_camera_and_lights(props["ground"])

    if LAYOUT_TEST:
        render_frames()
        annotate_frames(legs, dims)
        print("LAYOUT_TEST done:", sorted(str(p) for p in LABELED_DIR.glob("*.png")),
              str(PARTS_MAP_OUT))
        return

    save_outputs(physics_json(legs, dims))
    render_frames()
    annotate_frames(legs, dims)
    make_gif()
    make_contact_sheet()
    print(f"TORSO len={dims['len']:.4f} w={dims['width']:.4f} h={dims['height']:.4f} "
          f"center_z={dims['center_z']:.4f} mount_sep={dims['mount_sep']:.4f} "
          f"mount_z={dims['mount_z']:.4f}")
    for leg in legs:
        print(f"MOUNT {leg['spec']['name']} {tuple(leg['mount'].location)} "
              f"mirrored={leg['spec']['mirrored']}")
    for p in (BLEND_OUT, GLB_OUT, JSON_OUT, GIF_OUT, CONTACT_SHEET_OUT, PARTS_MAP_OUT):
        print("OUT", p)


if __name__ == "__main__":
    main()
