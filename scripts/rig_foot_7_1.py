#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Rig the 7-1-2026 test mesh as a robotic FOOT and render a labeled full-ROM animation.

Run with:
  blender --background --python scripts/rig_foot_7_1.py

Mechanism (geometric introspection + design intent from the user):
  - A purple WORM (axis Y, hip motor input) inside the fixed drive housing
    meshes with a C-shaped SECTOR GEAR (~20T, axis Z) whose toothed rim is a
    circle centered exactly on the axle pin held in the HOUSING.
  - The gear grips the red DRIVE FRAME through its C-opening; the frame's
    bearing boss rides the axle pin. Gear + frame + twin GUIDE RAILS + knee
    carrier + shin rail + distal carrier form ONE swinging leg link:
    worm -> sector gear (20:1) -> the whole leg swings about the axle pin.
  - The purple blade is TWO lengths hinged at the yellow TOE PIN: the straight
    center strip (UPPER length, knee pin -> toe hinge, r 75 mm) is the crank a
    separate KNEE MOTOR rotates so the toe pivots OUT IN FRONT of the knee;
    the curved perforated plates (LOWER length, toe hinge -> heel pin,
    L 100 mm) are the conrod whose ear wraps the HEEL PIN on top of the
    orange PUSHROD. The distal-carrier bushing restricts the pushrod to pure
    piston motion colinear with the shin rail, so the slider-crank closure
    lowers the heel pin 41 mm toward the ground at full sweep-out.

All motion is keyframed kinematics (no Bullet sim), matching the 6-23 pipeline.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

LAYOUT_TEST = os.environ.get("FOOT_RIG_LAYOUT_TEST", "") == "1"

import bpy
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Test_Mesh_Leg_7-1-2026.gltf"
OUT_DIR = ROOT / "build" / "foot_rig"
FRAMES_DIR = OUT_DIR / "frames"
LABELED_DIR = OUT_DIR / "frames_labeled"
BLEND_OUT = ROOT / "Test_Mesh_Leg_7-1-2026_physics.blend"
GLB_OUT = ROOT / "Test_Mesh_Leg_7-1-2026_physics.glb"
JSON_OUT = ROOT / "Test_Mesh_Leg_7-1-2026.physics.json"
GIF_OUT = ROOT / "Test_Mesh_Leg_7-1-2026_full_rom.gif"
CONTACT_SHEET_OUT = ROOT / "Test_Mesh_Leg_7-1-2026_motion_contact_sheet.png"
ANKLE_SHEET_OUT = ROOT / "Test_Mesh_Leg_7-1-2026_ankle_detail_contact_sheet.png"
DRIVE_SHEET_OUT = ROOT / "Test_Mesh_Leg_7-1-2026_drive_detail_contact_sheet.png"
PARTS_MAP_OUT = ROOT / "Test_Mesh_Leg_7-1-2026_parts_map.png"

FPS = 12
N_FRAMES = 72

# Joint centers measured from the source geometry (meters).
WORM_AXIS_POS = Vector((-0.2185, 0.0, 0.0))    # worm spins about +Y here
GEAR_PIVOT = Vector((-0.16269, 0.0, 0.0))      # sector-gear rim circle fit center (== axle pin)
KNEE_PIVOT = Vector((0.0, 0.0, 0.0))           # knee pin, axis +Z; the foot blade pivots here
HEEL_PIN_NEUTRAL = Vector((0.0, -0.025, 0.0))  # blade heel ear == pushrod clevis pin
BUSHING_CENTER = Vector((0.0, -0.2405, 0.0))   # distal-carrier bushing the pushrod slides through
TOE_PIN_LOCAL = Vector((0.0, 0.075, 0.0))      # toe-tip spacer pin (blade local == world at neutral)
PUSHROD_TIP_NEUTRAL = Vector((0.0, -0.283, 0.0))

# Range of motion (validated against rendered frames).
# Design intent: the TOE PIN pivots OUT IN FRONT of the knee pin (+X, away from the drive
# frame), never backwards toward it. The heel ear swings back through the carrier's internal
# slot and the pushrod visibly extends/retracts along the shin (~26 mm of tip travel).
ANKLE_MIN_DEG = -90.0   # CW: toe swept fully out front (blade horizontal, toe forward)
ANKLE_MAX_DEG = 10.0    # CCW: slight back-reach past vertical for a natural return
SWING_DEG = 25.0        # worm-driven leg-swing about the gear axle pin: +/-25 deg
WORM_GEAR_RATIO = 20.0  # single-start worm on a ~20T sector gear -> 20 worm revs per gear rev
SERVO_MODEL = "Waveshare ST3215-HS"
SERVO_STALL_TORQUE_NM = 20.0 * 0.0980665  # 20 kgf.cm @ 12 V
SERVO_MASS_KG = 0.068

# Rigid-body grouping: imported object names are stable (unnamed gltf nodes -> Mesh_0..Mesh_47).
# The sector gear grips the red drive frame through its C-opening, the frame's bearing boss rides
# the axle pin held in the housing: gear + frame + rails + carriers = ONE swinging leg link.
GROUPS = {
    "worm_input": ["Mesh_18", "Mesh_3"],
    "leg_swing_link": ["Mesh_19", "Mesh_17", "Mesh_39", "Mesh_40", "Mesh_41", "Mesh_42",
                       "Mesh_43", "Mesh_44", "Mesh_36", "Mesh_20", "Mesh_21", "Mesh_37",
                       "Mesh_9", "Mesh_10", "Mesh_11", "Mesh_46", "Mesh_47", "Mesh_45",
                       "Mesh_38", "Mesh_24", "Mesh_25"],
    # the "blade" is TWO lengths hinged at the yellow toe pin:
    # upper = straight center strip, knee pin -> toe hinge (the knee-motor crank)
    "blade_upper": ["Mesh_35", "Mesh_1", "Mesh_34"],
    # lower = curved perforated plates, toe hinge -> heel pin (the conrod)
    "blade_lower": ["Mesh_22", "Mesh_23", "Mesh_32", "Mesh_33"],
    # heel pin + spacers ride the piston (the lower length's ear wraps the pin on top of it)
    "pushrod": ["Mesh_29", "Mesh_30", "Mesh_31", "Mesh_26", "Mesh_0", "Mesh_27", "Mesh_28"],
}
CHASSIS = ["Mesh_5", "Mesh_2", "Mesh_6", "Mesh_7", "Mesh_8", "Mesh_4", "Mesh_12", "Mesh_13",
           "Mesh_14", "Mesh_15", "Mesh_16"]

# Human-readable part registry -> burned into labels, JSON and custom properties.
PART_REGISTRY = {
    "drive_housing": {"meshes": ["Mesh_5"], "note": "purple shell, rendered see-through so the gear shows"},
    "worm": {"meshes": ["Mesh_18"], "note": "purple helical worm, motor input, axis Y"},
    "worm_axle": {"meshes": ["Mesh_3"], "note": "red shaft through the worm"},
    "motor_mount_shaft": {"meshes": ["Mesh_2", "Mesh_6", "Mesh_7", "Mesh_8"],
                          "note": "outboard red shaft + 3 orange lugs, static"},
    "sector_gear": {"meshes": ["Mesh_19"],
                    "note": "C-shaped ~20T gear ring; grips the drive frame, swings the leg link"},
    "gear_axle_pin": {"meshes": ["Mesh_4", "Mesh_12", "Mesh_13", "Mesh_15", "Mesh_14", "Mesh_16"],
                      "note": "red pin held in the HOUSING + bushing + retainers; leg-swing axle"},
    "drive_frame": {"meshes": ["Mesh_17"],
                    "note": "red window frame; bearing boss rides the axle pin, swings with the gear"},
    "guide_rails": {"meshes": ["Mesh_39", "Mesh_40"],
                    "note": "twin black rails, frame -> ankle carrier; swing with the frame"},
    "rail_screws": {"meshes": ["Mesh_41", "Mesh_42", "Mesh_43", "Mesh_44"], "note": "orange clamp screws"},
    "knee_carrier": {"meshes": ["Mesh_36", "Mesh_20", "Mesh_21"],
                     "note": "purple block + two side plates that hold the knee pin; the KNEE"},
    "shin_rail": {"meshes": ["Mesh_37", "Mesh_9", "Mesh_10", "Mesh_11"],
                  "note": "vertical black rail + dark-red set screws; swings with the leg"},
    "blade_upper_length": {"meshes": ["Mesh_35"],
                           "note": "straight center strip, knee pin -> toe hinge; knee-motor crank"},
    "blade_lower_length": {"meshes": ["Mesh_22", "Mesh_23"],
                           "note": "curved perforated plates, toe hinge -> heel pin; the conrod"},
    "knee_pin": {"meshes": ["Mesh_38", "Mesh_24", "Mesh_25"], "note": "red pivot pin + orange bearings"},
    "toe_hinge_pin": {"meshes": ["Mesh_1", "Mesh_32", "Mesh_33", "Mesh_34"],
                      "note": "yellow HINGE joining the two blade lengths at the top"},
    "heel_pin": {"meshes": ["Mesh_0", "Mesh_27", "Mesh_28"],
                 "note": "pin atop the piston; the lower length's ear wraps it"},
    "pushrod": {"meshes": ["Mesh_29", "Mesh_30", "Mesh_31", "Mesh_26"],
                "note": "orange rod + collar + clevis + clevis cross-pin; pure piston"},
    "distal_carrier": {"meshes": ["Mesh_46", "Mesh_47", "Mesh_45"],
                       "note": "purple blocks + orange bushing the pushrod slides through"},
}


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_source() -> dict[str, bpy.types.Object]:
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    bpy.ops.import_scene.gltf(filepath=str(SRC))
    objs = {o.name: o for o in bpy.context.scene.objects if o.type == "MESH"}
    if len(objs) != 48:
        raise RuntimeError(f"expected 48 meshes, got {len(objs)}")
    for part, spec in PART_REGISTRY.items():
        for mesh_name in spec["meshes"]:
            objs[mesh_name]["part_name"] = part
            objs[mesh_name]["part_note"] = spec["note"]
            objs[mesh_name]["source_file"] = SRC.name
    return objs


def rebase_mesh_origin(obj: bpy.types.Object, pivot_world: Vector) -> None:
    """Rewrite vertex coords so the object's origin sits at the joint pivot."""
    world_verts = [obj.matrix_world @ vert.co for vert in obj.data.vertices]
    obj.parent = None
    obj.matrix_world = Matrix.Translation(pivot_world)
    inv = obj.matrix_world.inverted()
    for vert, world in zip(obj.data.vertices, world_verts):
        vert.co = inv @ world


def parent_at_pivot(obj: bpy.types.Object, parent: bpy.types.Object) -> None:
    obj.parent = parent
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.location = (0.0, 0.0, 0.0)


def add_empty(name: str, loc: Vector, size: float = 0.03) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = "ARROWS"
    obj.empty_display_size = size
    obj.location = loc
    bpy.context.collection.objects.link(obj)
    return obj


def add_marker(name: str, loc: Vector, radius: float = 0.0035, color=(0.95, 0.95, 1.0, 1.0)) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=radius, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.data.name = name + "_mesh"
    m = bpy.data.materials.new(name + "_mat")
    m.diffuse_color = color
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
    obj.data.materials.append(m)
    obj["physics_role"] = "rotation_marker"
    return obj


def add_arc(name: str, pivot: Vector, radius: float, start_deg: float, end_deg: float) -> bpy.types.Object:
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    curve.bevel_depth = 0.0012
    curve.bevel_resolution = 2
    spline = curve.splines.new("POLY")
    n = 48
    spline.points.add(n - 1)
    for i in range(n):
        a = math.radians(start_deg + (end_deg - start_deg) * i / (n - 1))
        spline.points[i].co = (radius * math.cos(a), radius * math.sin(a), 0.019, 1.0)
    obj = bpy.data.objects.new(name, curve)
    obj.location = pivot
    bpy.context.collection.objects.link(obj)
    m = bpy.data.materials.new(name + "_mat")
    m.diffuse_color = (0.0, 1.0, 0.65, 1.0)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.0, 1.0, 0.65, 1.0)
    obj.data.materials.append(m)
    obj["physics_role"] = "range_of_motion_arc"
    return obj


def make_housing_transparent(objs: dict[str, bpy.types.Object]) -> None:
    """Give the housing its own semi-transparent material so worm+gear are visible."""
    housing = objs["Mesh_5"]
    m = bpy.data.materials.new("mat_housing_glass")
    m.diffuse_color = (0.35, 0.12, 0.48, 0.22)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.35, 0.12, 0.48, 1.0)
        bsdf.inputs["Alpha"].default_value = 0.22
        bsdf.inputs["Roughness"].default_value = 0.4
    m.blend_method = "BLEND"
    housing.data.materials.clear()
    housing.data.materials.append(m)
    housing.show_transparent = True


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

def knee_angle_deg(t: float) -> float:
    """Sinusoidal blade sweep between ANKLE_MIN_DEG and ANKLE_MAX_DEG (starts mid-range)."""
    mid = 0.5 * (ANKLE_MIN_DEG + ANKLE_MAX_DEG)
    amp = 0.5 * (ANKLE_MAX_DEG - ANKLE_MIN_DEG)
    return mid + amp * math.sin(2.0 * math.pi * t)


def swing_angle_deg(t: float) -> float:
    """Worm-driven leg-swing about the gear axle pin."""
    return SWING_DEG * math.sin(2.0 * math.pi * t)


def swung(p: Vector, sigma_deg: float) -> Vector:
    """Rotate a neutral-pose point with the leg-swing link about the gear axle pin."""
    s = math.radians(sigma_deg)
    d = Vector(p) - GEAR_PIVOT
    return GEAR_PIVOT + Vector((d.x * math.cos(s) - d.y * math.sin(s),
                                d.x * math.sin(s) + d.y * math.cos(s), d.z))


# Slider-crank dimensions (measured; all in the leg-link frame):
#   crank: knee pin (0,0) -> toe hinge, r = 0.075
#   conrod: toe hinge -> heel pin, L = 0.100 (== |(0,0.075)-(0,-0.025)| exactly)
#   slider: heel pin on the piston, constrained to the shin line x = 0
CRANK_R = TOE_PIN_LOCAL.y
CONROD_L = (TOE_PIN_LOCAL - HEEL_PIN_NEUTRAL).length


def toe_hinge_pos(phi_deg: float) -> Vector:
    """Toe-hinge position: tip of the knee-motor crank (upper blade length)."""
    phi = math.radians(phi_deg)
    return Vector((-CRANK_R * math.sin(phi), CRANK_R * math.cos(phi), 0.0))


def heel_pin_y(phi_deg: float) -> float:
    """Heel-pin height from the slider-crank closure.

    The piston is restricted to the shin line (x=0), so the heel pin sits
    where the 100 mm lower length reaches down from the toe hinge: as the
    blade sweeps out front the heel pin lowers toward the ground (41 mm at
    full sweep), weight keeping the joints seated.
    """
    t = toe_hinge_pos(phi_deg)
    return t.y - math.sqrt(CONROD_L * CONROD_L - t.x * t.x)


def conrod_angle_deg(phi_deg: float) -> float:
    """Rotation of the lower blade length (neutral: hanging straight down)."""
    t = toe_hinge_pos(phi_deg)
    d = Vector((0.0, heel_pin_y(phi_deg), 0.0)) - t
    return math.degrees(math.atan2(d.x, -d.y))


def build_rig(objs: dict[str, bpy.types.Object]) -> dict:
    # kinematic joint empties; knee + pushrod live INSIDE the swinging leg link
    j_worm = add_empty("joint_worm_spin", WORM_AXIS_POS)
    j_swing = add_empty("joint_leg_swing", GEAR_PIVOT)
    j_knee = add_empty("joint_knee_blade_flexion", KNEE_PIVOT)
    j_conrod = add_empty("joint_toe_hinge_lower_length", Vector(TOE_PIN_LOCAL))
    j_rod = add_empty("joint_pushrod_follower", HEEL_PIN_NEUTRAL)
    # freshly created objects report identity matrix_world until the depsgraph runs;
    # evaluate now so the matrix_parent_inverse captures below are correct
    bpy.context.view_layer.update()

    pivots = {"worm_input": (WORM_AXIS_POS, j_worm), "leg_swing_link": (GEAR_PIVOT, j_swing),
              "blade_upper": (KNEE_PIVOT, j_knee), "blade_lower": (Vector(TOE_PIN_LOCAL), j_conrod),
              "pushrod": (HEEL_PIN_NEUTRAL, j_rod)}
    for group, names in GROUPS.items():
        pivot, parent = pivots[group]
        for name in names:
            rebase_mesh_origin(objs[name], pivot)
            parent_at_pivot(objs[name], parent)
            objs[name]["rigid_body_group"] = group

    for name in CHASSIS:
        objs[name]["rigid_body_group"] = "base_housing_and_axle"

    # knee crank, toe-hinge conrod and piston all ride the swinging leg
    for child in (j_knee, j_conrod, j_rod):
        mw = child.matrix_world.copy()
        child.parent = j_swing
        child.matrix_parent_inverse = j_swing.matrix_world.inverted()
        child.matrix_world = mw

    # joint metadata (exported to glb extras)
    j_worm["joint_type"] = "revolute"
    j_worm["axis"] = [0, 1, 0]
    j_worm["note"] = "hip motor input; drives leg_swing through the sector gear 20:1"
    j_swing["joint_type"] = "revolute"
    j_swing["axis"] = [0, 0, 1]
    j_swing["limit_deg"] = [-SWING_DEG, SWING_DEG]
    j_knee["joint_type"] = "revolute"
    j_knee["axis"] = [0, 0, 1]
    j_knee["limit_deg"] = [ANKLE_MIN_DEG, ANKLE_MAX_DEG]
    j_conrod["joint_type"] = "revolute_passive"
    j_conrod["axis"] = [0, 0, 1]
    j_conrod["note"] = "yellow toe-pin hinge joining the two blade lengths"
    j_rod["joint_type"] = "prismatic"
    j_rod["axis"] = [0, 1, 0]
    j_rod["note"] = "piston: heel pin height from slider-crank closure"

    # knee motor placeholder (gold ring above the knee pin), same convention as the 6-23 rig
    bpy.ops.mesh.primitive_torus_add(major_radius=0.013, minor_radius=0.0022,
                                     major_segments=48, minor_segments=8,
                                     location=KNEE_PIVOT + Vector((0.0, 0.0, 0.030)))
    knee_motor = bpy.context.object
    knee_motor.name = "actuator_knee_motor"
    m = bpy.data.materials.new("physics_knee_motor_gold")
    m.diffuse_color = (1.0, 0.62, 0.0, 1.0)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (1.0, 0.62, 0.0, 1.0)
    knee_motor.data.materials.append(m)
    knee_motor["physics_role"] = "torque_motor_placeholder"
    knee_motor["drives_joint"] = "knee_blade_flexion"
    knee_motor["command_interface"] = "commanded_torque_nm"
    knee_motor["motor_model"] = SERVO_MODEL
    knee_motor["max_torque_nm"] = SERVO_STALL_TORQUE_NM
    mw = knee_motor.matrix_world.copy()
    knee_motor.parent = j_swing
    knee_motor.matrix_parent_inverse = j_swing.matrix_world.inverted()
    knee_motor.matrix_world = mw

    # rotation markers so spin/swing direction is visible in renders
    m_worm = add_marker("marker_worm_spin", WORM_AXIS_POS + Vector((0.0, 0.020, 0.0125)), 0.0030)
    m_worm.parent = j_worm
    m_worm.matrix_parent_inverse = j_worm.matrix_world.inverted()
    m_gear = add_marker("marker_gear_rim", GEAR_PIVOT + Vector((0.0455, 0.0, 0.0)), 0.0035)
    m_gear.parent = j_swing
    m_gear.matrix_parent_inverse = j_swing.matrix_world.inverted()
    m_toe = add_marker("marker_toe_pin", Vector(TOE_PIN_LOCAL), 0.0030, (1.0, 0.85, 0.1, 1.0))
    m_toe.parent = j_knee
    m_toe.matrix_parent_inverse = j_knee.matrix_world.inverted()
    m_tip = add_marker("marker_pushrod_tip", Vector(PUSHROD_TIP_NEUTRAL), 0.0030, (1.0, 0.85, 0.1, 1.0))
    m_tip.parent = j_rod
    m_tip.matrix_parent_inverse = j_rod.matrix_world.inverted()

    # ROM arcs: blade-tip sweep (rides the leg link) and leg-swing range (fixed, in the base frame)
    arc_knee = add_arc("rom_arc_knee_blade", KNEE_PIVOT, 0.085,
                       90.0 + ANKLE_MIN_DEG, 90.0 + ANKLE_MAX_DEG)
    mw = arc_knee.matrix_world.copy()
    arc_knee.parent = j_swing
    arc_knee.matrix_parent_inverse = j_swing.matrix_world.inverted()
    arc_knee.matrix_world = mw
    add_arc("rom_arc_leg_swing", GEAR_PIVOT, 0.054, -SWING_DEG, SWING_DEG)

    # keyframes
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = N_FRAMES
    for frame in range(1, N_FRAMES + 1):
        t = (frame - 1) / (N_FRAMES - 1)
        phi = knee_angle_deg(t)
        sigma = swing_angle_deg(t)
        psi = -WORM_GEAR_RATIO * sigma   # physical 20:1 worm ratio

        j_knee.rotation_euler = (0.0, 0.0, math.radians(phi))
        j_swing.rotation_euler = (0.0, 0.0, math.radians(sigma))
        j_worm.rotation_euler = (0.0, math.radians(psi), 0.0)
        # lower blade length: pinned to the crank tip (toe hinge), oriented so its
        # heel ear lands exactly on the piston line
        j_conrod.location = toe_hinge_pos(phi)
        j_conrod.rotation_euler = (0.0, 0.0, math.radians(conrod_angle_deg(phi)))
        # pure piston: x stays 0, no rotation — colinear with shin rail + distal carrier
        j_rod.location = Vector((0.0, heel_pin_y(phi), 0.0))
        j_rod.rotation_euler = (0.0, 0.0, 0.0)
        for obj in (j_knee, j_swing, j_worm, j_conrod):
            obj.keyframe_insert(data_path="rotation_euler", frame=frame)
        j_conrod.keyframe_insert(data_path="location", frame=frame)
        j_rod.keyframe_insert(data_path="location", frame=frame)

    return {"markers": {"worm": m_worm, "gear": m_gear, "toe": m_toe, "tip": m_tip},
            "joints": {"worm": j_worm, "swing": j_swing, "knee": j_knee,
                       "conrod": j_conrod, "rod": j_rod}}


# ---------------------------------------------------------------------------
# Camera / render
# ---------------------------------------------------------------------------

def setup_camera_and_lights() -> None:
    scene = bpy.context.scene
    points = []
    sample_frames = sorted({1, N_FRAMES // 4, N_FRAMES // 2, (3 * N_FRAMES) // 4, N_FRAMES})
    render_meshes = [o for o in scene.objects if o.type == "MESH"]
    for frame in sample_frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        for obj in render_meshes:
            points.extend(obj.matrix_world @ Vector(c) for c in obj.bound_box)
    mn = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    mx = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    center = (mn + mx) * 0.5
    size_x = mx.x - mn.x
    size_y = mx.y - mn.y

    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    rx, ry = scene.render.resolution_x, scene.render.resolution_y
    # cover both extents with margin, accounting for the canvas aspect ratio
    cam_data.ortho_scale = max(size_x * 1.14, size_y * 1.14 * rx / ry, 0.25)
    cam_data.clip_end = 100
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    cam.location = center + Vector((0.16, -0.20, 1.0)).normalized() * max(size_x, size_y) * 3.0
    # build orientation manually so image-up stays world +Y (to_track_quat pulls up toward world Z)
    look = (center - cam.location).normalized()
    cam_z = -look
    cam_x = Vector((0.0, 1.0, 0.0)).cross(cam_z).normalized()
    cam_y = cam_z.cross(cam_x)
    rot = Matrix((cam_x, cam_y, cam_z)).transposed().to_4x4()
    cam.matrix_world = Matrix.Translation(cam.location) @ rot
    scene.camera = cam

    key = bpy.data.lights.new("key", type="SUN")
    key.energy = 3.2
    ko = bpy.data.objects.new("key", key)
    scene.collection.objects.link(ko)
    ko.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    fill = bpy.data.lights.new("fill", type="SUN")
    fill.energy = 1.0
    fo = bpy.data.objects.new("fill", fill)
    scene.collection.objects.link(fo)
    fo.rotation_euler = (center - (cam.location * Vector((-1, -1, 1)))).to_track_quat("-Z", "Y").to_euler()

    world = scene.world or bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.88, 0.88, 0.90, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0


def setup_render() -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 840
    scene.render.resolution_y = 800
    scene.render.fps = FPS
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"


def render_frames() -> None:
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.filepath = str(FRAMES_DIR / "frame_")
    if LAYOUT_TEST:
        for frame in (1, 19, 37, 55):
            scene.frame_set(frame)
            scene.render.filepath = str(FRAMES_DIR / f"frame_{frame:04d}")
            bpy.ops.render.render(write_still=True)
        return
    bpy.ops.render.render(animation=True)


# ---------------------------------------------------------------------------
# Labels (numbered so parts can be referenced unambiguously)
# ---------------------------------------------------------------------------

LABEL_TABLE = [
    ("[1] WORM (hip motor input) - spins FAST",    "#c65cff",  10,  26, "worm"),
    ("[2] SECTOR GEAR - swings the leg",           "#ff5cd0",  10,  58, "gear_rim"),
    ("[3] Gear axle pin = swing pivot",            "#ff6b6b",  10,  90, "gear_axle"),
    ("[4] Drive frame (red) - swings w/ gear",     "#ff3b30",  10, 122, "drive_frame"),
    ("[5] Drive housing (see-through) - FIXED",    "#b48ce0",  10, 154, "housing"),
    ("[6] Motor mount shaft - FIXED",              "#e0a080",  10, 186, "motor_mount"),
    ("[7] Guide rails (twin) - swing w/ frame",    "#b8c0c8", 330, 356, "rails"),
    ("[8] KNEE carrier + side plates",             "#9f6bff", 275,  40, "carrier"),
    ("[9] Shin rail (swings with leg)",            "#9aa4ad", 300, 500, "shin"),
    ("[10] BLADE UPPER length (knee crank)",       "#cf86ff", 545,  26, "blade"),
    ("[11] Knee pin = crank pivot",                "#ff6b6b", 650, 300, "knee"),
    ("[12] Heel pin (lower length -> piston)",     "#ffa040", 600, 332, "heel"),
    ("[13] Toe pin = YELLOW HINGE (joins lengths)", "#ffd21e", 545,  58, "toe"),
    ("[14] PUSHROD - piston, up/down only",        "#e8a020", 630, 470, "rod_mid"),
    ("[15] Pushrod tip",                           "#ffd21e", 630, 745, "rod_tip"),
    ("[16] Distal carrier + bushing",              "#b070e0", 290, 640, "distal"),
    ("[17] KNEE MOTOR (gold) - drives crank",      "#ffb020", 275,  72, "knee_motor"),
    ("[18] BLADE LOWER length (toe -> heel)",      "#e070ff", 545,  90, "blade_lower"),
]
ARROW_KEYS = [("toe", "#ffd21e"), ("heel", "#ffa040"), ("rod_tip", "#ffd21e"),
              ("gear_rim", "#ff5cd0"), ("worm", "#c65cff")]


def compute_label_tracks(rig: dict) -> dict:
    scene = bpy.context.scene
    cam = scene.camera
    rx, ry = scene.render.resolution_x, scene.render.resolution_y

    def px(world: Vector) -> tuple[int, int]:
        co = world_to_camera_view(scene, cam, Vector(world))
        return (int(min(max(co.x, 0.0), 1.0) * rx), int((1.0 - min(max(co.y, 0.0), 1.0)) * ry))

    mk = rig["markers"]
    fixed_anchors = {
        "gear_axle": Vector((-0.16269, 0.0, 0.031)),
        "housing": Vector((-0.235, -0.044, 0.014)),
        "motor_mount": Vector((-0.2366, 0.028, 0.0)),
    }
    # anchors that ride the swinging leg link (rotated about the gear axle per frame)
    swing_anchors = {
        "drive_frame": Vector((-0.163, 0.0266, 0.0)),
        "rails": Vector((-0.10, -0.0167, 0.0063)),
        "carrier": Vector((-0.029, 0.020, 0.010)),
        "shin": Vector((-0.0167, -0.15, 0.0063)),
        "knee": Vector((0.0, 0.0, 0.026)),
        "knee_motor": Vector((0.0, 0.0, 0.032)),
        "distal": Vector((-0.0083, -0.245, 0.010)),
    }

    proj: dict = {}
    hud_vals: dict = {}
    for frame in range(1, N_FRAMES + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        t = (frame - 1) / (N_FRAMES - 1)
        phi = knee_angle_deg(t)
        sigma = swing_angle_deg(t)
        pin_y = heel_pin_y(phi)
        toe_local = toe_hinge_pos(phi)
        tip_w = mk["tip"].matrix_world.translation
        knee_w = swung(Vector((0.0, 0.0, 0.016)), sigma)
        anchors = dict(fixed_anchors)
        anchors.update({k: swung(v, sigma) for k, v in swing_anchors.items()})
        anchors["worm"] = mk["worm"].matrix_world.translation
        anchors["gear_rim"] = mk["gear"].matrix_world.translation
        anchors["toe"] = mk["toe"].matrix_world.translation
        anchors["heel"] = swung(Vector((0.0, pin_y, 0.016)), sigma)
        anchors["blade"] = (mk["toe"].matrix_world.translation + knee_w) * 0.5
        # mid-point of the lower length (toe hinge -> heel pin), in world coords
        anchors["blade_lower"] = swung(Vector(((toe_local.x + 0.0) * 0.5 + 0.012,
                                               (toe_local.y + pin_y) * 0.5, 0.008)), sigma)
        anchors["rod_mid"] = (anchors["heel"] + Vector((tip_w.x, tip_w.y, 0.0))) * 0.5
        anchors["rod_tip"] = tip_w
        proj[frame] = {k: px(v) for k, v in anchors.items()}
        # piston drop measured in the leg-link frame (negative = lowered toward the ground)
        drop_mm = 1000.0 * (pin_y - HEEL_PIN_NEUTRAL.y)
        hud_vals[frame] = (phi, sigma, drop_mm)

    tracks: dict = {}
    for frame in range(1, N_FRAMES + 1):
        labels = [(text, color, tx, ty, proj[frame][key]) for (text, color, tx, ty, key) in LABEL_TABLE]
        fa, fb = max(1, frame - 1), min(N_FRAMES, frame + 1)
        arrows = []
        for key, color in ARROW_KEYS:
            sx, sy = proj[frame][key]
            (ax, ay), (bx, by) = proj[fa][key], proj[fb][key]
            vx, vy = bx - ax, by - ay
            mag = math.hypot(vx, vy)
            if mag < 0.8:
                continue
            length = max(16.0, min(64.0, mag * 5.0))
            arrows.append((color, (sx, sy), (int(sx + vx / mag * length), int(sy + vy / mag * length))))
        phi, sigma, drop_mm = hud_vals[frame]
        hud = (f"frame {frame}/{N_FRAMES}  |  leg swing {sigma:+.1f} deg (worm 20:1)  |  "
               f"knee blade {phi:+.1f} deg (- = toe out front)  |  pushrod drop {drop_mm:+.1f} mm")
        tracks[frame] = {"labels": labels, "arrows": arrows, "hud": hud}
    return tracks


def annotate_frames(tracks: dict) -> None:
    if LABELED_DIR.exists():
        shutil.rmtree(LABELED_DIR)
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    frame_list = (1, 19, 37, 55) if LAYOUT_TEST else range(1, N_FRAMES + 1)
    for frame in frame_list:
        src = FRAMES_DIR / f"frame_{frame:04d}.png"
        dst = LABELED_DIR / f"frame_{frame:04d}.png"
        spec = tracks.get(frame, {})
        cmd = ["magick", str(src), "-strokewidth", "2"]
        for _text, color, tx, ty, (ax, ay) in spec.get("labels", []):
            cmd += ["-fill", "none", "-stroke", color + "aa", "-draw", f"line {tx + 6},{ty - 5} {ax},{ay}"]
            cmd += ["-fill", color, "-stroke", "none", "-draw", f"circle {ax},{ay} {ax + 4},{ay}"]
        cmd += ["-strokewidth", "3"]
        for color, (sx, sy), (ex, ey) in spec.get("arrows", []):
            cmd += ["-fill", "none", "-stroke", color, "-draw", f"line {sx},{sy} {ex},{ey}"]
            ang = math.atan2(ey - sy, ex - sx)
            for da in (math.radians(150), math.radians(-150)):
                hx, hy = int(ex + 10 * math.cos(ang + da)), int(ey + 10 * math.sin(ang + da))
                cmd += ["-draw", f"line {ex},{ey} {hx},{hy}"]
        cmd += ["-strokewidth", "1", "-pointsize", "14"]
        for text, color, tx, ty, _anchor in spec.get("labels", []):
            cmd += ["-stroke", "none", "-undercolor", "#000000bb", "-fill", color,
                    "-annotate", f"+{tx}+{ty}", f" {text} "]
        cmd += ["-pointsize", "14", "-undercolor", "#000000cc", "-fill", "#cfe2ff",
                "-annotate", "+40+790", f" {spec.get('hud', '')} "]
        cmd += [str(dst)]
        subprocess.run(cmd, check=True, capture_output=True)


def make_gif() -> None:
    frames = sorted(LABELED_DIR.glob("frame_*.png"))
    if not frames:
        raise RuntimeError("No labeled frames")
    try:
        from PIL import Image
        images = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE) for p in frames]
        images[0].save(GIF_OUT, save_all=True, append_images=images[1:],
                       duration=int(1000 / FPS), loop=0, optimize=True)
    except ImportError:
        delay_cs = max(1, round(100 / FPS))
        subprocess.run(["magick", "-delay", str(delay_cs), "-loop", "0",
                        *map(str, frames), str(GIF_OUT)], check=True)


def sheet_frames() -> list[int]:
    return [1, N_FRAMES // 4, N_FRAMES // 2, (3 * N_FRAMES) // 4, N_FRAMES]


def make_contact_sheet() -> None:
    frames = [LABELED_DIR / f"frame_{f:04d}.png" for f in sheet_frames()]
    subprocess.run(["montage", *map(str, frames), "-tile", "5x1", "-geometry", "320x208+8+8",
                    "-background", "#202024", str(CONTACT_SHEET_OUT)], check=True)


def make_detail_sheet(out_path: Path, crop: str, resize: str) -> None:
    crops_dir = OUT_DIR / ("crops_" + out_path.stem[-20:])
    if crops_dir.exists():
        shutil.rmtree(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)
    crops = []
    for f in sheet_frames():
        src = FRAMES_DIR / f"frame_{f:04d}.png"
        dst = crops_dir / f"crop_{f:04d}.png"
        subprocess.run(["magick", str(src), "-crop", crop, "+repage", "-resize", resize, str(dst)],
                       check=True)
        crops.append(dst)
    subprocess.run(["montage", *map(str, crops), "-tile", "5x1", "-geometry", "+8+8",
                    "-background", "#202024", str(out_path)], check=True)


def make_parts_map() -> None:
    shutil.copyfile(LABELED_DIR / "frame_0001.png", PARTS_MAP_OUT)


# ---------------------------------------------------------------------------
# Sidecar physics JSON (schema-compatible with the 6-23 file)
# ---------------------------------------------------------------------------

def physics_json() -> dict:
    tips = [round(1000.0 * (heel_pin_y(phi) - HEEL_PIN_NEUTRAL.y), 2)
            for phi in (ANKLE_MIN_DEG, ANKLE_MAX_DEG)]
    return {
        "units": "meters_assumed",
        "source": SRC.name,
        "status": "third_pass_swing_link_and_knee_motor_per_design_intent",
        "hardware_contract": {
            "motor_model": SERVO_MODEL,
            "motor_mass_kg_each": SERVO_MASS_KG,
            "operating_voltage_v": 12.0,
            "stall_torque_nm_each": SERVO_STALL_TORQUE_NM,
        },
        "bodies": {
            "base_housing_and_axle": {
                "mass_kg": 0.5, "friction": 0.65, "status": "fixed_base",
                "mesh_names": sorted(CHASSIS),
                "parts": {k: v["meshes"] for k, v in PART_REGISTRY.items()
                          if set(v["meshes"]) <= set(CHASSIS)},
                "note": "housing + motor mount shaft + gear axle pin; mounts to the robot body",
            },
            "worm_input": {"mass_kg": 0.06, "friction": 0.45, "mesh_names": GROUPS["worm_input"],
                           "axis_xyz": [0, 1, 0], "note": "hip motor input shaft"},
            "leg_swing_link": {"mass_kg": 0.55, "friction": 0.6,
                               "mesh_names": GROUPS["leg_swing_link"], "axis_xyz": [0, 0, 1],
                               "parts": {k: v["meshes"] for k, v in PART_REGISTRY.items()
                                         if set(v["meshes"]) <= set(GROUPS["leg_swing_link"])},
                               "note": "sector gear + drive frame + guide rails + knee carrier + "
                                       "shin rail + distal carrier: ONE rigid link swinging on the "
                                       "gear axle pin, worm-driven"},
            "blade_upper": {"mass_kg": 0.05, "friction": 0.7, "mesh_names": GROUPS["blade_upper"],
                            "axis_xyz": [0, 0, 1],
                            "note": "straight center strip, knee pin -> toe hinge; the knee-motor "
                                    "crank (r = 75 mm)"},
            "blade_lower": {"mass_kg": 0.07, "friction": 0.7, "mesh_names": GROUPS["blade_lower"],
                            "axis_xyz": [0, 0, 1],
                            "note": "curved perforated plates, toe hinge -> heel pin; the conrod "
                                    "(L = 100 mm)"},
            "pushrod": {"mass_kg": 0.08, "friction": 0.6, "mesh_names": GROUPS["pushrod"],
                        "note": "pure piston (incl. heel pin): the distal-carrier bushing restricts "
                                "it to up/down motion colinear with the shin rail (no tilt)"},
        },
        "joints": {
            "worm_spin": {"type": "revolute", "parent": "base_housing_and_axle",
                          "child": "worm_input", "origin_xyz": list(WORM_AXIS_POS),
                          "axis_xyz": [0, 1, 0], "note": "hip motor input, fast"},
            "leg_swing": {"type": "revolute", "parent": "base_housing_and_axle",
                          "child": "leg_swing_link", "origin_xyz": list(GEAR_PIVOT),
                          "axis_xyz": [0, 0, 1], "limit_deg": [-SWING_DEG, SWING_DEG],
                          "note": "worm drives the sector gear 20:1; gear grips the drive frame "
                                  "through its C-opening, so frame+rails+leg swing together"},
            "knee_blade_flexion": {"type": "revolute", "parent": "leg_swing_link",
                                   "child": "blade_upper",
                                   "origin_xyz": list(KNEE_PIVOT), "axis_xyz": [0, 0, 1],
                                   "limit_deg": [ANKLE_MIN_DEG, ANKLE_MAX_DEG],
                                   "note": "KNEE-motor-driven crank; origin in leg-link coords"},
            "toe_hinge": {"type": "revolute_passive", "parent": "blade_upper",
                          "child": "blade_lower", "origin_xyz": list(TOE_PIN_LOCAL),
                          "axis_xyz": [0, 0, 1],
                          "note": "yellow hinge at the top of the blade joining the two lengths"},
            "heel_pin_joint": {"type": "revolute_passive", "parent": "blade_lower",
                               "child": "pushrod", "origin_xyz": list(HEEL_PIN_NEUTRAL),
                               "axis_xyz": [0, 0, 1],
                               "note": "the lower length's ear wraps the heel pin on top of the "
                                       "piston; slider-crank closure - as the blade sweeps out "
                                       "the heel pin lowers toward the ground (weight keeps the "
                                       "joints seated)"},
            "pushrod_bushing_slide": {"type": "prismatic", "parent": "leg_swing_link",
                                      "child": "pushrod", "origin_xyz": list(BUSHING_CENTER),
                                      "axis_xyz": [0, 1, 0],
                                      "travel_mm_at_limits": [tips[0], tips[1]],
                                      "note": "bushing restricts the rod to piston motion only, "
                                              "colinear with the shin rail and distal carrier"},
            "toe_pin_reference": {"type": "pin_visual", "origin_xyz": list(TOE_PIN_LOCAL),
                                  "axis_xyz": [0, 0, 1], "note": "blade tip spacer pin, radius 75 mm"},
        },
        "actuators": {
            "hip_worm_drive_motor": {"type": "torque_motor", "drives_joint": "leg_swing",
                                     "motor_model": SERVO_MODEL, "motor_mass_kg": SERVO_MASS_KG,
                                     "command_interface": "commanded_torque_nm",
                                     "max_torque_nm": SERVO_STALL_TORQUE_NM,
                                     "gear_ratio": WORM_GEAR_RATIO,
                                     "max_joint_torque_nm": SERVO_STALL_TORQUE_NM * WORM_GEAR_RATIO,
                                     "status": "worm on motor mount shaft; 20:1 from ~20T sector gear"},
            "knee_motor": {"type": "torque_motor", "drives_joint": "knee_blade_flexion",
                           "motor_model": SERVO_MODEL, "motor_mass_kg": SERVO_MASS_KG,
                           "command_interface": "commanded_torque_nm",
                           "max_torque_nm": SERVO_STALL_TORQUE_NM, "gear_ratio": 1.0,
                           "max_joint_torque_nm": SERVO_STALL_TORQUE_NM,
                           "status": "separate motor in the KNEE rotates the upper blade length "
                                     "(crank); lower length + piston follow via the slider-crank; "
                                     "gold-ring placeholder at the knee pin"},
        },
        "animation": {
            "name": "full_range_of_motion", "fps": FPS, "frames": N_FRAMES,
            "leg_swing_deg": [-SWING_DEG, SWING_DEG],
            "knee_blade_deg": [ANKLE_MIN_DEG, ANKLE_MAX_DEG],
            "worm_deg": f"{WORM_GEAR_RATIO:.0f}x inverse swing angle (physical worm ratio)",
            "pushrod_drop_mm_at_limits": [tips[0], tips[1]],
        },
        "notes": [
            "48 unnamed flat mesh nodes; groups assigned from measured bounds/colors/shape.",
            "Sector gear rim circle fit: center (-0.16269, 0.00000), R 0.0495, ~20 teeth over ~270 deg.",
            "Design intent per user: drive frame [4] + guide rails [7] are NOT fixed - they swing "
            "with the sector gear about the gear axle pin (worm-driven).",
            "A separate KNEE motor at the knee carrier drives the foot blade + pushrod.",
            "The blade is TWO lengths hinged at the yellow toe pin: upper straight strip (knee "
            "crank, r 75 mm) and lower curved perforated plates (conrod, L 100 mm) whose ear "
            "wraps the heel pin on top of the piston. Slider-crank: as the blade sweeps out the "
            "heel pin lowers 41 mm toward the ground, weight keeping the joints seated. The "
            "distal-carrier bushing restricts the piston to up/down motion colinear with the "
            "shin rail (no tilt).",
            "Knee-blade ROM per design intent: toe pin pivots OUT IN FRONT of the knee pin "
            "(-90 deg, blade horizontal toe-forward) with a small +10 deg back-reach; the heel "
            "ear swings back through the knee carrier's internal slot region. Swing ROM +/-25 deg. "
            "Both verified against rendered frames.",
            "Mass/friction values are placeholders for downstream schema testing.",
            "The GIF is keyframed kinematics, not a Bullet rigid-body simulation.",
        ],
    }


def save_outputs(props: dict) -> None:
    JSON_OUT.write_text(json.dumps(props, indent=2) + "\n")
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_OUT))
    try:
        bpy.ops.export_scene.gltf(filepath=str(GLB_OUT), export_format="GLB",
                                  export_animations=True, export_extras=True)
    except TypeError:
        bpy.ops.export_scene.gltf(filepath=str(GLB_OUT), export_format="GLB")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_scene()
    objs = import_source()
    make_housing_transparent(objs)
    rig = build_rig(objs)
    setup_render()
    setup_camera_and_lights()
    tracks = compute_label_tracks(rig)
    if LAYOUT_TEST:
        render_frames()
        annotate_frames(tracks)
        print("LAYOUT_TEST frames:", sorted(str(p) for p in LABELED_DIR.glob("*.png")))
        return
    save_outputs(physics_json())
    render_frames()
    annotate_frames(tracks)
    make_gif()
    make_contact_sheet()
    # ankle crop: right-center of frame; drive crop: left of frame (computed after seeing frame 1)
    make_detail_sheet(ANKLE_SHEET_OUT, "470x380+360+110", "470x380")
    make_detail_sheet(DRIVE_SHEET_OUT, "400x330+20+120", "400x330")
    make_parts_map()
    print(f"BLEND {BLEND_OUT}")
    print(f"GLB {GLB_OUT}")
    print(f"PHYSICS_JSON {JSON_OUT}")
    print(f"GIF {GIF_OUT}")
    print(f"CONTACT_SHEET {CONTACT_SHEET_OUT}")
    print(f"PARTS_MAP {PARTS_MAP_OUT}")
    print(f"FRAMES {FRAMES_DIR}")


if __name__ == "__main__":
    main()
