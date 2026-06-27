#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an annotated Blender physics scene and full-ROM GIF for the test leg.

Run with:
  blender --background --python scripts/animate_leg_physics.py

The source glTF has unnamed flat mesh nodes, so this script uses the stable
imported mesh indices and visible geometry to build a first-pass kinematic
physics scaffold. Mass/friction values are placeholders and are written both
as Blender custom properties and a sidecar JSON file.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Test_Mesh_Leg_6-23-2026.gltf"
OUT_DIR = ROOT / "build" / "blender_leg"
FRAMES_DIR = OUT_DIR / "frames"
LABELED_DIR = OUT_DIR / "frames_labeled"   # frames with overlaid part/motion labels (used for the GIF)
BLEND_OUT = ROOT / "Test_Mesh_Leg_6-23-2026_physics.blend"
GLB_OUT = ROOT / "Test_Mesh_Leg_6-23-2026_physics.glb"
JSON_OUT = ROOT / "Test_Mesh_Leg_6-23-2026.physics.json"
GIF_OUT = ROOT / "Test_Mesh_Leg_6-23-2026_full_rom.gif"
CONTACT_SHEET_OUT = ROOT / "Test_Mesh_Leg_6-23-2026_motion_contact_sheet.png"
DETAIL_CONTACT_SHEET_OUT = ROOT / "Test_Mesh_Leg_6-23-2026_knee_detail_contact_sheet.png"
GEAR_DETAIL_CONTACT_SHEET_OUT = ROOT / "Test_Mesh_Leg_6-23-2026_red_worm_detail_contact_sheet.png"

FPS = 12
N_FRAMES = 72

# Inferred joint centers, scene units assumed to be meters.
WORM_AXIS = Vector((-0.2185, 0.0000, 0.0000))
HIP = Vector((-0.1788, 0.0000, 0.0000))
UPPER_OUTBOARD = Vector((-0.0250, 0.0000, 0.0000))
KNEE_DRIVER = Vector((0.0000, -0.0750, 0.0000))
KNEE = Vector((0.0000, -0.1850, 0.0000))
DISTAL_LINK = Vector((-0.0167, -0.2410, 0.0000))
TOOL_TIP = Vector((0.0000, -0.4330, 0.0000))

# LINEAR mechanism: the hip motor drives the purple perforated leg AND the yellow piston STRAIGHT UP
# AND DOWN together (along world -Y, colinear with the vertical black stabilizer rod). The piston
# slides in-line inside the FIXED distal carrier. NOTHING hinges — the knee is a rigid bend, not a
# powered joint, and the piston never swings/pivots.
CARRIER_GUIDE = Vector((0.0000, -0.2280, 0.0000))   # fixed carrier bore the piston slides inside
LEG_STROKE = 0.040                                   # vertical up/down travel (m) of leg + piston

# Imported Blender object names are stable for this glTF because the mesh nodes
# are unnamed and ordered Mesh_0..Mesh_29, with three named duplicate nodes.
RED_WORM_NAMES = {"Mesh_9"}
RED_WORM_WHEEL_NAMES = {"Mesh_10"}
FIXED_DRIVE_NAMES = {"Mesh_11", "Mesh_12", "Mesh_13"}
PURPLE_HINGE_NAMES = {
    "Mesh_4",
    "Mesh_5",
    "Mesh_6",
    "Mesh_7",
    "Mesh_20",
    "Mesh_21",
    "Mesh_26",
}
YELLOW_ROD_NAMES = {"Mesh_8"}


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_source() -> list[bpy.types.Object]:
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    bpy.ops.import_scene.gltf(filepath=str(SRC))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects imported from {SRC}")
    for obj in meshes:
        obj["source_file"] = SRC.name
        obj["physics_note"] = "visual mesh; see rigid-body proxy objects and sidecar JSON"
    return meshes


def mat(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    m = bpy.data.materials.new(name)
    m.diffuse_color = color
    return m


MAT_PIN = None
MAT_PROXY_BASE = None
MAT_PROXY_UPPER = None
MAT_PROXY_LOWER = None
MAT_AXIS = None
MAT_ARC = None
MAT_GROUND = None
MAT_MOTOR = None


def make_materials() -> None:
    global MAT_PIN, MAT_PROXY_BASE, MAT_PROXY_UPPER, MAT_PROXY_LOWER, MAT_AXIS, MAT_ARC, MAT_GROUND, MAT_MOTOR
    MAT_PIN = mat("physics_pin_cyan", (0.0, 0.85, 1.0, 1.0))
    MAT_PROXY_BASE = mat("physics_proxy_base", (0.15, 0.45, 1.0, 0.22))
    MAT_PROXY_UPPER = mat("physics_proxy_upper", (0.05, 1.0, 0.25, 0.22))
    MAT_PROXY_LOWER = mat("physics_proxy_lower", (0.0, 0.95, 0.75, 0.16))
    MAT_AXIS = mat("physics_axis_white", (0.95, 0.95, 1.0, 1.0))
    MAT_ARC = mat("physics_rom_arc", (0.0, 1.0, 0.65, 1.0))
    MAT_GROUND = mat("mat_ground_dark", (0.055, 0.055, 0.06, 1.0))
    MAT_MOTOR = mat("physics_torque_motor_orange", (1.0, 0.42, 0.0, 1.0))
    for m in (MAT_PROXY_BASE, MAT_PROXY_UPPER, MAT_PROXY_LOWER):
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Alpha"].default_value = m.diffuse_color[3]
        m.blend_method = "BLEND"
        m.use_screen_refraction = True


def parent_keep(obj: bpy.types.Object, parent: bpy.types.Object) -> None:
    mw = obj.matrix_world.copy()
    obj.parent = parent
    obj.matrix_parent_inverse = parent.matrix_world.inverted()
    obj.matrix_world = mw


def rebase_mesh_origin(obj: bpy.types.Object, pivot_world: Vector) -> None:
    """Move a mesh object's origin to pivot_world while preserving geometry.

    The source glTF stores every mesh with object origin at world zero and
    absolute vertex coordinates. Kinematic parenting would otherwise rotate the
    visual geometry around world zero instead of the hinge. This rewrites vertex
    coordinates into a local frame centered on the intended physical pivot.
    """
    world_verts = [obj.matrix_world @ vert.co for vert in obj.data.vertices]
    obj.parent = None
    obj.matrix_world = Matrix.Translation(pivot_world)
    inv = obj.matrix_world.inverted()
    for vert, world in zip(obj.data.vertices, world_verts):
        vert.co = inv @ world


def parent_at_pivot(obj: bpy.types.Object, parent: bpy.types.Object) -> None:
    """Parent obj whose origin already matches parent origin."""
    obj.parent = parent
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.location = (0.0, 0.0, 0.0)


def parent_local_offset(obj: bpy.types.Object, parent: bpy.types.Object, local_offset: Vector) -> None:
    """Parent a generated primitive using an explicit local offset."""
    rot = obj.rotation_euler.copy()
    obj.parent = parent
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.location = local_offset
    obj.rotation_euler = rot


def add_empty(name: str, loc: Vector, display: str = "ARROWS", size: float = 0.03) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = display
    obj.empty_display_size = size
    obj.location = loc
    bpy.context.collection.objects.link(obj)
    return obj


def add_pin(name: str, loc: Vector, radius: float = 0.006, depth: float = 0.035) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius, depth=depth, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.data.name = name + "_mesh"
    obj.data.materials.append(MAT_PIN)
    obj["physics_role"] = "pin"
    obj["axis"] = [0, 0, 1]
    return obj


def add_marker(name: str, loc: Vector, radius: float = 0.0035) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=radius, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.data.name = name + "_mesh"
    obj.data.materials.append(MAT_AXIS)
    obj["physics_role"] = "rotation_marker"
    return obj


def cylinder_between(
    name: str,
    start: Vector,
    end: Vector,
    radius: float,
    material: bpy.types.Material,
    render: bool = True,
) -> bpy.types.Object:
    mid = (start + end) * 0.5
    length = (end - start).length
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=length, location=mid)
    obj = bpy.context.object
    obj.name = name
    obj.data.name = name + "_mesh"
    obj.rotation_euler = (end - start).to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(material)
    obj.show_transparent = True
    obj.hide_render = not render
    return obj


def add_rigid_body(obj: bpy.types.Object, kind: str, mass: float, friction: float) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add(type=kind)
    obj.rigid_body.mass = mass
    obj.rigid_body.friction = friction
    obj.rigid_body.restitution = 0.02
    obj.rigid_body.linear_damping = 0.35
    obj.rigid_body.angular_damping = 0.45
    obj["mass_kg"] = mass
    obj["friction"] = friction
    obj["restitution"] = 0.02
    obj["physical_values_status"] = "assumed_placeholder"


def add_hinge_constraint(
    name: str,
    loc: Vector,
    parent_body: bpy.types.Object,
    child_body: bpy.types.Object,
    lower_deg: float,
    upper_deg: float,
) -> bpy.types.Object:
    con = add_empty(name, loc, "SINGLE_ARROW", 0.045)
    bpy.context.view_layer.objects.active = con
    bpy.ops.rigidbody.constraint_add()
    con.rigid_body_constraint.type = "HINGE"
    con.rigid_body_constraint.object1 = parent_body
    con.rigid_body_constraint.object2 = child_body
    con["joint_type"] = "revolute"
    con["axis"] = [0, 0, 1]
    con["limit_lower_deg"] = lower_deg
    con["limit_upper_deg"] = upper_deg
    con["damping_nms_per_rad"] = 0.03
    con["dry_friction_nm"] = 0.01
    con["physical_values_status"] = "assumed_placeholder"
    return con


def add_arc(name: str, pivot: Vector, radius: float, start_deg: float, end_deg: float, parent=None) -> bpy.types.Object:
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
        x = radius * math.cos(a)
        y = radius * math.sin(a)
        z = 0.019
        spline.points[i].co = (x, y, z, 1.0)
    obj = bpy.data.objects.new(name, curve)
    obj.location = pivot
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(MAT_ARC)
    obj["physics_role"] = "range_of_motion_arc"
    if parent:
        parent_keep(obj, parent)
    return obj


def add_motor_actuator(
    name: str,
    loc: Vector,
    drives_joint: str,
    max_torque_nm: float,
    gear_ratio: float,
) -> bpy.types.Object:
    """Visible placeholder torque actuator with exportable custom properties."""
    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.013,
        minor_radius=0.0022,
        major_segments=48,
        minor_segments=8,
        location=loc + Vector((0, 0, 0.028)),
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.name = name + "_mesh"
    obj.data.materials.append(MAT_MOTOR)
    obj["physics_role"] = "torque_motor_placeholder"
    obj["actuator_type"] = "torque_motor"
    obj["drives_joint"] = drives_joint
    obj["command_interface"] = "commanded_torque_nm"
    obj["commanded_torque_nm"] = 0.0
    obj["max_torque_nm"] = max_torque_nm
    obj["gear_ratio"] = gear_ratio
    obj["rotor_inertia_kg_m2"] = 1.0e-5
    obj["viscous_friction_nms_per_rad"] = 1.0e-3
    obj["status"] = "placeholder_send_torque_only"
    return obj


def setup_camera(meshes: list[bpy.types.Object]) -> None:
    points = []
    render_meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    sample_frames = sorted({1, N_FRAMES // 4, N_FRAMES // 2, (3 * N_FRAMES) // 4, N_FRAMES})
    for frame in sample_frames:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for obj in render_meshes:
            points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    mn = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    mx = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    center = (mn + mx) * 0.5
    size = max(mx.x - mn.x, mx.y - mn.y, mx.z - mn.z, 0.25)

    bpy.ops.mesh.primitive_plane_add(size=size * 1.55, location=(center.x - 0.04, center.y - 0.12, mn.z - 0.025))
    ground = bpy.context.object
    ground.name = "snapshot_ground"
    ground.data.materials.append(MAT_GROUND)

    cam_pos = center + Vector((size * 1.0, -size * 1.20, size * 0.78))
    bpy.ops.object.camera_add(location=cam_pos)
    cam = bpy.context.object
    direction = center - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = size * 1.45
    cam.data.clip_end = 1000
    bpy.context.scene.camera = cam

    bpy.ops.object.light_add(type="AREA", location=center + Vector((size * 0.8, -size * 1.2, size * 1.8)))
    key = bpy.context.object
    key.name = "snapshot_key"
    key.data.energy = 600
    key.data.size = size * 1.3

    bpy.ops.object.light_add(type="AREA", location=center + Vector((-size * 1.2, size * 1.0, size * 1.0)))
    fill = bpy.context.object
    fill.name = "snapshot_fill"
    fill.data.energy = 200
    fill.data.size = size * 1.8


def add_two_stage_physics_and_animation(meshes: list[bpy.types.Object]) -> dict:
    """Two-stage kinematic scaffold inferred from rendered checks.

    The drive gear, parallel rods, black stabilizer rod, and distal purple
    carrier remain fixed. The hip motor (fed by the worm) drives the purple
    perforated leg AND the yellow piston straight UP and DOWN together, along the
    vertical stabilizer rod; the piston slides in-line inside the fixed distal
    carrier. The knee is a rigid bend (no motor) — nothing hinges or swings.
    """
    pivot = UPPER_OUTBOARD
    worm_ctrl = add_empty("joint_red_worm_shaft_spin", WORM_AXIS, "SINGLE_ARROW", 0.035)
    wheel_ctrl = add_empty("joint_red_worm_wheel_axis", HIP, "SINGLE_ARROW", 0.045)
    hip_ctrl = add_empty("joint_fixed_upper_carrier_black_rod_hold", pivot, "ARROWS", 0.055)
    hinge_ctrl = add_empty("joint_purple_hinge_driver", KNEE_DRIVER, "ARROWS", 0.045)
    rod_ctrl = add_empty("joint_yellow_piston_rod_pulled_by_leg", KNEE, "ARROWS", 0.035)
    parent_local_offset(hinge_ctrl, hip_ctrl, KNEE_DRIVER - pivot)
    # The yellow PISTON's TOP is pinned to the purple leg's END (KNEE): its controller rides the leg
    # (parented to hinge_ctrl at the KNEE offset) so the hinging knee physically LIFTS it. The piston
    # then slides/pivots through the FIXED distal carrier — the animation loop sets rod_ctrl's rotation
    # so the rod always points from the (moving) leg end toward CARRIER_GUIDE, drawing the piston UP
    # through the carrier (a slider-crank). (Bug fix: earlier passes either swung the whole rod ~70 deg
    # about the knee, or slid it in a wrongly-mobile cylinder — both detached it from the real linkage.)
    parent_local_offset(rod_ctrl, hinge_ctrl, KNEE - KNEE_DRIVER)

    worm_mesh_names = []
    wheel_mesh_names = []
    held_mesh_names = []
    hinge_mesh_names = []
    rod_mesh_names = []
    for obj in meshes:
        if obj.name in RED_WORM_NAMES:
            rebase_mesh_origin(obj, WORM_AXIS)
            parent_at_pivot(obj, worm_ctrl)
            obj["body"] = "red_worm_shaft"
            worm_mesh_names.append(obj.name)
            continue
        if obj.name in RED_WORM_WHEEL_NAMES:
            rebase_mesh_origin(obj, HIP)
            parent_at_pivot(obj, wheel_ctrl)
            obj["body"] = "red_worm_wheel"
            wheel_mesh_names.append(obj.name)
            continue
        if obj.name in FIXED_DRIVE_NAMES:
            obj["body"] = "drive_base_linkage"
            continue
        if obj.name in PURPLE_HINGE_NAMES:
            rebase_mesh_origin(obj, KNEE_DRIVER)
            parent_at_pivot(obj, hinge_ctrl)
            obj["body"] = "purple_hinge_link"
            hinge_mesh_names.append(obj.name)
            continue
        if obj.name in YELLOW_ROD_NAMES:
            rebase_mesh_origin(obj, KNEE)          # pivot at the piston TOP (pinned to the leg end)
            parent_at_pivot(obj, rod_ctrl)
            obj["body"] = "yellow_piston_rod"
            rod_mesh_names.append(obj.name)
            continue
        rebase_mesh_origin(obj, pivot)
        parent_at_pivot(obj, hip_ctrl)
        obj["body"] = "held_upper_carrier_black_rod"
        held_mesh_names.append(obj.name)

    drive_pin = add_pin("pin_drive_gear_axis_z", HIP, 0.007, 0.045)
    hip_pin = add_pin("pin_hip_bracket_revolute_axis_z", pivot, 0.0065, 0.042)
    knee_driver_pin = add_pin("pin_purple_hinge_driver_axis_z", KNEE_DRIVER, 0.0055, 0.038)
    knee_pin = add_pin("pin_purple_hinge_to_yellow_rod", KNEE, 0.0065, 0.042)
    distal_pin = add_pin("pin_fixed_purple_carrier_black_rod_hold", DISTAL_LINK, 0.005, 0.034)
    tool_pin = add_pin("pin_yellow_rod_tip_reference", TOOL_TIP, 0.004, 0.028)
    parent_local_offset(knee_driver_pin, hip_ctrl, KNEE_DRIVER - pivot)
    parent_local_offset(knee_pin, hinge_ctrl, KNEE - KNEE_DRIVER)
    parent_local_offset(distal_pin, hip_ctrl, DISTAL_LINK - pivot)
    parent_local_offset(tool_pin, rod_ctrl, TOOL_TIP - KNEE)

    drive_motor = add_motor_actuator(
        "actuator_red_worm_drive_torque_motor_placeholder",
        WORM_AXIS,
        "red_worm_to_wheel",
        max_torque_nm=1.2,
        gear_ratio=20.0,
    )
    hip_motor = add_motor_actuator(
        "actuator_hip_bracket_torque_motor_placeholder",
        pivot,
        "hip",
        max_torque_nm=3.0,
        gear_ratio=8.0,
    )
    # (no knee motor — the knee is NOT a powered joint; the hip motor drives the leg straight up/down)

    worm_marker = add_marker("marker_red_worm_shaft_spin", WORM_AXIS + Vector((0.010, 0.025, 0.010)), 0.0035)
    wheel_marker = add_marker("marker_red_worm_wheel_slow_rotation", HIP + Vector((0.030, 0.000, 0.010)), 0.0035)
    parent_local_offset(worm_marker, worm_ctrl, worm_marker.location - WORM_AXIS)
    parent_local_offset(wheel_marker, wheel_ctrl, wheel_marker.location - HIP)

    base_proxy = cylinder_between(
        "rigid_proxy_drive_base_linkage",
        HIP,
        pivot,
        0.006,
        MAT_PROXY_BASE,
        True,
    )
    held_proxy = cylinder_between(
        "rigid_proxy_held_upper_carrier_black_rod",
        pivot,
        DISTAL_LINK,
        0.005,
        MAT_PROXY_UPPER,
        True,
    )
    hinge_proxy = cylinder_between(
        "rigid_proxy_purple_hinge_link",
        KNEE_DRIVER,
        KNEE,
        0.003,
        MAT_PROXY_LOWER,
        True,
    )
    rod_proxy = cylinder_between(
        "rigid_proxy_yellow_piston_rod",
        KNEE,
        TOOL_TIP,
        0.002,
        MAT_PROXY_LOWER,
        True,
    )
    parent_local_offset(held_proxy, hip_ctrl, (pivot + DISTAL_LINK) * 0.5 - pivot)
    parent_local_offset(hinge_proxy, hinge_ctrl, (KNEE_DRIVER + KNEE) * 0.5 - KNEE_DRIVER)
    parent_local_offset(rod_proxy, rod_ctrl, (KNEE + TOOL_TIP) * 0.5 - KNEE)

    add_rigid_body(base_proxy, "PASSIVE", 0.35, 0.65)
    add_rigid_body(held_proxy, "PASSIVE", 0.20, 0.68)
    add_rigid_body(hinge_proxy, "ACTIVE", 0.08, 0.70)
    add_rigid_body(rod_proxy, "ACTIVE", 0.10, 0.74)
    add_hinge_constraint("constraint_black_rod_hold_base_to_carrier", pivot, base_proxy, held_proxy, 0, 0)
    hinge_constraint = add_hinge_constraint(
        "constraint_purple_leg_prismatic_vertical",
        KNEE_DRIVER,
        held_proxy,
        hinge_proxy,
        0,
        0,
    )
    hinge_constraint["joint_type"] = "prismatic_vertical"
    hinge_constraint["note"] = "hip motor drives the purple leg straight up/down along the stabilizer rod; not a hinge"
    parent_local_offset(hinge_constraint, hip_ctrl, KNEE_DRIVER - pivot)
    rod_pin_constraint = add_hinge_constraint(
        "constraint_yellow_rod_slides_in_fixed_carrier",
        CARRIER_GUIDE,
        held_proxy,
        rod_proxy,
        0,
        0,
    )
    rod_pin_constraint["joint_type"] = "prismatic_vertical"
    rod_pin_constraint["note"] = (
        "yellow piston moves straight up/down in-line inside the fixed distal carrier, together with the purple leg"
    )
    parent_local_offset(rod_pin_constraint, hip_ctrl, CARRIER_GUIDE - pivot)

    for name, loc, parent in [
        ("axis_red_worm_shaft_y", WORM_AXIS, None),
        ("axis_drive_gear_z", HIP, None),
        ("axis_hip_bracket_z", pivot, None),
        ("axis_knee_driver_z", KNEE_DRIVER, hip_ctrl),
        ("axis_yellow_rod_pin_z", KNEE, hinge_ctrl),
        ("axis_fixed_distal_carrier_z", DISTAL_LINK, hip_ctrl),
    ]:
        if name == "axis_red_worm_shaft_y":
            axis = cylinder_between(
                name,
                loc + Vector((0, -0.042, 0)),
                loc + Vector((0, 0.042, 0)),
                0.0014,
                MAT_AXIS,
                True,
            )
        else:
            axis = cylinder_between(
                name,
                loc + Vector((0, 0, -0.026)),
                loc + Vector((0, 0, 0.026)),
                0.0014,
                MAT_AXIS,
                True,
            )
        if parent == hip_ctrl:
            parent_local_offset(axis, parent, loc - pivot)
        elif parent == hinge_ctrl:
            parent_local_offset(axis, parent, loc - KNEE_DRIVER)

    # (no rotation arc: the motion is linear up/down — shown by the live motion arrows in the overlay)

    rod_ctrl.rotation_euler = (0.0, 0.0, 0.0)         # piston stays vertical (no swing); it only translates
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = N_FRAMES
    for frame in range(1, N_FRAMES + 1):
        t = (frame - 1) / (N_FRAMES - 1)
        bpy.context.scene.frame_set(frame)
        # ONE connected chain: worm/wheel turn the hip motor, which drives the purple leg + yellow
        # piston STRAIGHT up and down together (along -Y, colinear with the stabilizer rod). Nothing
        # hinges; the piston slides in-line inside the fixed carrier. s>0 up, s<0 down (full cycle).
        s = LEG_STROKE * math.sin(2.0 * math.pi * t)
        wheel_deg = 30.0 * math.sin(2.0 * math.pi * t)        # slow worm-wheel oscillation (the drive)
        worm_deg = -wheel_deg * 16.0                          # worm screw spins 16x faster (fast input)
        worm_ctrl.rotation_euler[1] = math.radians(worm_deg)
        wheel_ctrl.rotation_euler[2] = math.radians(wheel_deg)
        hip_ctrl.rotation_euler[2] = 0.0
        hinge_ctrl.location = (KNEE_DRIVER - pivot) + Vector((0.0, s, 0.0))   # leg (+ piston) translate up/down
        hip_motor.rotation_euler[2] = math.radians(55.0 * math.sin(2.0 * math.pi * t))   # driver motor turns
        drive_motor.rotation_euler[1] = math.radians(worm_deg)
        drive_motor["commanded_torque_nm"] = round(1.1 * math.sin(2.0 * math.pi * t), 5)
        hip_motor["commanded_torque_nm"] = round(1.6 * math.sin(2.0 * math.pi * t), 5)
        worm_ctrl.keyframe_insert(data_path="rotation_euler", frame=frame)
        wheel_ctrl.keyframe_insert(data_path="rotation_euler", frame=frame)
        hip_ctrl.keyframe_insert(data_path="rotation_euler", frame=frame)
        hinge_ctrl.keyframe_insert(data_path="location", frame=frame)
        hip_motor.keyframe_insert(data_path="rotation_euler", frame=frame)
        drive_motor.keyframe_insert(data_path="rotation_euler", frame=frame)
        drive_motor.keyframe_insert(data_path='["commanded_torque_nm"]', frame=frame)
        hip_motor.keyframe_insert(data_path='["commanded_torque_nm"]', frame=frame)

    scene_props = {
        "units": "meters_assumed",
        "source": SRC.name,
        "status": "first_pass_inferred_from_geometry_and_render",
        "bodies": {
            "red_worm_shaft": {
                "mass_kg": 0.05,
                "friction": 0.45,
                "mesh_names": sorted(worm_mesh_names),
                "axis_xyz": [0, 1, 0],
            },
            "red_worm_wheel": {
                "mass_kg": 0.08,
                "friction": 0.50,
                "mesh_names": sorted(wheel_mesh_names),
                "axis_xyz": [0, 0, 1],
            },
            "drive_base_linkage": {
                "mass_kg": 0.35,
                "friction": 0.65,
                "proxy": base_proxy.name,
                "mesh_names": sorted(FIXED_DRIVE_NAMES),
            },
            "held_upper_carrier_black_rod": {
                "mass_kg": 0.20,
                "friction": 0.68,
                "proxy": held_proxy.name,
                "mesh_names": sorted(held_mesh_names),
                "status": "held_fixed_by_black_stabilizer_rod",
            },
            "purple_hinge_link": {
                "mass_kg": 0.08,
                "friction": 0.70,
                "proxy": hinge_proxy.name,
                "mesh_names": sorted(hinge_mesh_names),
            },
            "yellow_piston_rod": {
                "mass_kg": 0.10,
                "friction": 0.74,
                "proxy": rod_proxy.name,
                "mesh_names": sorted(rod_mesh_names),
                "status": "moves_straight_up_and_down_in_line_inside_fixed_carrier",
            },
        },
        "joints": {
            "red_worm_shaft_spin": {
                "type": "revolute_visual",
                "parent": "drive_base_linkage",
                "child": "red_worm_shaft",
                "origin_xyz": list(WORM_AXIS),
                "axis_xyz": [0, 1, 0],
                "note": "fast worm shaft spin",
            },
            "red_worm_wheel_axis": {
                "type": "revolute_visual",
                "parent": "drive_base_linkage",
                "child": "red_worm_wheel",
                "origin_xyz": list(HIP),
                "axis_xyz": [0, 0, 1],
                "note": "slow worm-wheel rotation driven visually by red_worm_shaft_spin",
            },
            "black_rod_hold": {
                "type": "fixed_visual",
                "parent": "drive_base_linkage",
                "child": "held_upper_carrier_black_rod",
                "origin_xyz": list(pivot),
                "axis_xyz": [0, 0, 1],
                "limit_deg": [0, 0],
                "note": "black rod holds the distal purple carrier fixed in this approximation",
            },
            "purple_leg_slide": {
                "type": "prismatic",
                "parent": "held_upper_carrier_black_rod",
                "child": "purple_hinge_link",
                "origin_xyz": list(KNEE_DRIVER),
                "axis_xyz": [0, 1, 0],
                "travel_m": [-LEG_STROKE, LEG_STROKE],
                "note": "hip motor drives the purple perforated leg straight up/down along the stabilizer rod; the knee is a rigid bend, not a joint",
            },
            "yellow_rod_slide": {
                "type": "prismatic",
                "parent": "held_upper_carrier_black_rod",
                "child": "yellow_piston_rod",
                "origin_xyz": list(CARRIER_GUIDE),
                "axis_xyz": [0, 1, 0],
                "travel_m": [-LEG_STROKE, LEG_STROKE],
                "status": "piston moves straight up/down in-line inside the fixed distal carrier, together with the purple leg",
            },
            "drive_gear_axis": {
                "type": "pin_visual",
                "origin_xyz": list(HIP),
                "axis_xyz": [0, 0, 1],
            },
            "fixed_distal_carrier_pin": {
                "type": "pin_visual",
                "origin_xyz": list(DISTAL_LINK),
                "axis_xyz": [0, 0, 1],
                "status": "held by black stabilizer rod",
            },
            "yellow_rod_tip_reference": {
                "type": "pin_visual",
                "origin_xyz": list(TOOL_TIP),
                "axis_xyz": [0, 0, 1],
            },
        },
        "actuators": {
            "red_worm_drive_motor": {
                "type": "torque_motor",
                "drives_joint": "red_worm_shaft_spin",
                "command_interface": "commanded_torque_nm",
                "max_torque_nm": 1.2,
                "gear_ratio": 20.0,
                "status": "placeholder_send_torque_only_worm_wheel_keyframed",
            },
            "hip_drive_motor": {
                "type": "linear_actuator",
                "drives_joint": "purple_leg_slide",
                "command_interface": "commanded_torque_nm",
                "max_torque_nm": 3.0,
                "gear_ratio": 8.0,
                "status": "the DRIVER: turns the purple leg + piston straight up/down (worm-fed); knee has no motor",
            },
        },
        "animation": {
            "name": "full_range_of_motion",
            "fps": FPS,
            "frames": N_FRAMES,
            "hip_deg": [0, 0],
            "red_worm_shaft_deg": "16x inverse worm wheel angle",
            "red_worm_wheel_deg": [-30, 30],
            "leg_piston_travel_mm": [-LEG_STROKE * 1000, LEG_STROKE * 1000],
            "yellow_rod_motion": "straight up and down, in-line inside the fixed distal carrier (linear, no hinge / no swing)",
        },
        "notes": [
            "The source glTF has mostly unnamed nodes and no separated rigid-body hierarchy.",
            "LINEAR mechanism: the hip motor (worm-fed) drives the purple perforated leg AND the yellow piston straight UP and DOWN together, along -Y / the vertical black stabilizer rod.",
            "The red worm shaft spins fast (around Y); the worm wheel turns slowly (around Z) as the slow input.",
            "The black stabilizer rod and distal purple carrier are held fixed; the rod is the up/down rail/guide.",
            "The knee is a RIGID bend, not a powered joint, and there is NO motor at the knee.",
            "The yellow piston slides straight up/down IN-LINE inside the fixed distal carrier (it never swings or pivots).",
            "Mass/friction/damping values are placeholders for visualization and downstream schema testing.",
            "The GIF is keyframed kinematics, not a Bullet rigid-body simulation.",
        ],
    }
    bpy.context.scene["leg_physics"] = json.dumps(scene_props)
    return scene_props


def add_physics_and_animation(meshes: list[bpy.types.Object]) -> dict:
    return add_two_stage_physics_and_animation(meshes)


def setup_render() -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.world.color = (0.03, 0.03, 0.035)
    scene.render.resolution_x = 1000
    scene.render.resolution_y = 650
    scene.render.fps = FPS
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"


def render_frames() -> None:
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.filepath = str(FRAMES_DIR / "frame_")
    bpy.ops.render.render(animation=True)


def make_gif() -> None:
    src_dir = LABELED_DIR if LABELED_DIR.exists() else FRAMES_DIR
    frames = sorted(src_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError("No frames rendered")
    try:
        from PIL import Image

        images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frames]
        images[0].save(
            GIF_OUT,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / FPS),
            loop=0,
            optimize=True,
        )
    except Exception:
        delay_cs = max(1, round(100 / FPS))
        subprocess.run(
            ["magick", "-delay", str(delay_cs), "-loop", "0", *map(str, frames), str(GIF_OUT)],
            check=True,
        )


def make_contact_sheet() -> None:
    src_dir = LABELED_DIR if LABELED_DIR.exists() else FRAMES_DIR
    frame_numbers = [1, N_FRAMES // 4, N_FRAMES // 2, (3 * N_FRAMES) // 4, N_FRAMES]
    frames = [src_dir / f"frame_{frame:04d}.png" for frame in frame_numbers]
    missing = [path for path in frames if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing contact sheet frames: {missing}")
    subprocess.run(
        [
            "montage",
            *map(str, frames),
            "-tile",
            "5x1",
            "-geometry",
            "320x208+8+8",
            "-background",
            "#202024",
            str(CONTACT_SHEET_OUT),
        ],
        check=True,
    )


def make_detail_contact_sheet() -> None:
    frame_numbers = [1, N_FRAMES // 4, N_FRAMES // 2, (3 * N_FRAMES) // 4, N_FRAMES]
    frames = [FRAMES_DIR / f"frame_{frame:04d}.png" for frame in frame_numbers]
    missing = [path for path in frames if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing detail contact sheet frames: {missing}")

    crops_dir = OUT_DIR / "detail_crops"
    if crops_dir.exists():
        shutil.rmtree(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)
    crops = []
    for frame, path in zip(frame_numbers, frames):
        crop = crops_dir / f"knee_detail_{frame:04d}.png"
        subprocess.run(
            [
                "magick",
                str(path),
                "-crop",
                "850x430+70+10",
                "+repage",
                "-resize",
                "480x243",
                str(crop),
            ],
            check=True,
        )
        crops.append(crop)
    subprocess.run(
        [
            "montage",
            *map(str, crops),
            "-tile",
            "5x1",
            "-geometry",
            "480x243+8+8",
            "-background",
            "#202024",
            str(DETAIL_CONTACT_SHEET_OUT),
        ],
        check=True,
    )


def make_gear_detail_contact_sheet() -> None:
    frame_numbers = [1, N_FRAMES // 4, N_FRAMES // 2, (3 * N_FRAMES) // 4, N_FRAMES]
    frames = [FRAMES_DIR / f"frame_{frame:04d}.png" for frame in frame_numbers]
    missing = [path for path in frames if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing gear detail contact sheet frames: {missing}")

    crops_dir = OUT_DIR / "gear_crops"
    if crops_dir.exists():
        shutil.rmtree(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)
    crops = []
    for frame, path in zip(frame_numbers, frames):
        crop = crops_dir / f"gear_detail_{frame:04d}.png"
        subprocess.run(
            [
                "magick",
                str(path),
                "-crop",
                "520x330+230+25",
                "+repage",
                "-resize",
                "520x330",
                str(crop),
            ],
            check=True,
        )
        crops.append(crop)
    subprocess.run(
        [
            "montage",
            *map(str, crops),
            "-tile",
            "5x1",
            "-geometry",
            "520x330+8+8",
            "-background",
            "#202024",
            str(GEAR_DETAIL_CONTACT_SHEET_OUT),
        ],
        check=True,
    )


# Numbered part labels: (text, color, text_x, text_y[baseline], anchor_key). Numbers let a reviewer
# point at a part ("number 8 is wrong") without needing the jargon. anchor_key resolves per frame.
LABEL_TABLE = [
    ("[1] Worm shaft - spins FAST",            "#ff6b6b",  10,  26, "worm"),
    ("[2] Worm wheel - turns SLOW",            "#ff8a5b",  10,  58, "wheel"),
    ("[3] Drive motor (gold ring)",            "#ff9f1a",  10,  90, "drive_motor"),
    ("[7] Purple LEG - slides UP/DOWN",        "#cf86ff",  10, 300, "leg"),
    ("[8] Leg-end pin (drives piston)",        "#cf86ff",  10, 332, "legpin"),
    ("[11] Yellow PISTON - slides UP/DOWN",    "#ffd21e",  10, 566, "piston"),
    ("[12] Piston tip",                        "#ffd21e",  10, 598, "tip"),
    ("[15] Cyan posts = joint pins",           "#46d4ff",  10, 630, "cyanpin"),
    ("[4] Knee bend (rigid - NOT a joint)",    "#9fb3c0", 330,  22, "knee"),
    ("[5] Distal carrier - FIXED",             "#d6d6d6", 612,  22, "carrier"),
    ("[6] Upper black rods - FIXED",           "#cfcfcf", 700, 110, "blackrods"),
    ("[9] Stabilizer rod = up/down RAIL",      "#cfcfcf", 700, 150, "stab"),
    ("[14] Hip motor - DRIVES up/down",        "#ff9f1a", 700, 232, "hipmotor"),
    ("[16] White dot = spin marker",           "#ffffff", 700, 272, "whitemark"),
]
# Moving parts get a live arrow showing the instantaneous direction of travel (color, anchor_key).
ARROW_KEYS = [("legpin", "#cf86ff"), ("tip", "#ffd21e"), ("whitemark", "#ff6b6b"), ("wheelmark", "#ff8a5b")]


def compute_label_tracks() -> dict:
    """Per-frame label anchors, motion arrows, and a HUD readout (all camera-projected to pixels)."""
    scene = bpy.context.scene
    cam = scene.camera
    rx, ry = scene.render.resolution_x, scene.render.resolution_y

    def px(world: Vector) -> tuple[int, int]:
        co = world_to_camera_view(scene, cam, Vector(world))
        x = min(max(co.x, 0.0), 1.0) * rx
        y = (1.0 - min(max(co.y, 0.0), 1.0)) * ry
        return (int(x), int(y))

    rod = bpy.data.objects.get("joint_yellow_piston_rod_pulled_by_leg")
    tip = bpy.data.objects.get("pin_yellow_rod_tip_reference")
    wmark = bpy.data.objects.get("marker_red_worm_shaft_spin")
    whmark = bpy.data.objects.get("marker_red_worm_wheel_slow_rotation")

    def anchors_world(leg_end: Vector, tip_w: Vector) -> dict:
        return {
            "worm": Vector(WORM_AXIS), "wheel": Vector(HIP),
            "drive_motor": Vector(WORM_AXIS) + Vector((0, 0, 0.028)),
            "knee": Vector(KNEE_DRIVER), "carrier": Vector(CARRIER_GUIDE),
            "blackrods": Vector((-0.10, 0.017, 0.0)), "stab": Vector((-0.017, -0.114, 0.0)),
            "kneemotor": Vector(KNEE_DRIVER) + Vector((0, 0, 0.028)),
            "arc": Vector(KNEE_DRIVER) + Vector((0.061, -0.043, 0.019)),
            "hipmotor": Vector(UPPER_OUTBOARD) + Vector((0, 0, 0.028)),
            "cyanpin": Vector(UPPER_OUTBOARD),
            "leg": (Vector(KNEE_DRIVER) + Vector(leg_end)) * 0.5,
            "legpin": Vector(leg_end),
            "piston": (Vector(leg_end) + Vector(tip_w)) * 0.5,
            "tip": Vector(tip_w),
            "whitemark": (wmark.matrix_world.translation if wmark else Vector(WORM_AXIS)),
            "wheelmark": (whmark.matrix_world.translation if whmark else Vector(HIP)),
        }

    proj: dict = {}
    travel_mm_by_frame: dict = {}
    for frame in range(1, N_FRAMES + 1):                     # pass 1: project every anchor each frame
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        leg_end = rod.matrix_world.translation if rod else Vector(KNEE)
        tip_w = tip.matrix_world.translation if tip else Vector(TOOL_TIP)
        proj[frame] = {k: px(v) for k, v in anchors_world(leg_end, tip_w).items()}
        t = (frame - 1) / (N_FRAMES - 1)
        travel_mm_by_frame[frame] = 1000.0 * LEG_STROKE * math.sin(2.0 * math.pi * t)

    tracks: dict = {}
    for frame in range(1, N_FRAMES + 1):                     # pass 2: labels + central-difference arrows
        labels = [(text, color, tx, ty, proj[frame][key]) for (text, color, tx, ty, key) in LABEL_TABLE]
        fa, fb = max(1, frame - 1), min(N_FRAMES, frame + 1)
        arrows = []
        for key, color in ARROW_KEYS:
            sx, sy = proj[frame][key]
            (ax, ay), (bx, by) = proj[fa][key], proj[fb][key]
            vx, vy = bx - ax, by - ay
            mag = math.hypot(vx, vy)
            if mag < 0.8:                                    # part momentarily still -> no arrow
                continue
            length = max(16.0, min(64.0, mag * 5.0))
            arrows.append((color, (sx, sy), (int(sx + vx / mag * length), int(sy + vy / mag * length))))
        hud = (f"frame {frame}/{N_FRAMES}   |   leg+piston travel {travel_mm_by_frame[frame]:+.0f} mm (up/down)"
               "   |   arrows = direction each part is moving NOW")
        tracks[frame] = {"labels": labels, "arrows": arrows, "hud": hud}
    return tracks


def annotate_frames(tracks: dict) -> None:
    """Burn numbered labels, leader lines, live motion arrows, and a HUD onto each frame -> LABELED_DIR."""
    if LABELED_DIR.exists():
        shutil.rmtree(LABELED_DIR)
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    for frame in range(1, N_FRAMES + 1):
        src = FRAMES_DIR / f"frame_{frame:04d}.png"
        dst = LABELED_DIR / f"frame_{frame:04d}.png"
        spec = tracks.get(frame, {})
        labels = spec.get("labels", [])
        arrows = spec.get("arrows", [])
        hud = spec.get("hud", "")
        cmd = ["magick", str(src), "-strokewidth", "2"]
        for _text, color, tx, ty, (ax, ay) in labels:        # leader line + anchor dot
            cmd += ["-fill", "none", "-stroke", color + "aa", "-draw", f"line {tx + 6},{ty - 5} {ax},{ay}"]
            cmd += ["-fill", color, "-stroke", "none", "-draw", f"circle {ax},{ay} {ax + 4},{ay}"]
        cmd += ["-strokewidth", "3"]                          # motion arrows (shaft + 2 head lines)
        for color, (sx, sy), (ex, ey) in arrows:
            cmd += ["-fill", "none", "-stroke", color, "-draw", f"line {sx},{sy} {ex},{ey}"]
            ang = math.atan2(ey - sy, ex - sx)
            for da in (math.radians(150), math.radians(-150)):
                hx, hy = int(ex + 10 * math.cos(ang + da)), int(ey + 10 * math.sin(ang + da))
                cmd += ["-draw", f"line {ex},{ey} {hx},{hy}"]
        cmd += ["-strokewidth", "1", "-pointsize", "14"]      # label text with dark backing
        for text, color, tx, ty, _anchor in labels:
            cmd += ["-stroke", "none", "-undercolor", "#000000bb", "-fill", color,
                    "-annotate", f"+{tx}+{ty}", f" {text} "]
        cmd += ["-pointsize", "15", "-undercolor", "#000000cc", "-fill", "#cfe2ff",
                "-annotate", "+250+642", f" {hud} "]
        cmd += [str(dst)]
        subprocess.run(cmd, check=True)


def save_outputs(scene_props: dict) -> None:
    JSON_OUT.write_text(json.dumps(scene_props, indent=2) + "\n")
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_OUT))
    try:
        bpy.ops.export_scene.gltf(
            filepath=str(GLB_OUT),
            export_format="GLB",
            export_animations=True,
            export_extras=True,
        )
    except TypeError:
        bpy.ops.export_scene.gltf(filepath=str(GLB_OUT), export_format="GLB")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_scene()
    make_materials()
    meshes = import_source()
    scene_props = add_physics_and_animation(meshes)
    setup_camera(meshes)
    setup_render()
    label_tracks = compute_label_tracks()
    save_outputs(scene_props)
    render_frames()
    annotate_frames(label_tracks)
    make_gif()
    make_contact_sheet()
    make_detail_contact_sheet()
    make_gear_detail_contact_sheet()
    print(f"BLEND {BLEND_OUT}")
    print(f"GLB {GLB_OUT}")
    print(f"PHYSICS_JSON {JSON_OUT}")
    print(f"GIF {GIF_OUT}")
    print(f"CONTACT_SHEET {CONTACT_SHEET_OUT}")
    print(f"DETAIL_CONTACT_SHEET {DETAIL_CONTACT_SHEET_OUT}")
    print(f"GEAR_DETAIL_CONTACT_SHEET {GEAR_DETAIL_CONTACT_SHEET_OUT}")
    print(f"FRAMES {FRAMES_DIR}")


if __name__ == "__main__":
    main()
