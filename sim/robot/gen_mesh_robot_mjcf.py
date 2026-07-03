# SPDX-License-Identifier: MIT
"""gen_mesh_robot_mjcf.py — Robot_Assembly_7-3-2026.physics.json -> MJX-ready MJCF.

Converts the REAL leg mechanism (worm-pitch + toggle-press blade foot + yaw mount)
into a trainable robot XML, following the json's own `mjcf_conversion` recipe:

  * chain per leg: {L}_hip_yaw -> {L}_leg_swing -> {L}_knee_blade (+ passive
    {L}_toe_hinge, {L}_heel_pin*, {L}_pushrod_slide) at the json's origins/axes/limits
  * loop closure OPTION A: <equality><connect> pinning the heel ear (blade_lower
    local (0,-L,0)) to the pushrod's heel point — the slider-crank runs as a real
    dynamic loop, so the toggle-press force profile is PHYSICS, not scripting.
    (*heel_pin as a tree joint is subsumed by the connect: blade_lower's swing
    relative to the pushrod IS the constrained dof; adding it as a tree joint too
    would double-count. Documented deviation from the literal 6-joint chain —
    the census test counts the 5 tree joints + connect per leg.)
  * primitives only (MJX v1): every geom is a capsule/box/sphere sized from the
    mechanism dimensions; each records what it approximates in a comment.
  * honest transmissions:
      yaw   — motor 1.2 N.m through belt ratio TODO=1; SEA stiffness/damping are
              PLACEHOLDER joint damping (real model: rotor body + spring equality;
              bench-measure the rubber pulley to unblock)
      swing — worm 20:1: gear = 24 N.m, self-locking approximated by joint
              frictionloss = 2.0 N.m (> 2x the 0.99 N.m Drake peak gravity hold;
              true self-locking (infinite holding, zero backdrive) is not natively
              expressible in MuJoCo — frictionloss is the standard approximation)
      knee  — direct 3 N.m crank (toggle-press force comes from the loop)
  * per-axis armature = J_rotor x ratio^2 per the json's suggested_defaults.

CLI: python gen_mesh_robot_mjcf.py [--json PATH] [--out PATH] [--summary]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSEMBLY_JSON = HERE.parents[1] / "Robot_Assembly_7-3-2026.physics.json"

# mechanism constants (leg json / mjcf_conversion.closed_form)
CRANK_R = 0.075          # knee pin -> toe hinge (blade upper)
CONROD_L = 0.100         # toe hinge -> heel pin (blade lower)
TIP_LEN = 0.258          # heel pin -> pushrod tip; from the json's stomp reach:
                         # 0.025 + 0.04114 + tip = 0.3241  (sizing_reasoning.rule_height)
FOOT_R = 0.012           # pushrod tip contact sphere
TORSO_MASS_PLACEHOLDER = 3.0   # kg — json torso mass is "TODO placeholder"
WORM_FRICTIONLOSS = 2.0  # N.m — see module docstring
YAW_DAMPING_PLACEHOLDER = 0.08   # rubber-pulley SEA stand-in; bench-measure to replace
ARMATURE = {"hip_yaw": 1.35e-4,  # 1.5e-5 * 3^2 (belt ratio TODO; json example)
            "leg_swing": 0.006,  # 1.5e-5 * 20^2
            "knee_blade": 1.5e-5}


def slider_crank_s(phi: float) -> float:
    """Pushrod slide value vs knee angle (json closed_form): s(0)=0 at blade-vertical."""
    return (CRANK_R * math.cos(phi)
            - math.sqrt(CONROD_L ** 2 - (CRANK_R * math.sin(phi)) ** 2)
            - (CRANK_R - CONROD_L))


def loop_consistent_pose(phi: float) -> tuple[float, float]:
    """(toe_hinge, pushrod_slide) that close the slider-crank at knee angle phi.

    The conrod must counter-rotate so the heel stays on the slide lane:
    psi(phi) = asin(r sin(phi)/L) - phi. Setting knee qpos WITHOUT these two is a
    26 mm constraint violation at phi=-60 and the connect responds with ~1.2 kN —
    every reset/test that places the knee off zero MUST use this."""
    psi = math.asin(CRANK_R * math.sin(phi) / CONROD_L) - phi
    return psi, slider_crank_s(phi)


def load_assembly(path=ASSEMBLY_JSON) -> dict:
    return json.loads(Path(path).read_text())


def _leg_xml(L: str, leg: dict, torso_center) -> tuple[str, str]:
    """One leg subtree + its equality/exclude/actuator lines."""
    mt = leg["mount_transform"]
    jt = {j["name"].split("_", 1)[1]: j for j in leg["joints"]}
    mp = mt["attachment_point_world"]
    pos = (mp[0] - torso_center[0], mp[1] - torso_center[1], mp[2] - torso_center[2])
    quat = mt["quat_leg_local_to_world_wxyz"]
    yaw_lim = [math.radians(d) for d in jt["hip_yaw"]["limit_deg"]]
    swing_lim = [math.radians(d) for d in jt["leg_swing"]["limit_deg"]]
    knee_lim = [math.radians(d) for d in jt["knee_blade"]["limit_deg"]]
    slide_lim = jt["pushrod_slide"]["travel_m"]
    m = leg["masses_kg"]
    # leg-local stations (leg frame origin = knee pin; json origin_leg_local):
    yaw_x = jt["hip_yaw"]["origin_leg_local"][0]        # -0.219
    swing_x = jt["leg_swing"]["origin_leg_local"][0]    # -0.16269
    # housing mass folds in the worm rotor (its spin is motor-side; modeled as armature)
    housing_mass = m["base_housing_and_axle"] + m["worm_input"]
    cc0 = ' contype="0" conaffinity="0"'                # non-colliding (json contact_guidance)
    body = f'''
    <body name="{L}_base_housing" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"
          quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}">
      <joint name="{L}_hip_yaw" axis="0 1 0" range="{yaw_lim[0]:.7f} {yaw_lim[1]:.7f}"
             armature="{ARMATURE['hip_yaw']}" damping="{YAW_DAMPING_PLACEHOLDER}"/>
      <!-- housing+worm+pulley: box spanning yaw pivot to swing pivot (drive_housing bounds) -->
      <geom name="{L}_housing" type="box" pos="{(yaw_x + swing_x) / 2:.5f} 0 0"
            size="{abs(yaw_x - swing_x) / 2 + 0.02:.5f} 0.035 0.030" mass="{housing_mass}"{cc0}/>
      <body name="{L}_swing_link" pos="{swing_x:.6f} 0 0">
        <joint name="{L}_leg_swing" axis="0 0 1" range="{swing_lim[0]:.7f} {swing_lim[1]:.7f}"
               armature="{ARMATURE['leg_swing']}" frictionloss="{WORM_FRICTIONLOSS}"/>
        <!-- swing link: arm capsule pivot->knee (guide rails + drive frame) -->
        <geom name="{L}_arm" type="capsule" fromto="0 0 0 {-swing_x:.5f} 0 0"
              size="0.015" mass="{m['leg_swing_link'] * 0.5}"{cc0}/>
        <!-- shin rail: knee station down to the bushing (leg-local -Y) -->
        <geom name="{L}_shin" type="capsule"
              fromto="{-swing_x:.5f} 0 0 {-swing_x:.5f} -0.2405 0"
              size="0.012" mass="{m['leg_swing_link'] * 0.5}"{cc0}/>
        <body name="{L}_blade_upper" pos="{-swing_x:.6f} 0 0">
          <joint name="{L}_knee_blade" axis="0 0 1" range="{knee_lim[0]:.7f} {knee_lim[1]:.7f}"
                 armature="{ARMATURE['knee_blade']}"/>
          <!-- blade upper = knee crank, knee pin -> toe hinge (straight strip) -->
          <geom name="{L}_bladeU" type="capsule" fromto="0 0 0 0 {CRANK_R} 0"
                size="0.008" mass="{m['blade_upper']}"{cc0}/>
          <body name="{L}_blade_lower" pos="0 {CRANK_R} 0">
            <!-- passive-loop conditioning: the blade/pushrod bodies are ~50-80 g, and the
                 connect constraint is stiff; bearing-scale armature+damping keep the
                 loop well-posed at the toggle's TDC (phi=0) where its Jacobian is
                 singular. Values are numerical regularization, not bench data. -->
            <!-- toe range: working band psi(phi) spans [-0.05, +0.73] rad over the knee
                 ROM; stops at [-0.15, +0.85] mirror the real hinge's physical stops
                 (heel ear in the carrier slot) and block the flipped-elbow branch of
                 the slider-crank, which otherwise engages under zero-g whip (measured:
                 airborne strike flipped the loop to slide=+27 mm without this). -->
            <joint name="{L}_toe_hinge" axis="0 0 1" range="-0.15 0.85"
                   damping="0.1" armature="2e-4"/>
            <!-- blade lower = conrod, toe hinge -> heel ear (curved plates -> capsule);
                 COLLIDES: it is the strike face -->
            <geom name="{L}_bladeL" type="capsule" fromto="0 0 0 0 {-CONROD_L} 0"
                  size="0.010" mass="{m['blade_lower']}"/>
          </body>
        </body>
        <!-- pushrod: prismatic on the shin (bushing); body origin = heel point at s=0 -->
        <body name="{L}_pushrod" pos="{-swing_x:.6f} -0.025 0">
          <!-- slide range: working stroke is [-41.14, 0] mm (the connect enforces the
               exact relation); +5 mm headroom stops limit-vs-loop chatter at the
               toggle's TDC (s=0 at phi=0) while still blocking the flipped-elbow
               branch of the crank (which would need s=+200 mm). -->
          <joint name="{L}_pushrod_slide" type="slide" axis="0 1 0"
                 range="{slide_lim[0] - 0.002:.5f} 0.005" damping="0.3" armature="2e-4"/>
          <geom name="{L}_rod" type="capsule" fromto="0 0 0 0 {-TIP_LEN + FOOT_R:.5f} 0"
                size="0.008" mass="{m['pushrod'] * 0.8}"/>
          <geom name="{L}_foot" type="sphere" pos="0 {-TIP_LEN + FOOT_R:.5f} 0"
                size="{FOOT_R}" mass="{m['pushrod'] * 0.2}"/>
        </body>
      </body>
    </body>'''
    # loop closure (option_a): heel ear on blade_lower == heel point on pushrod.
    # anchor is in body1 (blade_lower) local coords: (0, -CONROD_L, 0).
    # solref default (0.02 1): the json's 0.002 suggestion violates MuJoCo's
    # timeconst >= 2*timestep stability rule at dt=0.004 and blows the loop up.
    extras = f'''    <connect name="{L}_heel_pin" body1="{L}_blade_lower" body2="{L}_pushrod"
             anchor="0 {-CONROD_L} 0"/>'''
    excludes = f'''    <exclude body1="{L}_blade_lower" body2="{L}_pushrod"/>
    <exclude body1="{L}_blade_upper" body2="{L}_pushrod"/>'''
    acts = f'''    <motor name="{L}_yaw_m" joint="{L}_hip_yaw" gear="1.2" ctrlrange="-1 1"
           forcerange="-1.2 1.2"/>
    <motor name="{L}_swing_m" joint="{L}_leg_swing" gear="24.0" ctrlrange="-1 1"
           forcerange="-24 24"/>
    <motor name="{L}_knee_m" joint="{L}_knee_blade" gear="3.0" ctrlrange="-1 1"
           forcerange="-3 3"/>'''
    return body, (extras, excludes, acts)


def build_mesh_robot(asm: dict | None = None, floor: bool = True) -> str:
    asm = asm or load_assembly()
    t = asm["torso"]
    c = t["center_world"]
    hx, hy, hz = (s / 2 for s in t["size_xyz_m"])
    bodies, connects, excludes, acts = [], [], [], []
    for L in ("FL", "FR", "RL", "RR"):
        b, (cn, ex, ac) = _leg_xml(L, asm["legs"][L], c)
        bodies.append(b); connects.append(cn); excludes.append(ex); acts.append(ac)
    floor_xml = ('    <geom name="floor" type="plane" size="0 0 0.1" pos="0 0 0" '
                 'rgba="0.4 0.5 0.4 1"/>\n' if floor else "")
    sk = asm["striker_placeholder"]
    return f'''<mujoco model="mesh_robot_7_3">
  <compiler angle="radian" autolimits="true"/>
  <!-- dt=0.002 (NOT the fleet's 0.004): the slider-crank connect near its toggle
       dead-center flips constraint-force direction as the piston reverses; at
       dt=0.004 that explicit stiff system explodes (measured: 10 m slide error,
       |qacc| 7e9), at dt=0.002 it tracks the closed form to <1 mm. Keep control
       at 50 Hz via frame_skip=10 instead of 5. -->
  <option timestep="0.002" integrator="implicitfast"/>
  <default>
    <joint damping="0.02"/>
    <geom friction="1.2 0.1 0.1" contype="1" conaffinity="1"/>
  </default>
  <worldbody>
{floor_xml}    <body name="torso" pos="{c[0]} {c[1]} {c[2]}">
      <freejoint name="root"/>
      <!-- torso: PLACEHOLDER box (structure not designed); mass placeholder {TORSO_MASS_PLACEHOLDER} kg -->
      <geom name="torso" type="box" size="{hx:.5f} {hy:.5f} {hz:.5f}"
            mass="{TORSO_MASS_PLACEHOLDER}" rgba="0.3 0.3 0.6 1"/>
      <!-- striker: COSMETIC placeholder, non-colliding, no actuator -->
      <geom name="striker_ph" type="cylinder" pos="{hx + sk['length_m'] / 2:.5f} 0 0"
            quat="0.707107 0 0.707107 0" size="{sk['radius_m']} {sk['length_m'] / 2:.5f}"
            mass="0.1" contype="0" conaffinity="0" rgba="0.8 0.8 0.85 1"/>
{"".join(bodies)}
    </body>
  </worldbody>
  <contact>
{chr(10).join(excludes)}
  </contact>
  <equality>
{chr(10).join(connects)}
  </equality>
  <actuator>
{chr(10).join(acts)}
  </actuator>
</mujoco>
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ASSEMBLY_JSON))
    ap.add_argument("--out", default=str(HERE / "mesh_robot.xml"))
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    asm = load_assembly(args.json)
    xml = build_mesh_robot(asm)
    Path(args.out).write_text(xml)
    print(f"mesh_robot: 4 legs (yaw 1.2 / worm-swing 24 / knee 3 N.m), "
          f"toggle-press loop via <connect>, torso placeholder {TORSO_MASS_PLACEHOLDER} kg "
          f"-> {args.out}")
    if args.summary:
        import mujoco
        m = mujoco.MjModel.from_xml_string(xml)
        print(f"  njnt={m.njnt} nv={m.nv} nu={m.nu} neq={m.neq} "
              f"mass={m.body_mass.sum():.3f} kg")


if __name__ == "__main__":
    main()
