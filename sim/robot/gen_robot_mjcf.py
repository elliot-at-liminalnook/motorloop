# SPDX-License-Identifier: MIT
"""Parametric quadruped generator: robot.toml -> MJCF (MJX-native).

Single provenance-tracked source -> a valid MuJoCo model, so "change a part" is a
config edit + regenerate. The actuator force limit is derived from the profiled
motor envelope (sim/tests/motors.py), tying the body to the measured motors the
same way derive_params.py ties the RTL to the plant.

  python gen_robot_mjcf.py [--config robot.toml] [--out model.xml]
  # or import: build_mjcf(load_spec(path), overrides={...})
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

try:
    import tomllib                       # py3.11+ stdlib
except ModuleNotFoundError:              # py3.10: pip install tomli
    import tomli as tomllib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "sim" / "tests"))
from motors import MOTORS  # noqa: E402


def load_spec(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def joint_torque_limit(spec: dict) -> float:
    """Peak joint torque from the motor envelope x gear (the provenance tie-in)."""
    a = spec["actuator"]
    m = MOTORS[a["motor"]]
    peak_motor = m.kt * a["peak_factor"] * m.rated_current_a    # N·m at the motor
    return peak_motor * a["gear"]


def _leg_xml(leg: dict, d: dict, tau: float, prefix: str = "") -> tuple[str, list[str]]:
    """One 3-DOF mammalian leg chain; returns (body_xml, [actuator_joint_names])."""
    n = prefix + leg["name"]
    px, py, pz = leg["pos"]
    sy = 1.0 if py >= 0 else -1.0                  # abduction stand-off direction
    r, tl, cl = d["link_radius"], d["thigh_len"], d["calf_len"]
    ho, fr = d["hip_offset"], d["foot_radius"]
    stiff = d["joint_stiffness"]
    spring = f' stiffness="{stiff}" springref="0"' if stiff > 0 else ""
    jn = [f"{n}_abd", f"{n}_flex", f"{n}_knee"]
    xml = f'''
      <body name="{n}_hip" pos="{px} {py} {pz}">
        <joint name="{jn[0]}" axis="1 0 0" range="{d['abd_range'][0]} {d['abd_range'][1]}"{spring}/>
        <geom name="{n}_hipg" type="capsule" fromto="0 0 0 0 {sy*ho:.3f} 0" size="{r}" mass="0.3"/>
        <body name="{n}_thigh" pos="0 {sy*ho:.3f} 0">
          <joint name="{jn[1]}" axis="0 1 0" range="{d['flex_range'][0]} {d['flex_range'][1]}"{spring}/>
          <geom name="{n}_thighg" type="capsule" fromto="0 0 0 0 0 {-tl}" size="{r}" mass="{d['thigh_mass']}"/>
          <body name="{n}_calf" pos="0 0 {-tl}">
            <joint name="{jn[2]}" axis="0 1 0" range="{d['knee_range'][0]} {d['knee_range'][1]}"{spring}/>
            <geom name="{n}_calfg" type="capsule" fromto="0 0 0 0 0 {-cl}" size="{r}" mass="{d['calf_mass']}"/>
            <geom name="{n}_foot" type="sphere" pos="0 0 {-cl}" size="{fr}" mass="0.05"/>'''
    if leg.get("is_weapon"):
        # a leg-weapon: a slim spear geom past the foot (kinetic; heater pending EO)
        xml += f'''
            <geom name="{n}_spear" type="capsule" fromto="0 0 {-cl} 0 0 {-cl-0.10}" size="0.008" mass="0.08" rgba="0.8 0.2 0.2 1"/>'''
    xml += '''
          </body>
        </body>
      </body>'''
    return xml, jn


def _robot_xml(spec, prefix="", pos=(0.0, 0.0, None), quat=(1, 0, 0, 0), rgba=None):
    """One robot's <body> block + its actuator lines (names prefixed for matches)."""
    t, d = spec["torso"], spec["leg_defaults"]
    tau = joint_torque_limit(spec)
    hx, hy, hz = t["half_extents"]
    z = t["spawn_height"] if pos[2] is None else pos[2]
    rgba = rgba or "0.3 0.3 0.6 1"
    legs_xml, joints = [], []
    for leg in spec["leg"]:
        lx, jn = _leg_xml(leg, d, tau, prefix)
        legs_xml.append(lx); joints += jn
    body = (f'<body name="{prefix}torso" pos="{pos[0]} {pos[1]} {z}" '
            f'quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">'
            f'<freejoint name="{prefix}root"/>'
            f'<geom name="{prefix}torso" type="box" size="{hx} {hy} {hz}" '
            f'mass="{t["mass"]}" rgba="{rgba}"/>{"".join(legs_xml)}</body>')
    acts = [f'    <motor name="{j}_m" joint="{j}" ctrlrange="-1 1" '
            f'forcerange="{-tau:.3f} {tau:.3f}"/>' for j in joints]
    return body, acts, joints


def _wrap(spec, bodies, acts):
    d = spec["leg_defaults"]
    return f'''<mujoco model="{spec['meta']['name']}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.004" integrator="implicitfast"/>
  <default>
    <joint damping="{d['joint_damping']}" armature="0.01"/>
    <geom friction="1 0.1 0.1" contype="1" conaffinity="1"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.1" pos="0 0 0" rgba="0.4 0.5 0.4 1"/>
    {"".join(bodies)}
  </worldbody>
  <actuator>
{chr(10).join(acts)}
  </actuator>
</mujoco>
'''


def build_mjcf(spec: dict, overrides: dict | None = None) -> str:
    if overrides:
        spec = _deep_merge(spec, overrides)
    body, acts, _ = _robot_xml(spec)
    return _wrap(spec, [body], acts)


def build_match(spec_a: dict, spec_b: dict, sep: float = 2.4) -> str:
    """Two robots facing each other for a self-play match (A = ours, B = attacker)."""
    ba, aa, _ = _robot_xml(spec_a, "A_", pos=(-sep / 2, 0.0, None),
                           quat=(1, 0, 0, 0), rgba="0.3 0.4 0.7 1")
    bb, ab, _ = _robot_xml(spec_b, "B_", pos=(sep / 2, 0.0, None),
                           quat=(0, 0, 0, 1), rgba="0.7 0.3 0.3 1")
    return _wrap(spec_a, [ba, bb], aa + ab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "robot.toml"))
    ap.add_argument("--out", default=str(HERE / "model.xml"))
    args = ap.parse_args()
    spec = load_spec(args.config)
    xml = build_mjcf(spec)
    Path(args.out).write_text(xml)
    print(f"{spec['meta']['name']}: {len(spec['leg'])} legs, joint torque limit "
          f"{joint_torque_limit(spec):.2f} N·m ({spec['actuator']['motor']} x "
          f"{spec['actuator']['gear']} gear) -> {args.out}")


if __name__ == "__main__":
    main()
