# SPDX-License-Identifier: MIT
"""Parametric quadruped generator: robot.toml -> MuJoCo-Warp-ready MJCF.

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
from motors import MOTORS, SERVOS  # noqa: E402

# Typed spec validation (plan V.2): a malformed robot.toml dies HERE at model build,
# not mid-training. spec_schema is a sibling module (sim/robot on sys.path for every
# caller); the only acceptable failure is a pod without pydantic — degrade LOUDLY.
try:
    from spec_schema import validate_spec  # noqa: E402
except ImportError as _e:
    import warnings
    warnings.warn(f"robot spec validation disabled ({_e}); "
                  "install pydantic==2.* to validate robot.toml at model build")

    def validate_spec(d: dict) -> dict:
        return d


def load_spec(path: str | Path) -> dict:
    with open(path, "rb") as f:
        spec = tomllib.load(f)
    return validate_spec(spec)


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
    if a["motor"] in SERVOS:
        servo = SERVOS[a["motor"]]
        voltage = float(a.get("voltage", max(servo.stall_torque_nm)))
        if voltage not in servo.stall_torque_nm:
            raise ValueError(
                f"{a['motor']} has no torque data at {voltage:g} V; "
                f"available points: {sorted(servo.stall_torque_nm)}")
        return servo.stall_torque_nm[voltage] * float(a["gear"])
    m = MOTORS[a["motor"]]
    peak_motor = m.kt * a["peak_factor"] * m.rated_current_a    # N·m at the motor
    return peak_motor * a["gear"]



def _armature(spec: dict) -> float:
    """Reflected rotor inertia at the joint = J_rotor x gear^2 — DERIVED from the
    motor entry, never hardcoded. The old constant 0.01 was 'close by luck' for
    gear 12 (6e-5 x 144 = 0.0086) — the same hidden-actuator-property class as
    the gear bug; the model contract asserts dof_armature against this."""
    a = spec["actuator"]
    if a["motor"] in SERVOS:
        return SERVOS[a["motor"]].output_inertia_kg_m2_est * float(a["gear"]) ** 2
    return MOTORS[a["motor"]].inertia_kg_m2 * float(a["gear"]) ** 2


def actuator_unit_mass(spec: dict) -> float:
    """Physical mass of one integrated actuator; discrete BLDC studies predate it."""
    return SERVOS[spec["actuator"]["motor"]].mass_kg \
        if spec["actuator"]["motor"] in SERVOS else 0.0

def striker_force(s: dict) -> float:
    """Pneumatic cylinder force F = pressure × piston area = P·π(bore/2)² (the strike push)."""
    import math
    return float(s["pressure"]) * math.pi * (float(s["bore"]) / 2.0) ** 2


def _leg_xml(leg: dict, d: dict, tau: float, prefix: str = "", cc: str = "",
             cc_upper: str | None = None, cc_calf: str | None = None,
             striker: dict | None = None,
             motor_mass: float = 0.0) -> tuple[str, list[str], list[str]]:
    """One 3-DOF mammalian leg chain; returns (body_xml, [hinge_joint_names], [strike_joint_names]).
    `cc` = ` contype/conaffinity` string for the STRIKING geoms (calf/foot/spear/rod); `cc_upper`
    overrides it for the upper geoms (hip/thigh) — pass ` contype="0" conaffinity="0"` to make
    them non-colliding (the F-SPEED reduced-collision mode: fewer geoms in the contact set, so
    the dominant A-B pairs shrink). Default `cc_upper=cc` (uniform); empty `cc` => model default.
    `striker` (the spec[striker] dict) makes this a PNEUMATIC striker leg: a powered `slide` DOF
    that shoots a steel rod out fast — returns its joint name so `_robot_xml` builds the cylinder
    actuator. The slide joint is EXCLUDED from the locomotion obs by the env (hinge-only)."""
    if cc_upper is None:
        cc_upper = cc
    if cc_calf is None:
        cc_calf = cc
    n = prefix + leg["name"]
    px, py, pz = leg["pos"]
    sy = 1.0 if py >= 0 else -1.0                  # abduction stand-off direction
    r, tl, cl = d["link_radius"], d["thigh_len"], d["calf_len"]
    ho, fr = d["hip_offset"], d["foot_radius"]
    hip_mass, foot_mass = float(d.get("hip_mass", 0.3)), float(d.get("foot_mass", 0.05))
    stiff = d["joint_stiffness"]
    stand_abd = float(d.get("stand_abd", 0.0))
    stand_flex = float(d.get("stand_flex", -0.4))
    stand_knee = float(d.get("stand_knee", -1.1))
    # PARALLEL-ELASTIC springs centered on the STANDING stance (springref = the env _q0 default angles),
    # so they passively HOLD the stand (zero spring torque at the stance, restoring force if it buckles)
    # rather than straightening the legs and fighting the actuator.
    def _sp(ref):
        return f' stiffness="{stiff}" springref="{ref}"' if stiff > 0 else ""
    sp_abd, sp_flex, sp_knee = _sp(stand_abd), _sp(stand_flex), _sp(stand_knee)
    jn = [f"{n}_abd", f"{n}_flex", f"{n}_knee"]
    strike_jn: list[str] = []
    xml = f'''
      <body name="{n}_hip" pos="{px} {py} {pz}">
        <joint name="{jn[0]}" axis="1 0 0" range="{d['abd_range'][0]} {d['abd_range'][1]}"{sp_abd}/>
        <geom name="{n}_hipg" type="capsule" fromto="0 0 0 0 {sy*ho:.3f} 0" size="{r}" mass="{hip_mass + motor_mass}"{cc_upper}/>
        <body name="{n}_thigh" pos="0 {sy*ho:.3f} 0">
          <joint name="{jn[1]}" axis="0 1 0" range="{d['flex_range'][0]} {d['flex_range'][1]}"{sp_flex}/>
          <geom name="{n}_thighg" type="capsule" fromto="0 0 0 0 0 {-tl}" size="{r}" mass="{d['thigh_mass'] + motor_mass}"{cc_upper}/>
          <body name="{n}_calf" pos="0 0 {-tl}">
            <joint name="{jn[2]}" axis="0 1 0" range="{d['knee_range'][0]} {d['knee_range'][1]}"{sp_knee}/>
            <geom name="{n}_calfg" type="capsule" fromto="0 0 0 0 0 {-cl}" size="{r}" mass="{d['calf_mass'] + motor_mass}"{cc_calf}/>
            <geom name="{n}_foot" type="sphere" pos="0 0 {-cl}" size="{fr}" mass="{foot_mass}"{cc}/>'''
    if striker:
        # PNEUMATIC striker: a carriage on a slide DOF carrying a steel rod, shot out fast by a
        # constant-force cylinder. `_rod` is a striking geom (legs-as-weapons damage credits it).
        s = striker; sj = f"{n}_strike"; strike_jn = [sj]
        st, rl, rr = s["stroke"], s["rod_len"], s["rod_radius"]
        # FORWARD striker: carriage at the knee, rod fires FORWARD (+x, toward the opponent) instead of
        # down into the floor (the old -z mount spawned underground -> launch bug, and could only stomp).
        xml += f'''
            <body name="{n}_carriage" pos="0 0 0">
              <joint name="{sj}" type="slide" axis="1 0 0" range="0 {st}" damping="2" armature="0.005" stiffness="{s['return_stiffness']}" springref="0" solreflimit="0.002 1"/>
              <!-- rod z+2mm: at qpos0 the rod axis EXACTLY intersected the thigh-guard axis -> undefined
            contact normal (C picked ±y, warp +x; cross-engine parity 3.5e-2). Degenerate geometry,
            not an engine bug — L-R1 finding. -->
              <geom name="{n}_rod" type="capsule" fromto="0 0 0.002 {rl} 0 0.002" size="{rr}" density="{s['rod_density']}" rgba="0.85 0.85 0.9 1"{cc}/>
            </body>'''
    elif leg.get("is_weapon"):
        # a rigid leg-weapon: a slim spear geom past the foot (kinetic; heater pending EO)
        xml += f'''
            <geom name="{n}_spear" type="capsule" fromto="0 0 {-cl} 0 0 {-cl-0.10}" size="0.008" mass="0.08" rgba="0.8 0.2 0.2 1"{cc}/>'''
    xml += '''
          </body>
        </body>
      </body>'''
    return xml, jn, strike_jn


def _striker_enabled(spec, striker):
    """Resolve the striker switch: explicit True/False overrides; None => the spec default."""
    if striker is None:
        return bool(spec.get("striker", {}).get("enabled", False))
    return bool(striker)


def _robot_xml(spec, prefix="", pos=(0.0, 0.0, None), quat=(1, 0, 0, 0), rgba=None, cc="",
               cc_upper=None, cc_calf=None, striker=None):
    """One robot's <body> block + its actuator lines (names prefixed for matches).
    `cc` = contype/conaffinity for torso + striking geoms; `cc_upper` for hip/thigh (pass
    "0 0" to drop them from collision — reduced-collision lean mode). `striker` (True/False/None,
    None=spec default) puts a PNEUMATIC striker on each FRONT leg (pos.x>0) — a powered slide DOF
    whose cylinder actuator is appended AFTER the hinge motors (so the env's action vector is
    [hinge…, strike…])."""
    t, d = spec["torso"], spec["leg_defaults"]
    tau = joint_torque_limit(spec)
    hx, hy, hz = t["half_extents"]
    z = t["spawn_height"] if pos[2] is None else pos[2]
    rgba = rgba or "0.3 0.3 0.6 1"
    use_striker = _striker_enabled(spec, striker)
    sspec = spec.get("striker", {}) if use_striker else {}
    motor_mass = actuator_unit_mass(spec)
    legs_xml, joints, strike_joints = [], [], []
    for leg in spec["leg"]:
        on = sspec if (use_striker and leg["pos"][0] > 0) else None      # front legs get the rod
        lx, jn, sj = _leg_xml(leg, d, tau, prefix, cc, cc_upper, cc_calf,
                              striker=on, motor_mass=motor_mass)
        legs_xml.append(lx); joints += jn; strike_joints += sj
    body = (f'<body name="{prefix}torso" pos="{pos[0]} {pos[1]} {z}" '
            f'quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">'
            f'<freejoint name="{prefix}root"/>'
            f'<geom name="{prefix}torso" type="box" size="{hx} {hy} {hz}" '
            f'mass="{t["mass"]}" rgba="{rgba}"{cc}/>{"".join(legs_xml)}</body>')
    # MuJoCo motor torque = gear × ctrl with ctrl clamped to ±1; forcerange only CLAMPS,
    # never amplifies. gear must equal tau or the actuator silently defaults to gear=1
    # (≈8% of design torque — the bug every pre-2026-07 training run ran under).
    acts = [f'    <motor name="{j}_m" joint="{j}" gear="{tau:.3f}" ctrlrange="-1 1" '
            f'forcerange="{-tau:.3f} {tau:.3f}"/>' for j in joints]
    if strike_joints:                                  # pneumatic cylinders (constant force + valve lag)
        F = striker_force(sspec); vt = sspec["valve_tau"]
        acts += [f'    <general name="{sj}_m" joint="{sj}" gaintype="fixed" gainprm="{F:.2f} 0 0" '
                 f'biastype="none" ctrlrange="0 1" forcerange="0 {F:.2f}" dyntype="filter" '
                 f'dynprm="{vt}"/>' for sj in strike_joints]
    return body, acts, joints


def _cc(ct, ca):
    return f' contype="{ct}" conaffinity="{ca}"'


def _calf_floor_enabled(spec: dict) -> bool:
    c = spec.get("contact", {})
    return bool(c.get("calf_floor", not c.get("disable_calf_floor", False)))


def _contact_pair_xml(pairs: list[str]) -> str:
    if not pairs:
        return ""
    return "  <contact>\n" + "\n".join(pairs) + "\n  </contact>\n"


def _floor_calf_pairs(*prefixed_specs: tuple[str, dict]) -> list[str]:
    pairs: list[str] = []
    for prefix, spec in prefixed_specs:
        contact = spec.get("contact", {})
        solref = contact.get("floor_calf_solref")
        if solref is None:
            continue
        solref_s = " ".join(str(float(x)) for x in solref)
        solimp = contact.get("floor_calf_solimp")
        solimp_s = ""
        if solimp is not None:
            solimp_s = ' solimp="' + " ".join(str(float(x)) for x in solimp) + '"'
        for leg in spec["leg"]:
            pairs.append(
                f'    <pair geom1="floor" geom2="{prefix}{leg["name"]}_calfg" '
                f'solref="{solref_s}"{solimp_s}/>'
            )
    return pairs


def _wrap(spec, bodies, acts, floor_cc="", contact_pairs: list[str] | None = None,
          lidar: bool = False, lidar_prefix: str = "A_", lidar_n_rays: int = 128,
          lidar_n_vertical: int = 16, lidar_max_range: float = 2.0):
    d = spec["leg_defaults"]
    contact = spec.get("contact", {})
    fr = contact.get("friction", [1, 0.1, 0.1])
    friction = " ".join(str(float(x)) for x in fr)
    contact_attrs = ""
    if contact.get("solref") is not None:
        contact_attrs += ' solref="' + " ".join(str(float(x)) for x in contact["solref"]) + '"'
    if contact.get("solimp") is not None:
        contact_attrs += ' solimp="' + " ".join(str(float(x)) for x in contact["solimp"]) + '"'
    # Keep one explicit material on lidar-enabled geoms so the uploaded Warp ray
    # scene has a complete material table.
    asset_xml = ""
    material_attr = ""
    if lidar:
        asset_xml = '\n  <asset>\n    <material name="mat0" rgba="0.5 0.5 0.5 1"/>\n  </asset>'
        material_attr = ' material="mat0"'
    # Insert lidar sites into the first body (A_torso) when lidar is enabled
    lidar_sites_xml = ""
    lidar_sensors_xml = ""
    if lidar:
        lidar_sites_xml, lidar_sensors_xml = _lidar_sites_xml(
            lidar_prefix, lidar_n_rays, lidar_max_range, lidar_n_vertical)
        # Insert sites before the first geom in the first body (A_torso)
        bodies = list(bodies)
        bodies[0] = bodies[0].replace(
            '<geom name="A_torso"',
            lidar_sites_xml + '<geom name="A_torso"')
    return f'''<mujoco model="{spec['meta']['name']}">
  <compiler angle="radian" autolimits="true"/>{asset_xml}
  <option timestep="0.004" integrator="implicitfast"/>
  <default>
    <joint damping="{d['joint_damping']}" armature="{_armature(spec):.5f}"/>
    <geom friction="{friction}" contype="1" conaffinity="1"{material_attr}{contact_attrs}/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.1" pos="0 0 0" rgba="0.4 0.5 0.4 1"{floor_cc}{material_attr}/>
    {"".join(bodies)}
  </worldbody>
{_contact_pair_xml(contact_pairs or [])}{lidar_sensors_xml}  <actuator>
{chr(10).join(acts)}
  </actuator>
</mujoco>
'''


def build_mjcf(spec: dict, overrides: dict | None = None, self_collision: bool = True,
               striker=False) -> str:
    # SINGLE-robot body (locomotor / parity / design / bench) defaults striker OFF — a walker
    # doesn't need the weapon, and this keeps every analysis tool on the legacy 12-action body.
    # The FIGHTER (build_match / AdversarialEnv) defaults striker ON (spec). Pass striker=True here
    # to view/analyze the armed body.
    if overrides:
        spec = _deep_merge(spec, overrides)
    if self_collision:                       # default: every geom collides (the validated model)
        body, acts, _ = _robot_xml(spec, striker=striker)
        return _wrap(spec, [body], acts, contact_pairs=_floor_calf_pairs(("", spec)))
    # F-SPEED lean scheme: robot collides with the floor only, not with itself
    body, acts, _ = _robot_xml(spec, cc=_cc(2, 1), striker=striker)
    return _wrap(spec, [body], acts, floor_cc=_cc(1, 2),
                 contact_pairs=_floor_calf_pairs(("", spec)))


import math


def _lidar_sites_xml(prefix: str, n_rays: int, max_range: float = 2.0,
                     n_vertical: int = 0, v_fov: float = 0.3) -> tuple[str, str]:
  """Generate lidar rangefinder sites on a torso and the corresponding <sensor> block.

  Returns (sites_xml, sensors_xml) to be inserted into the MJCF.
  Sites are placed at the torso center. Each ray is a rangefinder that shoots
  along the site's -z axis. Horizontal rays sweep 360 degrees; optional
  vertical rays fan forward at different pitch angles.

  A material is required on all geoms for the uploaded Warp ray scene; callers add
  ``<material name="mat0" .../>`` to <asset> and ``material="mat0"`` to the
  default <geom> when lidar is enabled.
  """
  sites = []
  sensors = []
  idx = 0
  # Horizontal sweep
  for i in range(n_rays):
    angle = 2.0 * math.pi * i / n_rays
    dx, dy = math.cos(angle), math.sin(angle)
    # Rotate (0,0,-1) -> (dx,dy,0): axis = cross((0,0,-1),(dx,dy,0)) = (dy,-dx,0)
    ax, ay = dy, -dx
    n = math.hypot(ax, ay)
    if n < 1e-8:
      quat = "1 0 0 0"
    else:
      ax, ay = ax / n, ay / n
      a = math.pi / 2
      w = math.cos(a / 2); s = math.sin(a / 2)
      quat = f"{w:.6f} {ax * s:.6f} {ay * s:.6f} 0.0"
    nm = f"{prefix}lidar_{idx}"
    sites.append(f'<site name="{nm}" pos="0 0 0.03" quat="{quat}" size="0.005"/>')
    sensors.append(f'    <rangefinder site="{nm}" cutoff="{max_range}"/>')
    idx += 1
  # Optional vertical fan (forward-facing only)
  if n_vertical > 0:
    for j in range(n_vertical):
      pitch = -v_fov + 2.0 * v_fov * (j + 1) / (n_vertical + 1)
      # Ray direction: (cos(pitch), 0, sin(pitch))
      dx = math.cos(pitch)
      # Rotate (0,0,-1) -> (dx,0,dz): axis = cross((0,0,-1),(dx,0,dz)) = (0,-dz,dx)... wait
      # cross((0,0,-1),(dx,0,dz)) = (0*dz - (-1)*0, (-1)*dx - 0*dz, 0*0 - 0*dx) = (0, -dx, 0)
      ax, ay, az = 0.0, -dx, 0.0
      n = math.sqrt(ax * ax + ay * ay + az * az)
      if n < 1e-8:
        quat = "1 0 0 0"
      else:
        ax, ay, az = ax / n, ay / n, az / n
        a = math.pi / 2
        w = math.cos(a / 2); s = math.sin(a / 2)
        quat = f"{w:.6f} {ax * s:.6f} {ay * s:.6f} {az * s:.6f}"
      nm = f"{prefix}lidar_v{idx}"
      sites.append(f'<site name="{nm}" pos="0 0 0.03" quat="{quat}" size="0.005"/>')
      sensors.append(f'    <rangefinder site="{nm}" cutoff="{max_range}"/>')
      idx += 1
  sites_xml = "\n      ".join(sites)
  sensors_xml = "  <sensor>\n" + "\n".join(sensors) + "\n  </sensor>\n"
  return sites_xml, sensors_xml


def _lidar_ray_count(n_rays: int, n_vertical: int) -> int:
  """Total number of rangefinder rays (horizontal + vertical)."""
  return n_rays + n_vertical


def build_match(spec_a: dict, spec_b: dict, sep: float = 2.4, self_collision: bool = True,
                striker=None, striker_b=False, lidar=False, lidar_n_rays: int = 128,
                lidar_n_vertical: int = 16, lidar_max_range: float = 2.0) -> str:
    """Two robots facing each other for a self-play match (A = ours, B = attacker).
    `self_collision=False` (F-SPEED) disables intra-robot self-collision via contype/conaffinity
    — keeps A↔B and X↔floor (the fight + support contacts) but drops the ~O(geoms²) self-pairs,
    the #1 throughput lever for the scene. Plain MuJoCo and Warp share the model.
    `striker` (None=spec default) arms A with the pneumatic striker; `striker_b` arms B too —
    needed for a SYMMETRIC self-play match (an A snapshot drives B → identical 14-action bodies)
    or a scripted armed attacker. Default `striker_b=False` keeps B passive (skill-curriculum)."""
    if self_collision:                       # default: the validated model (all pairs collide)
        ba, aa, _ = _robot_xml(spec_a, "A_", pos=(-sep / 2, 0.0, None), quat=(1, 0, 0, 0), rgba="0.3 0.4 0.7 1", striker=striker)
        bb, ab, _ = _robot_xml(spec_b, "B_", pos=(sep / 2, 0.0, None), quat=(0, 0, 0, 1), rgba="0.7 0.3 0.3 1", striker=striker_b)
        return _wrap(spec_a, [ba, bb], aa + ab,
                     contact_pairs=_floor_calf_pairs(("A_", spec_a), ("B_", spec_b)),
                     lidar=lidar, lidar_prefix="A_", lidar_n_rays=lidar_n_rays,
                     lidar_n_vertical=lidar_n_vertical, lidar_max_range=lidar_max_range)
    # lean: floor(1,6) ↔ A(2,5) ↔ B(4,3) — A-floor, B-floor, A-B collide; A-A, B-B do not.
    # AND drop hip/thigh from collision (cc_upper "0 0") so only torso/calf/foot/spear/rod collide
    # — shrinks the dominant A-B pair count (legs-as-weapons damage still works via calf/foot/spear/rod).
    off = _cc(0, 0)
    calf_a = calf_b = None
    if not (_calf_floor_enabled(spec_a) and _calf_floor_enabled(spec_b)):
        # Keep calf capsules available for A<->B contacts, but remove their floor contact bit.
        # Feet remain the floor support geometry. Extra high bits preserve calf-calf collisions
        # without re-enabling same-robot or floor pairs under the lean contact mask.
        calf_a = _cc(8, 20)   # collides with B default (4) and B no-floor calf (16), not floor (1/6)
        calf_b = _cc(16, 10)  # collides with A default (2) and A no-floor calf (8), not floor (1/6)
    ba, aa, _ = _robot_xml(spec_a, "A_", pos=(-sep / 2, 0.0, None), quat=(1, 0, 0, 0), rgba="0.3 0.4 0.7 1", cc=_cc(2, 5), cc_upper=off, cc_calf=calf_a, striker=striker)
    bb, ab, _ = _robot_xml(spec_b, "B_", pos=(sep / 2, 0.0, None), quat=(0, 0, 0, 1), rgba="0.7 0.3 0.3 1", cc=_cc(4, 3), cc_upper=off, cc_calf=calf_b, striker=striker_b)
    return _wrap(spec_a, [ba, bb], aa + ab, floor_cc=_cc(1, 6),
                 contact_pairs=_floor_calf_pairs(("A_", spec_a), ("B_", spec_b)),
                 lidar=lidar, lidar_prefix="A_", lidar_n_rays=lidar_n_rays,
                 lidar_n_vertical=lidar_n_vertical, lidar_max_range=lidar_max_range)


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
