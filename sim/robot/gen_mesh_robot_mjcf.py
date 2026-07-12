# SPDX-License-Identifier: MIT
"""Robot_Assembly_7-3-2026.physics.json -> MuJoCo-Warp-ready MJCF.

Converts the REAL leg mechanism (worm-pitch + toggle-press blade foot + yaw mount)
into a trainable robot XML, following the json's own `mjcf_conversion` recipe:

  * chain per leg: {L}_hip_yaw -> {L}_leg_swing -> {L}_knee_blade (+ passive
    {L}_toe_hinge, {L}_heel_pin*, {L}_pushrod_slide) at the json's origins/axes/limits
  * loop closure — two modes (notes/sim-engine-secret-sauce.md §8):
      OPTION B "polycoef" (DEFAULT): two 1-row <equality><joint> quartic couplings
        q_slide = poly_s(phi_knee), q_toe = poly_psi(phi_knee), least-squares fitted
        to the closed forms over the knee ROM (TDC-weighted). Each row's Jacobian is
        [1, -poly'(phi)] — the constant 1 keeps it FULL-RANK at the toggle's
        dead center where the connect's anchor Jacobian goes singular, and the
        single-valued polynomial makes the flipped-elbow branch unreachable.
        Toggle force amplification is preserved: slide-force/crank-torque = 1/poly'.
        This is what re-enables the fleet's dt=0.004 (see <option> comment).
      OPTION A "connect" (RETIRED, in git history): <connect> pinning the heel ear
        (blade_lower local (0,-L,0)) to the pushrod's heel point. Its anchor
        Jacobian is what went singular at TDC and forced dt=0.002; the closed
        forms live on as the fit reference and loop_consistent_pose(loop="connect").
    Either way the toggle-press force profile is PHYSICS, not scripting.
    (*heel_pin as a tree joint is subsumed by the loop constraint: blade_lower's
    swing relative to the pushrod IS the constrained dof; adding it as a tree joint
    too would double-count. Documented deviation from the literal 6-joint chain —
    the census test counts 5 tree joints + 2 couplings per leg.)
  * primitives only: every geom is a capsule/box/sphere sized from the
    mechanism dimensions; each records what it approximates in a comment.
  * honest transmissions (Waveshare ST3215-HS @12V on all 12 joints, 2026-07-09):
      yaw   — servo through belt ratio 6 (DESIGN DECISION: enables loaded-foot
              slew): 11.77 N.m, 1.85 rad/s free; SEA stiffness/damping still
              PLACEHOLDER joint damping (bench-measure the rubber pulley)
      swing — servo through worm 20:1: 39.23 N.m but 0.56 rad/s free — THE slow
              axis; self-locking approximated by frictionloss = 4.0 N.m
      knee  — servo direct on crank: 1.96 N.m; toggle TDC multiplication applies
    The env MUST derate drive torque by the WFREE torque-speed line or trained
    policies exploit servo speeds the hardware does not have.
  * mass is normalized to the 6 lb competition limit. All twelve 68 g servos are
    counted at datasheet mass; the assembly's placeholder non-servo masses retain
    their relative distribution and are scaled into the remaining budget.
  * per-axis armature = estimated servo-output inertia x external ratio^2.

CLI: python gen_mesh_robot_mjcf.py [--json PATH] [--out PATH] [--summary]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSEMBLY_JSON = HERE.parents[1] / "Robot_Assembly_7-3-2026.physics.json"
sys.path.insert(0, str(HERE.parents[1] / "sim" / "tests"))
from motors import SERVOS  # noqa: E402

# mechanism constants (leg json / mjcf_conversion.closed_form)
CRANK_R = 0.075          # knee pin -> toe hinge (blade upper)
CONROD_L = 0.100         # toe hinge -> heel pin (blade lower)
TIP_LEN = 0.258          # heel pin -> pushrod tip; from the json's stomp reach:
                         # 0.025 + 0.04114 + tip = 0.3241  (sizing_reasoning.rule_height)
FOOT_R = 0.012           # pushrod tip contact sphere
LB_TO_KG = 0.45359237
MAX_ROBOT_MASS_LB = 6.0
MAX_ROBOT_MASS_KG = MAX_ROBOT_MASS_LB * LB_TO_KG
SERVO_COUNT = 12
SERVO_MODEL = "waveshare_st3215_hs"
SERVO_VOLTAGE = 12.0
_SERVO = SERVOS[SERVO_MODEL]
SERVO_STALL_NM = _SERVO.stall_torque_nm[SERVO_VOLTAGE]
SERVO_FREE_RAD_S = _SERVO.no_load_speed_rad_s[SERVO_VOLTAGE]
SERVO_MASS = _SERVO.mass_kg
SERVO_STALL_CURRENT_A = _SERVO.stall_current_a[SERVO_VOLTAGE]
SERVO_NO_LOAD_CURRENT_A = _SERVO.no_load_current_a[SERVO_VOLTAGE]

# The assembly JSON explicitly labels its masses as placeholders. Preserve their
# relative distribution while forcing the compiled robot to the legal maximum.
# This is a conservative simulation mass envelope, not proof that the physical BOM
# fits: the real chassis, battery, wiring and fasteners still need to make this budget.
_RAW_TORSO_STRUCTURE_MASS = 2.0
_RAW_STRIKER_MASS = 0.1
_RAW_LEG_STRUCTURE_MASS = 1.31
_RAW_STRUCTURE_MASS = (_RAW_TORSO_STRUCTURE_MASS + _RAW_STRIKER_MASS
                       + 4 * _RAW_LEG_STRUCTURE_MASS)
NON_SERVO_MASS_BUDGET_KG = MAX_ROBOT_MASS_KG - SERVO_COUNT * SERVO_MASS
if NON_SERVO_MASS_BUDGET_KG <= 0:
    raise ValueError("the 12 selected servos alone exceed the 6 lb robot limit")
STRUCTURE_MASS_SCALE = NON_SERVO_MASS_BUDGET_KG / _RAW_STRUCTURE_MASS
TORSO_STRUCTURE_MASS = _RAW_TORSO_STRUCTURE_MASS * STRUCTURE_MASS_SCALE
STRIKER_MASS = _RAW_STRIKER_MASS * STRUCTURE_MASS_SCALE
LEG_STRUCTURE_MASS = _RAW_LEG_STRUCTURE_MASS * STRUCTURE_MASS_SCALE

WORM_FRICTIONLOSS = 4.0  # N.m — self-locking stand-in for the worm drive.
                         # This models non-backdrivability, not a
                         # mass-proportional viscous loss.
                         # True worm hold is infinite; keep >= 2x measured stance load.
YAW_DAMPING_PLACEHOLDER = 0.08   # rubber-pulley SEA stand-in; bench-measure to replace
# --- actuation: Waveshare ST3215-HS bus servo, ALL 12 joints ---
# Datasheet profile is single-sourced in sim/tests/motors.py SERVOS.
BELT_RATIO = 6.0           # DESIGN DECISION: 11.77 N·m yaw is above the loaded-foot
                           # drag threshold at the 6 lb limit. Costs speed: 1.85 rad/s
                           # free. Size the external pulley reduction to 6:1.
WORM_RATIO = 20.0          # existing worm pair (self-locking retained)
GEARS = {"hip_yaw": SERVO_STALL_NM * BELT_RATIO,     # 11.77 N·m
         "leg_swing": SERVO_STALL_NM * WORM_RATIO,   # 39.23 N·m
         "knee_blade": SERVO_STALL_NM}               # 1.96 N·m at the crank
# joint no-load speeds — the torque-speed line the env's servo model derates
# against (τ_drive <= stall·(1-|ω|/ω_free); braking torque is NOT derated)
WFREE = {"hip_yaw": SERVO_FREE_RAD_S / BELT_RATIO,   # 1.85 rad/s
         "leg_swing": SERVO_FREE_RAD_S / WORM_RATIO, # 0.56 rad/s — the slow axis
         "knee_blade": SERVO_FREE_RAD_S}             # 11.10 rad/s
# Reflected inertia at the servo output is NOT published for ST3215-HS. Keep the
# catalog's explicit estimate until it is measured, then apply external ratio².
# SEA yaw option (upgrade 6, 2026-07-04): rotor dof + spring coupling to the
# yaw joint models the rubber-belt series elasticity. Stiffness is an ESTIMATE
# (bench-measure the pulley!) so the option defaults OFF — training against a
# guessed k transfers worse than the rigid approximation. Resonance note:
# k=15 N·m/rad over estimated coupled yaw inertia 0.20 kg·m² gives
# f = sqrt(k/I)/2π ≈ 1.38 Hz — the same order as gait cadence, i.e. a tuned
# belt could store/return stride energy. Worth designing toward, on real data.
SEA_YAW_STIFFNESS_EST = 15.0   # N·m/rad — ESTIMATE, not a datasheet number
SEA_YAW_EFFECTIVE_INERTIA_EST = 0.20  # kg·m² at the 6 lb envelope; bench-fit
SERVO_OUTPUT_INERTIA_KG_M2_EST = _SERVO.output_inertia_kg_m2_est
ARMATURE = {"hip_yaw": SERVO_OUTPUT_INERTIA_KG_M2_EST * BELT_RATIO ** 2,
            "leg_swing": SERVO_OUTPUT_INERTIA_KG_M2_EST * WORM_RATIO ** 2,
            "knee_blade": SERVO_OUTPUT_INERTIA_KG_M2_EST}


def slider_crank_s(phi: float) -> float:
    """Pushrod slide value vs knee angle (json closed_form): s(0)=0 at blade-vertical."""
    return (CRANK_R * math.cos(phi)
            - math.sqrt(CONROD_L ** 2 - (CRANK_R * math.sin(phi)) ** 2)
            - (CRANK_R - CONROD_L))


def conrod_psi(phi: float) -> float:
    """Conrod counter-rotation vs knee angle (closed form): psi(0)=0 at TDC.

    The conrod must counter-rotate so the heel stays on the slide lane:
    psi(phi) = asin(r sin(phi)/L) - phi."""
    return math.asin(CRANK_R * math.sin(phi) / CONROD_L) - phi


def poly_eval(c, x: float) -> float:
    """Horner eval of ascending polycoef c0..c4 — the relation the model enforces."""
    return ((((c[4] * x + c[3]) * x + c[2]) * x + c[1]) * x + c[0])


def loop_consistent_pose(phi: float, loop: str = "polycoef") -> tuple[float, float]:
    """(toe_hinge, pushrod_slide) that close the slider-crank at knee angle phi.

    MODEL-exact for the given loop mode: evaluates the SAME fitted polynomials
    (POLY_TOE/POLY_SLIDE — single source of truth) that the build compiles into
    <equality><joint>, or the closed forms for loop="connect" (the retired weld,
    kept as the analytic reference). Setting knee qpos WITHOUT these two is a
    26 mm constraint violation at phi=-60 (a connect responds with ~1.2 kN) —
    every reset/test that places the knee off zero MUST use this."""
    if loop == "polycoef":
        return poly_eval(POLY_TOE, phi), poly_eval(POLY_SLIDE, phi)
    return conrod_psi(phi), slider_crank_s(phi)


def load_assembly(path=ASSEMBLY_JSON) -> dict:
    return json.loads(Path(path).read_text())


# --- quartic loop couplings: fitted ONCE at import, pure python (numpy-free).
# POLY_SLIDE/POLY_TOE are the single source of truth for the emitted XML,
# loop_consistent_pose, and the tests. Fit domain = knee ROM from the json.
KNEE_ROM = (math.radians(-90.0), math.radians(10.0))


def _fit_quartic(f, lo: float, hi: float, n: int = 721, tdc_w: float = 25.0):
    """Weighted least-squares quartic c0..c4 of f over [lo, hi] via 5x5 normal
    equations + Gaussian elimination (partial pivoting). Pure python, deterministic;
    the Gaussian weight bump at phi=0 pins the toggle's dead center."""
    xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    ws = [1.0 + (tdc_w - 1.0) * math.exp(-((x / 0.05) ** 2)) for x in xs]
    A = [[sum(w * x ** (i + j) for x, w in zip(xs, ws)) for j in range(5)]
         for i in range(5)]
    b = [sum(w * f(x) * x ** i for x, w in zip(xs, ws)) for i in range(5)]
    for col in range(5):
        piv = max(range(col, 5), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        b[col], b[piv] = b[piv], b[col]
        for r in range(col + 1, 5):
            k = A[r][col] / A[col][col]
            A[r] = [a - k * ac for a, ac in zip(A[r], A[col])]
            b[r] -= k * b[col]
    c = [0.0] * 5
    for i in range(4, -1, -1):
        c[i] = (b[i] - sum(A[i][j] * c[j] for j in range(i + 1, 5))) / A[i][i]
    return tuple(c)


POLY_SLIDE = _fit_quartic(slider_crank_s, *KNEE_ROM)   # q_slide = poly(phi_knee)
POLY_TOE = _fit_quartic(conrod_psi, *KNEE_ROM)         # q_toe   = poly(phi_knee)
_LOOP_RESID = None


def loop_polycoefs():
    """(POLY_SLIDE, POLY_TOE, resid_slide, resid_toe) with §8 gate (a) enforced:
    the generator refuses to build if the quartics can't hold 0.5 mm / 0.3 deg
    over the knee ROM. Single source of truth for helper, emission, and tests."""
    global _LOOP_RESID
    if _LOOP_RESID is None:
        lo, hi = KNEE_ROM
        xs = [lo + (hi - lo) * i / 720 for i in range(721)]
        rs = max(abs(poly_eval(POLY_SLIDE, x) - slider_crank_s(x)) for x in xs)
        rp = max(abs(poly_eval(POLY_TOE, x) - conrod_psi(x)) for x in xs)
        if rs > 5e-4:
            raise ValueError(f"slide quartic residual {rs * 1e3:.3f} mm > 0.5 mm — "
                             f"restrict ROM or build the connect option + dt=0.002")
        if rp > math.radians(0.3):
            raise ValueError(f"toe quartic residual {math.degrees(rp):.3f} deg > 0.3 deg")
        _LOOP_RESID = (rs, rp)
    return POLY_SLIDE, POLY_TOE, _LOOP_RESID[0], _LOOP_RESID[1]


def _leg_xml(L: str, leg: dict, torso_center, sea: bool = False,
             mass_scale: float = STRUCTURE_MASS_SCALE) -> tuple[str, str]:
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
    # + the swing servo body (68 g) rides the housing; yaw servo mass is added
    # at the torso mount (see build); knee servo rides the swing link
    housing_mass = ((m["base_housing_and_axle"] + m["worm_input"]) * mass_scale
                    + SERVO_MASS)
    cc0 = ' contype="0" conaffinity="0"'                # non-colliding (json contact_guidance)
    body = f'''
    <body name="{L}_base_housing" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"
          quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}">
      <joint name="{L}_hip_yaw" axis="0 1 0" range="{yaw_lim[0]:.7f} {yaw_lim[1]:.7f}"
             armature="{0.0 if sea else ARMATURE['hip_yaw']}" damping="{YAW_DAMPING_PLACEHOLDER}"/>
      <!-- housing+worm+pulley: box spanning yaw pivot to swing pivot (drive_housing bounds) -->
      <geom name="{L}_housing" type="box" pos="{(yaw_x + swing_x) / 2:.5f} 0 0"
            size="{abs(yaw_x - swing_x) / 2 + 0.02:.5f} 0.035 0.030" mass="{housing_mass}"{cc0}/>
      <body name="{L}_swing_link" pos="{swing_x:.6f} 0 0">
        <joint name="{L}_leg_swing" axis="0 0 1" range="{swing_lim[0]:.7f} {swing_lim[1]:.7f}"
               armature="{ARMATURE['leg_swing']}" frictionloss="{WORM_FRICTIONLOSS}"/>
        <!-- swing link: arm capsule pivot->knee (guide rails + drive frame) -->
        <geom name="{L}_arm" type="capsule" fromto="0 0 0 {-swing_x:.5f} 0 0"
              size="0.015" mass="{m['leg_swing_link'] * 0.5 * mass_scale}"{cc0}/>
        <!-- shin rail: knee station down to the bushing (leg-local -Y) -->
        <geom name="{L}_shin" type="capsule"
              fromto="{-swing_x:.5f} 0 0 {-swing_x:.5f} -0.2405 0"
              size="0.012" mass="{m['leg_swing_link'] * 0.5 * mass_scale + SERVO_MASS}"{cc0}/>
        <body name="{L}_blade_upper" pos="{-swing_x:.6f} 0 0">
          <joint name="{L}_knee_blade" axis="0 0 1" range="{knee_lim[0]:.7f} {knee_lim[1]:.7f}"
                 armature="{ARMATURE['knee_blade']}"/>
          <!-- blade upper = knee crank, knee pin -> toe hinge (straight strip) -->
          <geom name="{L}_bladeU" type="capsule" fromto="0 0 0 0 {CRANK_R} 0"
                size="0.008" mass="{m['blade_upper'] * mass_scale}"{cc0}/>
          <body name="{L}_blade_lower" pos="0 {CRANK_R} 0">
            <!-- passive-loop conditioning: the blade/pushrod bodies are ~50-80 g and the
                 loop coupling is stiff; bearing-scale armature+damping keep the loop
                 well-posed through the toggle's TDC (phi=0). Values are numerical
                 regularization, not bench data. -->
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
                  size="0.010" mass="{m['blade_lower'] * mass_scale}"/>
          </body>
        </body>
        <!-- pushrod: prismatic on the shin (bushing); body origin = heel point at s=0 -->
        <body name="{L}_pushrod" pos="{-swing_x:.6f} -0.025 0">
          <!-- slide range: working stroke is [-41.14, 0] mm (the loop coupling enforces
               the exact relation); +5 mm headroom stops limit-vs-loop chatter at the
               toggle's TDC (s=0 at phi=0) while still blocking the flipped-elbow
               branch of the crank (which would need s=+200 mm). -->
          <joint name="{L}_pushrod_slide" type="slide" axis="0 1 0"
                 range="{slide_lim[0] - 0.002:.5f} 0.005" damping="0.3" armature="2e-4"/>
          <geom name="{L}_rod" type="capsule" fromto="0 0 0 0 {-TIP_LEN + FOOT_R:.5f} 0"
                size="0.008" mass="{m['pushrod'] * 0.8 * mass_scale}"/>
          <geom name="{L}_foot" type="sphere" pos="0 {-TIP_LEN + FOOT_R:.5f} 0"
                size="{FOOT_R}" mass="{m['pushrod'] * 0.2 * mass_scale}"/>
        </body>
      </body>
    </body>'''
    # loop closure via 1-row QUARTIC JOINT COUPLINGS (secret-sauce §8), replacing the
    # 3-row <connect> whose Jacobian went rank-deficient at the toggle dead center and
    # forced dt=0.002. q_slide = poly_s(q_knee), q_toe = poly_psi(q_knee); efc Jacobian
    # [1, -poly'] is full-rank at TDC by construction, and poly'->0 there reproduces the
    # toggle-press force amplification exactly. Fit residuals gated at build (<0.5 mm /
    # <0.3 deg over the ROM, Gaussian TDC weighting). Side benefit: the polynomial is
    # single-valued, so the flipped-elbow branch is unreachable by construction.
    # NEAR-HARD coupling rows (measured, 2026-07-03): solref tc=0.008 = the 2*dt
    # refsafe bound at the fleet dt=0.004, dmax=0.9999 = MuJoCo's own impedance
    # ceiling for ~hard constraints. Soft rows (tc 0.02-0.04, dmax 0.9) could NOT
    # bear load through the 80 g pushrod (force capacity scales with the coupled
    # inertia): the loaded stomp sagged to the +limit and lifted nothing. At
    # 0.008/0.9999: loaded stomp lifts +31 mm (vs +12 mm in the old connect model),
    # slides reach -36 mm under body weight, unloaded sweep tracks the closed form
    # to 0.116 mm (= the fit residual) at dt=0.004, and the free system holds
    # consistency to 1e-13. The full-rank [1, -poly'] Jacobian is what tolerates
    # this stiffness at TDC — the connect could not. If dt ever rises above 0.004,
    # solref must rise with it (tc >= 2*dt).
    cs, cp = POLY_SLIDE, POLY_TOE      # SINGLE source of truth, shared with loop_consistent_pose
    extras = f'''    <joint name="{L}_loop_slide" joint1="{L}_pushrod_slide" joint2="{L}_knee_blade"
           solref="0.008 1" solimp="0.95 0.9999 0.001"
           polycoef="{cs[0]:.17g} {cs[1]:.17g} {cs[2]:.17g} {cs[3]:.17g} {cs[4]:.17g}"/>
    <joint name="{L}_loop_toe" joint1="{L}_toe_hinge" joint2="{L}_knee_blade"
           solref="0.008 1" solimp="0.95 0.9999 0.001"
           polycoef="{cp[0]:.17g} {cp[1]:.17g} {cp[2]:.17g} {cp[3]:.17g} {cp[4]:.17g}"/>'''
    if sea:
        # spring = soft equality row: constraint stiffness k = I_eff/(tc·ζ)² →
        # tc = sqrt(I_eff/k) at ζ=1 (I_eff ≈ 0.35 kg·m² leg-about-yaw + armature)
        tc = math.sqrt(SEA_YAW_EFFECTIVE_INERTIA_EST / SEA_YAW_STIFFNESS_EST)
        extras += f'''
    <joint name="{L}_yaw_sea" joint1="{L}_yaw_rotor" joint2="{L}_hip_yaw"
           polycoef="0 1 0 0 0" solref="{tc:.4f} 1"/>'''
        body += f'''
    <body name="{L}_yaw_rotor_body" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}">
      <joint name="{L}_yaw_rotor" axis="0 1 0" armature="{ARMATURE['hip_yaw']}" damping="0.01"/>
      <geom name="{L}_yaw_rotor_g" type="sphere" size="0.012" mass="{SERVO_MASS}"
            contype="0" conaffinity="0" rgba="0.9 0.6 0.2 0.5"/>
    </body>'''
    excludes = f'''    <exclude body1="{L}_blade_lower" body2="{L}_pushrod"/>
    <exclude body1="{L}_blade_upper" body2="{L}_pushrod"/>'''
    gy, gs, gk = GEARS["hip_yaw"], GEARS["leg_swing"], GEARS["knee_blade"]
    yaw_drive = f"{L}_yaw_rotor" if sea else f"{L}_hip_yaw"
    acts = f'''    <motor name="{L}_yaw_m" joint="{yaw_drive}" gear="{gy:.4g}" ctrlrange="-1 1"
           forcerange="-{gy:.4g} {gy:.4g}"/>
    <motor name="{L}_swing_m" joint="{L}_leg_swing" gear="{gs:.4g}" ctrlrange="-1 1"
           forcerange="-{gs:.4g} {gs:.4g}"/>
    <motor name="{L}_knee_m" joint="{L}_knee_blade" gear="{gk:.4g}" ctrlrange="-1 1"
           forcerange="-{gk:.4g} {gk:.4g}"/>'''
    return body, (extras, excludes, acts)


def build_mesh_robot(asm: dict | None = None, floor: bool = True, sea: bool = False) -> str:
    loop_polycoefs()                     # §8 gates fire here — a bad fit refuses to build
    asm = asm or load_assembly()
    t = asm["torso"]
    c = t["center_world"]
    hx, hy, hz = (s / 2 for s in t["size_xyz_m"])
    raw_structure_mass = (_RAW_TORSO_STRUCTURE_MASS + _RAW_STRIKER_MASS
                          + sum(float(asm["legs"][L]["leg_total_mass_kg"])
                                for L in ("FL", "FR", "RL", "RR")))
    mass_scale = NON_SERVO_MASS_BUDGET_KG / raw_structure_mass
    torso_structure_mass = _RAW_TORSO_STRUCTURE_MASS * mass_scale
    striker_mass = _RAW_STRIKER_MASS * mass_scale
    bodies, connects, excludes, acts = [], [], [], []
    for L in ("FL", "FR", "RL", "RR"):
        b, (cn, ex, ac) = _leg_xml(L, asm["legs"][L], c, sea=sea,
                                   mass_scale=mass_scale)
        bodies.append(b); connects.append(cn); excludes.append(ex); acts.append(ac)
    floor_xml = ('    <geom name="floor" type="plane" size="0 0 0.1" pos="0 0 0" '
                 'rgba="0.4 0.5 0.4 1"/>\n' if floor else "")
    sk = asm["striker_placeholder"]
    return f'''<mujoco model="mesh_robot_7_3">
  <compiler angle="radian" autolimits="true"/>
  <!-- dt=0.004 (fleet standard) RESTORED by the quartic loop couplings: the old
       <connect> Jacobian went rank-deficient at the toggle dead center and needed
       dt=0.002 (measured: 10 m slide error, |qacc| 7e9 at 0.004). The polynomial
       coupling row [1, -poly'] stays full-rank at TDC — see secret-sauce §8.
       Control at 50 Hz via frame_skip=5, matching the paramquad fleet. -->
  <option timestep="0.004" integrator="implicitfast"/>
  <default>
    <joint damping="0.02"/>
    <geom friction="1.2 0.1 0.1" contype="1" conaffinity="1"/>
  </default>
  <worldbody>
{floor_xml}    <body name="torso" pos="{c[0]} {c[1]} {c[2]}">
      <freejoint name="root"/>
      <!-- torso structure preserves the assembly's placeholder mass distribution;
           four yaw servos are lumped here unless represented by SEA rotor bodies -->
      <geom name="torso" type="box" size="{hx:.5f} {hy:.5f} {hz:.5f}"
            mass="{torso_structure_mass + (0 if sea else 4 * SERVO_MASS)}" rgba="0.3 0.3 0.6 1"/>
      <!-- striker: COSMETIC placeholder, non-colliding, no actuator -->
      <geom name="striker_ph" type="cylinder" pos="{hx + sk['length_m'] / 2:.5f} 0 0"
            quat="0.707107 0 0.707107 0" size="{sk['radius_m']} {sk['length_m'] / 2:.5f}"
            mass="{striker_mass}" contype="0" conaffinity="0" rgba="0.8 0.8 0.85 1"/>
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
    _, _, rs, rp = loop_polycoefs()
    print(f"mesh_robot: 4 legs / {SERVO_COUNT}x {_SERVO.label}, "
          f"toggle-press loop via quartic couplings (fit resid {rs * 1e3:.3f} mm / "
          f"{math.degrees(rp):.3f} deg), dt=0.004, "
          f"mass limit {MAX_ROBOT_MASS_LB:.1f} lb ({MAX_ROBOT_MASS_KG:.3f} kg) "
          f"-> {args.out}")
    if args.summary:
        import mujoco
        m = mujoco.MjModel.from_xml_string(xml)
        print(f"  njnt={m.njnt} nv={m.nv} nu={m.nu} neq={m.neq} "
              f"mass={m.body_mass.sum():.3f} kg")


if __name__ == "__main__":
    main()
