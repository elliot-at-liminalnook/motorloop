# SPDX-License-Identifier: MIT
"""synth_reference_gait.py — level-leg static crawl for the servo-true mesh robot;
writes sim/robot/reference_gait.json.

v2 (2026-07-03). v1 (deep pitch-stride bands, fore-aft-only COM sinusoid) FAILED
its open-loop proof: the planned hip-to-foot heights varied 31-42 mm through the
stance bands (the foot rides a wheel-leg arc, dz/dtheta up to 3.7 mm/deg), so the
rigid torso rocked +-10 deg, feet dropped out of contact one at a time, and the
robot fell sideways during startup (instrumented). v2 designs the rock out:

  * LEVEL-LEG STANCE — all four legs work the shallow branch theta in
    [+11, +18] deg where feet spread WIDE (front feet ~+0.16 m, rear ~-0.16 m:
    a 0.32 x 0.16 m support rectangle, no leg interference). The knee is
    SCHEDULED so hip-to-foot depth stays ~= Z_REF: extended (~-68 deg, weak
    toggle) only at the short-leg band end where the phase structure keeps that
    foot lightly loaded, retracted (strong, near-TDC) at the long end. Load
    peaks arrive mid-stance at knee ~-38 deg (69 N capacity >> 35 N worst).
  * TWO-AXIS COM SHIFTS — the support rectangle's diagonals pass through the
    COM, so every corner lift needs an active shift. Fore-aft = common-mode
    pitch; LATERAL = coordinated stance yaw (foot dy/dpsi = x_off = 50-79 mm/rad,
    validated exact against FK). Per-slot (dx, dy) values are optimized against
    a quasi-static evaluator (FK COM, support-polygon margin, min-norm contact
    loads, knee capacity, clearance) and ramped during the 4-stance windows.
  * CLEARANCE — front legs lift along their own swing arc (foot z rises 27 mm
    traversing +11 -> +18 deg); rear legs lift by knee retraction (~16 mm
    available at their -64 deg lift-end knee). 8.6 mm razor margins are gone.
  * LEADS — P-only servo (kp = 2/40/6) targets carry per-phase torque leads
    (worm frictionloss 4 N.m + load x dz/dtheta)/kp, smoothed to avoid steps.
  * SOFT START — emits "aux_mod" [n][12]: a stride-less weight-shift sway cycle
    (minus the prover's base pose) so the COM schedule is FULLY ACTIVE while
    the prover ramps stride/lift amplitude in (generalizes v1's aux_delta_x).

CLI: .venv-warp/bin/python synth_reference_gait.py [--out reference_gait.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gen_mesh_robot_mjcf import (build_mesh_robot, loop_consistent_pose,  # noqa: E402
                                 slider_crank_s, GEARS, WFREE, WORM_FRICTIONLOSS)

LEGS = ("FL", "FR", "RL", "RR")
JOINT_ORDER = [f"{L}_{j}" for L in LEGS for j in ("hip_yaw", "leg_swing", "knee_blade")]
FRONT = {"FL": True, "FR": True, "RL": False, "RR": False}

# ------------------------------------------------------------------ parameters
PERIOD_S = 8.0        # slow crawl: pitch/yaw rates land ~10x under the WFREE caps
N = 160
# swing slots (fraction of cycle), diagonal-sequence order FR, RL, FL, RR;
# the 0.10 gaps between slots are 4-stance weight-transfer windows.
SLOTS = {"FR": (0.10, 0.25), "RL": (0.35, 0.50), "FL": (0.60, 0.75), "RR": (0.85, 1.00)}
SLOT_SEQ = ("FR", "RL", "FL", "RR")
LIFT_TH = math.radians(11.0)    # local pitch band (shallow branch, feet spread)
PLANT_TH = math.radians(18.0)
Z_REF = -0.226        # hip-to-foot depth the knee schedule holds (m, rel mount)
KNEE_S_LO = math.radians(-72.0)   # deepest scheduled stance knee (cap 22 N)
KNEE_S_HI = math.radians(-3.0)    # shallowest (near-TDC, cap ~1 kN)
KNEE_SWING = math.radians(2.0)    # swing retraction target (TDC)
SWING_TRAV = (0.10, 0.90)     # xy traverse window within the slot
KNEE_RETRACT = 0.15           # knee retract complete by this slot fraction
KNEE_EXTEND = 0.82            # knee re-extend starts here
FLAG_WIN = (0.20, 0.90)       # feet_swing window reported to the prover
RAMP_LEN = 0.080              # shift-transfer ramp length (ends 0.005 before slot)
KP = np.array([2.0, 40.0, 6.0] * 4)
PSI_MAX = 0.76                # yaw usage cap (range 0.785)
TH_MAX = math.radians(24.0)
SHIFT_GRID = 0.032            # per-slot |dx|,|dy| search bound (m)
MASS_G = None                 # filled from model (W in N)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ------------------------------------------------------------------- probing
class Robot:
    """Probed kinematics: per-leg x_off(theta), z50(theta) (world, rel mount,
    knee=-50 ref), mounts; knee s(kappa) tables. foot = mount +
    Rz(psi) @ (x_off, 0, z) — validated exact against FK (self-check below)."""

    KNEE_REF = math.radians(-50.0)

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_string(build_mesh_robot())
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        self.jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
                    for n in [f"{L}_{j}" for L in LEGS for j in
                              ("hip_yaw", "leg_swing", "knee_blade",
                               "toe_hinge", "pushrod_slide")]}
        self.gid_foot = {L: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{L}_foot")
                         for L in LEGS}
        self.bid_mount = {L: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                               f"{L}_base_housing") for L in LEGS}
        self.torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.qadr = {n: int(m.jnt_qposadr[j]) for n, j in self.jid.items()}
        # mounts (torso frame)
        self._zero()
        mujoco.mj_forward(m, d)
        self.mount = {L: (d.xpos[self.bid_mount[L]] - d.xpos[self.torso])[:2].copy()
                      for L in LEGS}
        # per-leg theta maps at knee ref, yaw 0
        ths = np.linspace(math.radians(-26.0), math.radians(22.0), 241)
        self.ths = ths
        self.xoff, self.z50 = {}, {}
        psi_t, s_t = loop_consistent_pose(self.KNEE_REF)
        for L in LEGS:
            xo, zz = [], []
            for th in ths:
                self._zero()
                for LL in LEGS:
                    d.qpos[self.qadr[f"{LL}_knee_blade"]] = self.KNEE_REF
                    d.qpos[self.qadr[f"{LL}_toe_hinge"]] = psi_t
                    d.qpos[self.qadr[f"{LL}_pushrod_slide"]] = s_t
                d.qpos[self.qadr[f"{L}_leg_swing"]] = th
                mujoco.mj_forward(m, d)
                rel = d.geom_xpos[self.gid_foot[L]] - d.xpos[self.bid_mount[L]]
                xo.append(rel[0]); zz.append(rel[2])
            self.xoff[L] = np.array(xo)      # world x_off: monotonic in theta
            self.z50[L] = np.array(zz)
        # knee tables (monotonic increasing s on [-90, -3])
        ks = np.linspace(math.radians(-90.0), math.radians(-3.0), 175)
        self.k_grid = ks
        self.s_grid = np.array([slider_crank_s(k) for k in ks])
        self.s_ref = slider_crank_s(self.KNEE_REF)

    def _zero(self):
        self.d.qpos[:] = 0.0
        self.d.qpos[3] = 1.0

    def th_of_xoff(self, L, xo):
        xs = self.xoff[L]
        if xs[-1] > xs[0]:
            return float(np.interp(xo, xs, self.ths))
        return float(np.interp(xo, xs[::-1], self.ths[::-1]))

    def xoff_of(self, L, th):
        return float(np.interp(th, self.ths, self.xoff[L]))

    def z50_of(self, L, th):
        return float(np.interp(th, self.ths, self.z50[L]))

    def dz_dth(self, L, th):
        return float((self.z50_of(L, th + 5e-3) - self.z50_of(L, th - 5e-3)) / 1e-2)

    def k_of_s(self, s):
        return float(np.interp(s, self.s_grid, self.k_grid))

    def ds_dk(self, k):
        return float((slider_crank_s(k + 5e-4) - slider_crank_s(k - 5e-4)) / 1e-3)

    def knee_cap(self, k):
        d = abs(self.ds_dk(k))
        return GEARS["knee_blade"] / d if d > 1e-6 else 1e6

    def ik(self, L, fx, fy):
        """foot target (torso frame, xy) -> (theta, psi). Uses the leg's own
        signed x_off branch; exact for this mechanism (yaw axis vertical)."""
        dxm = fx - self.mount[L][0]
        dym = fy - self.mount[L][1]
        r = math.hypot(dxm, dym)
        sx = 1.0 if FRONT[L] else -1.0        # front legs: x_off > 0 branch here
        th = self.th_of_xoff(L, sx * r)
        psi = math.atan2(sx * dym, sx * dxm)
        return th, psi

    def fk_plan(self, q12, knee_override=None):
        """Set 12 actuated joints (+ loop-consistent passives), torso identity;
        return feet (4,3) and com (3,) in the torso frame."""
        d = self.d
        self._zero()
        for li, L in enumerate(LEGS):
            k = q12[3 * li + 2] if knee_override is None else knee_override[li]
            psi_t, s_t = loop_consistent_pose(min(k, math.radians(9.0)))
            d.qpos[self.qadr[f"{L}_hip_yaw"]] = q12[3 * li + 0]
            d.qpos[self.qadr[f"{L}_leg_swing"]] = q12[3 * li + 1]
            d.qpos[self.qadr[f"{L}_knee_blade"]] = k
            d.qpos[self.qadr[f"{L}_toe_hinge"]] = psi_t
            d.qpos[self.qadr[f"{L}_pushrod_slide"]] = s_t
        mujoco.mj_forward(self.m, d)
        feet = np.array([d.geom_xpos[self.gid_foot[L]] - d.xpos[self.torso]
                         for L in LEGS])
        com = (d.subtree_com[self.torso] - d.xpos[self.torso]).copy()
        return feet, com


# ------------------------------------------------------------------ gait plan
class Plan:
    """Foot-space crawl: feet march -x rel torso through stance (stride), swing
    slots re-plant them; a per-slot COM-shift schedule (dx, dy) moves the body
    inside each support triangle. stride=0 + no swing => the aux sway cycle."""

    def __init__(self, rb: Robot, shifts=None, stride=None, sway_only=False):
        self.rb = rb
        self.sway = sway_only
        x_lo = rb.xoff_of("FL", LIFT_TH)
        x_hi = rb.xoff_of("FL", PLANT_TH)
        self.stride = 0.0 if sway_only else (x_hi - x_lo if stride is None else stride)
        x_mid = 0.5 * (x_lo + x_hi)
        self.home = {}
        for L in LEGS:
            sx = 1.0 if FRONT[L] else -1.0
            self.home[L] = np.array([rb.mount[L][0] + sx * x_mid, rb.mount[L][1]])
        # per-slot shift values, order SLOT_SEQ
        self.shifts = np.zeros((4, 2)) if shifts is None else np.asarray(shifts, float)

    def shift_at(self, phi):
        """(dx, dy) COM shift: hold per-slot values, cosine ramps in the gaps."""
        phi = phi % 1.0
        out = None
        for j, Lj in enumerate(SLOT_SEQ):
            s0 = SLOTS[Lj][0]
            r0, r1 = s0 - RAMP_LEN - 0.005, s0 - 0.005
            prev = self.shifts[(j - 1) % 4]
            if phi < r0 % 1.0 and j == 0:
                out = self.shifts[3]          # before FR's ramp: RR values held
            if r0 <= phi < r1:
                t = smoothstep((phi - r0) / RAMP_LEN)
                out = prev + (self.shifts[j] - prev) * t
            elif r1 <= phi < SLOTS[Lj][1]:
                out = self.shifts[j].copy()
        if out is None:                       # in a gap after a slot, value holds
            for j, Lj in enumerate(SLOT_SEQ):
                s1 = SLOTS[Lj][1]
                nxt0 = SLOTS[SLOT_SEQ[(j + 1) % 4]][0] - RAMP_LEN - 0.005
                if s1 <= phi < nxt0 or (j == 3 and (phi >= s1 % 1.0 or phi < nxt0)):
                    out = self.shifts[j].copy()
        return out if out is not None else self.shifts[3].copy()

    def in_swing(self, L, phi):
        s0, s1 = SLOTS[L]
        x = (phi % 1.0 - s0) % 1.0
        if x < s1 - s0:
            return x / (s1 - s0)
        return None

    def foot_xy(self, L, phi):
        """Planned foot target rel torso + swing fraction (None if stance)."""
        dx, dy = self.shift_at(phi)
        hx, hy = self.home[L]
        s0, s1 = SLOTS[L]
        xi = None if self.sway else self.in_swing(L, phi)
        if xi is None:
            slot_len = s1 - s0
            zeta = ((phi % 1.0 - s1) % 1.0) / (1.0 - slot_len)
            fx = hx + (0.5 - zeta) * self.stride - dx
            return np.array([fx, hy - dy]), None
        dj = self.shifts[SLOT_SEQ.index(L)]
        a = np.array([hx - 0.5 * self.stride - dj[0], hy - dj[1]])
        b = np.array([hx + 0.5 * self.stride - dj[0], hy - dj[1]])
        t = smoothstep((xi - SWING_TRAV[0]) / (SWING_TRAV[1] - SWING_TRAV[0]))
        return a + (b - a) * t, xi

    def knee_stance(self, L, th):
        """Schedule kappa so hip-to-foot depth == Z_REF (clipped to capacity)."""
        s_t = (Z_REF - self.rb.z50_of(L, th)) + self.rb.s_ref
        s_t = min(max(s_t, slider_crank_s(KNEE_S_LO)), slider_crank_s(KNEE_S_HI))
        return self.rb.k_of_s(s_t)

    def at(self, phi):
        """-> q_plan (12), swing flags (4)  [unleaded plan]"""
        q = np.zeros(12)
        flags = np.zeros(4, dtype=bool)
        for li, L in enumerate(LEGS):
            (fx, fy), xi = self.foot_xy(L, phi)
            th, psi = self.rb.ik(L, fx, fy)
            if xi is None:
                k = self.knee_stance(L, th)
            else:
                k_lift = self.knee_stance(L, self.rb.ik(L, *self.foot_xy(
                    L, SLOTS[L][0] - 1e-6)[0])[1] if False else th)  # placeholder
                # knee: retract from the lift-end stance value to KNEE_SWING,
                # hold, re-extend to the plant-end stance value
                (fxa, fya), _ = self.foot_xy(L, SLOTS[L][0] - 1e-6)
                (fxb, fyb), _ = self.foot_xy(L, SLOTS[L][1] + 1e-6)
                k_a = self.knee_stance(L, self.rb.ik(L, fxa, fya)[0])
                k_b = self.knee_stance(L, self.rb.ik(L, fxb, fyb)[0])
                if xi < KNEE_RETRACT:
                    k = k_a + (KNEE_SWING - k_a) * smoothstep(xi / KNEE_RETRACT)
                elif xi > KNEE_EXTEND:
                    k = KNEE_SWING + (k_b - KNEE_SWING) * smoothstep(
                        (xi - KNEE_EXTEND) / (1.0 - KNEE_EXTEND))
                else:
                    k = KNEE_SWING
                flags[li] = FLAG_WIN[0] <= xi <= FLAG_WIN[1]
            q[3 * li + 0] = psi
            q[3 * li + 1] = th
            q[3 * li + 2] = k
        return q, flags


# ----------------------------------------------------------------- evaluator
def hull_margin(pts, p):
    """Signed distance of p to the convex hull of pts (>0 inside)."""
    pts = np.asarray(pts)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    dmin = 1e9
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        e = b - a
        L = math.hypot(*e)
        if L < 1e-9:
            continue
        d = (e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])) / L
        dmin = min(dmin, d)
    return dmin


def contact_loads(feet_xy, com_xy, W):
    """Min-norm static normal forces (>=0 attempted) for the stance set."""
    k = len(feet_xy)
    A = np.vstack([np.ones(k), feet_xy[:, 0], feet_xy[:, 1]])
    b = np.array([W, W * com_xy[0], W * com_xy[1]])
    try:
        F = A.T @ np.linalg.solve(A @ A.T, b)
    except np.linalg.LinAlgError:
        return np.full(k, W / k)
    for _ in range(2):
        neg = F < -1e-9
        if not neg.any() or (~neg).sum() < 3:
            break
        idx = np.where(~neg)[0]
        Ai = A[:, idx]
        try:
            Fi = Ai.T @ np.linalg.solve(Ai @ Ai.T, b)
        except np.linalg.LinAlgError:
            break
        F = np.zeros(k)
        F[idx] = Fi
    return F


def evaluate(rb: Robot, plan: Plan, phis):
    """Quasi-static per-phase metrics on the UNLEADED plan."""
    W = MASS_G
    out = {"phi": [], "margin": [], "loads": [], "clear": [], "tilt": [],
           "psi_max": 0.0, "th_rng": (1e9, -1e9), "cap_util": [], "swing_leg": []}
    for phi in phis:
        q, flags = plan.at(phi)
        feet, com = rb.fk_plan(q)
        st = ~flags if not plan.sway else np.ones(4, bool)
        fxy = feet[st][:, :2]
        # LS ground plane through stance feet -> attitude-corrected COM
        Amat = np.hstack([fxy, np.ones((len(fxy), 1))])
        coef, *_ = np.linalg.lstsq(Amat, feet[st][:, 2], rcond=None)
        a, b_, c = coef
        com_h = com[2] - (a * com[0] + b_ * com[1] + c)  # COM height above plane
        com_xy = com[:2] + np.array([a, b_]) * com_h     # tilt-induced shift
        mg = hull_margin(fxy, com_xy)
        F = contact_loads(fxy, com_xy, W)
        loads = np.zeros(4)
        loads[st] = np.maximum(F, 0.0)
        caps = np.array([rb.knee_cap(q[3 * i + 2]) for i in range(4)])
        util = np.where(st, loads / caps, 0.0)
        clear = 1e9
        if flags.any():
            i = int(np.where(flags)[0][0])
            clear = feet[i, 2] - (a * feet[i, 0] + b_ * feet[i, 1] + c)
            out["swing_leg"].append((phi, i, clear))
        out["phi"].append(phi); out["margin"].append(mg)
        out["loads"].append(loads); out["clear"].append(clear)
        out["tilt"].append(math.hypot(a, b_)); out["cap_util"].append(util.max())
        psis = np.abs(q[0::3]).max()
        out["psi_max"] = max(out["psi_max"], psis)
        out["th_rng"] = (min(out["th_rng"][0], q[1::3].min()),
                         max(out["th_rng"][1], q[1::3].max()))
    for k in ("margin", "loads", "clear", "tilt", "cap_util"):
        out[k] = np.asarray(out[k])
    return out


def slot_score(rb, plan, j, phis_j):
    """min margin over the slot minus soft penalties (caps, yaw/pitch range)."""
    ev = evaluate(rb, plan, phis_j)
    pen = 0.0
    if ev["psi_max"] > PSI_MAX:
        pen += 10.0 * (ev["psi_max"] - PSI_MAX)
    lo, hi = ev["th_rng"]
    if hi > TH_MAX:
        pen += 10.0 * (hi - TH_MAX)
    if lo < -TH_MAX:
        pen += 10.0 * (-TH_MAX - lo)
    over = np.maximum(ev["cap_util"] - 0.92, 0.0)
    pen += 0.15 * over.sum()
    return float(ev["margin"].min()) - pen


def optimize_shifts(rb, plan):
    """Per-slot (dx, dy) grid search (margins in a slot depend only on that
    slot's values — shifts are held during slots); 2 refinement passes."""
    for _ in range(2):
        for j, Lj in enumerate(SLOT_SEQ):
            s0, s1 = SLOTS[Lj]
            phis_j = np.linspace(s0 + 0.01, s1 - 0.01, 7)
            # include the preceding transfer end + following gap (loads there)
            phis_j = np.concatenate([phis_j, [s1 + 0.04 if s1 + 0.04 < 1 else s1 + 0.04 - 1]])
            best, best_v = plan.shifts[j].copy(), -1e9
            for dx in np.linspace(-SHIFT_GRID, SHIFT_GRID, 9):
                for dy in np.linspace(-SHIFT_GRID, SHIFT_GRID, 9):
                    plan.shifts[j] = (dx, dy)
                    v = slot_score(rb, plan, j, phis_j)
                    if v > best_v:
                        best_v, best = v, np.array([dx, dy])
            # local refine
            for ddx in np.linspace(-0.004, 0.004, 3):
                for ddy in np.linspace(-0.004, 0.004, 3):
                    plan.shifts[j] = best + (ddx, ddy)
                    v = slot_score(rb, plan, j, phis_j)
                    if v > best_v:
                        best_v, best = v, best + (ddx, ddy)
            plan.shifts[j] = best
    return plan


# ------------------------------------------------------------------- targets
def build_targets(rb, plan, phis, loads_per_phase):
    """plan -> P-servo targets with per-phase torque leads (smoothed)."""
    n = len(phis)
    qt = np.zeros((n, 12))
    plans = np.zeros((n, 12))
    flags = np.zeros((n, 4), dtype=bool)
    for i, phi in enumerate(phis):
        plans[i], flags[i] = plan.at(phi)
    dphi = 1.0 / n
    dq = (np.roll(plans, -1, axis=0) - np.roll(plans, 1, axis=0)) / (2 * dphi * PERIOD_S)
    for i in range(n):
        q = plans[i]
        for li in range(4):
            F = loads_per_phase[i][li]
            th, k = q[3 * li + 1], q[3 * li + 2]
            thd = dq[i, 3 * li + 1]
            # pitch: hold against contact (F*dz/dth) + worm friction breakaway
            tau_p = -F * rb.dz_dth(LEGS[li], th)
            if abs(thd) > 5e-3:
                tau_p -= WORM_FRICTIONLOSS * math.copysign(1.0, thd) * (-1.0)
                # friction opposes motion: servo adds torque in motion direction
                tau_p = -F * rb.dz_dth(LEGS[li], th) + WORM_FRICTIONLOSS * math.copysign(1.0, thd)
            qt[i, 3 * li + 0] = q[3 * li + 0]
            qt[i, 3 * li + 1] = th + tau_p / KP[1]
            tau_k = -F * rb.ds_dk(k)
            qt[i, 3 * li + 2] = k + tau_k / KP[2]
    # smooth the lead-carrying columns (circular, ~0.03 cycle) to avoid steps
    kern = np.array([1.0, 2.0, 3.0, 2.0, 1.0]); kern /= kern.sum()
    for c in range(12):
        if c % 3 == 0:
            continue
        col = qt[:, c]
        sm = np.zeros_like(col)
        for o, w in zip(range(-2, 3), kern):
            sm += w * np.roll(col, o)
        qt[:, c] = sm
    qt[:, 1::3] = np.clip(qt[:, 1::3], -math.radians(25) + 1e-3, math.radians(25) - 1e-3)
    qt[:, 2::3] = np.clip(qt[:, 2::3], -math.pi / 2 + 1e-3, math.radians(9.5))
    return qt, flags, plans


def main():
    global MASS_G
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "reference_gait.json"))
    args = ap.parse_args()

    rb = Robot()
    MASS_G = float(rb.m.body_mass.sum() * 9.81)
    # IK/FK round-trip self-check (guards the planar-yaw kinematic model)
    for L in LEGS:
        tgt = rb.mount[L] + np.array([0.065 if FRONT[L] else -0.065, 0.02])
        th, psi = rb.ik(L, *tgt)
        q = np.zeros(12)
        li = LEGS.index(L)
        q[3 * li], q[3 * li + 1], q[3 * li + 2] = psi, th, rb.KNEE_REF
        feet, _ = rb.fk_plan(q)
        err = np.linalg.norm(feet[li][:2] - tgt)
        assert err < 3e-3, f"IK self-check failed for {L}: {err * 1e3:.1f} mm"

    plan = Plan(rb)
    # seed shifts toward each slot's support-triangle centroid direction
    seed = {"FR": (-0.02, 0.02), "RL": (0.02, -0.02),
            "FL": (-0.02, -0.02), "RR": (0.02, 0.02)}
    plan.shifts = np.array([seed[L] for L in SLOT_SEQ])
    plan = optimize_shifts(rb, plan)

    phis = (np.arange(N) + 0.5) / N
    ev = evaluate(rb, plan, phis)
    loads = ev["loads"]
    qt, flags, plans = build_targets(rb, plan, phis, loads)

    # aux sway cycle (stride/lift suppressed, shifts identical)
    sway = Plan(rb, shifts=plan.shifts, sway_only=True)
    ev_sway = evaluate(rb, sway, phis)
    qt0, _, _ = build_targets(rb, sway, phis, ev_sway["loads"])
    base = qt.mean(axis=0)
    kc = [2, 5, 8, 11]
    base[kc] = qt[:, kc].min(axis=0)
    aux_mod = qt0 - base

    # velocities for reference-state initialization (central difference)
    dt = PERIOD_S / N
    qv = (np.roll(qt, -1, axis=0) - np.roll(qt, 1, axis=0)) / (2 * dt)

    stride = plan.stride
    v_exp = stride / PERIOD_S
    worst = float(ev["margin"].min())
    sw = ev["swing_leg"]
    min_clear = {L: min([c for (_, i, c) in sw if i == li], default=float("nan"))
                 for li, L in enumerate(LEGS)}
    prov = (
        "level-leg static crawl v2 (synth_reference_gait.py, 2026-07-03): order "
        f"FR,RL,FL,RR, duty 0.85, period {PERIOD_S} s; shallow pitch band "
        f"({math.degrees(LIFT_TH):.0f},{math.degrees(PLANT_TH):.0f}) deg both axles "
        "(feet spread ~0.32 m); knee SCHEDULED to hold hip-to-foot depth "
        f"{-Z_REF * 1e3:.0f} mm (extended-when-light, near-TDC at load peaks); "
        "COM shifts: common-mode pitch (fore-aft) + coordinated stance yaw "
        f"(lateral), per-slot values {np.round(plan.shifts * 1e3, 0).tolist()} mm "
        f"optimized on a quasi-static attitude-aware margin model (worst planned "
        f"margin {worst * 1e3:.1f} mm); front swing clears via the pitch arc, rear "
        "via knee retraction; targets carry per-phase torque leads for the P-only "
        f"kp=(2,40,6) servo model with worm frictionloss {WORM_FRICTIONLOSS} N.m "
        f"and torque-speed derating (stall {GEARS['leg_swing']:.1f} N.m, wfree "
        f"{WFREE['leg_swing']:.2f} rad/s); kinematic stride {stride * 1e3:.1f} mm/cycle; "
        "aux_mod = stride-less sway cycle for the prover's soft start; validated "
        "open-loop by prove_reference_gait.py"
    )
    out = {
        "period_s": PERIOD_S,
        "n": N,
        "joint_order": JOINT_ORDER,
        "qpos_targets": [[round(float(v), 6) for v in row] for row in qt],
        "qvel_targets": [[round(float(v), 6) for v in row] for row in qv],
        "feet_swing": [[bool(b) for b in row] for row in flags],
        "expected_speed_ms": round(v_exp, 4),
        "gait_class": "static_crawl",
        "provenance": prov,
        "aux_mod": [[round(float(v), 6) for v in row] for row in aux_mod],
    }
    Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"shifts (mm): " + ", ".join(
        f"{L}({plan.shifts[j][0] * 1e3:+.0f},{plan.shifts[j][1] * 1e3:+.0f})"
        for j, L in enumerate(SLOT_SEQ)))
    print(f"planned worst static margin {worst * 1e3:.1f} mm; "
          f"margin p05 {np.percentile(ev['margin'], 5) * 1e3:.1f} mm")
    print(f"loads: max {loads.max():.1f} N; knee cap util max {ev['cap_util'].max():.2f}")
    print(f"swing clearance min per leg (mm): " +
          ", ".join(f"{L} {min_clear[L] * 1e3:.1f}" for L in LEGS))
    print(f"tilt max {ev['tilt'].max():.3f} rad; psi max {ev['psi_max']:.2f} rad; "
          f"theta range [{math.degrees(ev['th_rng'][0]):.1f}, "
          f"{math.degrees(ev['th_rng'][1]):.1f}] deg")
    print(f"stride {stride * 1e3:.1f} mm/cycle -> {v_exp * 1e3:.1f} mm/s expected")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
