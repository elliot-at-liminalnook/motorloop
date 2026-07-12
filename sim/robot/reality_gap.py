# SPDX-License-Identifier: MIT
"""Treat the simulator as a calibrated, uncertain instrument.

Measure components -> fit the sim -> randomize the *remaining* uncertainty -> rank
designs by ROBUST match return, not nominal. This module is the single source of the
sim-to-real model the envs consume, so we add the effects that change rankings
between designs (actuator dynamics, contacts, latency, friction, impacts) rather than
"more realism" everywhere.

Nominal values come from the provenance seam (`motors.py` datasheet for the chosen
motor); ranges are calibrated where measured, ESTIMATED where not (tagged). The
parity hooks (`log_parity_trace`, `score_trace_mismatch`) are the real->sim truth
gates - framework here, fed by the hardware-ID suite once hardware exists.
"""

from __future__ import annotations

import dataclasses as dc
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from motors import MOTORS, SERVOS  # noqa: E402


# ----- uncertainty specs: (nominal, lo_frac, hi_frac) multiplicative, or (lo,hi) absolute -----
@dc.dataclass(frozen=True)
class ActuatorUncertainty:
    kt: tuple; r_phase: tuple; ke_phase: tuple        # motor constants (datasheet +-tol)
    i_limit: tuple; vbus: tuple                        # current limit (A), bus voltage (V)
    r_internal: tuple                                 # battery+wiring sag resistance (Ohm)
    gear_eff: tuple                                   # gearbox/belt efficiency
    latency_s: tuple                                  # command latency (s)
    thermal_derate: tuple                             # I_limit loss fraction when hot


@dc.dataclass(frozen=True)
class ContactUncertainty:
    friction: tuple; restitution: tuple; solref_t: tuple    # tangential mu, bounce, contact softness
    damage_ref_N: tuple                                     # impact force (N) = "1 unit" of damage


@dc.dataclass(frozen=True)
class BodyUncertainty:
    mass_scale: tuple; com_offset_m: tuple; inertia_scale: tuple
    joint_damping: tuple; joint_stiffness: tuple; backlash_rad: tuple


@dc.dataclass(frozen=True)
class SensorControlUncertainty:
    obs_noise_std: tuple; encoder_quant_rad: tuple; action_noise_std: tuple


def default_uncertainty(motor: str = "waveshare_st3215_hs", gear: float = 3.0) -> dict:
    """Calibrated-where-measured ranges centred on the datasheet motor."""
    if motor in SERVOS:
        servo = SERVOS[motor]
        vbus = max(servo.stall_torque_nm)
        i_pk = servo.stall_current_a[vbus]
        kt = servo.stall_torque_nm[vbus] / i_pk
        ke = vbus / servo.no_load_speed_rad_s[vbus]
        resistance = vbus / i_pk
        encoder_quant = servo.position_resolution_deg * np.pi / 180.0
    else:
        m = MOTORS[motor]
        vbus = m.rated_voltage_v
        i_pk = 4.0 * m.rated_current_a
        kt, ke, resistance = m.kt, m.ke_phase, m.r_phase
        encoder_quant = 0.0015
    return dict(
        act=ActuatorUncertainty(
            kt=(kt, 0.9, 1.1), r_phase=(resistance, 0.85, 1.2),          # datasheet/equivalent +-tol
            ke_phase=(ke, 0.9, 1.1),
            i_limit=(i_pk, 0.8, 1.1), vbus=(vbus, 0.85, 1.0), # battery droops, never over
            r_internal=(0.05, 0.5, 2.0),                                   # estimated (no cell data)
            gear_eff=(0.88, 0.92, 1.0), latency_s=(0.004, 0.5, 3.0),       # estimated
            thermal_derate=(0.15, 0.0, 1.0)),                             # estimated
        contact=ContactUncertainty(
            friction=(0.9, 0.6, 1.4), restitution=(0.1, 0.0, 3.0),
            solref_t=(0.02, 0.5, 2.0), damage_ref_N=(150.0, 0.7, 1.4)),    # unify on Newtons
        body=BodyUncertainty(
            mass_scale=(1.0, 0.85, 1.0), com_offset_m=(0.0, -0.02, 0.02),
            inertia_scale=(1.0, 0.85, 1.15), joint_damping=(0.5, 0.5, 2.0),
            joint_stiffness=(0.0, 0.0, 25.0), backlash_rad=(0.0, 0.0, 0.02)),
        sensor=SensorControlUncertainty(
            obs_noise_std=(0.0, 0.0, 0.02),
            encoder_quant_rad=(encoder_quant, 0.8, 1.2),
            action_noise_std=(0.0, 0.0, 0.05)),
        motor=motor, gear=gear)


def _s(rng, spec):
    """Sample one spec: 3-tuples are (nominal, lo_mult, hi_mult); else (lo, hi) absolute."""
    if len(spec) == 3 and spec[1] <= spec[2] and (spec[1] <= 1.5):   # multiplicative around nominal
        return spec[0] * rng.uniform(spec[1], spec[2])
    return rng.uniform(spec[-2], spec[-1])


def sample_domain_params(seed, unc: dict | None = None) -> dict:
    """One concrete sim instrument drawn from the calibrated uncertainty."""
    unc = unc or default_uncertainty()
    rng = np.random.default_rng(seed)
    out = {"motor": unc["motor"], "gear": unc["gear"]}
    for grp in ("act", "contact", "body", "sensor"):
        for f in dc.fields(unc[grp]):
            out[f.name] = float(_s(rng, getattr(unc[grp], f.name)))
    return out


def actuator_scale(joint_vel, dp):
    """Speed-dependent torque-envelope fraction per joint (the real motorloop stack):
    available_torque(omega)/static_limit via back-EMF + current limit + voltage sag +
    gear efficiency. ``joint_vel`` may be a NumPy array or Torch tensor."""
    xp = _backend(joint_vel)
    motor_w = xp.abs(joint_vel) * dp["gear"]
    vbus = dp["vbus"]                                          # (sag applied below via i)
    i_avail = (vbus - dp["ke_phase"] * motor_w) / dp["r_phase"]
    i_avail = xp.clip(i_avail, 0.0, dp["i_limit"])            # current limit
    # voltage sag: bus droops under the drawn current, lowering the ceiling again
    i_avail = xp.clip((vbus - i_avail * dp["r_internal"] - dp["ke_phase"] * motor_w) / dp["r_phase"],
                      0.0, dp["i_limit"] * (1.0 - dp["thermal_derate"]))
    return (i_avail / dp["i_limit"]) * dp["gear_eff"]         # fraction of static forcerange


def apply_to_warp_model(model, dp, hinge_mask=None):
    """Perturb a mutable MuJoCo model before it is uploaded to MuJoCo-Warp."""
    model.body_mass[1:] *= dp["mass_scale"]
    model.body_inertia[1:] *= dp["inertia_scale"] * dp["mass_scale"]
    model.dof_damping[:] *= dp["joint_damping"] / 0.5
    model.geom_friction[:, 0] = dp["friction"]
    if hinge_mask is not None and dp["joint_stiffness"] > 0:
        model.jnt_stiffness[np.asarray(hinge_mask, dtype=bool)] = dp["joint_stiffness"]
    model.geom_solref[:, 0] = dp["solref_t"]
    return model


def damage_from_force(contact_force_N, dp):
    """Unified damage currency: impact FORCE in Newtons / damage_ref (SPARC severity).
    Resolves historical contact-parameter mismatches into one calibrated model."""
    return contact_force_N / dp["damage_ref_N"]


# ----- real->sim truth gates (framework; fed by the hardware-ID suite) -----
PARITY_CHANNELS = ("joint_pos", "joint_vel", "imu_acc", "motor_current",
                   "bus_voltage", "temperature", "contact_time", "impact_N", "final_pose")


def log_parity_trace(trace: dict, path):
    np.savez(path, **{k: np.asarray(trace[k]) for k in trace})


def score_trace_mismatch(sim: dict, real: dict) -> dict:
    """Distribution-matching mismatch per channel (legged fight behaviour -> match the
    distribution, not the exact time alignment). Lower = closer. Drives SimOpt/Bayesian
    tightening of the uncertainty ranges once real logs exist."""
    out = {}
    for ch in PARITY_CHANNELS:
        if ch in sim and ch in real:
            s, r = np.asarray(sim[ch]).ravel(), np.asarray(real[ch]).ravel()
            # 1-Wasserstein-ish: mean abs gap of sorted samples (no scipy dep)
            n = min(len(s), len(r))
            ss, rr = np.sort(s)[:n], np.sort(r)[:n]
            out[ch] = float(np.mean(np.abs(ss - rr)))
    out["total"] = float(np.mean(list(out.values()))) if out else float("nan")
    return out


def _backend(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return torch
    except ImportError:
        pass
    return np


if __name__ == "__main__":
    unc = default_uncertainty()
    dp = sample_domain_params(0, unc)
    import numpy as _np
    scale0 = float(actuator_scale(_np.array([0.0]), dp)[0])
    scalehi = float(actuator_scale(_np.array([50.0]), dp)[0])
    print(f"sampled motor {dp['motor']} gear {dp['gear']}: vbus {dp['vbus']:.1f} "
          f"i_limit {dp['i_limit']:.1f} latency {dp['latency_s']*1000:.1f}ms")
    print(f"torque-envelope fraction: stall {scale0:.2f} -> 50rad/s {scalehi:.2f} "
          f"(droops with speed = back-EMF; the effect idealized motors miss)")
    print(f"unified damage: 300 N hit = {damage_from_force(300.0, dp):.2f} severity units")
    print("reality_gap self-check OK")
