# SPDX-License-Identifier: MIT
"""Motor profiles for the motor-selection study (notes/motor-selection-checklist.md).

Three concrete, available motors as provenance-tagged profiles. Each stores the
datasheet MEASURABLES (line-to-line R/L, peak line-to-line Ke, pole_count, rotor
inertia) + the bench-only extras (B, trapezoid_blend, align_offset). The
measurable -> per-phase conversions reuse the same relations the global
derive_params uses (R=Rll/2, L=Lll/2, Ke=Ke_ll/sqrt3, Kt=Ke, pp=poles/2), so a
profile is consistent with the bench's existing motor derivation.

Provenance is honest per field: `datasheet` where the part publishes it,
`assumed`/`estimate` where it must be measured on the bench (the gimbal's L/J,
all align offsets) - these stay blocked on Q1 until a motor-ID session.

The plant pole_pairs and the RTL `POLE_PAIRS` (gen_rtl_params) must match, so a
motor with a different pole count needs a regen + re-Verilate (build_motor.sh).
The DB42 (4 pp) matches the current build; the GM2804 (7) and EC 45 (8) do not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Motor:
    name: str
    label: str
    # Datasheet measurables (line-to-line / terminal).
    r_line_line: float          # Ohm
    l_line_line: float          # H
    ke_line_line_peak: float    # V*s/rad (peak line-to-line BEMF constant)
    pole_count: int             # magnet poles (pole_pairs = pole_count/2)
    inertia_kg_m2: float        # rotor (+nothing) inertia
    damping: float              # viscous B
    trapezoid_blend: float      # 0 = sinusoidal (the FOC assumption)
    rated_current_a: float      # datasheet rated phase current
    rated_voltage_v: float      # datasheet rated voltage
    price_usd: float
    provenance: str             # per-motor honest summary

    # --- measurable -> per-phase (the derive_params relations) ---
    @property
    def r_phase(self) -> float:
        return self.r_line_line / 2.0

    @property
    def l_phase(self) -> float:
        return self.l_line_line / 2.0

    @property
    def ke_phase(self) -> float:
        return self.ke_line_line_peak / math.sqrt(3.0)

    @property
    def kt(self) -> float:            # N*m/A, = Ke by energy consistency
        return self.ke_phase

    @property
    def pole_pairs(self) -> int:
        return self.pole_count // 2

    # --- analytical comparison metrics (closed-form, no bench run) ---
    @property
    def elec_tau_s(self) -> float:   # electrical time constant L/R
        return self.l_line_line / self.r_line_line

    def no_load_speed_rad_s(self, vbus: float) -> float:
        return vbus / self.ke_phase

    def stall_torque_nm(self, vbus: float, i_max: float | None = None) -> float:
        i = vbus / self.r_line_line
        if i_max is not None:
            i = min(i, i_max)
        return self.kt * i

    def accel_rad_s2(self, current_a: float) -> float:
        return self.kt * current_a / self.inertia_kg_m2

    def efficiency(self, current_a: float, omega_rad_s: float) -> float:
        p_mech = self.kt * current_a * omega_rad_s
        p_cu = current_a * current_a * self.r_line_line
        return p_mech / max(p_mech + p_cu, 1e-12)

    def latency_torque_loss(self, omega_rad_s: float, t_latency_s: float) -> float:
        """Fraction of torque lost to commutation misalignment from angle
        latency: 1 - cos(pole_pairs * omega * t_latency). The motor-sensor
        coupling - it grows with pole_pairs."""
        err = self.pole_pairs * omega_rad_s * t_latency_s
        return 1.0 - math.cos(err)

    def cfg_motor(self, base_motor: dict) -> dict:
        """Override dict for the bench cfg['motor'] (plant params)."""
        m = dict(base_motor)
        m.update(resistance_ohm=self.r_phase, inductance_h=self.l_phase,
                 ke_v_s_per_rad=self.ke_phase, inertia_kg_m2=self.inertia_kg_m2,
                 damping_n_m_s_per_rad=self.damping, pole_pairs=self.pole_pairs,
                 trapezoid_blend=self.trapezoid_blend)
        return m


def kv_to_ke_line_line(kv_rpm_per_v: float) -> float:
    return 60.0 / (2.0 * math.pi * kv_rpm_per_v)


# Sensor effective angle latencies (from the part-comparison study) for M8.
SENSOR_LATENCY_S = {"AS5600": 90e-6, "AS5047P": 0.35e-6}


MOTORS: dict[str, Motor] = {
    # Budget: iPower GM2804 gimbal (12N14P), ships with an AS5048A magnetic
    # encoder. Sparse datasheet - R quoted, Kv from no-load RPM, L/J estimated.
    "gm2804": Motor(
        name="gm2804", label="iPower GM2804 (gimbal)",
        r_line_line=9.0,                  # quoted "internal resistance" (assumed)
        l_line_line=4.0e-3,               # NOT published - estimate (assumed/Q1)
        ke_line_line_peak=kv_to_ke_line_line(165.0),  # Kv~165 from no-load RPM
        pole_count=14, inertia_kg_m2=1.5e-5,   # J estimated (assumed/Q1)
        damping=2.0e-5, trapezoid_blend=0.0,
        rated_current_a=1.0, rated_voltage_v=12.0, price_usd=30.0,
        provenance="pole_count datasheet; R/Kv from listing (assumed); L,J,B "
                   "estimated (Q1 - measure on bench); sinusoidal gimbal"),
    # Mid: Nanotec DB42S03 - full datasheet (the provenance win).
    "db42s03": Motor(
        name="db42s03", label="Nanotec DB42S03",
        r_line_line=1.5,                  # datasheet
        l_line_line=2.1e-3,               # datasheet
        ke_line_line_peak=0.060,          # from rated torque/current (datasheet)
        pole_count=8, inertia_kg_m2=4.8e-6,    # datasheet rotor inertia
        damping=1.0e-5, trapezoid_blend=0.0,
        rated_current_a=1.79, rated_voltage_v=24.0, price_usd=90.0,
        provenance="R,L,pole_count datasheet; Ke from rated torque/current; J "
                   "datasheet; 4 pp matches the current POLE_PAIRS=4 build"),
    # Premium: maxon EC 45 flat 50 W 12 V (p/n 251601). Datasheet-typical -
    # confirm exact values against the maxon 251601 datasheet.
    "maxon_ec45": Motor(
        name="maxon_ec45", label="maxon EC 45 flat 50W",
        r_line_line=0.8,                  # datasheet-typical (confirm 251601)
        l_line_line=0.56e-3,              # datasheet-typical
        ke_line_line_peak=0.0467,         # Kt~27 mNm/A -> Ke (datasheet-typical)
        pole_count=16, inertia_kg_m2=9.2e-6,   # ~92 gcm^2
        damping=1.0e-5, trapezoid_blend=0.0,
        rated_current_a=3.2, rated_voltage_v=12.0, price_usd=200.0,
        provenance="EC45-flat-50W-12V datasheet-typical values; confirm exact "
                   "against maxon 251601; full datasheet available"),
}

TIERS = ["gm2804", "db42s03", "maxon_ec45"]   # budget -> premium
