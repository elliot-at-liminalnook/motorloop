# SPDX-License-Identifier: MIT
"""Motor profiles for the motor-selection study (notes/motor-selection-checklist.md).

Discrete BLDC motors and integrated servos as provenance-tagged profiles. A
`Motor` stores the
datasheet MEASURABLES (line-to-line R/L, peak line-to-line Ke, pole_count, rotor
inertia) + the bench-only extras (B, trapezoid_blend, align_offset). The
measurable -> per-phase conversions reuse the same relations the global
derive_params uses (R=Rll/2, L=Lll/2, Ke=Ke_ll/sqrt3, Kt=Ke, pp=poles/2), so a
profile is consistent with the bench's existing motor derivation.

Provenance is honest per field: `datasheet` where the part publishes it,
`assumed`/`estimate` where it must be measured on the bench (the gimbal's L/J,
all align offsets) - these stay blocked on Q1 until a motor-ID session.

The plant pole_pairs and the RTL `POLE_PAIRS` (gen_rtl_params) must match for a
discrete motor, so a
motor with a different pole count needs a regen + re-Verilate (build_motor.sh).
The DB42 (4 pp) matches the current build; the GM2804 (7) and EC 45 (8) do not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


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
    # QDD actuator (MIT Mini-Cheetah / T-Motor AK80-class): high torque-density motor for a LOW gear
    # ratio (~6:1) -> high torque AND high speed AND backdrivable. The training-ease upgrade: ~3x the
    # db42s03 joint torque at a lower gear, so the body stands with headroom and recovers from impacts.
    "qdd_mc": Motor(
        name="qdd_mc", label="QDD actuator (Mini-Cheetah / AK80 class)",
        r_line_line=0.13,                 # big low-R stator
        l_line_line=0.05e-3,
        ke_line_line_peak=0.13,           # high Kt (torque-dense) -> ~6.7 N·m joint at gear 6, peak_factor 4
        pole_count=21, inertia_kg_m2=6.0e-5,   # larger rotor (heavier but proximal-mounted)
        damping=2.0e-5, trapezoid_blend=0.0,
        rated_current_a=3.6, rated_voltage_v=24.0, price_usd=150.0,
        provenance="QDD robot actuator class (MIT Mini-Cheetah / T-Motor AK80-9): high Kt + low gear "
                   "for torque+speed+backdrivability; values representative, confirm vs the chosen unit"),
    # Premium: maxon EC 45 flat 50 W 12 V (p/n 251601). Datasheet-typical -
    # confirm exact values against the maxon 251601 datasheet.
    "go_m8010": Motor(
        name="go_m8010", label="Unitree GO-M8010-6 (motor side; 6.33:1 in robot.toml)",
        r_line_line=0.23,                 # assumed (datasheet lists actuator-level only)
        l_line_line=0.06e-3,
        ke_line_line_peak=0.162,          # kt~=0.094: 23.7 N.m peak output / 6.33 / (4x 10A) - datasheet-derived
        pole_count=28, inertia_kg_m2=1.2e-4,   # large-gap outrunner rotor (assumed)
        damping=2.0e-5, trapezoid_blend=0.0,
        rated_current_a=10.0, rated_voltage_v=24.0, price_usd=280.0,
        provenance="Unitree GO-M8010-6 standalone actuator: 23.7 N.m peak / ~30 rad/s at output, 530 g, "
                   "integrated driver+encoder. Motor-side constants back-derived from actuator datasheet; "
                   "R/L/poles/inertia ASSUMED - bench-verify before hardware commit."),
    "diy_qdd_8308": Motor(
        name="diy_qdd_8308", label="DIY QDD: Eaglepower EA8308 KV90 + 8:1 printed planetary (own FOC stack)",
        r_line_line=0.09,                 # typical 8308 KV90 stator
        l_line_line=0.045e-3,
        ke_line_line_peak=0.184,          # kt~=0.106 from KV90 (9.55/KV), datasheet-derived
        pole_count=40, inertia_kg_m2=1.4e-4,   # large pancake rotor (assumed)
        damping=2.5e-5, trapezoid_blend=0.05,
        rated_current_a=9.0, rated_voltage_v=24.0, price_usd=150.0,
        provenance="DIY quasi-direct-drive: ~$85 EA8308 KV90 + printed 8:1 planetary + this repo's "
                   "DRV8316R/AS5047P FOC stack. Cheapest N.m; gear wear + thermal limits ASSUMED "
                   "(9A rated is conservative with airflow); bench-measure like the others."),
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

@dataclass(frozen=True)
class Servo:
    """Integrated servo with its own controller and position sensor.

    Deliberately NOT a Motor: no meaningful R/L/Ke at the user interface —
    the honest datasheet surface is stall torque + no-load speed per voltage.
    Sim mapping: P-only position actuator, torque clipped to stall_torque and
    derated linearly to zero at no_load_speed (torque-speed line)."""
    name: str
    label: str
    stall_torque_nm: dict         # vbus -> N*m
    no_load_speed_rad_s: dict     # vbus -> rad/s
    stall_current_a: dict         # vbus -> A
    no_load_current_a: dict       # vbus -> A
    travel_deg: float
    position_resolution_deg: float
    control_interface: str
    operating_voltage_v: tuple[float, float]
    baudrate_bps: tuple[int, int] | None
    output_inertia_kg_m2_est: float  # not published; simulation placeholder
    mass_kg: float
    price_usd: float
    source_urls: tuple[str, ...]
    provenance: str

    def joint(self, vbus: float, ratio: float) -> tuple[float, float]:
        """(stall torque, no-load speed) at a joint behind an external ratio."""
        return (self.stall_torque_nm[vbus] * ratio,
                self.no_load_speed_rad_s[vbus] / ratio)


SERVOS: dict[str, Servo] = {
    # Historical 2026-07-03 baseline, retained for comparison only.
    "gobilda_2000_5t": Servo(
        name="gobilda_2000_5t",
        label="ServoCity/goBILDA 2000 Series 5-Turn Dual-Mode (25-3, Speed)",
        stall_torque_nm={4.8: 0.775, 6.0: 0.912, 7.4: 1.059},   # 7.9/9.3/10.8 kg*cm
        no_load_speed_rad_s={4.8: 9.42, 6.0: 12.04, 7.4: 15.18},  # 90/115/145 RPM
        stall_current_a={4.8: 2.0, 6.0: 2.5, 7.4: 3.0},
        no_load_current_a={},            # not published
        travel_deg=1800.0,            # 5-turn default mode, pot feedback
        position_resolution_deg=3.6,  # 4 us deadband x 0.90 deg/us
        control_interface="PWM",
        operating_voltage_v=(4.8, 7.4), baudrate_bps=None,
        output_inertia_kg_m2_est=2.73375e-3,
        mass_kg=0.060, price_usd=49.99,
        source_urls=(),
        provenance="servocity.com product page 2026-07-03 (datasheet table): "
                   "torque/speed/current at 4.8/6.0/7.4 V, 135:1 steel gears, "
                   "H25T spline, dual ball bearing, 500-2500 us PWM at 0.90 "
                   "deg/us. Duty cycle / continuous torque NOT published - "
                   "derate stall for sustained load (assume <=50% conservatively)."),
    # DECIDED 2026-07-09: all 12 robot joints use this exact bus servo.
    "waveshare_st3215_hs": Servo(
        name="waveshare_st3215_hs",
        label="Waveshare ST3215-HS 20kg.cm High-Speed Bus Servo",
        stall_torque_nm={12.0: 20.0 * 0.0980665},       # 20 kgf.cm @ 12 V
        no_load_speed_rad_s={12.0: 106.0 * 2.0 * math.pi / 60.0},
        stall_current_a={12.0: 2.4},
        no_load_current_a={12.0: 0.240},
        travel_deg=360.0,
        position_resolution_deg=360.0 / 4096.0,
        control_interface="TTL UART serial bus",
        operating_voltage_v=(6.0, 12.6),
        baudrate_bps=(38_400, 1_000_000),
        # Waveshare does not publish output inertia. Keep the previous integrated-
        # servo estimate explicit until a coast-down/pendulum identification exists.
        output_inertia_kg_m2_est=2.7e-3,
        mass_kg=0.068, price_usd=27.81,
        source_urls=(
            "https://www.waveshare.com/st3215-hs-servo-motor.htm",
            "https://www.waveshare.com/wiki/ST3215-HS_Servo_Motor",
            "https://www.robotshop.com/products/waveshare-20kgcm-bus-servo-motor-106rpm-high-speed-large-torque-w-360-deg-high-precision-magnetic-encoder",
        ),
        provenance="Waveshare ST3215-HS product page/wiki and RobotShop RB-Wav-1556, "
                   "accessed 2026-07-09: 20 kg.cm @ 12 V, 106 RPM, 6-12.6 V, "
                   "240 mA no-load, 2.4 A locked rotor, 12-bit 360-degree magnetic "
                   "encoder, 68 g. Continuous-duty torque and output inertia are "
                   "not published and must be bench-characterized."),
}

TIERS = ["gm2804", "db42s03", "maxon_ec45"]   # budget -> premium
