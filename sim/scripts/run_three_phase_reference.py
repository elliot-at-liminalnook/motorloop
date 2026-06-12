#!/usr/bin/env python3
"""Three-phase averaged reference simulation (executable spec).

Mirrors the averaged mode of sim/cpp/src/three_phase_plant.cpp and the
Modelica oracle model, statement for statement where possible:

- per-phase R/L, position-dependent EMF (sinusoid-to-trapezoid blend),
  isolated neutral (currents sum to zero over connected legs),
- ideal averaged six-step bridge: driven high = duty*vbus, driven low = 0,
  the unselected leg freewheels through ideal clamps (0 / vbus) while its
  current decays, then floats with its current held at ~0,
- open-loop sector schedule: quadratic electrical-phase ramp to a final
  frequency, sector held constant within each outer step (evaluated at the
  step start),
- RK4 with leg modes re-resolved at each stage, diode-blocking clamps at
  substep ends.

All parameters come from sim/config/params.toml (provenance-flagged).
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import sim_params

TWO_PI = 2.0 * math.pi
PHASE_SHIFT = TWO_PI / 3.0

# Six-step table, sector -> (high phase, low phase); 0=A, 1=B, 2=C.
SIX_STEP = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]

DRIVEN_HIGH, DRIVEN_LOW, DIODE_LOW, DIODE_HIGH, FLOAT = range(5)


@dataclass(frozen=True)
class MotorParams:
    resistance_ohm: float
    inductance_h: float
    ke_v_s_per_rad: float
    inertia_kg_m2: float
    damping_n_m_s_per_rad: float
    pole_pairs: int
    trapezoid_blend: float
    load_torque_n_m: float


@dataclass(frozen=True)
class BridgeParams:
    vbus_v: float


@dataclass
class State:
    current_a: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    omega_rad_s: float = 0.0
    theta_rad: float = 0.0

    def copy(self) -> "State":
        return State(list(self.current_a), self.omega_rad_s, self.theta_rad)


def emf_shape(theta_e: float, blend: float) -> float:
    s = math.sin(theta_e)
    trap = max(-1.0, min(1.0, 2.0 * s))
    return (1.0 - blend) * s + blend * trap


def open_loop_phase_e(t_s: float, f_final: float, ramp_time: float) -> float:
    if t_s < ramp_time:
        return TWO_PI * (0.5 * f_final * t_s * t_s / ramp_time)
    return TWO_PI * (0.5 * f_final * ramp_time + f_final * (t_s - ramp_time))


def open_loop_sector(t_s: float, f_final: float, ramp_time: float) -> int:
    phase = open_loop_phase_e(t_s, f_final, ramp_time)
    return int(math.floor(phase / (math.pi / 3.0))) % 6


class ThreePhaseAveragedPlant:
    def __init__(self, motor: MotorParams, bridge: BridgeParams,
                 current_epsilon_a: float, max_substep_s: float):
        self.m = motor
        self.b = bridge
        self.eps = current_epsilon_a
        self.max_substep = max_substep_s
        self.state = State()
        self.sector = 0
        self.duty = 0.0

    def set_averaged(self, sector: int, duty: float) -> None:
        self.sector = sector % 6
        self.duty = max(0.0, min(1.0, duty))

    def resolve_modes(self, s: State) -> list[int]:
        hi, lo = SIX_STEP[self.sector]
        modes = []
        for k in range(3):
            i = s.current_a[k]
            if k == hi:
                modes.append(DRIVEN_HIGH)
            elif k == lo:
                modes.append(DRIVEN_LOW)
            elif i > self.eps:
                modes.append(DIODE_LOW)
            elif i < -self.eps:
                modes.append(DIODE_HIGH)
            else:
                modes.append(FLOAT)
        return modes

    def evaluate(self, s: State, mode: list[int] | None = None) -> dict:
        m, b = self.m, self.b
        if mode is None:
            mode = self.resolve_modes(s)
        mode = list(mode)
        theta_e = m.pole_pairs * s.theta_rad
        shape = [emf_shape(theta_e - k * PHASE_SHIFT, m.trapezoid_blend)
                 for k in range(3)]
        emf = [m.ke_v_s_per_rad * shape[k] * s.omega_rad_s for k in range(3)]

        vt = [0.0, 0.0, 0.0]
        for k in range(3):
            if mode[k] == DRIVEN_HIGH:
                vt[k] = self.duty * b.vbus_v
            elif mode[k] in (DRIVEN_LOW, DIODE_LOW):
                vt[k] = 0.0
            elif mode[k] == DIODE_HIGH:
                vt[k] = b.vbus_v

        connected = [mode[k] != FLOAT for k in range(3)]
        n = sum(connected)
        if n >= 2:
            vn = sum(vt[k] - emf[k] for k in range(3) if connected[k]) / n
        else:
            vn = -(emf[0] + emf[1] + emf[2]) / 3.0
            for k in range(3):
                if connected[k]:
                    connected[k] = False
                    mode[k] = FLOAT

        for k in range(3):
            if mode[k] == FLOAT:
                vt[k] = vn + emf[k]

        torque = sum(m.ke_v_s_per_rad * shape[k] * s.current_a[k]
                     for k in range(3))
        return {"vt": vt, "emf": emf, "vn": vn, "mode": mode,
                "torque": torque}

    def derivatives(self, s: State, mode: list[int]) -> tuple[list[float], float, float]:
        m = self.m
        out = self.evaluate(s, mode)
        di = [0.0, 0.0, 0.0]
        for k in range(3):
            if out["mode"][k] != FLOAT:
                di[k] = (out["vt"][k] - out["vn"] - out["emf"][k]
                         - m.resistance_ohm * s.current_a[k]) / m.inductance_h
        domega = (out["torque"] - m.damping_n_m_s_per_rad * s.omega_rad_s
                  - m.load_torque_n_m) / m.inertia_kg_m2
        return di, domega, s.omega_rad_s

    def _stage(self, s0: State, d: tuple, scale: float) -> State:
        di, domega, dtheta = d
        s = s0.copy()
        for k in range(3):
            s.current_a[k] += scale * di[k]
        s.omega_rad_s += scale * domega
        s.theta_rad += scale * dtheta
        return s

    def substep(self, dt: float) -> None:
        # Leg modes resolved ONCE at substep start and held for all RK4
        # stages (stage-level re-resolution lets diode modes flip-flop around
        # zero crossings); post-step clamps handle the crossing itself.
        # Mirrors ThreePhasePlant::substep in three_phase_plant.cpp.
        s0 = self.state
        mode = self.resolve_modes(s0)
        k1 = self.derivatives(s0, mode)
        k2 = self.derivatives(self._stage(s0, k1, dt / 2.0), mode)
        k3 = self.derivatives(self._stage(s0, k2, dt / 2.0), mode)
        k4 = self.derivatives(self._stage(s0, k3, dt), mode)

        s = s0.copy()
        for k in range(3):
            s.current_a[k] += dt * (k1[0][k] + 2.0 * k2[0][k]
                                    + 2.0 * k3[0][k] + k4[0][k]) / 6.0
        s.omega_rad_s += dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        s.theta_rad += dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0

        for k in range(3):
            if mode[k] == DIODE_LOW and s.current_a[k] < 0.0:
                s.current_a[k] = 0.0
            elif mode[k] == DIODE_HIGH and s.current_a[k] > 0.0:
                s.current_a[k] = 0.0
            elif mode[k] == FLOAT:
                s.current_a[k] = 0.0
        self.state = s

    def advance(self, dt: float) -> None:
        remaining = dt
        while remaining > 1e-15:
            h = min(remaining, self.max_substep)
            self.substep(h)
            remaining -= h


def simulate(motor: MotorParams, bridge: BridgeParams, *, duty: float,
             f_elec_final_hz: float, ramp_time_s: float, stop_s: float,
             dt_s: float, current_epsilon_a: float,
             max_substep_s: float) -> list[dict[str, float]]:
    plant = ThreePhaseAveragedPlant(motor, bridge, current_epsilon_a,
                                    max_substep_s)
    rows: list[dict[str, float]] = []
    steps = int(round(stop_s / dt_s))
    for step in range(steps + 1):
        t_s = step * dt_s
        sector = open_loop_sector(t_s, f_elec_final_hz, ramp_time_s)
        plant.set_averaged(sector, duty)

        s = plant.state
        out = plant.evaluate(s)
        rows.append({
            "time_s": t_s,
            "ia_a": s.current_a[0], "ib_a": s.current_a[1],
            "ic_a": s.current_a[2],
            "va_v": out["vt"][0], "vb_v": out["vt"][1], "vc_v": out["vt"][2],
            "ea_v": out["emf"][0], "eb_v": out["emf"][1],
            "ec_v": out["emf"][2],
            "neutral_v": out["vn"],
            "torque_n_m": out["torque"],
            "omega_rad_s": s.omega_rad_s,
            "theta_rad": s.theta_rad,
            "sector": float(sector),
        })
        if step < steps:
            plant.advance(dt_s)
    return rows


def motor_from_config(config: sim_params.SimParams) -> MotorParams:
    return MotorParams(
        resistance_ohm=config.value("motor.R"),
        inductance_h=config.value("motor.L"),
        ke_v_s_per_rad=config.value("motor.Ke"),
        inertia_kg_m2=config.value("motor.J"),
        damping_n_m_s_per_rad=config.value("motor.B"),
        pole_pairs=int(config.value("motor.pole_pairs")),
        trapezoid_blend=config.value("motor.emf_trapezoid_blend"),
        load_torque_n_m=config.value(
            "scenario.three_phase_open_loop.load_torque"),
    )


def scenario_from_config(config: sim_params.SimParams) -> dict:
    prefix = "scenario.three_phase_open_loop"
    return {
        "duty": config.value(f"{prefix}.duty"),
        "f_elec_final_hz": config.value(f"{prefix}.f_elec_final"),
        "ramp_time_s": config.value(f"{prefix}.ramp_time"),
        "stop_s": config.value(f"{prefix}.stop"),
        "dt_s": config.value(f"{prefix}.dt"),
        "current_epsilon_a": config.value("sim.current_epsilon"),
        "max_substep_s": config.value(f"{prefix}.max_substep"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path,
        default=Path("sim/build/three_phase_reference.csv"))
    args = parser.parse_args()

    config = sim_params.load(args.config)
    print(config.banner_text())

    rows = simulate(motor_from_config(config),
                    BridgeParams(vbus_v=config.value("bus.vbus")),
                    **scenario_from_config(config))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    sidecar = config.write_sidecar(args.output)

    final = rows[-1]
    print(f"wrote {args.output}")
    print(f"wrote {sidecar}")
    print(f"rows: {len(rows)}")
    print(f"final omega: {final['omega_rad_s']:.3f} rad/s")
    print(f"final currents: {final['ia_a']:.3f} {final['ib_a']:.3f} "
          f"{final['ic_a']:.3f} A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
