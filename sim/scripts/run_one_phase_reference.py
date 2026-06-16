#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reference simulation for the first one-phase plant slice.

This intentionally mirrors the equations in:
sim/modelica/BldcCosimTestbench/package.mo

All parameters come from sim/config/params.toml, which carries a provenance
status per parameter. The assumption banner printed at startup (and written
as a sidecar next to the CSV) lists every unconfirmed value; do not treat
results as hardware predictions while that list is non-empty.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import sim_params


@dataclass(frozen=True)
class Params:
    resistance_ohm: float
    inductance_h: float
    back_emf_v_s_per_rad: float
    torque_n_m_per_a: float
    inertia_kg_m2: float
    damping_n_m_s_per_rad: float
    vbus_v: float
    load_torque_n_m: float


def params_from_config(config: sim_params.SimParams) -> Params:
    return Params(
        resistance_ohm=config.value("motor.R"),
        inductance_h=config.value("motor.L"),
        back_emf_v_s_per_rad=config.value("motor.Ke"),
        torque_n_m_per_a=config.value("motor.Kt"),
        inertia_kg_m2=config.value("motor.J"),
        damping_n_m_s_per_rad=config.value("motor.B"),
        vbus_v=config.value("bus.vbus"),
        load_torque_n_m=config.value("scenario.one_phase_reference.load_torque"),
    )


@dataclass(frozen=True)
class State:
    current_a: float
    omega_rad_s: float
    theta_rad: float


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def make_duty_schedule(segments: list[list[float]]):
    """Piecewise-constant schedule from [[t_start_s, duty], ...] pairs.

    Shared verbatim (via params.toml) with the C++ and Modelica
    implementations so parity is exact.
    """

    def duty_schedule(t_s: float) -> float:
        duty = 0.0
        for t_start, value in segments:
            if t_s < t_start:
                break
            duty = value
        return duty

    return duty_schedule


def derivatives(state: State, duty: float, params: Params) -> State:
    duty_limited = clamp(duty, -1.0, 1.0)
    phase_voltage_v = duty_limited * params.vbus_v
    back_emf_v = params.back_emf_v_s_per_rad * state.omega_rad_s

    d_current = (
        phase_voltage_v
        - params.resistance_ohm * state.current_a
        - back_emf_v
    ) / params.inductance_h

    electromagnetic_torque = params.torque_n_m_per_a * state.current_a
    d_omega = (
        electromagnetic_torque
        - params.damping_n_m_s_per_rad * state.omega_rad_s
        - params.load_torque_n_m
    ) / params.inertia_kg_m2

    return State(
        current_a=d_current,
        omega_rad_s=d_omega,
        theta_rad=state.omega_rad_s,
    )


def add_scaled(state: State, delta: State, scale: float) -> State:
    return State(
        current_a=state.current_a + scale * delta.current_a,
        omega_rad_s=state.omega_rad_s + scale * delta.omega_rad_s,
        theta_rad=state.theta_rad + scale * delta.theta_rad,
    )


def rk4_step(state: State, t_s: float, dt_s: float, params: Params, duty_schedule) -> State:
    k1 = derivatives(state, duty_schedule(t_s), params)
    k2 = derivatives(add_scaled(state, k1, dt_s / 2.0), duty_schedule(t_s + dt_s / 2.0), params)
    k3 = derivatives(add_scaled(state, k2, dt_s / 2.0), duty_schedule(t_s + dt_s / 2.0), params)
    k4 = derivatives(add_scaled(state, k3, dt_s), duty_schedule(t_s + dt_s), params)

    return State(
        current_a=state.current_a + dt_s * (k1.current_a + 2.0 * k2.current_a + 2.0 * k3.current_a + k4.current_a) / 6.0,
        omega_rad_s=state.omega_rad_s + dt_s * (k1.omega_rad_s + 2.0 * k2.omega_rad_s + 2.0 * k3.omega_rad_s + k4.omega_rad_s) / 6.0,
        theta_rad=state.theta_rad + dt_s * (k1.theta_rad + 2.0 * k2.theta_rad + 2.0 * k3.theta_rad + k4.theta_rad) / 6.0,
    )


def row_for(t_s: float, state: State, params: Params, duty_schedule) -> dict[str, float]:
    duty = duty_schedule(t_s)
    duty_limited = clamp(duty, -1.0, 1.0)
    phase_voltage_v = duty_limited * params.vbus_v
    back_emf_v = params.back_emf_v_s_per_rad * state.omega_rad_s
    torque_n_m = params.torque_n_m_per_a * state.current_a
    return {
        "time_s": t_s,
        "duty": duty,
        "duty_limited": duty_limited,
        "phase_voltage_v": phase_voltage_v,
        "back_emf_v": back_emf_v,
        "current_a": state.current_a,
        "torque_n_m": torque_n_m,
        "omega_rad_s": state.omega_rad_s,
        "theta_rad": state.theta_rad,
    }


def simulate(stop_s: float, dt_s: float, params: Params, duty_schedule) -> list[dict[str, float]]:
    state = State(current_a=0.0, omega_rad_s=0.0, theta_rad=0.0)
    rows: list[dict[str, float]] = []

    steps = int(round(stop_s / dt_s))
    for step in range(steps + 1):
        t_s = step * dt_s
        rows.append(row_for(t_s, state, params, duty_schedule))
        if step < steps:
            state = rk4_step(state, t_s, dt_s, params, duty_schedule)

    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None,
                        help="parameter file (default: sim/config/params.toml)")
    parser.add_argument("--stop-s", type=float, default=None,
                        help="override scenario.one_phase_reference.stop")
    parser.add_argument("--dt-s", type=float, default=None,
                        help="override scenario.one_phase_reference.dt")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sim/build/one_phase_reference.csv"),
    )
    args = parser.parse_args()

    config = sim_params.load(args.config)
    print(config.banner_text())

    params = params_from_config(config)
    stop_s = args.stop_s if args.stop_s is not None else config.value(
        "scenario.one_phase_reference.stop")
    dt_s = args.dt_s if args.dt_s is not None else config.value(
        "scenario.one_phase_reference.dt")

    duty_schedule = make_duty_schedule(
        config.value("scenario.one_phase_reference.duty_schedule"))
    rows = simulate(stop_s, dt_s, params, duty_schedule)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sidecar = config.write_sidecar(args.output)

    peak_current = max(abs(row["current_a"]) for row in rows)
    peak_omega = max(abs(row["omega_rad_s"]) for row in rows)
    final = rows[-1]

    print(f"wrote {args.output}")
    print(f"wrote {sidecar}")
    print(f"rows: {len(rows)}")
    print(f"peak current: {peak_current:.3f} A")
    print(f"peak omega: {peak_omega:.3f} rad/s")
    print(f"final current: {final['current_a']:.3f} A")
    print(f"final omega: {final['omega_rad_s']:.3f} rad/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
