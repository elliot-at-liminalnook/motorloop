#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the Modelica oracle (three-phase averaged open-loop) through omc.

Every parameter is overridden at simulate time from sim/config/params.toml,
so the .mo literals can never silently drift from the config (the stronger
form of the literal-consistency check). Output is converted to the shared
trace schema used by the C++ bench and the Python reference.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_params  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_MO = PROJECT_ROOT / "sim" / "modelica" / "BldcCosimTestbench" / "package.mo"
MODEL = "BldcCosimTestbench.Plants.ThreePhaseAveragedOpenLoop"

COLUMN_MAP = {
    "time": "time_s",
    "i[1]": "ia_a",
    "i[2]": "ib_a",
    "i[3]": "ic_a",
    "omega": "omega_rad_s",
    "theta": "theta_rad",
    "torque": "torque_n_m",
}


def overrides_from_config(config: sim_params.SimParams) -> dict[str, float]:
    prefix = "scenario.three_phase_open_loop"
    return {
        "R": config.value("motor.R"),
        "L": config.value("motor.L"),
        "Ke": config.value("motor.Ke"),
        "J": config.value("motor.J"),
        "B": config.value("motor.B"),
        "polePairs": int(config.value("motor.pole_pairs")),
        "blend": config.value("motor.emf_trapezoid_blend"),
        "loadTorque": config.value(f"{prefix}.load_torque"),
        "vBus": config.value("bus.vbus"),
        "duty": config.value(f"{prefix}.duty"),
        "fElecFinal": config.value(f"{prefix}.f_elec_final"),
        "rampTime": config.value(f"{prefix}.ramp_time"),
        "iEps": config.value("sim.current_epsilon"),
    }


def run_oracle(config: sim_params.SimParams, workdir: Path) -> dict[str, list[float]]:
    """Simulate the oracle; returns columns in the shared trace schema."""
    workdir.mkdir(parents=True, exist_ok=True)
    prefix = "scenario.three_phase_open_loop"
    stop = config.value(f"{prefix}.stop")
    dt = config.value(f"{prefix}.dt")
    intervals = int(round(stop / dt))

    overrides = ",".join(
        f"{k}={v}" for k, v in overrides_from_config(config).items())
    mos = workdir / "run_oracle.mos"
    mos.write_text(
        f'loadFile("{PACKAGE_MO}"); getErrorString();\n'
        f'simulate({MODEL}, stopTime={stop}, numberOfIntervals={intervals},\n'
        f'  tolerance=1e-8, outputFormat="csv",\n'
        f'  variableFilter="i\\\\[[1-3]\\\\]|omega|theta|torque",\n'
        f'  simflags="-override {overrides}"); getErrorString();\n'
    )
    result = subprocess.run(
        ["omc", mos.name], cwd=workdir, capture_output=True, text=True,
        timeout=600)
    csv_path = workdir / f"{MODEL}_res.csv"
    if result.returncode != 0 or not csv_path.is_file():
        raise RuntimeError(
            f"oracle run failed:\n{result.stdout}\n{result.stderr}")
    if "Failed" in result.stdout:
        raise RuntimeError(f"oracle simulation error:\n{result.stdout}")

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    columns: dict[str, list[float]] = {}
    for src, dst in COLUMN_MAP.items():
        if src in rows[0]:
            columns[dst] = [float(r[src]) for r in rows]
    return columns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path,
        default=Path("sim/build/three_phase_oracle.csv"))
    parser.add_argument(
        "--workdir", type=Path, default=Path("sim/build/oracle"))
    args = parser.parse_args()

    config = sim_params.load(args.config)
    print(config.banner_text())

    columns = run_oracle(config, args.workdir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    keys = list(columns.keys())
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*(columns[k] for k in keys)))
    sidecar = config.write_sidecar(args.output)

    print(f"wrote {args.output}")
    print(f"wrote {sidecar}")
    print(f"rows: {len(columns['time_s'])}")
    print(f"final omega: {columns['omega_rad_s'][-1]:.3f} rad/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
