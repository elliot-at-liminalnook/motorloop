#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run a named lockstep-bench scenario and write trace artifacts.

Scenarios: init, open_loop, closed_loop.
Writes sim/build/bench_<scenario>.csv (+ .assumptions.txt sidecar, and
optionally a VCD for waveform debugging).

Usage:
  python3 sim/scripts/run_bench_scenario.py closed_loop [--seconds 1.2]
      [--vcd] [--config sim/config/params.toml]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "sim" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "sim" / "tests"))
sys.path.insert(0, str(PROJECT_ROOT / "sim" / "build" / "cpp"))

import sim_params  # noqa: E402
from bench_factory import bench_config, expected_init_time, freq_word  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario",
                        choices=["init", "open_loop", "closed_loop"])
    parser.add_argument("--seconds", type=float, default=1.2)
    parser.add_argument("--target-speed", type=int, default=80)
    parser.add_argument("--duty", type=float, default=0.5)
    parser.add_argument("--f-elec", type=float, default=30.0)
    parser.add_argument("--vcd", action="store_true",
                        help="dump full-rate waveforms; roughly 6 GB per "
                             "simulated second - use short --seconds")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = sim_params.load(args.config)
    print(config.banner_text())

    try:
        import bldcsim
    except ImportError:
        raise SystemExit("bldcsim module not built; run "
                         "sim/scripts/build_bench.sh first")

    b = bldcsim.Bench(bench_config(config))
    output = PROJECT_ROOT / "sim" / "build" / f"bench_{args.scenario}.csv"
    if args.vcd:
        vcd_path = output.with_suffix(".vcd")
        b.open_vcd(str(vcd_path))

    b.run_for(expected_init_time(config))
    print(f"init done at {b.time_s*1e3:.1f} ms, configured={b.configured}")

    if args.scenario == "open_loop":
        clk = config.value("rtl.clock_frequency")
        b.set_open_loop(freq_word(args.f_elec, clk), 1)
        b.set_duty(int(args.duty * 625))
        b.set_mode(1)
        b.run_for(args.seconds)
    elif args.scenario == "closed_loop":
        b.set_align_offset(int(config.value("rtl.sensored_align_offset")))
        b.set_target_speed(args.target_speed)
        b.set_mode(2)
        b.run_for(args.seconds)

    if args.vcd:
        b.close_vcd()
        print(f"wrote {vcd_path}")

    trace = b.trace
    keys = list(trace.keys())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*(trace[k] for k in keys)))
    sidecar = config.write_sidecar(output)

    print(f"wrote {output} ({len(trace['time_s'])} rows)")
    print(f"wrote {sidecar}")
    print(f"final: omega={b.omega:.2f} rad/s, faults={b.fault_count}, "
          f"mismatches={b.mismatch_count}, "
          f"shoot-through={b.shoot_through_violations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
