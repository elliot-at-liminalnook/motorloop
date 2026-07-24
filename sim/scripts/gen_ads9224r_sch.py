#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate the open ADS9224R module's passive-network schematic from
sim/config/params.toml [circuit.ads9224r_module] (ads9224r-open-board-checklist
§3).

Like gen_kicad_sch.py, the param table is the primary source and the schematic
is a generated, kicad-cli-round-trip-checkable mirror of the board's measurable
passives: the THS4551 gain network (Rf/Rg), the ADC charge-buckets (Rflt/Cflt),
the phase shunts, and the reference reservoir. The active parts (ADS9224R,
THS4551, REF6041) appear as the labelled nets the passives connect to - the same
convention as the feedback-circuits schematic (which draws only board passives).
Two channels (A, B) share one CONVST, so both currents sample simultaneously.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_kicad_sch  # noqa: E402
import sim_params  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCH_DIR = PROJECT_ROOT / "hw" / "ads9224r-module"
SCH_PATH = SCH_DIR / "module.kicad_sch"
PROJECT = "ads9224r_module"
TITLE = "Open ADS9224R simultaneous current-sense module"
NAMESPACE = uuid.UUID("00000000-0000-0000-0000-0000ad592240")


def det_uuid(name: str) -> str:
    return str(uuid.uuid5(NAMESPACE, name))


def components_from_params(params: sim_params.SimParams):
    """(ref, lib, value_str, net_top, net_bottom, x, y) per passive. Each FOC
    channel: shunt -> FDA gain net (Rg, Rf) -> charge-bucket (Rflt, Cflt) ->
    ADC input. Plus the shared reference reservoir."""
    m = params.circuit_values("circuit.ads9224r_module")
    to_si = gen_kicad_sch.to_si
    rf = to_si(m["fda_rf"], ""); rg = to_si(m["fda_rg"], "")
    rflt = to_si(m["flt_r"], ""); cflt = to_si(m["flt_c"], "")
    rsh = to_si(m["shunt"], ""); cref = to_si(m["ref_reservoir_c"], "")
    rows = []
    for ch, y0 in (("A", 50.8), ("B", 101.6)):
        rows += [
            (f"RSH{ch}", "R", rsh, f"PH{ch}_P", f"PH{ch}_N", 50.8, y0),
            (f"RG{ch}", "R", rg, f"PH{ch}_P", f"FDA{ch}_FB", 76.2, y0),
            (f"RF{ch}", "R", rf, f"FDA{ch}_FB", f"FDA{ch}_OUT", 101.6, y0),
            (f"RFLT{ch}", "R", rflt, f"FDA{ch}_OUT", f"AIN{ch}", 127.0, y0),
            (f"CFLT{ch}", "C", cflt, f"AIN{ch}", "GND", 127.0, y0 + 25.4),
        ]
    rows.append(("CREF", "C", cref, "VREF", "GND", 152.4, 50.8))
    return rows


def render_schematic(params: sim_params.SimParams) -> str:
    return gen_kicad_sch.render_schematic(
        params, components=components_from_params(params),
        title=TITLE, project=PROJECT, uuid_fn=det_uuid)


def check(params: sim_params.SimParams, netlist_path: Path) -> list[str]:
    return gen_kicad_sch.check(params, netlist_path,
                               components=components_from_params(params))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=SCH_PATH)
    args = parser.parse_args()
    params = sim_params.load(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_schematic(params))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
