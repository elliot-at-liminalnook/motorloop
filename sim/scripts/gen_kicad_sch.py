#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate the feedback-circuits KiCad schematic from sim/config/params.toml.

Tier 3 of the derivation layer (notes/derivation-checklist.md): the schematic
is a generated, human-reviewable mirror of the [circuit.*] component tables —
the tables stay the primary source. `check` mode exports a SPICE netlist via
kicad-cli and verifies component values and connectivity survive the round
trip; the PDF export is the wiring/measurement reference for bring-up.

Only the physically-measurable board passives are drawn (EMF divider+filter,
bus divider). The IOUT amp and ADC sample network are silicon-internal or
behavioral and live in sim/circuits/*.cir instead.
"""

from __future__ import annotations

import argparse
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_params  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCH_DIR = PROJECT_ROOT / "hw" / "feedback-circuits"
SCH_PATH = SCH_DIR / "feedback_circuits.kicad_sch"
DEVICE_LIB = Path("/usr/share/kicad/symbols/Device.kicad_sym")

NAMESPACE = uuid.UUID("00000000-0000-0000-0000-00000000b1dc")
PIN_OFFSET = 3.81  # Device:R and Device:C pin 1 at (0,+3.81), pin 2 (0,-3.81)


def det_uuid(name: str) -> str:
    return str(uuid.uuid5(NAMESPACE, name))


def to_si(value: float, unit: str) -> str:
    """Engineering notation matching KiCad value conventions (95.3k, 100n)."""
    prefixes = [(1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"),
                (1e-6, "u"), (1e-9, "n"), (1e-12, "p")]
    for scale, prefix in prefixes:
        if value >= scale:
            scaled = value / scale
            text = f"{scaled:.10g}"
            return f"{text}{prefix}{unit}"
    return f"{value:g}{unit}"


def extract_symbol(lib_text: str, name: str) -> str:
    marker = f'(symbol "{name}"'
    start = lib_text.index(marker)
    depth = 0
    for i in range(start, len(lib_text)):
        if lib_text[i] == "(":
            depth += 1
        elif lib_text[i] == ")":
            depth -= 1
            if depth == 0:
                block = lib_text[start:i + 1]
                return block.replace(f'(symbol "{name}"',
                                     f'(symbol "Device:{name}"', 1)
    raise ValueError(f"symbol {name} not found")


def components_from_params(params: sim_params.SimParams):
    """(ref, lib, value_str, net_top, net_bottom, x, y) for each passive."""
    emf = params.circuit_values("circuit.emf_channel")
    bus = params.circuit_values("circuit.bus_divider")
    return [
        ("R1", "R", to_si(emf["r_top"], ""), "EMF_IN", "EMF_MID", 50.8, 50.8),
        ("R2", "R", to_si(emf["r_bottom"], ""), "EMF_MID", "GND", 50.8, 76.2),
        ("R3", "R", to_si(emf["r_series"], ""), "EMF_MID", "EMF_OUT", 76.2, 50.8),
        ("C1", "C", to_si(emf["c_filter"], ""), "EMF_OUT", "GND", 76.2, 76.2),
        ("R4", "R", to_si(bus["r_top"], ""), "BUS_IN", "BUS_OUT", 127.0, 50.8),
        ("R5", "R", to_si(bus["r_bottom"], ""), "BUS_OUT", "GND", 127.0, 76.2),
    ]


def render_schematic(params: sim_params.SimParams, components=None,
                     title: str = "BLDC feedback measurement circuits",
                     project: str = "feedback_circuits", uuid_fn=None) -> str:
    """Render a KiCad schematic of a passive network. Defaults reproduce the
    feedback-circuits schematic byte-for-byte; pass `components` (the
    components_from_params tuple list), `title`, `project` and a deterministic
    `uuid_fn` to render a different board (e.g. the ADS9224R module)."""
    uuid_fn = uuid_fn or det_uuid
    if components is None:
        components = components_from_params(params)
    lib_text = DEVICE_LIB.read_text()
    root_uuid = uuid_fn("root")
    parts = [
        '(kicad_sch',
        '  (version 20250114)',
        '  (generator "gen_kicad_sch")',
        '  (generator_version "9.0")',
        f'  (uuid "{root_uuid}")',
        '  (paper "A4")',
        f'  (title_block (title "{title}")'
        ' (comment 1 "GENERATED from sim/config/params.toml - do not edit"))',
        '  (lib_symbols',
        extract_symbol(lib_text, "R"),
        extract_symbol(lib_text, "C"),
        '  )',
    ]

    label_done: set[tuple[str, float, float]] = set()
    for ref, lib, value, net_top, net_bot, x, y in components:
        sym_uuid = uuid_fn(f"sym-{ref}")
        parts.append(f'''  (symbol (lib_id "Device:{lib}") (at {x} {y} 0) (unit 1)
    (exclude_from_sim no) (in_bom yes) (on_board yes) (dnp no)
    (uuid "{sym_uuid}")
    (property "Reference" "{ref}" (at {x + 2.54} {y - 1.27} 0)
      (effects (font (size 1.27 1.27)) (justify left)))
    (property "Value" "{value}" (at {x + 2.54} {y + 1.27} 0)
      (effects (font (size 1.27 1.27)) (justify left)))
    (property "Footprint" "" (at {x} {y} 0)
      (effects (font (size 1.27 1.27)) (hide yes)))
    (pin "1" (uuid "{uuid_fn(f"pin1-{ref}")}"))
    (pin "2" (uuid "{uuid_fn(f"pin2-{ref}")}"))
    (instances (project "{project}"
      (path "/{root_uuid}" (reference "{ref}") (unit 1))))
  )''')
        # Global labels at the exact pin connection points (pin 1 above the
        # anchor in schematic coordinates, pin 2 below).
        for net, py_ in ((net_top, y - PIN_OFFSET), (net_bot, y + PIN_OFFSET)):
            key = (net, x, py_)
            if key in label_done:
                continue
            label_done.add(key)
            parts.append(
                f'  (global_label "{net}" (shape input) (at {x} {py_} 0)\n'
                f'    (effects (font (size 1.27 1.27)) (justify left))\n'
                f'    (uuid "{uuid_fn(f"lbl-{net}-{x}-{py_}")}"))')

    parts.append(f'  (sheet_instances (path "/" (page "1")))')
    parts.append(')')
    return "\n".join(parts) + "\n"


def parse_spice_netlist(text: str):
    """{ref: (value_str, frozenset(nets))} for R/C lines."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^([RC]\w*)\s+(\S+)\s+(\S+)\s+(\S+)", line.strip())
        if m:
            ref, n1, n2, value = m.groups()
            out[ref.upper()] = (value, frozenset({n1.upper(), n2.upper()}))
    return out


def check(params: sim_params.SimParams, netlist_path: Path,
          components=None) -> list[str]:
    """Compare an exported SPICE netlist against the component tables."""
    if components is None:
        components = components_from_params(params)
    exported = parse_spice_netlist(netlist_path.read_text())
    errors = []
    for ref, _lib, value, net_top, net_bot, _x, _y in components:
        if ref not in exported:
            errors.append(f"{ref} missing from exported netlist")
            continue
        got_value, got_nets = exported[ref]
        if got_value.upper() != value.upper():
            errors.append(f"{ref}: value {got_value} != {value}")
        want = frozenset(n.upper() for n in (net_top, net_bot))
        if got_nets != want:
            errors.append(f"{ref}: nets {sorted(got_nets)} != {sorted(want)}")
    return errors


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
