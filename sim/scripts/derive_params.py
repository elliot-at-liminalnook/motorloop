#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Derive simulation parameters from the codified circuit specifications.

The source of truth for circuit-derived parameters is the component level:
[circuit.*] and [motor_spec] tables in sim/config/params.toml. This script
re-derives every parameter that carries a `derived_from` reference.

  --check                  recompute and compare against committed values
                           (exit 1 on mismatch) - also run by the test suite
  --update                 rewrite mismatched values in params.toml in place
                           (comments and provenance fields preserved)
  --measurement-checklist  emit the component worksheet for the Q7/Q1 bench
                           sessions, with the derived parameters each
                           measurement unblocks
  --write-spice-params     render sim/build/spice/components.param so the
                           ngspice netlists share the same component values

Unit-conversion notes (the silent-error traps, encoded once):
  - The plant is a wye-equivalent model: per-phase R and L are HALF the
    line-to-line (terminal) measurements, regardless of the internal wye or
    delta connection (any balanced 3-terminal network has this equivalent).
  - motor.Ke is the PEAK PER-PHASE BEMF constant in V*s/rad (mechanical),
    matching emf_shape()'s unit-peak convention: Ke = ke_line_line_peak /
    sqrt(3) for a sinusoidal machine.
  - Kt = Ke exactly, by the energy consistency of the plant's torque
    coupling (torque = Ke * sum(f_k * i_k)).
  - If only Kv [RPM/V] is known: ke_line_line_peak ~= 60/(2*pi*Kv).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_params  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPICE_PARAM_PATH = PROJECT_ROOT / "sim" / "build" / "spice" / "components.param"


def kv_to_ke_line_line(kv_rpm_per_v: float) -> float:
    """Peak line-to-line BEMF constant from a hobby-style Kv rating."""
    return 60.0 / (2.0 * math.pi * kv_rpm_per_v)


def parallel(a: float, b: float) -> float:
    return a * b / (a + b)


# ---------------------------------------------------------------------------
# Derivation registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Derivation:
    target: str                       # parameter path in params.toml
    fn: Callable[[sim_params.SimParams], object]
    uses: tuple[str, ...]             # component paths consumed (for the
                                      # measurement checklist + orphan check)


def _emf_source_impedance(p: sim_params.SimParams) -> float:
    c = p.circuit_values("circuit.emf_channel")
    topology = c["filter_topology"]
    if topology == "post_divider":
        return parallel(c["r_top"], c["r_bottom"]) + c["r_series"]
    if topology == "series_only":
        return c["r_series"]
    raise ValueError(f"unknown emf filter_topology '{topology}'")


def _emf_cutoff(p: sim_params.SimParams) -> float:
    c = p.circuit_values("circuit.emf_channel")
    return 1.0 / (2.0 * math.pi * _emf_source_impedance(p) * c["c_filter"])


def _sample_residual_unbuffered(p: sim_params.SimParams,
                                source_impedance: float) -> float:
    """Residual for a channel with NO local capacitor at the ADC pin: the
    sample cap must settle through the full source impedance within the
    1.5-clock aperture."""
    c = p.circuit_values("circuit.adc_frontend")
    t_window = 1.5 / p.value("adc.sclk")  # conservative (budget SCLK)
    tau = (source_impedance + c["sample_switch_r"]) * c["sample_cap"]
    return math.exp(-t_window / tau)


def _sample_residual_cap_buffered(p: sim_params.SimParams,
                                  local_cap: float) -> float:
    """Residual for a channel with a local filter cap at the ADC pin: the
    sample cap charge-shares with the reservoir (settling through the
    switch alone, tau ~ 20 ns, is complete). The reservoir droop this
    causes persists and recovers through the channel's RC - that dynamic
    is applied in-simulation via the sample-theft feedback, not folded
    into this per-sample fraction."""
    c = p.circuit_values("circuit.adc_frontend")
    return c["sample_cap"] / (c["sample_cap"] + local_cap)


def _ads9224r_full_scale_a(p: sim_params.SimParams) -> float:
    """Differential +/- full-scale current of the open ADS9224R module:
    ref_v / (shunt * FDA gain), FDA gain = fda_rf/fda_rg."""
    m = p.circuit_values("circuit.ads9224r_module")
    return m["ref_v"] / (m["shunt"] * (m["fda_rf"] / m["fda_rg"]))


def _ads9224r_acq_residual(p: sim_params.SimParams) -> float:
    """Single-pole charge-bucket settling estimate at the end of the
    acquisition window: exp(-t_acq / (flt_r * flt_c))."""
    m = p.circuit_values("circuit.ads9224r_module")
    tau = m["flt_r"] * m["flt_c"]
    return math.exp(-p.value("adc.ads9224r_acq_window_s") / tau)


DERIVATIONS: list[Derivation] = [
    # Motor (wye-equivalent per-phase from line-to-line measurables).
    Derivation("motor.R",
               lambda p: p.value("motor_spec.r_line_line") / 2.0,
               ("motor_spec.r_line_line",)),
    Derivation("motor.L",
               lambda p: p.value("motor_spec.l_line_line") / 2.0,
               ("motor_spec.l_line_line",)),
    Derivation("motor.Ke",
               lambda p: p.value("motor_spec.ke_line_line_peak") / math.sqrt(3.0),
               ("motor_spec.ke_line_line_peak",)),
    Derivation("motor.Kt",
               lambda p: p.value("motor_spec.ke_line_line_peak") / math.sqrt(3.0),
               ("motor_spec.ke_line_line_peak",)),
    Derivation("motor.pole_pairs",
               lambda p: int(p.value("motor_spec.pole_count")) // 2,
               ("motor_spec.pole_count",)),
    # Current channels.
    Derivation("feedback.current.shunt",
               lambda p: p.value("circuit.iout_channel.shunt"),
               ("circuit.iout_channel.shunt",)),
    Derivation("feedback.current.offset",
               lambda p: p.value("circuit.iout_channel.ref_pin_v") / 2.0,
               ("circuit.iout_channel.ref_pin_v",)),
    Derivation("drv8301.amp_vref",
               lambda p: p.value("circuit.iout_channel.ref_pin_v") / 2.0,
               ("circuit.iout_channel.ref_pin_v",)),
    # EMF channels.
    Derivation("feedback.emf.divider_ratio",
               lambda p: p.value("circuit.emf_channel.r_bottom")
               / (p.value("circuit.emf_channel.r_top")
                  + p.value("circuit.emf_channel.r_bottom")),
               ("circuit.emf_channel.r_top", "circuit.emf_channel.r_bottom")),
    Derivation("feedback.emf.source_impedance", _emf_source_impedance,
               ("circuit.emf_channel.r_top", "circuit.emf_channel.r_bottom",
                "circuit.emf_channel.r_series",
                "circuit.emf_channel.filter_topology")),
    Derivation("feedback.emf.rc_cutoff", _emf_cutoff,
               ("circuit.emf_channel.r_top", "circuit.emf_channel.r_bottom",
                "circuit.emf_channel.r_series", "circuit.emf_channel.c_filter",
                "circuit.emf_channel.filter_topology")),
    # Bus divider.
    Derivation("feedback.bus_voltage.divider_ratio",
               lambda p: p.value("circuit.bus_divider.r_bottom")
               / (p.value("circuit.bus_divider.r_top")
                  + p.value("circuit.bus_divider.r_bottom")),
               ("circuit.bus_divider.r_top", "circuit.bus_divider.r_bottom")),
    # ADC shared-sample-cap residuals. EMF channels have the filter cap AT
    # the ADC pin (charge share with the local reservoir); the bus divider
    # has no local cap (must settle through ~8.4 kOhm); IOUT is op-amp
    # buffered (negligible).
    Derivation("adc.sample_residual_emf",
               lambda p: _sample_residual_cap_buffered(
                   p, p.value("circuit.emf_channel.c_filter")),
               ("circuit.adc_frontend.sample_cap",
                "circuit.emf_channel.c_filter")),
    Derivation("adc.sample_residual_bus",
               lambda p: _sample_residual_unbuffered(
                   p, parallel(p.value("circuit.bus_divider.r_top"),
                               p.value("circuit.bus_divider.r_bottom"))),
               ("circuit.adc_frontend.sample_switch_r",
                "circuit.adc_frontend.sample_cap")),
    Derivation("adc.sample_residual_iout",
               lambda p: round(_sample_residual_unbuffered(
                   p, p.value("circuit.adc_frontend.iout_source_impedance")),
                   12),
               ("circuit.adc_frontend.sample_switch_r",
                "circuit.adc_frontend.sample_cap",
                "circuit.adc_frontend.iout_source_impedance")),
    # Open ADS9224R module: FDA gain, full-scale current, and the codes/A
    # scaling all fall out of the shunt + FDA resistors + reference.
    Derivation("feedback.current_ads9224r.fda_gain",
               lambda p: p.value("circuit.ads9224r_module.fda_rf")
               / p.value("circuit.ads9224r_module.fda_rg"),
               ("circuit.ads9224r_module.fda_rf",
                "circuit.ads9224r_module.fda_rg")),
    Derivation("feedback.current_ads9224r.full_scale_a", _ads9224r_full_scale_a,
               ("circuit.ads9224r_module.shunt", "circuit.ads9224r_module.fda_rf",
                "circuit.ads9224r_module.fda_rg", "circuit.ads9224r_module.ref_v")),
    Derivation("feedback.current_ads9224r.codes_per_amp",
               lambda p: 32768.0 / _ads9224r_full_scale_a(p),
               ("circuit.ads9224r_module.shunt", "circuit.ads9224r_module.fda_rf",
                "circuit.ads9224r_module.fda_rg", "circuit.ads9224r_module.ref_v")),
    Derivation("adc.acq_settle_residual_ads9224r", _ads9224r_acq_residual,
               ("circuit.ads9224r_module.flt_r", "circuit.ads9224r_module.flt_c")),
    Derivation("feedback.current_ads9224r.signal_bw_hz",
               lambda p: 1.0 / (2.0 * math.pi
                                * p.value("circuit.ads9224r_module.fda_rf")
                                * p.value("circuit.ads9224r_module.fda_fb_c")),
               ("circuit.ads9224r_module.fda_rf",
                "circuit.ads9224r_module.fda_fb_c")),
    # Ground-shift disturbance coefficients (realism stage 3) come straight
    # from the codified harness components.
    Derivation("disturbance.gnd_shift_r",
               lambda p: p.value("circuit.harness.r_return"),
               ("circuit.harness.r_return",)),
    Derivation("disturbance.gnd_shift_l",
               lambda p: p.value("circuit.harness.l_return"),
               ("circuit.harness.l_return",)),
]

# Spec components consumed directly by the bench (no derived scalar between
# them and the simulation) - exempt from the orphan check.
DIRECTLY_CONSUMED = {
    "circuit.gate_pulldowns.en_gate_pulldown",  # bench config-window model
    # ADS9224R front-end SPICE-model + ENOB-helper components: consumed directly
    # by the ngspice netlists / the noise-ENOB calc (sim-validation Tiers 2-3),
    # not by a closed-form scalar derivation.
    "circuit.ads9224r_module.ref_reservoir_c",
    "circuit.ths4551.vnoise_density", "circuit.ths4551.inoise_density",
    "circuit.ths4551.gbw", "circuit.ths4551.slew", "circuit.ths4551.zout",
    "circuit.ths4551.vos", "circuit.ths4551.aol_db", "circuit.ths4551.iq",
    "circuit.ads9224r_adc.csh", "circuit.ads9224r_adc.input_bw",
    "circuit.ads9224r_adc.snr_db", "circuit.ads9224r_adc.thd_db",
    "circuit.ads9224r_adc.transition_noise_lsb", "circuit.ads9224r_adc.vcm",
}


def check(params: sim_params.SimParams, rel_tol: float = 1e-6):
    """Returns (mismatches, missing): derived params off their derivation,
    and derived_from-tagged params with no registered derivation."""
    by_target = {d.target: d for d in DERIVATIONS}
    mismatches = []
    missing = []
    for entry in params.derived_entries():
        derivation = by_target.get(entry.path)
        if derivation is None:
            missing.append(entry.path)
            continue
        expected = derivation.fn(params)
        actual = entry.value
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            scale = max(abs(float(expected)), abs(float(actual)), 1e-30)
            ok = abs(float(expected) - float(actual)) <= rel_tol * scale \
                or abs(float(expected) - float(actual)) < 1e-12
        else:
            ok = expected == actual
        if not ok:
            mismatches.append((entry.path, actual, expected))
    return mismatches, missing


def registry_targets_exist(params: sim_params.SimParams) -> list[str]:
    return [d.target for d in DERIVATIONS if d.target not in params.entries]


def unused_components(params: sim_params.SimParams) -> list[str]:
    used = {path for d in DERIVATIONS for path in d.uses} | DIRECTLY_CONSUMED
    components = [
        e.path for e in params.entries.values()
        if e.path.startswith("circuit.") or e.path.startswith("motor_spec.")
    ]
    return [c for c in components if c not in used]


def update_params_file(config_path: Path, mismatches) -> None:
    """Rewrite only the value=... portion of mismatched parameter lines,
    preserving comments and provenance."""
    text = config_path.read_text()
    lines = text.splitlines(keepends=True)
    section = ""
    for idx, line in enumerate(lines):
        header = re.match(r"^\[([^\]]+)\]", line)
        if header:
            section = header.group(1)
            continue
        m = re.match(r"^([A-Za-z0-9_]+)\s*=\s*\{", line)
        if not m:
            continue
        path = f"{section}.{m.group(1)}" if section else m.group(1)
        for target, _actual, expected in mismatches:
            if path == target:
                new_value = repr(expected) if isinstance(expected, float) \
                    else str(expected)
                lines[idx] = re.sub(r"(value\s*=\s*)[^,}]+",
                                    lambda mm: mm.group(1) + new_value,
                                    line, count=1)
    config_path.write_text("".join(lines))


def measurement_checklist(params: sim_params.SimParams) -> str:
    used_by: dict[str, list[str]] = {}
    for d in DERIVATIONS:
        for component in d.uses:
            used_by.setdefault(component, []).append(d.target)
    lines = [
        "Bench measurement worksheet (Q7 board session / Q1 motor ID)",
        "=" * 72,
    ]
    for e in params.entries.values():
        if not (e.path.startswith("circuit.") or e.path.startswith("motor_spec.")):
            continue
        if e.status == "measured":
            continue
        targets = ", ".join(used_by.get(e.path, ["(unused?)"]))
        blocked = f" [{e.blocked_by}]" if e.blocked_by else ""
        lines.append(f"{e.path}")
        lines.append(f"  baseline: {e.value} {e.unit} ({e.status}){blocked}")
        if e.source:
            lines.append(f"  source:   {e.source}")
        lines.append(f"  unblocks: {targets}")
    lines.append("=" * 72)
    lines.append("After measuring: update values + statuses to 'measured',")
    lines.append("then run: python3 sim/scripts/derive_params.py --update")
    return "\n".join(lines)


def write_spice_params(params: sim_params.SimParams) -> Path:
    SPICE_PARAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["* AUTO-GENERATED from sim/config/params.toml - DO NOT EDIT"]
    for e in params.entries.values():
        if e.path.startswith("circuit.") and isinstance(e.value, (int, float)):
            name = e.path.replace("circuit.", "").replace(".", "_")
            lines.append(f".param {name}={e.value:g}")
    # Non-circuit values the netlists also need.
    lines.append(f".param adc_t_window={1.5 / params.value('adc.sclk'):g}")
    lines.append(f".param drv_amp_gain={params.value('drv8301.amp_gain'):g}")
    lines.append(f".param adc_vref={params.value('adc.vref'):g}")
    lines.append(f".param adc_acq_window_ads9224r="
                 f"{params.value('adc.ads9224r_acq_window_s'):g}")
    SPICE_PARAM_PATH.write_text("\n".join(lines) + "\n")
    return SPICE_PARAM_PATH


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--measurement-checklist", action="store_true")
    parser.add_argument("--write-spice-params", action="store_true")
    args = parser.parse_args()

    params = sim_params.load(args.config)

    if args.measurement_checklist:
        print(measurement_checklist(params))
        return 0
    if args.write_spice_params:
        print(f"wrote {write_spice_params(params)}")
        return 0

    mismatches, missing = check(params)
    for path in missing:
        print(f"NO DERIVATION REGISTERED: {path}")
    for path, actual, expected in mismatches:
        print(f"MISMATCH {path}: committed {actual} vs derived {expected}")

    if args.update and mismatches:
        update_params_file(params.config_path, mismatches)
        print(f"updated {len(mismatches)} value(s) in {params.config_path}")
        return 0

    if not mismatches and not missing:
        print(f"all {len(params.derived_entries())} derived parameters "
              "consistent with their circuit specs")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
