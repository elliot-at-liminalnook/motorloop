#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate FuseSoC CAPI2 .core files for the motorloop HDL library
(trusted-library-checklist stage 2.4).

Each reusable leaf gets its own core so a third party can pull a single module
standalone (`fusesoc run motorloop:ip:<name>`). Most leaves are self-contained
after parameterization (stage 1); the few composites list their children. The
top `motorloop:bldc:controller` core (the integrated system) is in
`../motorloop.core`.

Run from the repo root:  python3 cores/gen_cores.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cores"

# module -> (one-line description, [extra rtl files], [gen include files])
# The module's own rtl/<name>.v is always included; extras are child modules.
LEAVES = {
    # pure combinational datapath (self-contained)
    "clarke":       ("Clarke transform (3-phase -> alpha/beta), combinational", [], []),
    "park":         ("Park transform (alpha/beta -> d/q), combinational", [], []),
    "inv_park":     ("Inverse Park (d/q -> alpha/beta), combinational", [], []),
    "svpwm":        ("Space-vector PWM by min/max injection, combinational", [], []),
    "circle_limit": ("Voltage-circle magnitude limiter (isqrt), combinational", [], []),
    "commutation":  ("Six-step sector -> per-leg drive decode, combinational", [], []),
    # sequential leaves
    "sincos":       ("Quarter-wave sin/cos LUT (Q15), 16-bit angle in", [],
                     ["rtl/gen/sincos_init.vh"]),
    "current_pi":   ("Single-axis current PI with anti-windup clamp (PROVEN)", [], []),
    "speed_pi":     ("Six-step speed PI with down-slew limit", [], []),
    "speed_iq_pi":  ("FOC outer speed loop -> iq* with clamp (PROVEN)", [], []),
    "speed_meter":  ("Sector-edge speed measurement", ["rtl/divider32.v"], []),
    "divider32":    ("32-bit sequential divider", [], []),
    "open_loop_ramp": ("Open-loop commutation frequency ramp", [], []),
    "pwm_generator": ("Center-aligned 3-phase complementary PWM + dead-time "
                      "(shoot-through-free, PROVEN)", [], []),
    "drv_manager":  ("Gate-driver bring-up/SPI-config/fault FSM (legality PROVEN)",
                     [], []),
    "spi_drv_master": ("SPI master, mode-1 16-bit (gate driver)", [], []),
    "adc_spi_master": ("SPI master for the MCP3208 12-bit ADC", [], []),
    "as5047p_spi_master": ("AS5047P SPI angle reader (framing PROVEN)", [], []),
    "ads9224r_master": ("ADS9224R dual-simultaneous 16-bit ADC master "
                        "(framing PROVEN)", [], []),
    "as5600_pwm_capture": ("AS5600 PWM-output angle capture", [], []),
    "adc_sequencer": ("Sector-aware ADC schedule + FOC current sampling "
                      "(pulse well-formedness PROVEN)", [], []),
    "uart_rx":      ("8N1 UART receiver", [], []),
    "uart_tx":      ("8N1 UART transmitter", [], []),
    "uart_regfile": ("UART command/telemetry register file",
                     ["rtl/uart_rx.v", "rtl/uart_tx.v"], []),
    "foc_core":     ("FOC datapath: Clarke/Park/PI/circle-limit/inv-Park/SVPWM",
                     ["rtl/sincos.v", "rtl/clarke.v", "rtl/park.v",
                      "rtl/inv_park.v", "rtl/current_pi.v", "rtl/circle_limit.v",
                      "rtl/svpwm.v"], ["rtl/gen/sincos_init.vh"]),
}

CORE = """CAPI=2:
# SPDX-License-Identifier: MIT
name: motorloop:ip:{name}:0.1.0
description: "{desc}"
filesets:
  rtl:
    files:
{files}    file_type: verilogSource
{gen}targets:
  default: &default
    filesets: [rtl{gen_fs}]
    toplevel: {name}
  lint:
    <<: *default
    default_tool: verilator
    tools:
      verilator:
        mode: lint-only
"""


def main():
    OUT.mkdir(exist_ok=True)
    for name, (desc, extras, gens) in LEAVES.items():
        # Paths are relative to this .core file, which lives in cores/ - so
        # one level up to reach rtl/.
        files = "".join(f"      - ../{f}\n"
                        for f in [f"rtl/{name}.v", *extras])
        gen_block = ""
        gen_fs = ""
        if gens:
            gen_files = "".join(
                f"      - ../{g}: {{is_include_file: true}}\n" for g in gens)
            gen_block = (f"  gen:\n    files:\n{gen_files}"
                         f"    file_type: verilogSource\n")
            gen_fs = ", gen"
        (OUT / f"{name}.core").write_text(
            CORE.format(name=name, desc=desc, files=files,
                        gen=gen_block, gen_fs=gen_fs))
    print(f"wrote {len(LEAVES)} leaf cores to {OUT}")


if __name__ == "__main__":
    main()
