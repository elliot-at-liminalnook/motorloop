#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Shared stimulus format (realism-checklist 7.1): a scenario described as
a JSON-serializable timeline, runnable identically against the lockstep
bench (direct ports or the UART path) and — once hardware exists — against
the real controller via the same UART register map.

  {"name": "...", "duration": 1.2,
   "timeline": [{"t": 0.0, "action": "set_target_speed", "args": [80]},
                {"t": 0.0, "action": "set_mode", "args": [2]}]}

Supported actions: set_mode, set_duty, set_target_speed, set_align_offset,
set_open_loop, set_load_torque (bench-only physical input).
"""

from __future__ import annotations

import json
from pathlib import Path

# UART register map (mirrors rtl/uart_regfile.v).
UART_REG = {"set_mode": 0, "set_duty": 1, "set_target_speed": 2,
            "set_align_offset": 3}


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def save(stim: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(stim, indent=2) + "\n")


def _apply_direct(bench, action: str, args: list) -> None:
    getattr(bench, action)(*args)


def _apply_uart(bench, action: str, args: list, baud: float) -> None:
    if action == "set_open_loop":
        word, ramp = args
        for addr, val in ((4, (word >> 16) & 0xFFFF), (5, word & 0xFFFF),
                          (6, (ramp >> 16) & 0xFFFF), (7, ramp & 0xFFFF)):
            bench.uart_send([0x80 | addr, (val >> 8) & 0xFF, val & 0xFF])
            bench.run_for(4 * 10 / baud)
        return
    if action == "set_load_torque":      # physical input, never a register
        bench.set_load_torque(*args)
        return
    addr = UART_REG[action]
    val = int(args[0]) & 0xFFFF
    bench.uart_send([0x80 | addr, (val >> 8) & 0xFF, val & 0xFF])
    bench.run_for(4 * 10 / baud)


def run(bench, stim: dict, *, via_uart: bool = False,
        uart_baud: float = 115200.0) -> None:
    """Execute the timeline on an initialized bench (DRV configured)."""
    if via_uart:
        bench.uart_send([0x80 | 8, 0, 1])  # UART takes the control mux
        bench.run_for(4 * 10 / uart_baud)

    t_base = bench.time_s
    events = sorted(stim["timeline"], key=lambda e: e["t"])
    for event in events:
        target = t_base + event["t"]
        if target > bench.time_s:
            bench.run_for(target - bench.time_s)
        if via_uart and event["action"] in UART_REG or (
                via_uart and event["action"] == "set_open_loop"):
            _apply_uart(bench, event["action"], event["args"], uart_baud)
        elif via_uart and event["action"] == "set_load_torque":
            bench.set_load_torque(*event["args"])
        else:
            _apply_direct(bench, event["action"], event["args"])
    end = t_base + stim["duration"]
    if end > bench.time_s:
        bench.run_for(end - bench.time_s)


def closed_loop_step(target1: int, target2: int, align: int,
                     duration: float = 1.6) -> dict:
    """Canonical stimulus used by the comparator self-tests."""
    return {
        "name": f"closed_loop_step_{target1}_{target2}",
        "duration": duration,
        "timeline": [
            {"t": 0.0, "action": "set_align_offset", "args": [align]},
            {"t": 0.0, "action": "set_target_speed", "args": [target1]},
            {"t": 0.0, "action": "set_mode", "args": [2]},
            {"t": duration * 0.6, "action": "set_target_speed",
             "args": [target2]},
        ],
    }
