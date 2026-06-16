# SPDX-License-Identifier: MIT
"""FOC stage 9.1: a closed-loop FOC run driven end-to-end over the UART
register file (mirrors the six-step S11 scenario). The host writes the FOC
align, the speed target, takes over the control mux, and commands mode 3; the
controller's outer speed loop then spins the PMSM. Telemetry is read back over
the same link. Placeholder motor params.
"""

from __future__ import annotations

import statistics

from bench_factory import (foc, expected_init_time, uart_write_frame,
                           uart_read_frame)


def _uart_write(b, addr, value, params):
    b.uart_send(uart_write_frame(addr, value))
    b.run_for(4 * 10 / params.value("rtl.uart_baud"))


def _uart_read(b, addr, params):
    b.uart_take_received()
    b.uart_send(uart_read_frame(addr))
    b.run_for(6 * 10 / params.value("rtl.uart_baud"))
    data = b.uart_take_received()
    assert len(data) >= 2, f"no UART response: {data}"
    return (data[-2] << 8) | data[-1]


def test_foc_closed_loop_over_uart(bldcsim, params):
    b = bldcsim.Bench(foc(params))
    b.run_for(expected_init_time(params))
    assert b.configured

    _uart_write(b, 3, int(params.value("foc.align_offset")), params)  # align
    _uart_write(b, 2, 70, params)                # target speed
    _uart_write(b, 8, 1, params)                 # UART takes the control mux
    _uart_write(b, 0, 3, params)                 # mode 3 = FOC
    b.run_for(1.2)

    # Average over a window (instantaneous dq currents carry ripple).
    speeds, ids = [], []
    for _ in range(1000):
        b.run_for(2e-4)
        speeds.append(b.omega)
        ids.append(b.foc_id)
    assert b.shoot_through_violations == 0
    assert statistics.mean(speeds) > 55, (
        f"UART-driven FOC failed to spin: {statistics.mean(speeds):.1f}")
    # Field orientation: id stays a small fraction of the torque current
    # (precise id regulation is covered by the stage-5/6 tests; here the point
    # is that the whole loop runs end-to-end over the UART link).
    assert abs(statistics.mean(ids)) < 20, (
        f"d-axis current not held near 0: {statistics.mean(ids):.1f}")

    # Telemetry readback agrees with the bench probe.
    speed = _uart_read(b, 16, params)
    assert abs(speed - b.speed) <= max(8, 0.15 * max(b.speed, 1))
    status = _uart_read(b, 20, params)
    assert status & 0x8, "configured bit missing in UART status"
    echoed = _uart_read(b, 2, params)
    assert echoed == 70
