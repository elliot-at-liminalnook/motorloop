# SPDX-License-Identifier: MIT
"""cocotb equivalence test for circle_limit_seq (robotics-ip-checklist stage 6.5).

The sequential, multi-cycle circle limiter must be BIT-EXACT to the
combinational `circle_limit` reference for every input - that is what lets
foc_core be pipelined (Fmax 3.3 -> 41 MHz) without changing its behaviour. The
`eq_circle_limit` wrapper runs both; this drives a sweep (edges, the saturation
boundary, axes, random) and asserts the outputs agree when the sequential block
pulses `done`. circle_limit itself is DOCUMENTED-not-proven (the isqrt+divide is
intractable for the open SMT engines), so this equivalence is its verification.
"""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


def _s18(v):
    v &= 0x3FFFF
    return v - 0x40000 if v & 0x20000 else v


async def _run_one(dut, vd, vq):
    dut.vd_in.value = vd & 0x3FFFF
    dut.vq_in.value = vq & 0x3FFFF
    await RisingEdge(dut.clk)
    # reference is combinational on the held inputs
    ref_vd, ref_vq = _s18(int(dut.ref_vd.value)), _s18(int(dut.ref_vq.value))
    ref_sat = int(dut.ref_sat.value)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(200):
        await RisingEdge(dut.clk)
        if int(dut.seq_done.value) == 1:
            break
    else:
        assert False, f"circle_limit_seq never asserted done for ({vd},{vq})"

    seq_vd, seq_vq = _s18(int(dut.seq_vd.value)), _s18(int(dut.seq_vq.value))
    seq_sat = int(dut.seq_sat.value)
    assert (seq_vd, seq_vq, seq_sat) == (ref_vd, ref_vq, ref_sat), (
        f"mismatch for ({vd},{vq}): seq=({seq_vd},{seq_vq},sat{seq_sat}) "
        f"ref=({ref_vd},{ref_vq},sat{ref_sat})")


@cocotb.test()
async def seq_matches_combinational(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.vd_in.value = 0
    dut.vq_in.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    VLIM = 594
    edges = [0, 1, -1, VLIM, -VLIM, VLIM - 1, VLIM + 1, 419, 420, 421,
             2500, -2500, 2499, -2499, 100, -100]
    for a in edges:
        for b in edges:
            await _run_one(dut, a, b)

    # boundary band (|v| ~ VLIM) + axes + random
    for a in range(-2500, 2501, 53):
        await _run_one(dut, a, 0)
        await _run_one(dut, 0, a)

    rng = random.Random(0xC0FFEE)
    for _ in range(400):
        await _run_one(dut, rng.randint(-2500, 2500), rng.randint(-2500, 2500))
