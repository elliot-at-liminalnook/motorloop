# SPDX-License-Identifier: MIT
"""cocotb equivalence test for svpwm_seq (FOC Fmax-optimization checklist, task 4).

svpwm_seq must be BIT-EXACT to the combinational svpwm reference for every input
- that is what lets foc_core use the pipelined variant without changing
behaviour. The eq_svpwm wrapper runs both; this drives a sweep (edges, the
clamp boundary, axes, random (valpha, vbeta)) and asserts the three per-leg
duties agree when svpwm_seq pulses `done`.
"""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


async def _run_one(dut, va, vb):
    dut.valpha.value = va & 0x3FFFF
    dut.vbeta.value = vb & 0x3FFFF
    await RisingEdge(dut.clk)
    ref = int(dut.ref_duty3.value)            # combinational, on held inputs
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    for _ in range(20):
        await RisingEdge(dut.clk)
        if int(dut.seq_done.value) == 1:
            break
    else:
        assert False, f"svpwm_seq never asserted done for ({va},{vb})"
    seq = int(dut.seq_duty3.value)
    assert seq == ref, (f"mismatch ({va},{vb}): seq={seq:#014x} ref={ref:#014x}")


@cocotb.test()
async def seq_matches_combinational(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.valpha.value = 0
    dut.vbeta.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # full-scale is ~ +-PWM_HALF_PERIOD (625); sweep beyond to exercise the clamp
    E = [0, 1, -1, 312, -312, 625, -625, 624, 626, 900, -900, 200, -200]
    for a in E:
        for b in E:
            await _run_one(dut, a, b)

    rng = random.Random(0xABCDE)
    for _ in range(500):
        await _run_one(dut, rng.randint(-1200, 1200), rng.randint(-1200, 1200))
