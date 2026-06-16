# SPDX-License-Identifier: MIT
"""cocotb bus test for axis_sampler (robotics-ip-checklist stage 5).

Receives beats with a real AXI-Stream sink (cocotbext-axi) and confirms the
packed telemetry matches the inputs. Pairs with the formal AXI-Stream
legality proof (axis_sampler_fv).
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotbext.axi import AxiStreamSink, AxiStreamBus


@cocotb.test()
async def streams_one_sample(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    sink = AxiStreamSink(AxiStreamBus.from_prefix(dut, "m"), dut.clk,
                         dut.rst_n, reset_active_level=False)

    dut.in_speed.value = 80
    dut.in_angle.value = 0x123
    dut.in_sector.value = 5
    dut.in_configured.value = 1
    dut.sample.value = 0

    dut.rst_n.value = 0
    for _ in range(6):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)

    # Strobe one sample.
    dut.sample.value = 1
    await RisingEdge(dut.clk)
    dut.sample.value = 0

    frame = await sink.recv()
    got = int.from_bytes(bytes(frame.tdata), "little")
    expected = (5 << 29) | (1 << 28) | (0x123 << 16) | 80
    assert got == expected, f"beat {got:#010x} != expected {expected:#010x}"
