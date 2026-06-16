# SPDX-License-Identifier: MIT
"""cocotb bus test for wb_regfile (robotics-ip-checklist stage 5).

Drives a Wishbone B4 classic master by hand (the bus is small), writing the
control registers + reading them back, and reading telemetry. Pairs with the
formal Wishbone legality proof (wb_regfile_fv).
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


async def wb_write(dut, addr, data):
    dut.wb_adr.value = addr
    dut.wb_dat_w.value = data
    dut.wb_we.value = 1
    dut.wb_stb.value = 1
    dut.wb_cyc.value = 1
    await RisingEdge(dut.clk)
    while int(dut.wb_ack.value) == 0:
        await RisingEdge(dut.clk)
    dut.wb_stb.value = 0
    dut.wb_cyc.value = 0
    dut.wb_we.value = 0
    await RisingEdge(dut.clk)


async def wb_read(dut, addr):
    dut.wb_adr.value = addr
    dut.wb_we.value = 0
    dut.wb_stb.value = 1
    dut.wb_cyc.value = 1
    await RisingEdge(dut.clk)
    while int(dut.wb_ack.value) == 0:
        await RisingEdge(dut.clk)
    val = int(dut.wb_dat_r.value)
    dut.wb_stb.value = 0
    dut.wb_cyc.value = 0
    await RisingEdge(dut.clk)
    return val


@cocotb.test()
async def register_round_trip(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    for sig in ("wb_adr", "wb_dat_w", "wb_we", "wb_stb", "wb_cyc"):
        getattr(dut, sig).value = 0
    dut.t_speed.value = 1234
    dut.t_fault_count.value = 0
    dut.t_mismatch_count.value = 0
    dut.t_angle.value = 0x555
    dut.t_noctw_count.value = 0
    dut.t_sector.value = 3
    dut.t_configured.value = 1
    dut.t_flags.value = 0

    dut.rst_n.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    await wb_write(dut, 1, 500)      # duty
    await wb_write(dut, 2, 80)       # target_speed
    await wb_write(dut, 0, 3)        # mode = FOC

    assert await wb_read(dut, 1) == 500
    assert await wb_read(dut, 2) == 80
    assert await wb_read(dut, 0) == 3
    assert int(dut.r_duty.value) == 500
    assert int(dut.r_mode.value) == 3
    assert await wb_read(dut, 16) == 1234            # speed
    assert await wb_read(dut, 20) == ((1 << 3) | 3)  # {configured, sector}
