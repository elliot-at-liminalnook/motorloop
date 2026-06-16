# SPDX-License-Identifier: MIT
"""cocotb bus test for axil_regfile (robotics-ip-checklist stage 5).

Drives the wrapper with a real AXI-Lite master (cocotbext-axi): write the
control registers + read them back, confirm the control outputs reflect the
writes, and read the telemetry registers. Pairs with the formal AXI-Lite
protocol-legality proof (axil_regfile_fv).
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotbext.axi import AxiLiteMaster, AxiLiteBus


@cocotb.test()
async def register_round_trip(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    master = AxiLiteMaster(AxiLiteBus.from_prefix(dut, "s"), dut.clk,
                           dut.rst_n, reset_active_level=False)

    # Telemetry stimulus.
    dut.t_speed.value = 1234
    dut.t_fault_count.value = 0
    dut.t_mismatch_count.value = 0
    dut.t_angle.value = 0x555
    dut.t_noctw_count.value = 0
    dut.t_sector.value = 3
    dut.t_configured.value = 1
    dut.t_flags.value = 0

    dut.rst_n.value = 0
    for _ in range(6):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)

    # Write control registers (byte address = index*4).
    await master.write_dword(1 * 4, 500)     # duty
    await master.write_dword(2 * 4, 80)      # target_speed
    await master.write_dword(3 * 4, 0xABC)   # align
    await master.write_dword(0 * 4, 3)       # mode = FOC

    # Read them back over AXI.
    assert await master.read_dword(1 * 4) == 500
    assert await master.read_dword(2 * 4) == 80
    assert await master.read_dword(3 * 4) == 0xABC
    assert await master.read_dword(0 * 4) == 3

    # Control outputs reflect the writes (drop-in for uart_regfile).
    assert int(dut.r_duty.value) == 500
    assert int(dut.r_target_speed.value) == 80
    assert int(dut.r_align.value) == 0xABC
    assert int(dut.r_mode.value) == 3

    # Telemetry reads.
    assert await master.read_dword(16 * 4) == 1234            # speed
    assert await master.read_dword(18 * 4) == 0x555           # angle
    assert await master.read_dword(20 * 4) == ((1 << 3) | 3)  # {configured,sector}
