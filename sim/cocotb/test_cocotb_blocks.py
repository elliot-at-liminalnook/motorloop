# SPDX-License-Identifier: MIT
"""cocotb block-test runner (robotics-ip-checklist stage 4).

Builds each reusable block standalone with Verilator and runs its cocotb
testbench. Block tests exercise protocol/FSM/datapath behaviour in isolation -
complementary to the full co-sim and the formal proofs. Run with the cocotb
venv (cocotb does not support the system Python 3.14 directly):

    ~/.local/share/cocotb-venv/bin/python -m pytest sim/cocotb/test_cocotb_blocks.py -q

(The CI installs cocotb in a 3.12 venv and runs the same.)
"""

from pathlib import Path

import pytest
from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# (toplevel, [rtl sources], test_module, build parameters)
# drv_manager's deep timers are shrunk for the test via parameters - exactly the
# reuse payoff: a parameterized block is fast to exercise standalone.
BLOCKS = [
    ("pwm_generator", ["rtl/pwm_generator.v"], "tb_pwm_generator", {}),
    ("spi_drv_master", ["rtl/spi_drv_master.v"], "tb_spi_drv_master", {}),
    ("drv_manager", ["rtl/drv_manager.v"], "tb_drv_manager",
     {"CLK_HZ": 40000, "EN_READY_CYCLES": 40, "QUICK_RESET_CYC": 10}),
    ("axil_regfile", ["rtl/bus/axil_regfile.v"], "tb_axil_regfile", {}),
    ("wb_regfile", ["rtl/bus/wb_regfile.v"], "tb_wb_regfile", {}),
    ("axis_sampler", ["rtl/bus/axis_sampler.v"], "tb_axis_sampler", {}),
    # circle_limit_seq vs the combinational circle_limit: bit-exact equivalence
    # (the correctness claim behind the pipelined foc_core, stage 6.5).
    ("eq_circle_limit",
     ["sim/cocotb/eq_circle_limit.v", "rtl/circle_limit.v",
      "rtl/circle_limit_seq.v", "rtl/divider32.v"],
     "tb_circle_limit_seq", {}),
    ("as5047p_spi_master", ["rtl/as5047p_spi_master.v"],
     "tb_as5047p_spi_master", {"DRV_SPI_DIV": 4}),
    ("speed_iq_pi", ["rtl/speed_iq_pi.v"], "tb_speed_iq_pi", {}),
    # svpwm_seq vs combinational svpwm: bit-exact equivalence (task 4).
    ("eq_svpwm",
     ["sim/cocotb/eq_svpwm.v", "rtl/svpwm.v", "rtl/svpwm_seq.v"],
     "tb_svpwm_seq", {}),
    # the pipelined FOC datapath, latency-aware (stages 4 + 6.5).
    ("foc_core",
     ["rtl/foc_core.v", "rtl/sincos.v", "rtl/clarke.v", "rtl/park.v",
      "rtl/inv_park.v", "rtl/current_pi.v", "rtl/circle_limit_seq.v",
      "rtl/divider32.v", "rtl/svpwm_seq.v"],
     "tb_foc_core", {}),
]

# The reference-SoC wrapper (tier2-adoption §2) needs the full controller_top
# hierarchy, so its source list is the whole rtl tree (minus the foc_math test
# harness) + the AXI-Lite slave + the wrapper.
import glob  # noqa: E402
_SOC_SRC = ["rtl/soc/motorloop_axil_top.v", "rtl/bus/axil_regfile.v"] + [
    f"rtl/{Path(p).name}" for p in sorted(glob.glob(str(ROOT / "rtl" / "*.v")))
    if "foc_math" not in p]
BLOCKS.append(("motorloop_axil_top", _SOC_SRC, "tb_motorloop_axil_top", {}))


@pytest.mark.parametrize(
    "toplevel,sources,test_module,parameters", BLOCKS,
    ids=[b[0] for b in BLOCKS])
def test_block(toplevel, sources, test_module, parameters):
    runner = get_runner("verilator")
    runner.build(
        sources=[str(ROOT / s) for s in sources],
        hdl_toplevel=toplevel,
        parameters=parameters,
        build_dir=str(HERE / "build" / toplevel),
        build_args=["-Wno-fatal", "--trace",
                    f"-I{ROOT / 'rtl'}", f"-I{ROOT / 'rtl' / 'gen'}"],
        always=True,
    )
    runner.test(
        hdl_toplevel=toplevel,
        test_module=test_module,
        test_dir=str(HERE),
    )
