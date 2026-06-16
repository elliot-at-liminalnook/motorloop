# SPDX-License-Identifier: MIT
# Timing constraints for the Quartus flow (release-and-portability §4.3).
# 25 MHz bring-up clock; TimeQuest reports the achieved Fmax.
create_clock -name clk -period 40.000 [get_ports clk]
derive_clock_uncertainty
