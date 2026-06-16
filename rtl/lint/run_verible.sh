#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Verible gate (robotics-ip-checklist stage 3): lint is enforced; format is
# advisory (verible-format is opinionated and would reformat the hand-written
# style wholesale, so we report rather than fail on it).
set -euo pipefail
cd "$(dirname "$0")/../.."

RULES="rtl/lint/.rules.verible_lint"
# foc_math.v is a verification harness, not shipped IP. Includes the bus
# wrappers (rtl/bus/) and the reference-SoC wrapper (rtl/soc/).
FILES=$(ls rtl/*.v rtl/bus/*.v rtl/soc/*.v 2>/dev/null | grep -v foc_math)

echo "[verible] lint (enforced)"
verible-verilog-lint --rules_config "$RULES" $FILES

echo "[verible] format (advisory)"
if verible-verilog-format --verify $FILES >/dev/null 2>&1; then
  echo "  all files match verible canonical format"
else
  echo "  note: some files are not in verible canonical format (advisory;"
  echo "        the library uses a hand-maintained style - lint is the gate)"
fi
echo "[verible] OK"
