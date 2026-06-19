#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Build the co-sim bench for a specific motor's pole count (motor-selection
# checklist §0.3). Pole pairs are BUILD-TIME (POLE_PAIRS / speed_num / EXTRAP_NUM
# in rtl/gen/rtl_params.vh), so a motor with a different pole count needs a regen
# + re-Verilate - unlike the runtime BOM/sensor swaps.
#
#   ./build_motor.sh gm2804      # 7 pp  -> rebuild
#   ./build_motor.sh db42s03     # 4 pp  -> matches the default build
#   ./build_motor.sh maxon_ec45  # 8 pp  -> rebuild
#
# Restore the default build afterwards with: bash sim/scripts/build_bench.sh
set -euo pipefail
NAME="${1:?usage: build_motor.sh <gm2804|db42s03|maxon_ec45>}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

PP="$(python3 -c "import sys; sys.path.insert(0, '$ROOT/sim/tests'); \
from motors import MOTORS; print(MOTORS['$NAME'].pole_pairs)")"

echo "building bench for $NAME ($PP pole pairs) ..."
MOTORLOOP_POLE_PAIRS="$PP" bash "$ROOT/sim/scripts/build_bench.sh"
echo "done. run the motor's scenarios, then restore the default build with:"
echo "  bash $ROOT/sim/scripts/build_bench.sh"
