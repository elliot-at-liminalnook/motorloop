#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Configure + build the C++ bench module into sim/build/cpp.
set -eu

root="$(cd "$(dirname "$0")/../.." && pwd)"
src="$root/sim/cpp"
build="$root/sim/build/cpp"
python_bin="${PYTHON:-python3}"
python_exe="$("$python_bin" -c 'import sys; print(sys.executable)')"

"$python_bin" "$root/sim/scripts/gen_rtl_params.py" >/dev/null
cmake -G Ninja -S "$src" -B "$build" \
  -DPython3_EXECUTABLE="$python_exe" >/dev/null
ninja -C "$build"
