#!/usr/bin/env bash
# Configure + build the C++ bench module into sim/build/cpp.
set -eu

root="$(cd "$(dirname "$0")/../.." && pwd)"
src="$root/sim/cpp"
build="$root/sim/build/cpp"

python3 "$root/sim/scripts/gen_rtl_params.py" >/dev/null
cmake -G Ninja -S "$src" -B "$build" >/dev/null
ninja -C "$build"
