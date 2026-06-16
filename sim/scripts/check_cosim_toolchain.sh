#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Toolchain check for the lockstep verification bench (see notes/architecture.md).
# Required: everything the C++ bench, RTL verilation, and pytest suite need.
# Optional: oracle and debug tooling that the critical path can run without.
set -u

missing=0

check_bin() {
  local req="$1" name="$2"
  shift 2

  if command -v "$name" >/dev/null 2>&1; then
    printf '%-14s %s\n' "$name" "$(command -v "$name")"
    "$name" "$@" 2>&1 | sed 's/^/  /' | head -n 1
  else
    printf '%-14s missing (%s)\n' "$name" "$req"
    if [ "$req" = "required" ]; then
      missing=1
    fi
  fi
}

check_py_module() {
  local req="$1" name="$2"

  if python3 -c "import $name" >/dev/null 2>&1; then
    printf '%-14s python module %s\n' "$name" \
      "$(python3 -c "import $name; print(getattr($name, '__version__', 'ok'))")"
  else
    printf '%-14s missing python module (%s)\n' "$name" "$req"
    if [ "$req" = "required" ]; then
      missing=1
    fi
  fi
}

echo "== required (lockstep bench) =="
check_bin required python3 --version
check_bin required g++ --version
check_bin required cmake --version
check_bin required ninja --version
check_bin required verilator --version
check_py_module required pybind11
check_py_module required pytest

echo
echo "== required (oracle tier) =="
check_bin required omc --version

echo
echo "== required (derivation tier) =="
check_bin required ngspice --version

echo
echo "== optional =="
check_py_module optional matplotlib
check_bin optional gtkwave --version
# OMSimulator is intentionally optional: FMI is off the critical path.
check_bin optional OMSimulator --version

exit "$missing"
