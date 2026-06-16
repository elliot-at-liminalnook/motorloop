#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Verify the open-source formal toolchain (foc/formal-checklist stage 0.2).
# Required: yosys + sby (SymbiYosys) + yosys-smtbmc + at least one SMT solver.
# The YosysHQ OSS CAD Suite bundles all of these; if it is installed at
# ~/oss-cad-suite we add it to PATH automatically.
set -u

if [ -d "$HOME/oss-cad-suite/bin" ]; then
  export PATH="$HOME/oss-cad-suite/bin:$PATH"
fi

ok=0
need() {
  local name="$1"; shift
  if command -v "$name" >/dev/null 2>&1; then
    printf "  %-14s %s\n" "$name" "$("$@" 2>&1 | head -1)"
  else
    printf "  %-14s MISSING\n" "$name"
    ok=1
  fi
}
opt() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    printf "  %-14s present\n" "$name"
    return 0
  fi
  printf "  %-14s (optional, absent)\n" "$name"
  return 1
}

echo "Required:"
need yosys yosys --version
need sby sby --help >/dev/null 2>&1 && printf "  %-14s present\n" "sby" \
  || { command -v sby >/dev/null 2>&1 || { printf "  %-14s MISSING\n" "sby"; ok=1; }; }
need yosys-smtbmc yosys-smtbmc --help >/dev/null 2>&1 \
  && printf "  %-14s present\n" "yosys-smtbmc" || true

echo "SMT / proof engines (need at least one):"
have_solver=1
for s in boolector bitwuzla yices-smt2 z3; do
  if opt "$s"; then have_solver=0; fi
done
opt abc || true
[ "$have_solver" -ne 0 ] && { echo "  !! no SMT solver found"; ok=1; }

if [ "$ok" -eq 0 ]; then
  echo "formal toolchain: OK"
else
  echo "formal toolchain: INCOMPLETE"
  echo "  install the YosysHQ OSS CAD Suite into ~/oss-cad-suite:"
  echo "    https://github.com/YosysHQ/oss-cad-suite-build/releases"
fi
exit "$ok"
