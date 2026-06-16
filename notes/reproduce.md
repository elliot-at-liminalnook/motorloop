<!-- SPDX-License-Identifier: MIT -->
# Reproduce from scratch

The audit artifact: a clean clone → the pinned toolchain → all four gates green,
with the expected numbers. Versions are pinned in `toolchain.lock`.

## 1. Toolchain (one pinned tarball covers everything)

```sh
# yosys + Verilator + SymbiYosys + solvers + nextpnr-ecp5 + ecppack
cd ~ && TAG=2026-06-14 && STAMP=${TAG//-/}
curl -sL -o oss.tgz \
  "https://github.com/YosysHQ/oss-cad-suite-build/releases/download/${TAG}/oss-cad-suite-linux-x64-${STAMP}.tgz"
tar xzf oss.tgz
source ~/oss-cad-suite/environment      # puts all tools on PATH
pip install pybind11 pytest
```

## 2. Build the co-sim bench

```sh
bash sim/scripts/build_bench.sh          # cmake + Verilator + pybind11 (idempotent)
```

## 3. The four gates

```sh
# (1) Lint
verilator --lint-only -Wall -Irtl -Irtl/gen rtl/controller_top.v \
  $(ls rtl/*.v | grep -v foc_math)

# (2) Simulation  -> expect: 401 passed, 1 skipped (test_synth needs yosys on PATH)
python3 -m pytest sim/tests -q

# (3) Formal      -> expect: 12 PROVEN + 1 DOCUMENTED, 0 unexpected
python3 formal/run_formal.py --check
python3 formal/gen_proof_report.py

# (4) Synth       -> expect: fits ECP5-85F; Fmax ~3.3 MHz (NOT 25 MHz; see report)
python3 synth/run_synth.py               # synth -> PnR -> bitstream + report
```

## 4. Pull a single IP module (the reuse acceptance test)

```sh
fusesoc library add motorloop .
fusesoc run --target lint motorloop:ip:pwm_generator   # standalone, no motorloop includes
```

## Expected numbers (the 2026-06-14 snapshot)

| Gate | Expected |
| --- | --- |
| Sim | 401 passed, 1 skipped |
| Formal | 12 PROVEN + 1 DOCUMENTED |
| Synth | fits -85F (I/O 10%, DSP 14%, ~14.3k LUT4); Fmax ≈ 3.3 MHz |

If any differs, the change is not behavior-preserving — investigate before
merging. (The leaf parameterization was validated byte-identical against this
baseline.)
