<!-- SPDX-License-Identifier: MIT -->
# Reproduce from scratch

The audit artifact: a clean clone → the pinned toolchain → every gate green with
the expected numbers. Versions are pinned in `toolchain.lock`.

## 0. One command (recommended)

```sh
# Containerized: pinned toolchain + deps, zero host setup.
docker build -f Containerfile -t motorloop .  # toolchain.lock pins the HDL stack
docker run --rm motorloop         # runs `make verify` (the full gate set)
```

Or locally, once the toolchain (§1) is installed:

```sh
make verify        # cores, lint, reuse, coverage, contracts, test, cocotb,
                   # formal, synth-check, asic, ipxact, docs - the CI gate set
make all           # verify + a full place&route Fmax run
make help          # list every target
```

`make` is the single entry point; the targets below are what it orchestrates.

## 1. Toolchain (one pinned tarball covers the HDL stack)

```sh
# yosys + Verilator + SymbiYosys + solvers + nextpnr-ecp5 + ecppack
cd ~ && TAG=2026-06-14 && STAMP=${TAG//-/}
curl -sL -o oss.tgz \
  "https://github.com/YosysHQ/oss-cad-suite-build/releases/download/${TAG}/oss-cad-suite-linux-x64-${STAMP}.tgz"
tar xzf oss.tgz
source ~/oss-cad-suite/environment      # puts all tools on PATH
pip install -r requirements.txt          # build/test/lint (pybind11, pytest, numpy, reuse)
# Verible (lint), Bender (consume), and the cocotb/docs venvs: see the Containerfile.
```

## 2. The gates (what `make verify`/`make all` run)

```sh
make bench     # cmake + Verilator + pybind11 co-sim (idempotent)
make test      # full pytest regression  -> 400 passed, 2 skipped
make cocotb    # per-block cocotb suite   -> 11 blocks pass
make lint      # Verible (enforced) + Verilator advisory
make reuse     # REUSE/SPDX               -> compliant
make coverage  # every core proven-or-sim-only
make contracts # every block has a finished datasheet (30/30)
make formal    # SymbiYosys               -> 12 PROVEN + 1 DOCUMENTED
make synth     # ECP5 PnR + report        -> fits -85F; Fmax ~64 MHz (meets 25 MHz)
make asic      # yosys ASIC-readiness      -> 14/14 clean (no latches/loops/multidriver)
make docs      # assemble + build the docs site (mkdocs --strict)
```

## 3. Pull a single IP module (the reuse acceptance test)

```sh
fusesoc library add motorloop .
fusesoc run --target lint motorloop:ip:pwm_generator   # standalone, no motorloop includes
bender script flist                                    # the same source map, Bender-side
```

## 4. The reference SoC (RISC-V drives the controller over AXI-Lite)

```sh
# See soc/README.md: build the LiteX SoC for the ULX3S, or run it in litex_sim.
make soc-sim    # (where LiteX is installed) RISC-V boots + drives the controller
```

## Expected numbers (current snapshot)

| Gate | Expected |
| --- | --- |
| Sim (`make test`) | 400 passed, 2 skipped |
| cocotb (`make cocotb`) | 11 blocks pass |
| Formal (`make formal`) | 12 PROVEN + 1 DOCUMENTED, 0 unexpected |
| Synth (`make synth`) | fits -85F; **Fmax ≈ 64 MHz** (meets the 25 MHz target) |
| ASIC (`make asic`) | 14/14 blocks gate-clean |
| Contracts (`make contracts`) | 30/30 finished datasheets |

If any differs, the change is not behavior-preserving — investigate before
merging. (The FOC datapath was pipelined to ~64 MHz with every step verified
bit-exact / latency-only against this baseline; see
`notes/foc-fmax-optimization-checklist.md`.)
