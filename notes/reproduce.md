<!-- SPDX-License-Identifier: MIT -->
# Reproduce from scratch

> **Document status:** Current · **Audience:** Developers and release reviewers · **Last reviewed:** 2026-07-12 · **Canonical for:** Full repository setup and reproduction

The audit artifact: a clean clone → the pinned toolchain → every required gate
green with generated evidence. Versions are pinned in `toolchain.lock`.

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

### Robot simulation and RL

The robot/RL stack has one stricter entry point. The local form is only a fast
precheck:

```sh
bash scripts/run_pre_gpu_tests.sh
```

Full verification must run in a CUDA environment:

```sh
bash sim/robot/setup_warp_pod.sh
source out/warp_env.sh
bash scripts/run_pre_gpu_tests.sh --require-gpu
```

That full command includes the complete CPU-only Verilator/component regression,
parallelized across CPU workers on the GPU host, followed by exact CPU robot
oracles and the CUDA rollout, PPO, and repeatability tiers. A local precheck exit
of zero is not authorization to start a long simulation or RL run. See
[`pre-gpu-test-entrypoint.md`](pre-gpu-test-entrypoint.md).

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
make test      # full component pytest regression
make cocotb    # per-block cocotb suite
make lint      # Verible (enforced) + Verilator advisory
make reuse     # REUSE/SPDX               -> compliant
make coverage  # every core proven-or-sim-only
make contracts # every packaged block has a finished contract
make formal    # SymbiYosys + generated proof report
make synth     # ECP5 PnR + generated fit/timing report
make asic      # yosys ASIC-readiness (no latches/loops/multidriver)
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

## Expected outcomes and authoritative evidence

| Gate | Passing outcome | Authoritative evidence |
| --- | --- | --- |
| Sim (`make test`) | All required collected tests pass; optional-tool skips are explicit | Current pytest output |
| cocotb (`make cocotb`) | Every collected block test passes | Current pytest output |
| Formal (`make formal`) | Every manifest expectation is met and no proof is unexpectedly incomplete | `formal/proof_report.md` |
| Synth (`make synth`) | The ECP5 design fits and meets the configured target clock | `synth/synth_report.md` |
| ASIC (`make asic`) | Every selected block is gate-clean under the smoke criteria | `synth/asic_smoke_report.md` |
| Contracts (`make contracts`) | Every packaged block has a complete contract | Checker output and `rtl/contracts/` |

Volatile counts and timing values belong in generated reports, not this runbook.
If the current command disagrees with its generated evidence, investigate before
merging or promoting the artifact.
