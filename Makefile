# SPDX-License-Identifier: MIT
# One-command reproducibility (tier2-adoption-checklist §3A). Every target
# orchestrates an existing script - this file is the single entry point, not a
# reimplementation. `make all` reproduces the full result set; `make help` lists
# targets. Toolchain is pinned in toolchain.lock; see notes/reproduce.md.
SHELL := /bin/bash

# Tool locations (override on the command line if yours differ).
OSS       ?= $(HOME)/oss-cad-suite/environment        # yosys/sby/nextpnr/verilator
COCOTB_PY ?= $(HOME)/.local/share/cocotb-venv/bin/python
MKDOCS    ?= $(HOME)/.local/share/docs-venv/bin/mkdocs
LITEX_PY  ?= $(HOME)/litex-venv/bin/python             # the LiteX install (soc/)
# Sourcing OSS CAD shadows the system numpy, so the pytest suite uses the system
# python3 and ONLY the synth/formal targets source $(OSS) (in-recipe).

.PHONY: help all verify deps cores bench test cocotb lint reuse coverage \
        contracts version portability formal synth synth-check asic fmax ipxact \
        bender docs clean soc-sim soc-build

help:  ## list targets
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) | \
	  awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

## --- generators (single source -> cores / bender / ip-xact / docs) ---
cores:  ## (re)generate FuseSoC cores + Bender.yml
	python3 cores/gen_cores.py
ipxact:  ## generate + (best-effort) validate IP-XACT
	python3 scripts/gen_ipxact.py
docs:  ## assemble site-src/ and build the docs site
	python3 scripts/gen_docs.py
	$(MKDOCS) build --strict

## --- lint / hygiene gates (fast, no heavy tools) ---
lint:  ## Verible (enforced) + Verilator advisory
	bash rtl/lint/run_verible.sh
reuse:  ## REUSE/SPDX licence gate
	reuse lint
coverage:  ## every core proven-or-sim-only
	python3 formal/check_coverage.py
contracts:  ## every block has a finished datasheet
	python3 scripts/check_contracts.py
version:  ## release version is consistent (CITATION/cores/IP-XACT/CHANGELOG)
	python3 scripts/check_version.py
portability:  ## RTL maps to Xilinx/Intel/Gowin (yosys, resource estimates)
	source $(OSS) && python3 synth/portability.py --check

## --- build + simulate (system python; do NOT source OSS - numpy shadow) ---
deps:  ## verify the toolchain is present
	bash sim/scripts/check_cosim_toolchain.sh
	bash formal/check_formal_toolchain.sh
bench:  ## build the C++/Verilator co-sim bench
	bash sim/scripts/build_bench.sh
test: bench  ## full pytest regression (system python3 + numpy)
	python3 -m pytest sim/tests -q
cocotb:  ## per-block cocotb suite
	$(COCOTB_PY) -m pytest sim/cocotb/test_cocotb_blocks.py -q

## --- proofs / synthesis / ASIC (need the OSS CAD Suite, sourced in-recipe) ---
formal:  ## run + check all formal proofs
	source $(OSS) && python3 formal/run_formal.py --check
synth:  ## ECP5 synth + place&route + post-route Fmax
	source $(OSS) && python3 synth/run_synth.py
synth-check:  ## ECP5 synthesis-only gate (maps + fits)
	source $(OSS) && python3 synth/run_synth.py --check
asic:  ## yosys ASIC-readiness smoke (no latches/loops/multidriver)
	source $(OSS) && python3 synth/asic_smoke.py --check
fmax:  ## per-module standalone Fmax (FOC blocks)
	source $(OSS) && python3 synth/fmax_module.py
bender:  ## resolve the Bender source list
	bender script flist > /dev/null && echo "bender: resolved OK"

## --- reference SoC (LiteX/RISC-V over AXI-Lite; needs the LiteX install) ---
soc-sim:  ## run the reference SoC in litex_sim (RISC-V boots + the controller)
	source $(OSS) && $(LITEX_PY) soc/motorloop_soc.py --sim
soc-build:  ## build the reference SoC gateware (ULX3S bitstream)
	source $(OSS) && $(LITEX_PY) soc/motorloop_soc.py --build

## --- aggregate ---
verify: cores lint reuse coverage contracts version test cocotb formal synth-check asic portability ipxact docs  ## the CI gate set
all: verify synth  ## verify + a full place&route Fmax run

clean:  ## remove generated/build artifacts
	rm -rf build sim/build formal/work synth/work sim/cocotb/build \
	       sim/cocotb/sim_build site site-src
