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
RL_PY     ?= $(HOME)/rl-venv/bin/python                # MuJoCo+SB3 RL env (sim/rl)
LITEX_PY  ?= $(HOME)/litex-venv/bin/python             # the LiteX install (soc/)
# Sourcing OSS CAD shadows the system numpy, so the pytest suite uses the system
# python3 and ONLY the synth/formal targets source $(OSS) (in-recipe).

.PHONY: help all verify deps cores bench test cocotb lint reuse coverage \
        contracts version portability formal synth synth-check asic fmax ipxact \
        bender docs clean soc-sim soc-build compare ads9224r stress motors rl-figures rl-train rl-eval rl-dodge-train rl-dodge-eval rl-combat-train rl-combat-eval robot \
        gpu-baseline gpu-mjx-train gpu-adversarial gpu-codesign gpu-coevolve gpu-selfplay \
        gpu-match gpu-parity gpu-rederive gpu-extra gpu-e2e gpu-validate \
        gpu-residual gpu-rma gpu-robust-codesign gpu-active-id codesign-rs

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
compare: bench  ## part-comparison study: render the 10 sensor/ADC figures
	python3 sim/scripts/gen_comparison_figures.py
ads9224r:  ## open ADS9224R module: regenerate schematic + figures
	python3 sim/scripts/gen_ads9224r_sch.py
	python3 sim/scripts/gen_ads9224r_figures.py
stress: bench  ## extreme-scenario / stress study: render the 11 stress figures
	python3 sim/scripts/gen_stress_figures.py
motors:  ## motor-selection study: render the motor-comparison figures
	python3 sim/scripts/gen_motor_figures.py
rl-figures:  ## RL motor-coupling figures (system python, no torch)
	python3 sim/scripts/gen_rl_figures.py
rl-train:  ## train the RL locomotion policy (needs ~/rl-venv; see requirements-rl.txt)
	MUJOCO_GL=osmesa $(RL_PY) sim/rl/train.py --steps 1500000 --n-envs 16
rl-eval:  ## eval + render the trained RL policy
	MUJOCO_GL=osmesa $(RL_PY) sim/rl/eval.py --model sim/build/rl/ppo_HalfCheetah-v5_db42s03.zip --video --tag halfcheetah_db42
	MUJOCO_GL=osmesa $(RL_PY) sim/rl/render_rollout.py --traj sim/build/rl/halfcheetah_db42_traj.npz --tag halfcheetah_db42
rl-dodge-train:  ## train the dodge-balance quadruped (perception + threats + curriculum)
	MUJOCO_GL=osmesa $(RL_PY) sim/rl/train_dodge.py --steps 2000000 --n-envs 16 --max-difficulty 0.6
rl-dodge-eval:  ## eval + render the dodge policy (objects flying at the legs)
	MUJOCO_GL=osmesa $(RL_PY) sim/rl/eval_dodge.py --model sim/build/rl/ppo_dodge.zip --difficulty 0.6 --video --tag dodge_after
	MUJOCO_GL=osmesa $(RL_PY) sim/rl/render_rollout.py --traj sim/build/rl/dodge_after_traj.npz --tag dodge_after
rl-combat-train:  ## train the combat skill-ladder: stand -> hop -> dodge (see notes/rl-combat-dodge-report.md)
	MUJOCO_GL=osmesa $(RL_PY) sim/rl/train_combat.py --steps 1500000 --max-difficulty 0.0 --tag combat_stand
	MUJOCO_GL=osmesa $(RL_PY) sim/rl/train_combat.py --steps 1200000 --max-difficulty 0.0 --hop-reward \
	  --init-model sim/build/rl/ppo_combat_stand.zip --tag combat_hop
	MUJOCO_GL=osmesa $(RL_PY) sim/rl/train_combat.py --steps 2500000 --max-difficulty 0.2 --hop-reward --no-lethal \
	  --init-model sim/build/rl/ppo_combat_hop.zip --tag combat_h2
rl-combat-eval:  ## eval/render the hopper (high-steps) + an honest dodge engagement
	MUJOCO_GL=osmesa $(RL_PY) sim/rl/eval_combat.py --model sim/build/rl/ppo_combat_hop.zip --difficulty 0.0 --video --tag combat_hop
	MUJOCO_GL=osmesa $(RL_PY) sim/rl/render_rollout.py --traj sim/build/rl/combat_hop_traj.npz --tag combat_hop
	MUJOCO_GL=osmesa $(RL_PY) sim/rl/eval_combat.py --model sim/build/rl/ppo_combat_h2.zip --difficulty 0.13 --video --tag combat_after
	MUJOCO_GL=osmesa $(RL_PY) sim/rl/render_rollout.py --traj sim/build/rl/combat_after_traj.npz --tag combat_after
robot:  ## generate + prove the parametric body + co-design (robot.toml -> MJCF -> optimize)
	$(RL_PY) sim/robot/gen_robot_mjcf.py
	MUJOCO_GL=osmesa $(RL_PY) sim/robot/prove_robot.py
	MUJOCO_GL=osmesa $(RL_PY) sim/robot/train_mjx.py --smoke
	MUJOCO_GL=osmesa $(RL_PY) sim/robot/optimize_design.py --gens 8
	MUJOCO_GL=osmesa $(RL_PY) sim/robot/coevolve.py --rounds 6
	MUJOCO_GL=osmesa $(RL_PY) sim/robot/match_env.py --prove

## --- GPU/MJX co-design (run on a CUDA box; pip install -r requirements-gpu.txt;
##     see notes/gpu-runbook.md for provision -> install -> run order -> wall-clock) ---
gpu-mjx-train: gpu-baseline  ## alias: Phase 1 MJX baseline policy
gpu-baseline:  ## Phase 0/1: throughput baseline + train the MJX baseline locomotion policy
	python3 sim/robot/bench_throughput.py
	python3 sim/robot/train_codesign.py --steps 3000000
gpu-parity:  ## Phase 1/9 gate: MJX<->MuJoCo parity (qpos/qvel/reward + SPARC twin)
	python3 sim/robot/test_parity.py
gpu-adversarial:  ## Stage B: warm-start the adversarial design-conditioned (SPARC) policy
	python3 sim/robot/train_adversarial.py --steps 12000000 --resume /root/proj/out/universal_ckpt.pkl
gpu-codesign:  ## Phases 2/3/5: universal policy + trained-return co-design + Pareto
	python3 sim/robot/codesign_gpu.py --steps 4000000
gpu-rederive:  ## Phase R1/R7: re-derive design rankings under the calibrated sim (robust/CVaR)
	python3 sim/robot/rederive_r7.py
gpu-coevolve:  ## Phase 4/8b: co-evolve two generated bodies (real-physics, HoF + abs benchmark)
	MUJOCO_GL=osmesa python3 sim/robot/coevolve.py --rounds 6
gpu-selfplay:  ## Phase 4: MJX self-play with a Hall-of-Fame league (two-policy)
	python3 sim/robot/selfplay_mjx.py --steps 3000000
gpu-match:  ## Phase 4: single-learner MJX SPARC match (force-weighted damage)
	python3 sim/robot/match_mjx.py --steps 4000000
gpu-extra:  ## Phases 6/7: topology GA + differentiable co-design
	python3 sim/robot/codesign_extra.py
	python3 sim/robot/codesign_diff.py
gpu-e2e:  ## lightweight instrumented end-to-end loop check (profiles each stage; appends e2e_history.jsonl)
	CODESIGN_OUT=$${CODESIGN_OUT:-sim/build/gpu} python3 sim/robot/e2e.py
gpu-validate:  ## tiny sequential leak-test of EVERY GPU stage before any long run (E2E-first)
	CODESIGN_OUT=$${CODESIGN_OUT:-sim/build/gpu} bash sim/robot/validate_gpu.sh

## --- Real2Sim2Real (Phase R/RS): framework-now, sim-to-sim verified (CPU, no hardware) ---
codesign-rs:  ## run ALL the Phase-R/RS sim-to-sim self-tests (reality-gap calibrated co-design)
	$(RL_PY) sim/robot/reality_gap.py
	$(RL_PY) sim/robot/design_codec.py
	$(RL_PY) sim/robot/domain_model.py
	$(RL_PY) sim/robot/hardware_id.py
	MUJOCO_GL=osmesa $(RL_PY) sim/robot/test_contact.py
	$(RL_PY) sim/robot/robust_codesign.py
	$(RL_PY) sim/robot/reality_gap_eval.py
	$(RL_PY) sim/robot/multifidelity.py
	$(RL_PY) sim/robot/nsga2.py
gpu-residual:  ## RS2/RS3: learned actuator + contact residuals (sim-to-sim; bench fit hardware-gated)
	$(RL_PY) sim/robot/actuator_residual.py
	$(RL_PY) sim/robot/contact_residual.py
gpu-rma:  ## RS4: teacher->student online adaptation (RMA) — adaptation gap (sim-to-sim)
	$(RL_PY) sim/robot/adaptive_policy.py
gpu-robust-codesign:  ## R6/RS6: CVaR robust ranking + MAP-Elites QD archive of robust bodies
	$(RL_PY) sim/robot/robust_codesign.py
gpu-active-id:  ## RS5/RS8: info-gain test selection + proxy/nominal/robust ranking correlation
	$(RL_PY) sim/robot/reality_gap_eval.py

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
	       sim/cocotb/sim_build site site-src sim/robot/model.xml
