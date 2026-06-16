<!-- SPDX-License-Identifier: MIT -->
# Tier-2 adoption checklist — runnable reference SoC, one-command repro, citable release, per-block contracts

Implements items 2–4 of [adoption-roadmap.md](adoption-roadmap.md): make the
library **trivial to adopt and to cite**. Ordered, precise, with exact files and
interfaces. North star unchanged: every task must make a real block easier to
*adopt*; no conformance for its own sake.

**Constraint:** the release step (§3B) involves `git tag` / a GitHub release /
minting a Zenodo DOI — those are **Elliot's actions**. This checklist *prepares*
every artifact (CITATION.cff, CHANGELOG, badge markdown, Zenodo metadata) so the
tag-and-publish is one click; it never commits, tags, or pushes.

## Recommended execution order (dependency-aware)

1. **§4 contracts** — independent, mechanical, unblocks the docs story and the
   SoC's `axil_regfile` datasheet.
2. **§3A one-command repro** (Makefile + container) — the foundation everything
   else plugs into.
3. **§2 reference SoC** — the headline; its build target plugs into the Makefile,
   its datasheet into §4.
4. **§3B citable release** — capstone, once the SoC works and the build is
   reproducible.

---

# §4 — A contract (datasheet) for every reusable block

**State:** 30 leaf cores; only `pwm_generator` and `foc_core` have a contract in
`rtl/contracts/`. The docs site (`mkdocs.yml` nav) lists only those two.

**Goal:** every reusable block ships a one-page contract — *claim · assumptions ·
interface · timing · parameters · proof-or-sim status · reuse* — rendered in the
docs catalog. A stranger should adopt a block without reading its RTL.

### 4.1 Scaffold generator (do the mechanical 80% automatically)
- [ ] Add `scripts/gen_contract_stubs.py`:
  - For each `*.core` at the repo root (skip `motorloop`), read the block name +
    one-line description from the `.core`.
  - Parse the module port list from `rtl/<name>.v` (or `rtl/bus/<name>.v`) —
    reuse the port parser in `synth/fmax_module.py` (`ports()`), which already
    extracts dir/signed/width/name. Emit the **Interface** table.
  - Pull the **proof status** from `formal/manifest.toml` (PROVEN/DOCUMENTED) or
    `formal/sim_only.toml` (the per-block reason) — one source, no hand-typing.
  - Emit a stub `rtl/contracts/<name>.md` with the section skeleton + the
    auto-filled interface/params/status, and `TODO:` markers for the prose
    (claim, assumptions, timing notes).
  - Mirror the existing two contracts' structure exactly (headings, the port
    table columns: Port | Dir | Width | Signed | Reset | Semantics).
- [ ] `# REUSE-IgnoreStart/End` around any SPDX string the generator emits into
  the stub (the established trap — see `gen_cores.py`/`asic_smoke.py`).

### 4.2 Fill the prose (the human 20%, per block)
- [ ] For each of the 28 blocks below, complete the `TODO:` prose: the **claim**
  (what it guarantees), **assumptions** (clock domain, input ranges, reset),
  **timing** (latency/throughput; cite the proof or the contracted timing), and
  **reuse notes** (deps, `fusesoc run` line). Group by kind to reuse wording:
  - *Combinational transforms:* `clarke`, `park`, `inv_park`, `svpwm`, `sincos`,
    `circle_limit`, `commutation` (bit-exact vs the Python reference;
    `test_foc_math`).
  - *Sequential datapath:* `circle_limit_seq`, `svpwm_seq`, `divider32`,
    `current_pi`, `speed_pi`, `speed_iq_pi`, `open_loop_ramp`, `speed_meter`.
  - *Protocol/FSM:* `spi_drv_master`, `adc_spi_master`, `as5047p_spi_master`,
    `ads9224r_master`, `as5600_pwm_capture`, `drv_manager`, `adc_sequencer`,
    `uart_rx`, `uart_tx`, `uart_regfile`.
  - *Bus wrappers:* `axil_regfile`, `wb_regfile`, `axis_sampler` (include the
    register map for the regfiles — reuse the `REGS` table from
    `scripts/gen_ipxact.py`, the one source).
- [ ] Each contract carries its **WaveDrom timing** where non-trivial; the
  protocol blocks reference the diagrams already in `site-src/timing.md`.

### 4.3 Wire into the docs site
- [ ] Make the `mkdocs.yml` Contracts nav complete. Prefer **auto-generating the
  nav**: have `scripts/gen_docs.py` glob `rtl/contracts/*.md` and emit the nav
  list (so a new contract appears without editing `mkdocs.yml`), or use
  `mkdocs-awesome-pages`/literate-nav. Keep it generated, not hand-maintained.
- [ ] `mkdocs build --strict` clean with all contracts.

### 4.4 Gate it
- [ ] Add a CI check (extend the coverage gate or a new `scripts/check_contracts.py`):
  **every core has a `rtl/contracts/<name>.md`** and it contains no `TODO:`.
  Fails CI on a block shipped without a finished datasheet — the §2 "manifest as
  truth source" discipline, applied to docs.
- **Done-when:** 30/30 blocks have a complete contract; the docs site renders all
  of them; CI fails on a missing/unfinished one.

---

# §3A — One-command reproducibility (Makefile + container)

**State:** `toolchain.lock` pins the OSS CAD Suite / Verilator / cmake / python /
pybind11. Build entry points are scattered: `sim/scripts/build_bench.sh`,
`python3 -m pytest sim/tests`, `formal/run_formal.py`, `synth/run_synth.py`,
`synth/asic_smoke.py`, `scripts/gen_docs.py`, `scripts/gen_ipxact.py`,
`cores/gen_cores.py`, the cocotb suite. No `Makefile`.

**Goal:** `make all` (or `make verify`) reproduces **every** result on a clean
machine; a container removes "works on my box."

### 3A.1 Root Makefile (the single orchestrator)
- [ ] Add `Makefile` with documented phony targets, each shelling to the existing
  script (don't reimplement — orchestrate):
  - `make deps` — verify the toolchain (`sim/scripts/check_cosim_toolchain.sh`,
    `formal/check_formal_toolchain.sh`) and the venvs (cocotb, docs).
  - `make bench` — `sim/scripts/build_bench.sh`.
  - `make test` — `python3 -m pytest sim/tests` (the 400-test suite).
  - `make cocotb` — the cocotb venv runs `sim/cocotb/test_cocotb_blocks.py`.
  - `make formal` — `formal/run_formal.py --check`.
  - `make lint` — Verible (`rtl/lint/run_verible.sh`) + the Verilator advisory.
  - `make synth` — `synth/run_synth.py` (ECP5 Fmax).
  - `make asic` — `synth/asic_smoke.py --check`.
  - `make fmax` — `synth/fmax_module.py` over the FOC blocks.
  - `make cores` / `make bender` / `make ipxact` / `make docs` — the generators.
  - `make reuse` — `reuse lint`.
  - `make coverage` — `formal/check_coverage.py`.
  - `make all` — the dependency-correct union (deps → cores → lint/reuse →
    bench → test → cocotb → formal → synth → asic → ipxact → docs).
  - `make clean` — remove `build/`, `sim/build/`, `formal/work/`, `synth/work/`,
    `sim/cocotb/build/`, `site/`, `site-src/` (the gitignored artifacts).
  - Each target sources `~/oss-cad-suite/environment` where needed **in a
    subshell** (it shadows system numpy — the pytest target must NOT source it).
- [ ] `make help` (self-documenting target list).

### 3A.2 Pinned Python deps
- [ ] Add `requirements.txt` (and a cocotb `requirements-cocotb.txt`) pinning
  pybind11, pytest, numpy, reuse, cocotb, cocotbext-axi, mkdocs + material +
  pymdownx — the exact versions in `toolchain.lock`'s spirit. CI + the container
  install from these.

### 3A.3 Container (Docker, with a Nix flake as the reproducible-by-hash option)
- [ ] `Containerfile` (Docker/Podman): base image, fetch the **exact** OSS CAD
  Suite tag from `toolchain.lock` (`OSS_CAD_SUITE`), install Verible + Bender +
  the pinned `requirements*.txt`, set `PATH`. Entry point runs `make all`.
- [ ] `.devcontainer/devcontainer.json` referencing the Containerfile, so VS Code
  / Codespaces "just works."
- [ ] (Optional, stronger) `flake.nix` pinning all tools by hash for
  bit-reproducible builds; document `nix develop` + `nix flake check`.
- [ ] `notes/reproduce.md` (already referenced in `toolchain.lock`): the canonical
  "clone → `docker build` → `make all`" steps + the no-container local path.

### 3A.4 CI uses the same one command
- [ ] Refactor `.github/workflows/ci.yml` to call the Makefile targets (so CI and
  local repro are the *same* path, not two definitions that drift). Keep the
  per-gate step names for readable logs.
- **Done-when:** a fresh checkout + `docker build . && docker run … make all` (or
  `make all` after `make deps`) reproduces tests + formal + synth + docs green,
  with zero manual setup, pinned to `toolchain.lock`.

---

# §2 — Runnable reference SoC (RISC-V drives the controller over AXI-Lite)

**State:** `rtl/bus/axil_regfile.v` is a standalone, formally-proven AXI4-Lite
slave whose control surface (`use_axi, r_mode, r_duty, r_target_speed, r_align,
r_ol_freq_word, r_ol_ramp_inc`) and telemetry inputs (`t_speed, t_fault_count,
t_mismatch_count, t_angle, t_noctw_count, t_sector, t_configured, t_flags`)
**mirror `uart_regfile`** — which `controller_top` already consumes (the wiring
template is the `uart_regfile` instance in `controller_top.v`). The register map
is the `REGS` source in `scripts/gen_ipxact.py`. It is **not yet wired into any
top**.

**Goal:** a LiteX (RISC-V) SoC with the controller as an AXI-Lite peripheral, C
firmware that spins the motor + streams telemetry, proven first in `litex_sim`
(CI-able) then on a real board (feeds Tier-1 hardware correlation).

### 2.1 RTL: the integration wrapper (`rtl/soc/motorloop_axil_top.v`)
- [ ] New module instantiating `axil_regfile` + `controller_top`:
  - Expose the **AXI4-Lite slave** ports at the top (the SoC connects here).
  - Connect `axil_regfile.r_mode → controller_top.ctrl_mode`,
    `r_duty → ctrl_duty`, `r_target_speed → ctrl_target_speed`,
    `r_ol_freq_word → ctrl_ol_freq_word`, `r_ol_ramp_inc → ctrl_ol_ramp_inc`,
    `r_align → ctrl_align_offset` (widths already match: 2/16/16/32/32/12).
  - Tie the platform straps to a chosen default BOM (zonri_drv8301):
    `ctrl_drv_hw_mode=0, ctrl_angle_spi_mode=0, ctrl_adc_dual_mode=0,
    ctrl_cur_norm_shift=…, ctrl_id_target=0`, and route `ctrl_iq_target`/
    `ctrl_foc_*` either from constants or from **added registers** (see 2.2).
  - Drive `controller_top.uart_rx_pin` tied off (or keep the UART too — the
    wrapper can expose both; `use_axi` selects the AXI register source).
  - Connect telemetry back: `controller_top` `speed/angle/sector/dbg_* →
    axil_regfile.t_*` exactly as the `uart_regfile` instance does.
  - Bring the controller's peripheral pins (gate driver SPI, ADC SPI/CONVST,
    angle, `inh/inl/en_gate`, `nfault/noctw`) to the wrapper top for the board.
- [ ] Verilator-lint + Verible clean; add a `.core` (`gen_cores.py`) +
    `Bender.yml` entry + the contract (§4) for `motorloop_axil_top`.
- [ ] **Extend the register map** if FOC torque control over the bus is wanted:
  add `iq_target`, `foc_speed_loop`, `foc_extrap` registers to `axil_regfile`
  **and** the `REGS` source in `gen_ipxact.py` (one source → IP-XACT + docs
  regenerate). Re-run the `axil_regfile` formal proof (protocol legality is
  width-agnostic, should stay PROVEN) and `tb_axil_regfile`.

### 2.2 Sim verification (CI-able, before any hardware)
- [ ] `sim/cocotb/tb_motorloop_axil_top.py` (cocotbext-axi master): reset →
  AXI-write `mode`, `target_speed`, `control.use_axi=1` → step the clock →
  AXI-read the telemetry registers and assert the controller responds
  (`configured` set, `sector`/`angle` advance when an angle stimulus is fed,
  fault counts sane). This proves the **CPU↔AXI↔controller** plumbing. Register
  it in `test_cocotb_blocks.py`.
- [ ] *Note honestly in the contract:* the cocotb test proves register-level
  integration; the **motor actually spinning** is proven by the C++ co-sim
  (cycle-accurate plant, already green) and by §2.4 hardware — the SoC test is
  not a plant sim.

### 2.3 LiteX SoC + firmware (the headline artifact, `soc/`)
- [ ] `soc/motorloop_soc.py`: a LiteX `SoCCore` (VexRiscv or PicoRV32) for a
  target board (ULX3S, matching `synth/` ECP5). Wrap `motorloop_axil_top` via
  LiteX `Instance()`; attach it as an **AXI-Lite peripheral** using LiteX's
  `AXILiteInterface` + the `wishbone2axilite`/`axilite` bridge, mapped at a CSR
  base. Add the source files via `platform.add_source(...)` (reuse the
  `synth_ecp5.ys` file list / the FuseSoC core).
  - *Path-of-least-resistance fallback:* LiteX is **Wishbone-native**, so
    `rtl/bus/wb_regfile.v` can attach with no bridge. Document both; lead with
    AXI-Lite (the roadmap's framing), offer Wishbone as the simpler variant.
- [ ] `soc/firmware/main.c`: bare-metal C using the generated `csr.h` MMIO base —
  configure the platform, set `mode=FOC`/`target_speed`, enable `use_axi`, then
  loop printing telemetry (speed/angle/sector/faults) over the LiteX UART.
- [ ] `soc/README.md`: exact build commands
  (`python3 soc/motorloop_soc.py --build`), the firmware build, and the
  `litex_sim` invocation.
- [ ] **`litex_sim` integration target:** run the SoC in LiteX's Verilator sim so
  the RISC-V boots, the firmware writes the registers, and telemetry reads back —
  a CI-able end-to-end "RISC-V drives the controller" proof (no real motor, but
  full gateware + CPU). Add `make soc-sim` to the Makefile.

### 2.4 Hardware bring-up (feeds Tier-1; manual, not CI)
- [ ] Build the LiteX gateware to ULX3S (`--build`), flash it + the firmware,
  wire a gate driver + BLDC, and spin it closed-loop. Capture telemetry over the
  CPU UART and (Tier-1) overlay phase/gate captures on the co-sim trace.
- **Done-when:** `litex_sim` shows the RISC-V configuring the controller and
  reading live telemetry over AXI-Lite (green in CI); `soc/README.md` reproduces
  it; the board build + a captured "it spins" log/scope shot are published.

---

# §3B — Citable release (Elliot executes the tag/publish)

**Goal:** a versioned, DOI-citable `v0.1.0` that researchers cite and developers
trust at a glance.

### 3B.1 Release artifacts to PREPARE (no git actions)
- [ ] `CITATION.cff` at the repo root: title, authors (Elliot
  <elliot@liminalnook.com>), license MIT, repo URL, version `0.1.0`, the
  (placeholder) Zenodo DOI, keywords (FOC, BLDC, HDL IP, formal verification).
- [ ] `CHANGELOG.md` `0.1.0` entry summarizing the bundle: parameterized leaf IP,
  29 FuseSoC cores + Bender + IP-XACT, 12 formal proofs, cocotb suite, the
  cycle-accurate co-sim, 8 platform BOMs, the pipelined FOC (foc_core 79 MHz,
  system 64 MHz), the reference SoC, OpenLane smoke. Follow Keep-a-Changelog.
- [ ] **README badges** (markdown ready to paste): CI status (ci.yml, formal.yml),
  REUSE-compliant, the Zenodo DOI badge, license, the docs-site link. Add a
  one-paragraph "what this is / isn't" (verified + proven + reference-SoC-run;
  **not** silicon-validated until hardware bring-up; gains placeholder pending
  motor ID) — honesty as a trust signal.
- [ ] `.zenodo.json` (or the GitHub-Zenodo metadata): title, creators, license,
  keywords, related identifiers — so the DOI mint is clean.
- [ ] Per-IP **semantic versioning**: the cores are already `…:0.1.0`; document
  the version-bump policy (when an interface/timing changes) in
  `notes/reproduce.md` or a `VERSIONING.md`.
- [ ] A short `notes/release-checklist.md` for Elliot: the exact steps —
  `git tag v0.1.0`, push, create the GitHub release (attach the docs site +
  proof report + synth report), enable the GitHub↔Zenodo hook, mint the DOI,
  paste the DOI back into `CITATION.cff`/README, re-tag if needed.

### 3B.2 Verify before handing off
- [ ] `reuse lint`, `make all`, `mkdocs build --strict` green on a clean checkout.
- [ ] Every new file (Makefile, Containerfile, soc/**, CITATION.cff, .zenodo.json)
  is REUSE-compliant (SPDX headers / REUSE.toml coverage for non-commentable
  files like JSON).
- **Done-when:** the release artifacts are complete + verified; Elliot can tag,
  release, and mint the DOI in one sitting; the README shows live badges + the
  DOI; `CITATION.cff` resolves.

---

## Cross-cutting verification ritual (after each section)
`make all` green (400 tests, 12 proofs, cocotb, synth, asic), `reuse lint`,
Verible, coverage gate, `mkdocs build --strict`. The reference SoC adds the
`litex_sim` integration test to the green set.

## What NOT to do
Don't fork the build logic between CI and the Makefile (one source). Don't
hand-maintain the docs nav or the register map (generate from the one source).
Don't claim "silicon-validated" until §2.4 hardware exists — the reference SoC in
sim proves *integration*, not the motor. Don't add a third bus to the SoC; one
working AXI-Lite (or Wishbone) path is the proof.

## Implemented (results)

All four sections done. **Board: ULX3S (ECP5 LFE5U-85F)** — matches the synth
flow's device, fully open toolchain (yosys/nextpnr-ecp5), first-class LiteX
support; 50 MHz sys-clk << the 64 MHz Fmax.

- **§4 contracts** — `scripts/gen_contract_stubs.py` scaffolds; all **30 blocks**
  have a finished datasheet (no `TODO:`); nav auto-generated by `gen_docs.py`;
  `scripts/check_contracts.py` gates CI; `mkdocs build --strict` clean.
- **§3A repro** — root `Makefile` (`make all/verify/test/...`, tested),
  `requirements*.txt`, `Containerfile` + `.devcontainer/`, refreshed
  `notes/reproduce.md`; CI routes `reuse`/`contracts`/`coverage` through `make`.
  (Container/devcontainer authored to mirror the validated CI installs; Docker
  isn't in this env so they're CI-built, not built here.)
- **§2 reference SoC** — `rtl/soc/motorloop_axil_top.v` (axil_regfile↔controller_top,
  default DRV8301 BOM, FOC torque from the speed PI); **cocotb integration test
  `tb_motorloop_axil_top` PASSES** (12 cocotb blocks green) — the CI-able
  CPU↔AXI↔controller proof. `soc/motorloop_soc.py` (LiteX/VexRiscv) **validated**:
  generates the full gateware + compiles the BIOS against real LiteX + RISC-V
  GCC; the SoC map shows `motor → motorloop_axil_top`, "AXI-Lite 32-bit →
  Wishbone". `soc/firmware/` (main.c + Makefile + linker), `soc/README.md`,
  `make soc-sim`/`soc-build`. litex_sim **builds the full SoC into `Vsim`** (+ the
  BIOS); the live BIOS boot needs a TTY (serial2console) and the multi-minute
  one-off Verilator build, so it's run interactively, not captured headless — the
  cocotb test is the CI-gated execution proof.
- **§3B release** — `CITATION.cff`, `.zenodo.json`, README badges + "what this
  is/isn't", `CHANGELOG` `[0.1.0]`, `notes/release-checklist.md`. Tagging + the
  Zenodo DOI are maintainer-run release actions.
