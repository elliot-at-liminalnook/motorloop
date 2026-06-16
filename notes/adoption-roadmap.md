<!-- SPDX-License-Identifier: MIT -->
# Adoption roadmap — earning trust & value from researchers and FPGA developers

The library already has a deep *verification & packaging* foundation: 12 formal
proofs (SymbiYosys), an 11-block cocotb suite, a 400-test cycle-accurate
Verilator co-sim against golden device models, bit-exact fixed-point references,
Verible/Verilator lint gates, FuseSoC + Bender + validated IP-XACT, REUSE/SPDX,
WaveDrom timing docs + an mkdocs site, a real ECP5 synth flow (64 MHz) + an
OpenLane ASIC-readiness smoke, and 8 datasheet-backed platform BOMs.

That foundation answers *"is it specified, proven, tested, and packaged?"* What
it does **not** yet answer is the question an outside researcher or FPGA
developer asks first: **"has it actually run on real silicon driving a real
motor, and can I get it running in my flow in an afternoon?"** This roadmap is
ordered by how much each item closes *that* gap — trust and adoption, not more
standards. (North star unchanged: every item must make a real block easier to
*trust* or *adopt*; don't add a badge that serves no consumer.)

## The one thing (highest leverage by far)

### 1. Hardware-in-the-loop validation — silicon correlation
Everything today is simulation + formal + synthesis. The biggest single trust
multiplier is to **run the controller on a real board driving a real BLDC and
show the hardware matches the cycle-accurate sim.**
- Bring up `board_top` on the already-targeted ULX3S (ECP5) or Tang Primer, wire
  a gate driver + motor, and spin it closed-loop (six-step first, then FOC).
- Capture phase currents / gate timing / rotor angle with a logic analyzer or
  scope and **overlay them on the sim trace** (the bench is cycle-accurate, so
  this is a direct comparison, not hand-waving). Publish the overlay.
- **Resolve the placeholder gains (Q1):** do a real motor identification so the
  FOC genuinely *controls* — today the gains are explicitly "placeholder-grade."
  A verified datapath that has never closed a loop on hardware is the credibility
  gap reviewers will probe first.
- *Why it wins trust:* it converts every "the sim says X" claim into "silicon
  does X," and it is the project's own stated trust ceiling. *Effort:* high (real
  hardware), but nothing else substitutes for it.

## Tier 2 — make it trivial to adopt and to cite

### 2. A complete, runnable reference SoC integration
Ship a worked example, not just wrappers in isolation: a **LiteX (or minimal
VexRiscv/PicoRV32) SoC** that instantiates the controller through the AXI-Lite
wrapper, with firmware that spins the motor and reads telemetry, plus the
AXI-Stream telemetry into a DMA/logger. Buildable in N documented commands.
- *Why:* FPGA developers adopt what they can see integrated and running on their
  fabric. "Here's a proven AXI wrapper" is a promise; "here's a RISC-V SoC that
  spins a motor over it" is proof. *Effort:* medium.

### 3. One-command reproducibility + a versioned, citable release
- **Containerize the toolchain** (Docker / Nix / devcontainer) pinned to
  `toolchain.lock`, so `make all` reproduces every result (sim, formal, synth,
  docs) on any machine with one command and zero setup.
- **Tag `v0.1.0`, mint a Zenodo DOI**, add CI status badges to the README, a
  `CHANGELOG`, and semantic versioning per IP (the cores are already `:0.1.0`).
- *Why:* researchers cite DOIs and trust "it built first try, green badges";
  developers abandon projects that don't build in 10 minutes. *Effort:* low–med.

### 4. Quickstart + a contract (datasheet) for every reusable block
- A 10-minute **getting-started**: clone → build the bench → run one block test →
  see a WaveDrom/VCD waveform.
- **Complete the per-block contracts** — only `pwm_generator` and `foc_core` have
  one; every reusable block needs its claim · assumptions · interface · timing ·
  proof-or-sim status (the docs site already renders them). This is the per-block
  "datasheet" an integrator reads before pulling a core.
- *Why:* a self-describing block is one a stranger can adopt without reading the
  RTL. *Effort:* low–med (mechanical, high payoff).

## Tier 3 — breadth of credibility

### 5. Multi-vendor FPGA portability results
ECP5 is niche. The RTL is deliberately Verilog-2005 (portable) — *prove it*:
publish synth results (LUT/FF/DSP, Fmax) on **Xilinx 7-series/UltraScale
(Vivado)** and **Intel (Quartus)**, ideally Gowin too. A small results table per
family. *Why:* "it'll map to *my* board" is a precondition for most adopters.
*Effort:* medium (needs vendor tools, but the flow scripts generalize).

### 6. Benchmarks vs prior art + a crisp scope/limitations page
- A feature / resource / Fmax / latency comparison against the existing open
  FOC/BLDC HDL projects.
- An explicit **"what this is and isn't"**: verified datapath + proofs + bit-exact
  sim, **not** silicon-validated until Tier 1; gains placeholder until motor ID;
  OpenLane is a *smoke*, not a tapeout. *Why:* honest, quantified positioning
  earns more trust than claims. *Effort:* low.

## Tier 4 — lower the contribution barrier

### 7. Project hygiene for contributors
`CONTRIBUTING.md`, issue/PR templates, a public roadmap, a code of conduct,
"good first issue" labels, and a short **architecture/design-rationale** doc (why
Verilog-2005, the role-abstraction, formal+sim+synth philosophy). *Why:* turns
onlookers into contributors and signals the project is maintained. *Effort:* low.

### 8. Published verification-coverage metrics
Surface functional-coverage numbers (cocotb coverage, formal cover closure) so
"how well tested" is a number, not an adjective. *Effort:* low–med.

## Sequencing (recommended)

1. **Tier 2 first if the goal is reach** (#3 reproducibility + #4 docs + #2
   reference SoC): cheap, and they remove every excuse not to try it.
2. **Tier 1 in parallel if the goal is depth** (#1 hardware correlation): the
   long pole, but the decisive trust signal — start the board bring-up early.
3. Then #5 multi-vendor and #6 benchmarks to widen the audience, and #7–#8 to
   sustain a community.

## What NOT to do (avoid the standards-as-product trap)
More IP-XACT/SPDX/lint polish, additional bus protocols with no consuming SoC,
or chasing Fmax past what any target clock needs — none of these move adoption.
The gap is **real-world proof and frictionless first use**, not more conformance.
