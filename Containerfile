# SPDX-License-Identifier: MIT
# Reproducible build environment (tier2-adoption-checklist §3A). Pins the exact
# HDL toolchain from toolchain.lock + the Python deps, so `make all` reproduces
# every result on any machine. Build + run:
#
#   docker build -f Containerfile -t motorloop .
#   docker run --rm motorloop            # runs `make verify`
#
# The tool versions below MUST match toolchain.lock (bump both together).
FROM ubuntu:24.04

ARG OSS_CAD_TAG=2026-06-14
ARG OSS_CAD_STAMP=20260614
ARG VERIBLE_VER=v0.0-4075-g795c204a
ARG BENDER_VER=0.32.0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git curl ca-certificates xz-utils \
        python3 python3-pip python3-venv libxml2-utils \
    && rm -rf /var/lib/apt/lists/*

# OSS CAD Suite (yosys + SymbiYosys + solvers + nextpnr-ecp5 + ecppack + verilator)
RUN cd /opt \
    && curl -sL -o oss.tgz \
       "https://github.com/YosysHQ/oss-cad-suite-build/releases/download/${OSS_CAD_TAG}/oss-cad-suite-linux-x64-${OSS_CAD_STAMP}.tgz" \
    && tar xzf oss.tgz && rm oss.tgz

# Verible (enforced lint gate)
RUN mkdir -p /opt/verible && cd /opt/verible \
    && curl -sL -o v.tar.gz \
       "https://github.com/chipsalliance/verible/releases/download/${VERIBLE_VER}/verible-${VERIBLE_VER}-linux-static-x86_64.tar.gz" \
    && tar xzf v.tar.gz --strip-components=1 && rm v.tar.gz
ENV PATH="/opt/verible/bin:${PATH}"

# Bender (PULP/Bender consumers)
RUN mkdir -p /opt/bender && cd /opt/bender \
    && curl -sL -o b.tar.xz \
       "https://github.com/pulp-platform/bender/releases/download/v${BENDER_VER}/bender-x86_64-unknown-linux-gnu.tar.xz" \
    && tar xJf b.tar.xz
ENV PATH="/opt/bender/bender-x86_64-unknown-linux-gnu:${PATH}"

WORKDIR /work
COPY requirements*.txt ./
# One python for everything (Ubuntu 24.04 ships 3.12, within cocotb's support).
RUN pip3 install --break-system-packages --no-cache-dir \
        -r requirements.txt -r requirements-cocotb.txt -r requirements-docs.txt

COPY . .
# Override the Makefile tool-location vars to the container layout.
ENV OSS=/opt/oss-cad-suite/environment COCOTB_PY=python3 MKDOCS=mkdocs
CMD ["make", "verify"]
