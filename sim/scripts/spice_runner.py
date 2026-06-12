#!/usr/bin/env python3
"""Cached ngspice batch runner for the codified circuit netlists.

Each run gets a working directory under sim/build/spice keyed by the hash of
the netlist text, the generated components.param, any extra .param
overrides, and auxiliary includes — so re-runs with unchanged inputs are
free and the pytest derivation tier stays fast.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import derive_params  # noqa: E402
import sim_params  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CIRCUITS_DIR = PROJECT_ROOT / "sim" / "circuits"
CACHE_ROOT = PROJECT_ROOT / "sim" / "build" / "spice"

DRV8301_LIB = (
    PROJECT_ROOT / "docs" / "ti-simulation-models"
    / "spnm068-drv8301-tina-ti-spice-model" / "DRV8301" / "Release_TI"
    / "TINA" / "DRV8301_TINA_AIO" / "DRV8301_TINA_AIO_SPICE_MODEL"
    / "DRV8301.LIB"
)


class SpiceError(RuntimeError):
    pass


def _apply_param_overrides(netlist: str, overrides: dict[str, float]) -> str:
    for name, value in overrides.items():
        pattern = re.compile(rf"^\.param {re.escape(name)}=.*$", re.MULTILINE)
        if not pattern.search(netlist):
            raise SpiceError(f"no .param {name}= line to override")
        netlist = pattern.sub(f".param {name}={value:g}", netlist)
    return netlist


def run_netlist(name: str, params: sim_params.SimParams,
                overrides: dict[str, float] | None = None,
                aux_files: dict[str, Path] | None = None,
                compat: str | None = None,
                ) -> dict[str, list[list[float]]]:
    """Run sim/circuits/<name>.cir; returns {output filename: rows of
    floats} for every wrdata output the netlist produced. `compat` selects
    an ngspice compatibility mode (e.g. 'psa' for PSpice-dialect vendor
    models) via a working-directory .spiceinit."""
    netlist_path = CIRCUITS_DIR / f"{name}.cir"
    netlist = netlist_path.read_text()
    if overrides:
        netlist = _apply_param_overrides(netlist, overrides)

    components = derive_params.write_spice_params(params).read_text()
    aux_files = aux_files or {}
    aux_blob = "".join(
        f"{dst}:{src.read_text(errors='ignore')}"
        for dst, src in sorted(aux_files.items()))

    digest = hashlib.sha256(
        (netlist + components + aux_blob + (compat or "")).encode()
    ).hexdigest()[:16]
    workdir = CACHE_ROOT / f"{name}-{digest}"
    done_marker = workdir / ".done"

    if not done_marker.is_file():
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True)
        (workdir / f"{name}.cir").write_text(netlist)
        (workdir / "components.param").write_text(components)
        if compat:
            (workdir / ".spiceinit").write_text(
                f"set ngbehavior={compat}\n")
        for dst, src in aux_files.items():
            shutil.copy(src, workdir / dst)
        result = subprocess.run(
            ["ngspice", "-b", f"{name}.cir", "-o", "ngspice.log"],
            cwd=workdir, capture_output=True, text=True, timeout=300)
        outputs = list(workdir.glob("*.out"))
        if result.returncode != 0 or not outputs:
            log = (workdir / "ngspice.log")
            log_text = log.read_text() if log.is_file() else result.stderr
            raise SpiceError(
                f"ngspice failed for {name} (rc={result.returncode}):\n"
                f"{log_text[-3000:]}")
        done_marker.write_text("ok")

    data: dict[str, list[list[float]]] = {}
    for out in workdir.glob("*.out"):
        rows = []
        for line in out.read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            try:
                rows.append([float(f) for f in fields])
            except ValueError:
                continue
        data[out.name] = rows
    return data


if __name__ == "__main__":
    p = sim_params.load()
    for circuit in ("emf_channel", "iout_channel", "adc_frontend_emf",
                    "adc_frontend_bus"):
        result = run_netlist(circuit, p)
        for fname, rows in result.items():
            print(f"{circuit}: {fname}: {len(rows)} rows")
