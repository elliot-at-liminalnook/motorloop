#!/usr/bin/env python3
"""Hardware-vs-simulation trace comparator (realism-checklist 7.2).

Aligns two shared-schema trace CSVs (optional time-shift search), resamples
onto a common uniform grid, and reports per-channel error metrics plus
dominant spectral peaks. The acceptance test for the comparator itself is
sim-vs-sim with deliberately perturbed parameters (test_model_form_harness).

Usage:
  python3 sim/scripts/compare_traces.py reference.csv candidate.csv \
      [--align-channel omega_rad_s] [--report report.md]
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

DEFAULT_CHANNELS = ["ia_a", "ib_a", "ic_a", "omega_rad_s", "vbus_v",
                    "cur_code", "emf_code"]


def load_csv(path: Path) -> dict[str, np.ndarray]:
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))

    def to_f(v):
        if v == "True":
            return 1.0
        if v == "False":
            return 0.0
        return float(v)

    return {k: np.array([to_f(r[k]) for r in rows]) for k in rows[0]}


def resample(trace: dict, grid: np.ndarray) -> dict[str, np.ndarray]:
    t = trace["time_s"]
    return {k: np.interp(grid, t, v) for k, v in trace.items()
            if k != "time_s"}


def find_time_shift(ref: dict, cand: dict, channel: str,
                    max_shift_s: float = 0.05) -> float:
    """Shift (added to candidate time) minimizing RMS error on `channel`."""
    t0 = max(ref["time_s"][0], cand["time_s"][0]) + max_shift_s
    t1 = min(ref["time_s"][-1], cand["time_s"][-1]) - max_shift_s
    if t1 <= t0:
        return 0.0
    grid = np.linspace(t0, t1, 2000)
    ref_v = np.interp(grid, ref["time_s"], ref[channel])
    best, best_err = 0.0, math.inf
    for shift in np.linspace(-max_shift_s, max_shift_s, 101):
        cand_v = np.interp(grid, cand["time_s"] + shift, cand[channel])
        err = float(np.sqrt(np.mean((ref_v - cand_v) ** 2)))
        if err < best_err:
            best, best_err = shift, err
    return best


def spectral_peaks(values: np.ndarray, dt: float, n_peaks: int = 3):
    spec = np.abs(np.fft.rfft(values - values.mean()))
    freqs = np.fft.rfftfreq(len(values), dt)
    order = np.argsort(spec)[::-1]
    peaks = []
    for idx in order:
        if freqs[idx] == 0.0:
            continue
        if all(abs(freqs[idx] - f) > 2.0 for f, _ in peaks):
            peaks.append((float(freqs[idx]), float(spec[idx])))
        if len(peaks) >= n_peaks:
            break
    return peaks


def compare(ref: dict, cand: dict, channels=None, align_channel=None,
            skip_initial_s: float = 0.0) -> dict:
    channels = [c for c in (channels or DEFAULT_CHANNELS)
                if c in ref and c in cand]
    shift = (find_time_shift(ref, cand, align_channel)
             if align_channel else 0.0)
    cand_t = {**cand, "time_s": cand["time_s"] + shift}

    t0 = max(ref["time_s"][0], cand_t["time_s"][0]) + skip_initial_s
    t1 = min(ref["time_s"][-1], cand_t["time_s"][-1])
    dt = float(np.median(np.diff(ref["time_s"])))
    grid = np.arange(t0, t1, dt)
    ref_r = resample(ref, grid)
    cand_r = resample(cand_t, grid)

    result = {"shift_s": shift, "channels": {}}
    for ch in channels:
        err = ref_r[ch] - cand_r[ch]
        ref_rms = float(np.sqrt(np.mean(ref_r[ch] ** 2)))
        result["channels"][ch] = {
            "rms_error": float(np.sqrt(np.mean(err ** 2))),
            "peak_error": float(np.max(np.abs(err))),
            "rms_ratio": (float(np.sqrt(np.mean(err ** 2))) / ref_rms
                          if ref_rms > 1e-12 else math.inf),
            "ref_peaks_hz": spectral_peaks(ref_r[ch], dt),
            "cand_peaks_hz": spectral_peaks(cand_r[ch], dt),
        }
    return result


def report_markdown(result: dict, ref_name: str, cand_name: str) -> str:
    lines = [
        f"# Trace comparison: {ref_name} vs {cand_name}",
        "",
        f"Applied time shift: {result['shift_s'] * 1e3:.2f} ms",
        "",
        "| channel | RMS err | peak err | RMS ratio |",
        "|---|---|---|---|",
    ]
    for ch, m in result["channels"].items():
        lines.append(f"| {ch} | {m['rms_error']:.4g} | {m['peak_error']:.4g}"
                     f" | {m['rms_ratio']:.3f} |")
    lines.append("")
    for ch, m in result["channels"].items():
        ref_p = ", ".join(f"{f:.1f} Hz" for f, _ in m["ref_peaks_hz"])
        cand_p = ", ".join(f"{f:.1f} Hz" for f, _ in m["cand_peaks_hz"])
        lines.append(f"- {ch}: ref peaks [{ref_p}] vs cand [{cand_p}]")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--align-channel", default="omega_rad_s")
    parser.add_argument("--skip-initial", type=float, default=0.0)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    result = compare(load_csv(args.reference), load_csv(args.candidate),
                     align_channel=args.align_channel,
                     skip_initial_s=args.skip_initial)
    text = report_markdown(result, args.reference.name, args.candidate.name)
    if args.report:
        args.report.write_text(text)
        print(f"wrote {args.report}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
