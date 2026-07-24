# SPDX-License-Identifier: MIT
"""Metrics for the part-comparison study (notes/part-comparison-checklist.md §0.3).

Pure functions over sampled bench arrays - no bench dependency, so the same
metric definitions back both the pytest assertions (test_part_comparison.py) and
the rendered figures (gen_comparison_figures.py). Everything here is relative:
the study holds the FOC controller fixed and changes one part, so these quantify
the *difference* a part makes, not an absolute hardware spec (placeholder gains,
Q1).
"""

from __future__ import annotations

import numpy as np

# A commutation angle error of 90 electrical degrees flips the sign of the
# torque-producing current projection (cos > 0 -> cos < 0): past it the loop
# pushes the rotor the wrong way. 30 deg is the "degraded but still driving"
# mark. Both are used as break thresholds.
COMMUTATION_INVERT_DEG = 90.0


def wrap_pi(x):
    """Wrap to (-pi, pi]."""
    return (np.asarray(x, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def angle_error_elec_rad(meas_mech, truth_mech, pole_pairs):
    """Electrical-angle error (rad) between a measured and the true mechanical
    angle - the commutation-relevant quantity. Mechanical lag x pole_pairs is
    the electrical lag the FOC dq frame actually rotates by."""
    return wrap_pi(pole_pairs * (np.asarray(meas_mech, dtype=float)
                                 - np.asarray(truth_mech, dtype=float)))


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def peak_abs(x):
    x = np.asarray(x, dtype=float)
    return float(np.max(np.abs(x))) if x.size else float("nan")


def torque_efficiency(angle_err_elec_rad_arr):
    """Fraction of the commanded current that makes torque, given the
    commutation misalignment: mean(cos(theta_err)). 1.0 = perfect; <0 = the
    misalignment is braking. The honest "so-what" of angle latency (T8)."""
    return float(np.mean(np.cos(np.asarray(angle_err_elec_rad_arr, dtype=float))))


def detrend(sig, window=51):
    """Remove the slow fundamental with a centered moving average, leaving the
    ripple/noise. Window is clamped odd and < len."""
    sig = np.asarray(sig, dtype=float)
    n = sig.size
    if n < 3:
        return sig - np.mean(sig) if n else sig
    w = min(window, n if n % 2 else n - 1)
    if w % 2 == 0:
        w -= 1
    w = max(w, 1)
    kern = np.ones(w) / w
    base = np.convolve(sig, kern, mode="same")
    return sig - base


def dq_ripple(idv, iqv):
    """Combined dq current ripple (RMS of the detrended id,iq vector) - the
    measurement noise the FOC sees. Scale-consistent across the platforms in a
    comparison (verified: foc/truth code ratio matches)."""
    rid = detrend(idv)
    riq = detrend(iqv)
    return float(np.sqrt(np.mean(rid * rid + riq * riq)))


def noise_floor_fft(sig, fs):
    """One-sided power spectrum (dB) of the detrended signal. Returns
    (freqs_hz, psd_db). The floor away from the fundamental is the
    quantization/skew measurement-noise floor (T4)."""
    sig = detrend(np.asarray(sig, dtype=float))
    n = sig.size
    if n < 4:
        return np.array([0.0]), np.array([-300.0])
    win = np.hanning(n)
    spec = np.fft.rfft(sig * win)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = (np.abs(spec) ** 2) / (np.sum(win ** 2) * fs)
    psd_db = 10.0 * np.log10(psd + 1e-30)
    return freqs, psd_db


def is_locked(angle_err_elec_deg_arr, fault=False,
              thr_deg=COMMUTATION_INVERT_DEG):
    """Commutation lock: peak electrical angle error stayed below the
    torque-inversion threshold AND no fault latched. The shared lock/break
    classifier for the reversal cliff (T2) and the envelope map (T10)."""
    if fault:
        return False
    return peak_abs(angle_err_elec_deg_arr) < thr_deg


def expected_order(a, b, better="lower", eps=0.0):
    """Compare two scalar metrics for a pair (a = first part, b = second).
    Returns (ok, no_diff): ok True if the predicted ordering holds; no_diff True
    if |a-b| <= eps (the models don't distinguish the parts on this axis - the
    honesty sentinel, recorded rather than asserted)."""
    if abs(a - b) <= eps:
        return True, True
    if better == "lower":
        return (a < b), False
    return (a > b), False
