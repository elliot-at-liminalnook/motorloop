#!/usr/bin/env python3
"""Motor parameter-fit bootstrap (realism-checklist 7.4): the exact fitting
toolchain the Q1 identification session will use, self-tested against
synthetic bench traces so motor-ID day is measurement, not development.

Fits:
  fit_rl       - R and L from a locked-rotor voltage-step current trace.
  fit_ke_ll    - line-to-line peak BEMF constant from a coast capture
                 (omega + line-to-line voltage arrays, i.e. a scope shot).
  fit_spin_down- B and Coulomb friction from a spin-down omega(t) trace
                 with known J (linear regression of dw/dt against w).
"""

from __future__ import annotations

import math

import numpy as np


def fit_rl(t: np.ndarray, i: np.ndarray, v_applied: float):
    """i(t) = (V/R)(1 - exp(-tR/L)) for a series R-L step at t=0.
    Returns (R, L). Uses the steady tail for R and a log-linear fit of the
    approach for tau."""
    i_inf = float(np.mean(i[int(0.8 * len(i)):]))
    r = v_applied / i_inf
    resid = 1.0 - i / i_inf
    mask = (resid > 0.05) & (resid < 0.95)
    if mask.sum() < 4:
        raise ValueError("step too fast/slow for the capture window")
    slope, _ = np.polyfit(t[mask], np.log(resid[mask]), 1)
    tau = -1.0 / slope
    return r, tau * r


def fit_ke_ll(omega: np.ndarray, v_ll: np.ndarray):
    """Peak line-to-line BEMF per mech rad/s from a coast capture: the
    envelope of v_ll against omega. Uses windowed peaks so amplitude decay
    during the coast is handled."""
    n_windows = 12
    points = []
    size = len(omega) // n_windows
    for w in range(n_windows):
        sl = slice(w * size, (w + 1) * size)
        if sl.stop > len(omega):
            break
        # Pair the voltage peak with omega AT the peak instant - pairing
        # against the window mean biases Ke high while omega decays.
        idx = int(np.argmax(np.abs(v_ll[sl]))) + sl.start
        v_peak = float(abs(v_ll[idx]))
        w_at_peak = float(omega[idx])
        if w_at_peak > 1.0:
            points.append((w_at_peak, v_peak))
    if len(points) < 3:
        raise ValueError("not enough usable windows")
    ws = np.array([p[0] for p in points])
    vs = np.array([p[1] for p in points])
    # Through-origin least squares: ke = sum(w*v)/sum(w^2).
    return float(np.dot(ws, vs) / np.dot(ws, ws))


def fit_spin_down(t: np.ndarray, omega: np.ndarray, j_kg_m2: float):
    """J dw/dt = -(B w + tau_c): linear regression of -J*dw/dt against w
    gives slope B and intercept tau_c. Returns (B, tau_c)."""
    dw = np.gradient(omega, t)
    y = -j_kg_m2 * dw
    mask = omega > 2.0  # stay away from the stiction-regularized tail
    slope, intercept = np.polyfit(omega[mask], y[mask], 1)
    return float(slope), float(intercept)


def kv_from_ke_ll(ke_ll: float) -> float:
    """Convenience inverse of the derive_params helper."""
    return 60.0 / (2.0 * math.pi * ke_ll)
