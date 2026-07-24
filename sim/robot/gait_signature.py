#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""V.7b gait signature: reduce a (T, 4) boolean foot-contact history to three
gait-describing numbers. Pure NumPy so it can run on logged rollouts
anywhere.

  stride_hz   dominant frequency of the mean-removed per-foot contact signal
              (rFFT peak, DC excluded), averaged over the four feet
  duty_factor mean fraction of time each foot spends in contact
  diag_sync   corr(FL, RR) - corr(FL, FR) with feet ordered [FL, FR, RL, RR];
              a trot (diagonal pairs in phase, lateral pairs in antiphase)
              scores strongly positive, a pace/bound goes negative or ~0
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# foot column order contract (matches commanded_env/train_adversarial feet order)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, 0.0 when either signal is constant."""
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / denom) if denom > 0.0 else 0.0


def gait_signature(foot_contacts: np.ndarray, dt: float) -> dict:
    """foot_contacts: (T, 4) bool/0-1 array in [FL, FR, RL, RR] order; dt: seconds
    per sample. Returns dict(stride_hz, duty_factor, diag_sync) of floats."""
    fc = np.asarray(foot_contacts, dtype=np.float64)
    if fc.ndim != 2 or fc.shape[1] != 4:
        raise ValueError(f"foot_contacts must be (T, 4), got {fc.shape}")
    T = fc.shape[0]
    if T < 4:
        raise ValueError(f"need at least 4 samples, got {T}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    freqs = np.fft.rfftfreq(T, d=dt)
    per_foot_hz = []
    for i in range(4):
        sig = fc[:, i] - fc[:, i].mean()
        spec = np.abs(np.fft.rfft(sig))
        spec[0] = 0.0                       # exclude DC (duty offset, not stride)
        per_foot_hz.append(float(freqs[int(np.argmax(spec))]))
    stride_hz = float(np.mean(per_foot_hz))

    duty_factor = float(fc.mean())

    fl, fr, _rl, rr = fc[:, 0], fc[:, 1], fc[:, 2], fc[:, 3]
    diag_sync = _corr(fl, rr) - _corr(fl, fr)

    return {
        "stride_hz": stride_hz,
        "duty_factor": duty_factor,
        "diag_sync": float(diag_sync),
    }


def test_gait_signature_synthetic():
    """Perfect 2 Hz trot: diagonal pairs (FL+RR / FR+RL) as antiphase square
    waves with duty 0.5 — the signature must recover all three properties."""
    dt = 0.01
    t = np.arange(500) * dt                 # 5 s of contact history
    phase = (t * 2.0) % 1.0                 # 2 Hz cycle
    diag_a = phase < 0.5                    # FL & RR on the ground
    diag_b = ~diag_a                        # FR & RL on the ground
    contacts = np.stack([diag_a, diag_b, diag_b, diag_a], axis=1)  # FL FR RL RR

    sig = gait_signature(contacts, dt)
    assert abs(sig["stride_hz"] - 2.0) <= 0.2, f"stride_hz={sig['stride_hz']}"
    assert abs(sig["duty_factor"] - 0.5) <= 0.05, f"duty_factor={sig['duty_factor']}"
    assert sig["diag_sync"] > 0.5, f"diag_sync={sig['diag_sync']} (trot must be positive)"


if __name__ == "__main__":
    test_gait_signature_synthetic()
    print("gait_signature: synthetic trot recovered (stride 2 Hz, duty 0.5, diag sync +)")
