# SPDX-License-Identifier: MIT
"""SNR/ENOB helpers for the ADS9224R sim-validation tiers
(notes/ads9224r-sim-validation-checklist.md §0).

Pure functions over ngspice .noise spectra + datasheet numbers. The front-end's
noise-limited SNR is combined (root-sum-square) with the ADC's own datasheet SNR
so ENOB is the *system* number, not just the amplifier.
"""

from __future__ import annotations

import math


def integrate_noise_rms(freqs, density):
    """Total RMS from a one-sided noise spectral density (V/sqrtHz) sampled at
    `freqs` (Hz): sqrt(integral of density^2 df), trapezoidal."""
    total = 0.0
    for (f0, d0), (f1, d1) in zip(zip(freqs, density), zip(freqs[1:], density[1:])):
        total += 0.5 * (d0 * d0 + d1 * d1) * (f1 - f0)
    return math.sqrt(max(total, 0.0))


def snr_db_from_noise(full_scale_amplitude_v, noise_rms_v):
    """SNR (dB) of a full-scale sine vs an RMS noise. The signal RMS is
    amplitude/sqrt(2)."""
    sig_rms = full_scale_amplitude_v / math.sqrt(2.0)
    return 20.0 * math.log10(sig_rms / max(noise_rms_v, 1e-30))


def enob_from_snr(snr_db):
    """Effective number of bits from SNR (the ideal-SNR relation)."""
    return (snr_db - 1.76) / 6.02


def current_noise_rms_a(v_noise_rms, fda_gain, shunt_ohm):
    """Refer a front-end output-voltage noise back to the sensed phase current:
    i_noise = v_noise / (gain * shunt)."""
    return v_noise_rms / (fda_gain * shunt_ohm)


def rss(*terms):
    """Root-sum-square of independent (RMS) contributions."""
    return math.sqrt(sum(t * t for t in terms))


def combine_snr_db(*snr_db_terms):
    """Combine independent noise sources given as SNR (dB) into a system SNR:
    each SNR maps to a noise power 10^(-SNR/10); powers add; back to dB."""
    power = sum(10.0 ** (-s / 10.0) for s in snr_db_terms)
    return -10.0 * math.log10(power)
