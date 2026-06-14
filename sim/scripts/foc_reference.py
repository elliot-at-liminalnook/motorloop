#!/usr/bin/env python3
"""Reference FOC math: a floating-point executable spec and a bit-faithful
fixed-point twin of the RTL, plus a steady-state dq PMSM model.

Two layers:
  * float layer (clarke/park/inv_park/inv_clarke/svpwm_float) - the clean
    control math, used by the stage-1 dq cross-check and as the oracle the
    fixed-point layer is checked against;
  * fixed layer (clarke_fx/park_fx/.../svpwm_fx, sincos_table) - mirrors the
    RTL arithmetic exactly (Q-format and shifts per notes/foc-fixed-point.md),
    used by the stage-3 parity tests.

Sign/orientation convention (see notes/foc-fixed-point.md): the plant EMF is
e_a = Ke*omega*sin(theta_e); with the Park below, a current in phase with the
EMF lands on +iq (positive torque), id = 0.
"""

from __future__ import annotations

import math

# ----------------------------------------------------------------------------
# Float reference (the clean control math)
# ----------------------------------------------------------------------------


def clarke(ia, ib, ic=None):
    """Amplitude-invariant Clarke. Balanced => ic = -ia-ib (ignored)."""
    ialpha = ia
    ibeta = (ia + 2.0 * ib) / math.sqrt(3.0)
    return ialpha, ibeta


def park(ialpha, ibeta, theta_e):
    c, s = math.cos(theta_e), math.sin(theta_e)
    id_ = ialpha * c + ibeta * s
    iq = ialpha * s - ibeta * c
    return id_, iq


def inv_park(vd, vq, theta_e):
    # The Park matrix [[c, s],[s, -c]] is its own inverse (symmetric,
    # orthogonal, det = -1), so inverse Park reuses the same form.
    c, s = math.cos(theta_e), math.sin(theta_e)
    valpha = vd * c + vq * s
    vbeta = vd * s - vq * c
    return valpha, vbeta


def inv_clarke(valpha, vbeta):
    va = valpha
    vb = -0.5 * valpha + (math.sqrt(3.0) / 2.0) * vbeta
    vc = -0.5 * valpha - (math.sqrt(3.0) / 2.0) * vbeta
    return va, vb, vc


def svpwm_float(valpha, vbeta):
    """min/max common-mode injection. Inputs in [-1, 1] (fraction of the
    available phase swing); returns three duties in [0, 1] centered at 0.5."""
    ra, rb, rc = inv_clarke(valpha, vbeta)
    cm = 0.5 * (max(ra, rb, rc) + min(ra, rb, rc))
    return tuple(0.5 + 0.5 * (r - cm) for r in (ra, rb, rc))


# ----------------------------------------------------------------------------
# Steady-state dq PMSM model (surface PMSM, Ld = Lq = L)
# ----------------------------------------------------------------------------


def lambda_m(ke_v_s_per_rad, pole_pairs):
    """Peak rotor flux linkage. e_peak(phase) = Ke*omega_mech = lambda_m*omega_e
    with omega_e = pole_pairs*omega_mech, so lambda_m = Ke/pole_pairs."""
    return ke_v_s_per_rad / pole_pairs


def dq_voltage(id_, iq, omega_e, r_ohm, l_h, lam):
    """Steady-state (d/dt = 0) dq voltages for THIS transform convention.

    The Park above puts the q-axis at the negation of the textbook q-axis
    (iq = -iq_standard), which flips the sign of the speed cross-coupling
    terms relative to the textbook equations; the EMF term we*lam stays
    positive (it tracks the rotation/torque direction). Verified against the
    abc plant at clamped speed across operating points (test_foc_plant)."""
    vd = r_ohm * id_ + omega_e * l_h * iq
    vq = r_ohm * iq - omega_e * l_h * id_ + omega_e * lam
    return vd, vq


def torque_from_iq(iq, ke_v_s_per_rad):
    """Surface-PMSM torque (amplitude-invariant transform): T = 1.5*Ke*iq."""
    return 1.5 * ke_v_s_per_rad * iq


# ----------------------------------------------------------------------------
# Fixed-point twin of the RTL (Q-format per notes/foc-fixed-point.md)
# ----------------------------------------------------------------------------

Q15 = 15
ONE_OVER_SQRT3 = round(0.5773502691896258 * (1 << Q15))   # 18919
SQRT3_OVER_2 = round(0.8660254037844386 * (1 << Q15))     # 28378


def _sat(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def sat16(x):
    return _sat(int(x), -32768, 32767)


def asr(x, n):
    """Arithmetic shift right (matches Verilog >>> on a signed value).
    Python's >> already floors toward -inf for negative ints, as does a
    two's-complement arithmetic shift, so they agree bit-for-bit."""
    return int(x) >> n


def sincos_table(bits):
    """Quarter-wave sin table, Q15. table[i] = round(sin((i+0.5)/N * pi/2)*32767)
    for i in [0, N). The +0.5 centers each sample in its cell (matches RTL)."""
    n = 1 << bits
    return [round(math.sin((i + 0.5) / n * (math.pi / 2.0)) * 32767)
            for i in range(n)]


def sincos_fx(theta16, bits, table=None):
    """16-bit angle (0..65535 = 0..2pi) -> (cos_q15, sin_q15) using the
    quarter-wave table with quadrant folding."""
    if table is None:
        table = sincos_table(bits)
    n = 1 << bits
    theta16 &= 0xFFFF

    def sin_lookup(a16):
        quad = (a16 >> 14) & 0x3
        idx = (a16 >> (14 - bits)) & (n - 1)
        if quad == 0:
            return table[idx]
        if quad == 1:
            return table[n - 1 - idx]
        if quad == 2:
            return -table[idx]
        return -table[n - 1 - idx]

    sin_v = sin_lookup(theta16)
    cos_v = sin_lookup((theta16 + 0x4000) & 0xFFFF)  # cos = sin(theta + pi/2)
    return cos_v, sin_v


def clarke_fx(ia, ib):
    ialpha = ia
    ibeta = asr((ia + 2 * ib) * ONE_OVER_SQRT3, Q15)
    return ialpha, ibeta


def park_fx(ialpha, ibeta, cos_q15, sin_q15):
    id_ = asr(ialpha * cos_q15 + ibeta * sin_q15, Q15)
    iq = asr(ialpha * sin_q15 - ibeta * cos_q15, Q15)
    return id_, iq


def inv_park_fx(vd, vq, cos_q15, sin_q15):
    valpha = asr(vd * cos_q15 + vq * sin_q15, Q15)
    vbeta = asr(vd * sin_q15 - vq * cos_q15, Q15)
    return valpha, vbeta


def svpwm_fx(valpha, vbeta, half):
    """Inputs vd/vq-domain (duty units, +-half). Returns three duties in
    [0, half], centered at half>>1. Matches rtl/svpwm.v exactly."""
    half_alpha = asr(valpha, 1)
    s3b = asr(SQRT3_OVER_2 * vbeta, Q15)
    ra = valpha
    rb = -half_alpha + s3b
    rc = -half_alpha - s3b
    cm = asr(max(ra, rb, rc) + min(ra, rb, rc), 1)
    center = half >> 1
    return tuple(_sat(center + (r - cm), 0, half) for r in (ra, rb, rc))


def circle_limit_fx(vd, vq, vlim):
    """Cap the vector magnitude to vlim, priority on vd (field-weakening
    headroom). Reference uses exact sqrt; the RTL approximates."""
    mag2 = vd * vd + vq * vq
    if mag2 <= vlim * vlim:
        return vd, vq
    mag = math.isqrt(mag2)
    if mag == 0:
        return vd, vq
    return (vd * vlim) // mag if vd >= 0 else -((-vd * vlim) // mag), \
           (vq * vlim) // mag if vq >= 0 else -((-vq * vlim) // mag)


# ----------------------------------------------------------------------------
# Full FOC control law (executable spec for the RTL controller, foc_core +
# speed_iq_pi). Fixed-point throughout; drives an averaged 3-phase plant.
# ----------------------------------------------------------------------------


class FocController:
    """One-update-per-PWM-period FOC controller in fixed-point: speed PI ->
    iq*, then id/iq current PIs -> circle limit -> inverse Park -> SVPWM. The
    behavioral twin of the RTL (foc_core.v + speed_iq_pi.v); used as the
    executable spec the RTL is checked against at the loop level."""

    def __init__(self, *, half, vlim, cur_kp, cur_kish, speed_kp, speed_kish,
                 iq_max, sincos_bits, id_target=0):
        self.half = half
        self.vlim = vlim
        self.cur_kp = cur_kp
        self.cur_kish = cur_kish
        self.speed_kp = speed_kp
        self.speed_kish = speed_kish
        self.iq_max = iq_max
        self.bits = sincos_bits
        self.id_target = id_target
        self.integ_d = 0
        self.integ_q = 0
        self.integ_s = 0
        self.table = sincos_table(sincos_bits)

    def step(self, ia_lsb, ib_lsb, theta16, speed_int, target_speed):
        """Returns (duty_a, duty_b, duty_c) in [0, half] plus (id, iq)."""
        # Outer speed loop -> iq*.
        err_s = target_speed - speed_int
        raw_s = self.speed_kp * err_s + asr(self.speed_kp * self.integ_s,
                                            self.speed_kish)
        iq_t = _sat(raw_s, -self.iq_max, self.iq_max)
        if not ((iq_t >= self.iq_max and err_s > 0)
                or (iq_t <= -self.iq_max and err_s < 0)):
            self.integ_s += err_s

        # Measurement.
        cos_q, sin_q = sincos_fx(theta16, self.bits, self.table)
        ialpha, ibeta = clarke_fx(ia_lsb, ib_lsb)
        id_m, iq_m = park_fx(ialpha, ibeta, cos_q, sin_q)

        # Current PIs.
        err_d = self.id_target - id_m
        err_q = iq_t - iq_m
        vmax = 4 * self.half
        vd_raw = _sat(self.cur_kp * err_d
                      + asr(self.cur_kp * self.integ_d, self.cur_kish),
                      -vmax, vmax)
        vq_raw = _sat(self.cur_kp * err_q
                      + asr(self.cur_kp * self.integ_q, self.cur_kish),
                      -vmax, vmax)
        vd, vq = circle_limit_fx(vd_raw, vq_raw, self.vlim)
        if not (vd_raw * vd_raw + vq_raw * vq_raw > self.vlim * self.vlim):
            self.integ_d += err_d
            self.integ_q += err_q

        valpha, vbeta = inv_park_fx(vd, vq, cos_q, sin_q)
        da, db, dc = svpwm_fx(valpha, vbeta, self.half)
        return (da, db, dc), id_m, iq_m
