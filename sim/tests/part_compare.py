# SPDX-License-Identifier: MIT
"""Part-comparison harness + experiment runners
(notes/part-comparison-checklist.md §0-§1).

Each runner builds bench(es), drives one stress scenario, and returns a result
dict of numpy arrays + scalars. The pytest suite (test_part_comparison.py) calls
them with coarse sizes and asserts the predicted ordering; the figure generator
(gen_comparison_figures.py) calls the same runners with finer sizes and plots -
one source of truth per experiment.

The honest frame (FRAME), enforced by construction:
  * Simulation, against the device *models*, not silicon.
  * Relative: every comparison holds the FOC controller fixed and changes
    exactly one part. Placeholder motor gains (Q1) -> absolute thresholds are
    indicative; report orderings/ratios.
  * No manufactured differences: where the models don't separate two parts, the
    runner returns the numbers and the caller records "no modeled difference".

What varies, and how it is isolated (verified empirically, see the report):
  * Angle sensor  AS5600 vs AS5047P  -> platforms zonri_drv8301 vs zonri_as5047p
    (identical except the angle role: driver, ADC, gains all held).
  * Current sampling  MCP3208 vs ADS9224R  -> the foc current_sample_scheme on a
    fixed stable platform: 1 = sequential single-ADC skew (MCP3208 reality),
    0 = dual simultaneous (ADS9224R reality). This isolates Q21 skew from the
    operating-point shift the whole-BOM adc_dual_mode swap causes under the
    shared placeholder loop.
"""

from __future__ import annotations

import numpy as np

import part_metrics as M
from bench_factory import expected_init_time, foc, platform


# The clean isolated-variable angle pair (verified: differ only in angle_name).
SENSOR_A = "zonri_drv8301"   # AS5600 (PWM, ~90 us latency)
SENSOR_B = "zonri_as5047p"   # AS5047P (SPI 14-bit, DAEC ~1.5 us)
SENSOR_LABEL = {SENSOR_A: "AS5600", SENSOR_B: "AS5047P"}

# The current-sampling toggle (MCP3208 sequential-skew vs ADS9224R simultaneous).
SCHEME_SIMULTANEOUS = 0      # ADS9224R reality
SCHEME_SEQUENTIAL = 1        # MCP3208 reality
SCHEME_LABEL = {SCHEME_SIMULTANEOUS: "ADS9224R (simultaneous)",
                SCHEME_SEQUENTIAL: "MCP3208 (sequential skew)"}

# The whole-BOM pair, for the system snapshot (T5).


# --------------------------------------------------------------------------- #
# low-level build / sample
# --------------------------------------------------------------------------- #
def _pp(params):
    return int(params.value("motor.pole_pairs"))


def _align(params):
    return int(params.value("foc.align_offset"))


def _build(params, bldcsim, cfg):
    b = bldcsim.Bench(cfg)
    b.run_for(expected_init_time(params))
    assert b.configured, "DRV init did not complete"
    return b


def _sensor_cfg(params, sensor):
    """Platform config for an angle-sensor comparison, pinned to the
    sinusoidal-PMSM plant (FOC convention) so the EMF shape is common-mode
    across the A/B pair and only the sensor differs."""
    return platform(params, sensor, motor={"trapezoid_blend": 0.0})


def _start_foc(b, params, omega, iq=40, clamp=True):
    """Bring a built bench to a clamped FOC operating point at +omega."""
    b.set_align_offset(_align(params))
    b.set_id_target(0)
    b.set_iq_target(int(iq))
    if clamp:
        b.set_speed_clamp(True, omega)
    b.set_mode(3)


def _flags(b):
    """Latched break/fault telemetry after a run."""
    return dict(shoot=int(b.shoot_through_violations),
                locked_out=bool(b.locked_out),
                stalled=bool(b.stalled),
                pvdd_uv=int(b.pvdd_uv_events),
                fault_count=int(b.fault_count),
                bus_v_max=float(b.bus_v_max),
                bus_v_min=float(b.bus_v_min))


def _any_fault(flags):
    return (flags["shoot"] > 0 or flags["locked_out"] or flags["stalled"]
            or flags["pvdd_uv"] > 0 or flags["fault_count"] > 0)


# --------------------------------------------------------------------------- #
# T1 / T6 / T8 - the speed sweep (angle error, lead/lag, torque penalty)
# --------------------------------------------------------------------------- #
def run_speed_sweep(params, bldcsim, sensor, omegas, iq=40,
                    settle=0.04, n=300, dt=2e-5):
    """At each clamped speed, measure the sensor's commutation angle error and
    the FOC torque consequence. Feeds T1 (rms error vs speed), T6 (signed
    lead/lag + max locked speed) and T8 (iq retained + torque efficiency)."""
    pp = _pp(params)
    rms_deg, mean_deg, iq_ret, cos_eff = [], [], [], []
    for omega in omegas:
        b = _build(params, bldcsim, _sensor_cfg(params, sensor))
        _start_foc(b, params, omega, iq=iq)
        b.run_for(settle)
        err, iqv = [], []
        for _ in range(n):
            b.run_for(dt)
            err.append(M.angle_error_elec_rad(b.encoder_angle_rad, b.theta, pp))
            iqv.append(b.foc_iq)
        err = np.asarray(err)
        rms_deg.append(np.degrees(M.rms(err)))
        mean_deg.append(float(np.degrees(np.mean(err))))
        iq_ret.append(float(np.mean(iqv)) / iq)
        cos_eff.append(M.torque_efficiency(err))
    return dict(omega=np.asarray(omegas, float),
                rms_deg=np.asarray(rms_deg),
                mean_deg=np.asarray(mean_deg),
                iq_retained=np.asarray(iq_ret),
                torque_eff=np.asarray(cos_eff),
                iq_cmd=iq, sensor=sensor, label=SENSOR_LABEL[sensor])


def max_locked_speed(sweep, thr_deg=M.COMMUTATION_INVERT_DEG):
    """Highest swept speed whose rms electrical angle error stayed below the
    threshold (the commutation 'max RPM' for T6 / the cliff value for T2)."""
    ok = sweep["rms_deg"] < thr_deg
    return float(sweep["omega"][ok].max()) if ok.any() else 0.0


# --------------------------------------------------------------------------- #
# T2 - the reversal cliff (prescribed dyno reversal)
# --------------------------------------------------------------------------- #
def run_reversal(params, bldcsim, sensor, peak_speeds, transition_s=0.01,
                 iq=40, settle=0.04, dt=2e-5, post_s=0.01):
    """Dyno-prescribe +Omega -> -Omega over transition_s at each peak speed;
    record the peak commutation error through the reversal. The cliff is the
    fastest reversal whose error stays below the inversion threshold."""
    pp = _pp(params)
    peak_err_deg, faulted = [], []
    for omega in peak_speeds:
        b = _build(params, bldcsim, _sensor_cfg(params, sensor))
        _start_foc(b, params, omega, iq=iq)
        b.run_for(settle)
        nt = max(int(transition_s / dt), 1)
        npost = int(post_s / dt)
        pk = 0.0
        for k in range(nt + npost):
            if k <= nt:
                b.set_speed_clamp(True, omega * (1 - 2 * k / nt))
            b.run_for(dt)
            pk = max(pk, abs(float(M.angle_error_elec_rad(
                b.encoder_angle_rad, b.theta, pp))))
        peak_err_deg.append(np.degrees(pk))
        faulted.append(_any_fault(_flags(b)))
    peak_err_deg = np.asarray(peak_err_deg)
    ok = peak_err_deg < M.COMMUTATION_INVERT_DEG
    cliff = float(np.asarray(peak_speeds)[ok].max()) if ok.any() else 0.0
    return dict(peak_speed=np.asarray(peak_speeds, float),
                peak_err_deg=peak_err_deg, faulted=np.asarray(faulted),
                cliff_speed=cliff, sensor=sensor, label=SENSOR_LABEL[sensor])


def run_reversal_waveform(params, bldcsim, sensor, omega=200.0,
                          transition_s=0.01, iq=40, settle=0.04, dt=2e-5,
                          post_s=0.01):
    """One reversal, full time series (truth vs sensor electrical angle, error,
    phase current) - the T2 waveform panel."""
    pp = _pp(params)
    b = _build(params, bldcsim, _sensor_cfg(params, sensor))
    _start_foc(b, params, omega, iq=iq)
    b.run_for(settle)
    nt = max(int(transition_s / dt), 1)
    npost = int(post_s / dt)
    t, om, err, ia = [], [], [], []
    for k in range(nt + npost):
        if k <= nt:
            b.set_speed_clamp(True, omega * (1 - 2 * k / nt))
        b.run_for(dt)
        t.append(k * dt)
        om.append(b.omega)
        err.append(np.degrees(float(M.angle_error_elec_rad(
            b.encoder_angle_rad, b.theta, pp))))
        ia.append(b.currents[0])
    return dict(t=np.asarray(t), omega=np.asarray(om), err_deg=np.asarray(err),
                ia=np.asarray(ia), peak_speed=omega,
                sensor=sensor, label=SENSOR_LABEL[sensor])


# --------------------------------------------------------------------------- #
# T3 / T4 - current-sampling skew (scheme toggle on a fixed platform)
# --------------------------------------------------------------------------- #
def run_skew_sweep(params, bldcsim, scheme, omegas, iq=60,
                   settle=0.05, n=400, dt=2e-5):
    """dq current ripple (measurement error) vs speed for a sampling scheme.
    di/dt of the phase currents grows with speed, so this is the skew error vs
    di/dt (T3). The fundamental is detrended out - only the ripple remains."""
    ripple = []
    for omega in omegas:
        b = _build(params, bldcsim, foc(params, sample_scheme=scheme))
        _start_foc(b, params, omega, iq=iq)
        b.run_for(settle)
        idv, iqv = [], []
        for _ in range(n):
            b.run_for(dt)
            idv.append(b.foc_id)
            iqv.append(b.foc_iq)
        ripple.append(M.dq_ripple(idv, iqv))
    return dict(omega=np.asarray(omegas, float), ripple=np.asarray(ripple),
                scheme=scheme, label=SCHEME_LABEL[scheme])


def run_skew_spectrum(params, bldcsim, scheme, omega=120.0, iq=60,
                      settle=0.05, n=2048, dt=2e-5):
    """Steady-state spectrum of the measured id current for a sampling scheme -
    the measurement noise floor (T4)."""
    b = _build(params, bldcsim, foc(params, sample_scheme=scheme))
    _start_foc(b, params, omega, iq=iq)
    b.run_for(settle)
    idv = []
    for _ in range(n):
        b.run_for(dt)
        idv.append(b.foc_id)
    freqs, psd_db = M.noise_floor_fft(idv, fs=1.0 / dt)
    return dict(freqs=freqs, psd_db=psd_db, floor_db=float(np.median(psd_db)),
                scheme=scheme, label=SCHEME_LABEL[scheme])


# --------------------------------------------------------------------------- #
# T5 - snap-reversal system snapshot (whole sensor pair)
# --------------------------------------------------------------------------- #
def run_snap(params, bldcsim, sensor, omega=200.0, iq=60, settle=0.05,
             dt=1e-5, pre_s=0.004, post_s=0.02):
    """Near-instant +Omega -> -Omega flip; capture the phase currents, bus
    voltage, commutation error and any fault events through the event."""
    pp = _pp(params)
    b = _build(params, bldcsim, _sensor_cfg(params, sensor))
    _start_foc(b, params, omega, iq=iq)
    b.run_for(settle)
    npre = int(pre_s / dt)
    npost = int(post_s / dt)
    t, ia, ib, ic, vbus, err = [], [], [], [], [], []
    flip_at = npre
    for k in range(npre + npost):
        if k == flip_at:
            b.set_speed_clamp(True, -omega)   # the snap
        b.run_for(dt)
        c = b.currents
        t.append(k * dt)
        ia.append(c[0]); ib.append(c[1]); ic.append(c[2])
        vbus.append(b.bus_v)
        err.append(np.degrees(float(M.angle_error_elec_rad(
            b.encoder_angle_rad, b.theta, pp))))
    flags = _flags(b)
    peak_i = float(np.max(np.abs([ia, ib, ic])))
    return dict(t=np.asarray(t), ia=np.asarray(ia), ib=np.asarray(ib),
                ic=np.asarray(ic), vbus=np.asarray(vbus),
                err_deg=np.asarray(err), t_flip=flip_at * dt,
                peak_current=peak_i, flags=flags, faulted=_any_fault(flags),
                sensor=sensor, label=SENSOR_LABEL[sensor])


# --------------------------------------------------------------------------- #
# T7 - delivered angular resolution (slow rotation staircase)
# --------------------------------------------------------------------------- #
def run_resolution(params, bldcsim, sensor, omega=2.0, iq=20,
                   settle=0.05, n=3000, dt=5e-5):
    """Slow rotation: the measured angle climbs in quantization steps. Returns
    the delivered LSB (median step) and a staircase trace (mechanical deg)."""
    b = _build(params, bldcsim, _sensor_cfg(params, sensor))
    _start_foc(b, params, omega, iq=iq)
    b.run_for(settle)
    t, enc, truth, dbg = [], [], [], []
    for k in range(n):
        b.run_for(dt)
        t.append(k * dt)
        enc.append(b.encoder_angle_rad)
        truth.append(b.theta)
        dbg.append(b.angle)
    enc_u = np.unwrap(np.asarray(enc))
    d = np.diff(enc_u)
    steps = np.abs(d[np.abs(d) > 1e-9])
    lsb_deg = float(np.degrees(np.median(steps))) if steps.size else float("nan")
    return dict(t=np.asarray(t), enc_deg=np.degrees(enc_u),
                truth_deg=np.degrees(np.unwrap(np.asarray(truth))),
                dbg_levels=int(np.unique(dbg).size), lsb_deg=lsb_deg,
                sensor=sensor, label=SENSOR_LABEL[sensor])


# --------------------------------------------------------------------------- #
# T9 - dirty bench (realism layers, per layer breakdown)
# --------------------------------------------------------------------------- #
def _angle_err_std_deg(b, params, n, dt):
    pp = _pp(params)
    err = []
    for _ in range(n):
        b.run_for(dt)
        err.append(np.degrees(float(M.angle_error_elec_rad(
            b.encoder_angle_rad, b.theta, pp))))
    return float(np.std(err))


def run_dirty_bench(params, bldcsim, sensor, omega=60.0, iq=40,
                    settle=0.05, n=1500, dt=5e-5):
    """Angle-error std under each realism layer, isolated, for one sensor. The
    breakdown shows which imperfections an IC upgrade actually retires (latency-
    coupled) and which it cannot (mechanical eccentricity hits both)."""
    from bench_factory import realism
    over = dict(_sensor_cfg(params, sensor))  # platform sub-config
    layers = {"clean": [], "eccentricity": ["sensor"],
              "disturbance": ["disturbance"], "all": ["sensor", "disturbance"]}
    out = {}
    plat_over = {k: over[k] for k in over
                 if k in ("driver_name", "adc_name", "angle_name",
                          "drv_hw_mode", "angle_spi_mode", "adc_dual_mode",
                          "cur_norm_shift", "platform")}
    for name, groups in layers.items():
        cfg = realism(params, *groups, **plat_over)
        b = _build(params, bldcsim, cfg)
        _start_foc(b, params, omega, iq=iq)
        b.run_for(settle)
        out[name] = _angle_err_std_deg(b, params, n, dt)
    return dict(layers=out, sensor=sensor, label=SENSOR_LABEL[sensor])


# --------------------------------------------------------------------------- #
# T10 - operating-envelope map (speed x reversal abruptness)
# --------------------------------------------------------------------------- #
def run_envelope(params, bldcsim, sensor, peak_speeds, transitions,
                 iq=40, settle=0.03, dt=3e-5, post_s=0.006):
    """For each (peak reversal speed, transition time) cell, classify whether
    the commutation stays locked through the reversal. Returns a 2-D boolean
    grid (rows = transitions, cols = peak_speeds)."""
    pp = _pp(params)
    grid = np.zeros((len(transitions), len(peak_speeds)), dtype=bool)
    for i, trans in enumerate(transitions):
        for j, omega in enumerate(peak_speeds):
            b = _build(params, bldcsim, _sensor_cfg(params, sensor))
            _start_foc(b, params, omega, iq=iq)
            b.run_for(settle)
            nt = max(int(trans / dt), 1)
            npost = int(post_s / dt)
            pk = 0.0
            for k in range(nt + npost):
                if k <= nt:
                    b.set_speed_clamp(True, omega * (1 - 2 * k / nt))
                b.run_for(dt)
                pk = max(pk, abs(float(M.angle_error_elec_rad(
                    b.encoder_angle_rad, b.theta, pp))))
            locked = (np.degrees(pk) < M.COMMUTATION_INVERT_DEG
                      and not _any_fault(_flags(b)))
            grid[i, j] = locked
    return dict(peak_speeds=np.asarray(peak_speeds, float),
                transitions=np.asarray(transitions, float), locked=grid,
                locked_frac=float(grid.mean()),
                sensor=sensor, label=SENSOR_LABEL[sensor])
