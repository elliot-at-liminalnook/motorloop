# SPDX-License-Identifier: MIT
"""Part-comparison study - the 10 tests (notes/part-comparison-checklist.md §2).

Each test holds the FOC controller fixed and changes exactly one part, then
asserts the *predicted ordering* as a regression guard (or records "no modeled
difference" where the models don't separate the parts - the honesty clause).
The experiments live in part_compare.py so the figures (gen_comparison_figures.py)
render from the same runners. Sizes here are coarse for CI speed; the figure
generator re-runs them finer. Heavy rendering is on demand: `make compare`.

Pairs:
  * angle sensor  AS5600 vs AS5047P  (T1,T2,T5,T6,T7,T8,T9,T10)
  * current sampling  MCP3208 (sequential skew) vs ADS9224R (simultaneous)  (T3,T4)
"""

from __future__ import annotations

import numpy as np
import pytest

import part_compare as PC
import part_metrics as M


# A shared speed sweep (T1/T6/T8 all read it) - computed once per module.
SWEEP_OMEGAS = [40, 120, 300, 600, 1000]


@pytest.fixture(scope="module")
def speed_sweep(params, bldcsim):
    return {s: PC.run_speed_sweep(params, bldcsim, s, SWEEP_OMEGAS, n=160)
            for s in (PC.SENSOR_A, PC.SENSOR_B)}


# --------------------------------------------------------------------------- #
# T1 - latency curve: angle error vs speed
# --------------------------------------------------------------------------- #
def test_t1_latency_curve(speed_sweep):
    a = speed_sweep[PC.SENSOR_A]   # AS5600
    s = speed_sweep[PC.SENSOR_B]   # AS5047P
    # AS5047P is more precise at every speed above standstill.
    assert np.all(s["rms_deg"][1:] < a["rms_deg"][1:])
    # AS5600 error grows monotonically with speed (the latency signature).
    assert np.all(np.diff(a["rms_deg"]) > 0)
    # The gap is large where it matters (top speed): >= 20x.
    assert a["rms_deg"][-1] / max(s["rms_deg"][-1], 1e-9) > 20.0


# --------------------------------------------------------------------------- #
# T2 - reversal cliff: max reversal speed before commutation inverts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("transition_s", [0.01])
def test_t2_reversal_cliff(params, bldcsim, transition_s):
    speeds = [100, 300, 600, 900, 1300]
    a = PC.run_reversal(params, bldcsim, PC.SENSOR_A, speeds,
                        transition_s=transition_s)
    s = PC.run_reversal(params, bldcsim, PC.SENSOR_B, speeds,
                        transition_s=transition_s)
    # AS5047P survives a strictly faster reversal than AS5600.
    assert s["cliff_speed"] > a["cliff_speed"]
    # AS5600 breaks somewhere in the swept range; AS5047P clears it all.
    assert a["cliff_speed"] < speeds[-1]
    assert s["cliff_speed"] == speeds[-1]


# --------------------------------------------------------------------------- #
# T3 - current-sample skew vs di/dt (MCP3208 sequential vs ADS9224R simultaneous)
# --------------------------------------------------------------------------- #
def test_t3_skew_vs_didt(params, bldcsim):
    omegas = [60, 120, 240]
    sim = PC.run_skew_sweep(params, bldcsim, PC.SCHEME_SIMULTANEOUS, omegas,
                            n=200)
    seq = PC.run_skew_sweep(params, bldcsim, PC.SCHEME_SEQUENTIAL, omegas,
                            n=200)
    # The sequential single-ADC skew injects more dq ripple than simultaneous
    # sampling, at every speed (Q21). Compare the means to be robust to the
    # high-speed regime where the fundamental di/dt dominates both.
    assert seq["ripple"].mean() > sim["ripple"].mean()
    assert seq["ripple"][0] > sim["ripple"][0]   # clearest at low speed


# --------------------------------------------------------------------------- #
# T4 - quantization / measurement noise floor (FFT)
# --------------------------------------------------------------------------- #
def test_t4_noise_floor(params, bldcsim):
    sim = PC.run_skew_spectrum(params, bldcsim, PC.SCHEME_SIMULTANEOUS, n=1024)
    seq = PC.run_skew_spectrum(params, bldcsim, PC.SCHEME_SEQUENTIAL, n=1024)
    # The sequential path has the higher measurement-noise floor.
    assert seq["floor_db"] > sim["floor_db"]


# --------------------------------------------------------------------------- #
# T5 - snap-reversal: commutation integrity through a full-speed flip
# --------------------------------------------------------------------------- #
def test_t5_snap_reversal(params, bldcsim):
    a = PC.run_snap(params, bldcsim, PC.SENSOR_A, omega=200.0)
    s = PC.run_snap(params, bldcsim, PC.SENSOR_B, omega=200.0)
    # During the snap, AS5600's commutation error is large; AS5047P stays tight.
    assert M.peak_abs(a["err_deg"]) > 10.0
    assert M.peak_abs(s["err_deg"]) < 2.0
    assert M.peak_abs(a["err_deg"]) > 5.0 * M.peak_abs(s["err_deg"])
    # Honest: with the stiff-supply default, neither latches a fault - the
    # difference is commutation integrity, not a tripped protection.
    assert a["faulted"] == s["faulted"]


# --------------------------------------------------------------------------- #
# T6 - commutation phase margin: max locked speed
# --------------------------------------------------------------------------- #
def test_t6_phase_margin(speed_sweep):
    a = speed_sweep[PC.SENSOR_A]
    s = speed_sweep[PC.SENSOR_B]
    # Signed mean error = commutation lag (negative: the sensor trails truth).
    # The lag magnitude grows with speed - the phase-margin erosion.
    assert abs(a["mean_deg"][-1]) > abs(a["mean_deg"][0])
    # AS5047P holds commutation across the whole sweep; AS5600 does not.
    assert PC.max_locked_speed(s) >= PC.max_locked_speed(a)
    assert PC.max_locked_speed(a) < SWEEP_OMEGAS[-1]


# --------------------------------------------------------------------------- #
# T7 - delivered angular resolution (slow-rotation staircase)
# --------------------------------------------------------------------------- #
def test_t7_resolution(params, bldcsim):
    a = PC.run_resolution(params, bldcsim, PC.SENSOR_A, n=2000)
    s = PC.run_resolution(params, bldcsim, PC.SENSOR_B, n=2000)
    # AS5047P delivers a finer angular step.
    assert s["lsb_deg"] < a["lsb_deg"]
    # Honest: the delivered advantage is ~2-3x, NOT the full 14-vs-12-bit (4x) -
    # the RTL/filter path compresses it. Guard that nuance against regression.
    ratio = a["lsb_deg"] / max(s["lsb_deg"], 1e-9)
    assert 1.5 < ratio < 4.5


# --------------------------------------------------------------------------- #
# T8 - the torque penalty of angle latency
# --------------------------------------------------------------------------- #
def test_t8_torque_penalty(speed_sweep):
    a = speed_sweep[PC.SENSOR_A]
    s = speed_sweep[PC.SENSOR_B]
    # Torque efficiency = mean(cos(angle error)). AS5047P stays ~1.0; AS5600
    # decays and eventually goes braking (cos<0) at high speed.
    assert s["torque_eff"][-1] > 0.95
    assert a["torque_eff"][-1] < s["torque_eff"][-1]
    assert a["torque_eff"][-1] < 0.6   # a real penalty at top speed


# --------------------------------------------------------------------------- #
# T9 - dirty bench: which imperfections an IC upgrade actually retires
# --------------------------------------------------------------------------- #
def test_t9_dirty_bench(params, bldcsim):
    a = PC.run_dirty_bench(params, bldcsim, PC.SENSOR_A, n=1000)
    s = PC.run_dirty_bench(params, bldcsim, PC.SENSOR_B, n=1000)
    # The electrical disturbance layer does not corrupt the digital angle path
    # (both sensors): angle-error std unchanged from clean.
    assert a["layers"]["disturbance"] == pytest.approx(a["layers"]["clean"],
                                                        rel=0.2)
    # Mechanical eccentricity hits BOTH sensors and dominates - the honest
    # finding: a better sensor IC does not fix a mounting/magnet problem.
    assert a["layers"]["eccentricity"] > 3 * a["layers"]["clean"]
    _, no_diff = M.expected_order(a["layers"]["eccentricity"],
                                  s["layers"]["eccentricity"],
                                  eps=0.5 * a["layers"]["eccentricity"])
    assert no_diff, "eccentricity should affect both sensors comparably"


# --------------------------------------------------------------------------- #
# T10 - operating-envelope map: speed x reversal abruptness
# --------------------------------------------------------------------------- #
def test_t10_envelope(params, bldcsim):
    speeds = [100, 400, 800, 1200]
    transitions = [0.02, 0.004]
    a = PC.run_envelope(params, bldcsim, PC.SENSOR_A, speeds, transitions)
    s = PC.run_envelope(params, bldcsim, PC.SENSOR_B, speeds, transitions)
    # AS5047P's locked envelope strictly contains AS5600's.
    assert s["locked_frac"] > a["locked_frac"]
    assert bool(s["locked"].all())
    # The boundary is set by reversal SPEED, not abruptness: within a column
    # (fixed speed) the lock verdict is the same across transition rates.
    for col in range(a["locked"].shape[1]):
        assert len(set(a["locked"][:, col].tolist())) == 1
