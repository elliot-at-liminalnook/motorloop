"""FOC stage 8: scenario findings.

8.3 dead-time distortion: FOC chops all three legs continuously (no floating
phase like six-step), so the bridge dead time and body-diode conduction
distort the phase current - largest as a fraction at low modulation. The
bench models both (the shoot-through checker and the dead-time figure verify
the timing), so the distortion is present in every FOC run. Here it is
measured and shown bounded - the loop tolerates it. (Field weakening, 8.2, is
not applicable to the placeholder motor; see the checklist findings.)
Placeholder motor params.
"""

from __future__ import annotations

import math
import statistics

import numpy as np

from bench_factory import foc, expected_init_time


def _current_distortion(bldcsim, params, omega, iq_cmd=80, n=2000, dt=2e-4):
    """Residual-to-fundamental ratio (%) of phase-A current at a clamped
    speed: the deviation from the ideal sinusoid, dominated by dead-time and
    body-diode conduction at this scale."""
    pp = int(params.value("motor.pole_pairs"))
    b = bldcsim.Bench(foc(params))
    b.run_for(expected_init_time(params))
    b.set_speed_clamp(True, omega)
    b.set_align_offset(int(params.value("foc.align_offset")))
    b.set_id_target(0)
    b.set_iq_target(iq_cmd)
    b.set_mode(3)
    b.run_for(0.1)
    ts, ia = [], []
    for k in range(n):
        b.run_for(dt)
        ts.append(k * dt)
        ia.append(b.currents[0])
    t = np.array(ts)
    y = np.array(ia)
    we = pp * omega
    basis = np.column_stack([np.sin(we * t), np.cos(we * t),
                             np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
    resid = y - basis @ coef
    fund_rms = math.hypot(coef[0], coef[1]) / math.sqrt(2)
    return (statistics.pstdev(resid) / max(fund_rms, 1e-6) * 100.0,
            b.shoot_through_violations)


def test_dead_time_distortion_present_and_bounded(bldcsim, params):
    """The continuous-modulation dead-time/body-diode distortion is present
    (a few percent) and bounded - the FOC loop runs cleanly through it with
    zero shoot-through. The 8.3 finding."""
    distortion, shoot = _current_distortion(bldcsim, params, omega=10.0)
    assert shoot == 0
    # Present: the continuous-leg dead time leaves a measurable signature.
    assert distortion > 0.3, (
        f"expected a measurable distortion signature, got {distortion:.2f}%")
    # Bounded: the loop tolerates it (no runaway harmonic).
    assert distortion < 8.0, f"distortion unexpectedly large: {distortion:.2f}%"


def test_foc_runs_clean_across_speed_range(bldcsim, params):
    """The loop holds zero shoot-through across the AS5600's usable speed
    range at a fixed torque current (a robustness sweep)."""
    for omega in (10.0, 40.0, 90.0):
        _, shoot = _current_distortion(bldcsim, params, omega, n=400)
        assert shoot == 0, f"shoot-through at omega={omega}"
