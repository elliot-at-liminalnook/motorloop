"""FOC stage 6: the full cascade - closed-loop speed control spinning the
simulated PMSM. The outer speed PI commands iq*; the inner current loop holds
id=0 and regulates iq; the modulator drives the bridge. Placeholder motor
params: this checks the control law, not a hardware number.
"""

from __future__ import annotations

import statistics

import pytest

from bench_factory import foc, expected_init_time


def _spin(bldcsim, params, target, seconds=1.2, load_nm=0.0, **focargs):
    b = bldcsim.Bench(foc(params, **focargs))
    b.run_for(expected_init_time(params))
    assert b.angle_valid
    b.set_align_offset(int(params.value("foc.align_offset")))
    b.set_id_target(0)
    b.set_foc_speed_loop(True)
    b.set_target_speed(target)
    if load_nm:
        b.set_load_torque(load_nm)
    b.set_mode(3)
    b.run_for(seconds)
    return b


def _tail_mean(b, attr, window=0.3, dt=2e-4):
    vals = []
    for _ in range(int(window / dt)):
        b.run_for(dt)
        vals.append(getattr(b, attr))
    return statistics.mean(vals)


@pytest.mark.parametrize("target", [60, 80, 100])
def test_foc_speed_reaches_target(bldcsim, params, target):
    """The FOC loop spins the PMSM to the commanded speed and holds it with
    id ~ 0 and zero shoot-through."""
    b = _spin(bldcsim, params, target)
    assert b.shoot_through_violations == 0
    omega = _tail_mean(b, "omega")
    assert omega == pytest.approx(target, rel=0.08), (
        f"settled at {omega:.1f}, target {target}")
    # Field orientation: the d-axis current stays near zero.
    b2 = _spin(bldcsim, params, target)
    id_mean = _tail_mean(b2, "foc_id")
    assert abs(id_mean) < 8, f"id not held near 0: {id_mean:.1f}"


def test_foc_iq_tracks_load(bldcsim, params):
    """Adding a load torque raises the steady-state iq command (the loop
    supplies more torque current), while speed is held at target."""
    target = 80
    no_load = _spin(bldcsim, params, target, load_nm=0.0)
    iq_noload = _tail_mean(no_load, "foc_iq")
    omega_noload = no_load.omega

    loaded = _spin(bldcsim, params, target, load_nm=0.03)
    iq_loaded = _tail_mean(loaded, "foc_iq")
    omega_loaded = loaded.omega

    assert omega_loaded == pytest.approx(target, rel=0.1), (
        f"speed not held under load: {omega_loaded:.1f}")
    assert iq_loaded > iq_noload + 10, (
        f"iq did not rise under load: {iq_noload:.0f} -> {iq_loaded:.0f}")
    del omega_noload


def test_foc_shoot_through_clean_through_transient(bldcsim, params):
    """A step in target speed (acceleration transient, high modulation) keeps
    the shoot-through checker clean - the per-leg SVPWM dead time holds."""
    b = bldcsim.Bench(foc(params))
    b.run_for(expected_init_time(params))
    b.set_align_offset(int(params.value("foc.align_offset")))
    b.set_id_target(0)
    b.set_foc_speed_loop(True)
    b.set_mode(3)
    for target in (40, 90, 20, 110):
        b.set_target_speed(target)
        b.run_for(0.4)
    assert b.shoot_through_violations == 0
    assert b.plant_shoot_through_requests == 0
