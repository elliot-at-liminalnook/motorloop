# SPDX-License-Identifier: MIT
"""Motor-selection study (notes/motor-selection-checklist.md): the three motor
profiles + their head-to-head comparison.

Most metrics are closed-form from the datasheet params (Kt, torque-speed, L/R,
acceleration, efficiency, the angle-latency M8 coupling) - so the comparison and
its assertions are rebuild-free. The DB42 (4 pp) additionally runs on the current
RTL build (POLE_PAIRS=4) as a bench cross-check; the GM2804 (7) and EC 45 (8)
need build_motor.sh (a re-Verilate) for a full bench run.
"""

from __future__ import annotations

import math

import derive_params
import numpy as np
from bench_factory import expected_init_time, foc
from motors import MOTORS, SENSOR_LATENCY_S, TIERS, kv_to_ke_line_line

VBUS = 12.0


# ----- §0 conversions consistent with the bench's motor derivation -----------
def test_conversions_match_derive_params():
    # The Kv->Ke relation is the same one derive_params publishes.
    assert math.isclose(kv_to_ke_line_line(100.0),
                        derive_params.kv_to_ke_line_line(100.0), rel_tol=1e-12)
    for m in MOTORS.values():
        assert math.isclose(m.r_phase, m.r_line_line / 2.0)        # R=Rll/2
        assert math.isclose(m.l_phase, m.l_line_line / 2.0)        # L=Lll/2
        assert math.isclose(m.ke_phase, m.ke_line_line_peak / math.sqrt(3.0))
        assert m.kt == m.ke_phase                                  # Kt=Ke
        assert m.pole_pairs == m.pole_count // 2


def test_profiles_present_and_tagged():
    assert TIERS == ["gm2804", "db42s03", "maxon_ec45"]
    for m in MOTORS.values():
        assert m.provenance and m.price_usd > 0 and m.pole_count > 0


# ----- §3 the performance comparison (analytical orderings) ------------------
def test_M6_efficiency_ordering():
    """The gimbal's high resistance makes it the least efficient; the
    characterized industrial/servo motors are far better at rated."""
    eff = {n: MOTORS[n].efficiency(MOTORS[n].rated_current_a, 200.0) for n in TIERS}
    assert eff["gm2804"] < eff["db42s03"]
    assert eff["gm2804"] < eff["maxon_ec45"]
    assert eff["gm2804"] < 0.6 and eff["db42s03"] > 0.6


def test_M2_stall_torque_ordering():
    """At 12 V the gimbal's 9 Ohm starves its stall current/torque; the low-R
    motors deliver much more."""
    st = {n: MOTORS[n].stall_torque_nm(VBUS) for n in TIERS}
    assert st["gm2804"] < st["db42s03"] < st["maxon_ec45"]
    assert st["gm2804"] < 0.1   # < 100 mN*m


def test_M3_electrical_time_constant_distinct():
    """Each motor has a materially different L/R -> different current-loop
    dynamics (and PI tuning)."""
    taus = sorted(MOTORS[n].elec_tau_s for n in TIERS)
    assert taus[-1] > 2 * taus[0]   # spread > 2x


def test_M8_latency_sensitivity_scales_with_pole_pairs():
    """The motor<->sensor coupling: at speed, angle-latency torque loss grows
    with pole pairs, so the 8 pp maxon / 7 pp gimbal are far more sensitive than
    the 4 pp DB42 - they *need* the AS5047P where the DB42 tolerates the AS5600."""
    omega = 400.0
    lat = SENSOR_LATENCY_S["AS5600"]
    loss = {n: MOTORS[n].latency_torque_loss(omega, lat) for n in TIERS}
    assert loss["maxon_ec45"] > loss["gm2804"] > loss["db42s03"]
    # And the AS5047P essentially removes it for every motor.
    for n in TIERS:
        s = MOTORS[n].latency_torque_loss(omega, SENSOR_LATENCY_S["AS5047P"])
        assert s < loss[n] / 50.0


# ----- §2 DB42 (4 pp) runs on the current build ------------------------------
def test_db42_runs_on_current_build(bldcsim, params):
    """The DB42 is 4 pole pairs = the current POLE_PAIRS build, so it runs the
    FOC loop with no re-Verilate: init, track torque current, zero shoot-through."""
    cfg = foc(params)
    cfg["motor"].update(MOTORS["db42s03"].cfg_motor(cfg["motor"]))
    b = bldcsim.Bench(cfg)
    b.run_for(expected_init_time(params))
    assert b.configured
    b.set_align_offset(int(params.value("foc.align_offset")))
    b.set_id_target(0); b.set_iq_target(60); b.set_speed_clamp(True, 80.0)
    b.set_mode(3); b.run_for(0.1)
    iq = []
    for _ in range(200):
        b.run_for(2e-5); iq.append(b.foc_iq)
    assert abs(np.mean(iq) - 60.0) < 15.0
    assert b.shoot_through_violations == 0
