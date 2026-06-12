"""Stage 7.2: three-way plant parity — C++ vs Python vs Modelica oracle.

The C++/Python pair share the integrator and match to ~1e-8 (stage 2 test).
The Modelica oracle uses dassl with event-located mode transitions and emits
extra output rows at event instants, so it is linearly interpolated onto the
fixed grid before comparison. Measured agreement is ~0.2% RMS on currents
and tighter on mechanical states; thresholds below give modest headroom.
"""

from __future__ import annotations

import math
import shutil
from bisect import bisect_left
from pathlib import Path

import pytest
from trajectory_compare import compare_interpolated

PROJECT_ROOT = Path(__file__).resolve().parents[2]

needs_omc = pytest.mark.skipif(shutil.which("omc") is None,
                               reason="omc not installed (stage 0.2)")


def rms(values):
    return math.sqrt(sum(v * v for v in values) / len(values))


def interp_column(ts, vs, t):
    i = bisect_left(ts, t)
    if i <= 0:
        return vs[0]
    if i >= len(ts):
        return vs[-1]
    t0, t1, v0, v1 = ts[i - 1], ts[i], vs[i - 1], vs[i]
    if t1 == t0:
        return v1
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)


@pytest.fixture(scope="module")
def oracle_trace(params):
    from run_three_phase_oracle import run_oracle

    return run_oracle(params, PROJECT_ROOT / "sim" / "build" / "oracle")


@pytest.fixture(scope="module")
def cpp_trace(params, bldcsim):
    prefix = "scenario.three_phase_open_loop"
    motor = {
        "resistance_ohm": params.value("motor.R"),
        "inductance_h": params.value("motor.L"),
        "ke_v_s_per_rad": params.value("motor.Ke"),
        "inertia_kg_m2": params.value("motor.J"),
        "damping_n_m_s_per_rad": params.value("motor.B"),
        "pole_pairs": int(params.value("motor.pole_pairs")),
        "trapezoid_blend": params.value("motor.emf_trapezoid_blend"),
        "load_torque_n_m": params.value(f"{prefix}.load_torque"),
    }
    bridge = {
        "vbus_v": params.value("bus.vbus"),
        "fet_rds_on_ohm": params.value("inverter.fet_rds_on"),
        "diode_vf_v": params.value("inverter.body_diode_vf"),
    }
    config = {
        "current_epsilon_a": params.value("sim.current_epsilon"),
        "max_substep_s": params.value(f"{prefix}.max_substep"),
    }
    return bldcsim.simulate_three_phase_open_loop(
        motor, bridge, config,
        params.value(f"{prefix}.duty"),
        params.value(f"{prefix}.f_elec_final"),
        params.value(f"{prefix}.ramp_time"),
        params.value(f"{prefix}.stop"),
        params.value(f"{prefix}.dt"),
    )


@needs_omc
def test_oracle_produced_full_run(params, oracle_trace, cpp_trace):
    # The oracle adds event rows on top of the regular grid.
    assert len(oracle_trace["time_s"]) >= len(cpp_trace["time_s"])
    assert abs(oracle_trace["time_s"][-1] - cpp_trace["time_s"][-1]) < 1e-9


@needs_omc
def test_oracle_smooth_states_pointwise(params, oracle_trace, cpp_trace):
    result = compare_interpolated(
        cpp_trace, oracle_trace, ["omega_rad_s", "theta_rad"],
        abs_tol=0.5, rel_tol=0.01)
    assert result.ok, result.report()


@needs_omc
def test_oracle_currents_rms(params, oracle_trace, cpp_trace):
    ot = oracle_trace["time_s"]
    for col in ("ia_a", "ib_a", "ic_a"):
        ov = oracle_trace[col]
        err = [v - interp_column(ot, ov, t)
               for t, v in zip(cpp_trace["time_s"], cpp_trace[col])]
        ref = rms(cpp_trace[col])
        assert ref > 0.1, f"{col}: implausibly quiet reference"
        ratio = rms(err) / ref
        # Measured ~0.002; commutation-instant timing differences dominate.
        assert ratio < 0.02, f"{col}: RMS mismatch {ratio:.4f}"


@needs_omc
def test_oracle_final_state_close(params, oracle_trace, cpp_trace):
    w_cpp = cpp_trace["omega_rad_s"][-1]
    w_oracle = oracle_trace["omega_rad_s"][-1]
    assert abs(w_cpp - w_oracle) < 0.005 * abs(w_oracle), (
        f"final omega: cpp {w_cpp} vs oracle {w_oracle}"
    )
