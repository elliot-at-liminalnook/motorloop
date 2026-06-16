# SPDX-License-Identifier: MIT
"""Realism-checklist stage 7: the model-form validation harness.

7.1 stimulus round-trip and direct-vs-UART equivalence, 7.3 comparator
self-test (perturbed parameters must be localized to the right channels),
7.4 parameter-fit bootstrap against synthetic traces."""

from __future__ import annotations

import json
import math

import compare_traces
import fit_motor_params
import numpy as np
import pytest
import stimulus
from bench_factory import bench_config, expected_init_time, realism


@pytest.fixture
def align_offset(params):
    return int(params.value("rtl.sensored_align_offset"))


def run_stim(bldcsim, params, cfg, stim, via_uart=False):
    b = bldcsim.Bench(cfg)
    b.run_for(expected_init_time(params))
    assert b.configured
    stimulus.run(b, stim, via_uart=via_uart,
                 uart_baud=params.value("rtl.uart_baud"))
    return b


def trace_arrays(b):
    tr = b.trace
    return {k: np.array([float(v) for v in tr[k]]) for k in tr}


# ---------------------------------------------------------------------------
# 7.1 - stimulus format
# ---------------------------------------------------------------------------

def test_stimulus_json_roundtrip(tmp_path, align_offset):
    stim = stimulus.closed_loop_step(60, 90, align_offset)
    path = tmp_path / "stim.json"
    stimulus.save(stim, path)
    loaded = stimulus.load(path)
    assert loaded == stim
    json.dumps(loaded)  # serializable


def test_stimulus_direct_vs_uart_equivalent(params, bldcsim, align_offset):
    """The same stimulus through the direct ports and through the UART
    register file must produce equivalent outcomes (within the latency of
    the serial writes)."""
    stim = stimulus.closed_loop_step(70, 70, align_offset, duration=1.0)
    direct = run_stim(bldcsim, params, bench_config(params), stim)
    uarted = run_stim(bldcsim, params, bench_config(params), stim,
                      via_uart=True)
    assert abs(direct.omega - uarted.omega) < 0.1 * max(direct.omega, 1.0)
    assert uarted.omega > 50


# ---------------------------------------------------------------------------
# 7.3 - comparator self-test: perturbations localized
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def comparator_traces(params, bldcsim):
    align = int(params.value("rtl.sensored_align_offset"))
    stim = stimulus.closed_loop_step(70, 100, align, duration=1.4)

    def run(cfg):
        b = bldcsim.Bench(cfg)
        b.run_for(expected_init_time(params))
        stimulus.run(b, stim)
        return trace_arrays(b)

    baseline = run(bench_config(params))
    baseline2 = run(bench_config(params))
    r_perturbed = run(bench_config(
        params, motor={"resistance_ohm": 1.2 * params.value("motor.R")}))
    return baseline, baseline2, r_perturbed


def test_comparator_identical_runs_report_zero(comparator_traces):
    baseline, baseline2, _ = comparator_traces
    result = compare_traces.compare(baseline, baseline2,
                                    align_channel="omega_rad_s",
                                    skip_initial_s=0.05)
    for ch, m in result["channels"].items():
        assert m["rms_ratio"] < 1e-9, f"{ch}: {m['rms_ratio']}"


def test_comparator_localizes_resistance_error(comparator_traces):
    """+20% R must show up dominantly in the current channels - the
    acceptance test for the comparator (checklist 7.3)."""
    baseline, _, perturbed = comparator_traces
    result = compare_traces.compare(baseline, perturbed,
                                    align_channel="omega_rad_s",
                                    skip_initial_s=0.1)
    cur = result["channels"]["ia_a"]["rms_ratio"]
    spd = result["channels"]["omega_rad_s"]["rms_ratio"]
    assert cur > 0.02, f"current channel blind to +20% R: {cur:.4f}"
    assert cur > 1.5 * spd, (
        f"not localized: current ratio {cur:.3f} vs speed {spd:.3f}"
    )


def test_report_renders(comparator_traces):
    baseline, _, perturbed = comparator_traces
    result = compare_traces.compare(baseline, perturbed,
                                    align_channel="omega_rad_s")
    text = compare_traces.report_markdown(result, "ref", "cand")
    assert "| ia_a |" in text and "RMS ratio" in text


# ---------------------------------------------------------------------------
# 7.4 - parameter-fit bootstrap (synthetic self-tests with noise)
# ---------------------------------------------------------------------------

def test_fit_rl_from_locked_rotor_step(params, bldcsim):
    """Locked rotor, both FETs full on: series pair sees vbus through
    2R + 2rds. The fit recovers the pair values within a few percent
    despite measurement noise."""
    cfg = bench_config(params, motor={"inertia_kg_m2": 1e9})
    plant = bldcsim.ThreePhasePlant(cfg["motor"], cfg["bridge"], cfg["plant"])
    plant.set_gates([True, False, False], [False, True, False])
    rng = np.random.default_rng(7)
    ts, cur = [], []
    t = 0.0
    for _ in range(800):
        plant.advance(2e-5)
        t += 2e-5
        ts.append(t)
        cur.append(plant.currents_a[0] + rng.normal(0, 0.02))
    r_pair_true = 2 * (params.value("motor.R")
                       + params.value("inverter.fet_rds_on"))
    r_fit, l_fit = fit_motor_params.fit_rl(
        np.array(ts), np.array(cur), params.value("bus.vbus"))
    assert abs(r_fit - r_pair_true) < 0.03 * r_pair_true
    l_pair_true = 2 * params.value("motor.L")
    assert abs(l_fit - l_pair_true) < 0.10 * l_pair_true


def test_fit_ke_from_coast_capture(params, bldcsim):
    """External spin then coast: the 'scope capture' is (omega, v_ll) from
    the floating terminals; the fit recovers ke_line_line_peak."""
    cfg = bench_config(params, motor={"inertia_kg_m2": 5e-5})
    plant = bldcsim.ThreePhasePlant(cfg["motor"], cfg["bridge"], cfg["plant"])
    plant.set_gates([False] * 3, [False] * 3)
    plant.set_load_torque(-0.02)
    plant.advance(0.5)
    plant.set_load_torque(0.0)
    rng = np.random.default_rng(11)
    omegas, vlls = [], []
    for _ in range(4000):
        plant.advance(2e-4)
        out = plant.outputs()
        omegas.append(plant.omega_rad_s)
        vlls.append(out["terminal_v"][0] - out["terminal_v"][1]
                    + rng.normal(0, 0.01))
    ke_fit = fit_motor_params.fit_ke_ll(np.array(omegas), np.array(vlls))
    ke_true = params.value("motor_spec.ke_line_line_peak")
    assert abs(ke_fit - ke_true) < 0.05 * ke_true, f"{ke_fit} vs {ke_true}"


def test_fit_spin_down_recovers_b_and_coulomb(params, bldcsim):
    cfg = realism(params, "mechanical")
    plant = bldcsim.ThreePhasePlant(cfg["motor"], cfg["bridge"], cfg["plant"])
    plant.set_gates([False] * 3, [False] * 3)
    plant.set_load_torque(-0.05)
    plant.advance(0.6)
    plant.set_load_torque(0.0)
    ts, ws = [], []
    t = 0.0
    while plant.omega_rad_s > 1.0 and t < 5.0:
        ts.append(t)
        ws.append(plant.omega_rad_s)
        plant.advance(2e-3)
        t += 2e-3
    b_fit, tc_fit = fit_motor_params.fit_spin_down(
        np.array(ts), np.array(ws), params.value("motor.J"))
    assert abs(b_fit - params.value("motor.B")) < 0.08 * params.value("motor.B")
    tc_true = params.value("motor.coulomb_friction")
    assert abs(tc_fit - tc_true) < 0.10 * tc_true


def test_kv_conversion_inverse(params):
    import derive_params
    kv = fit_motor_params.kv_from_ke_ll(0.0346410)
    assert math.isclose(derive_params.kv_to_ke_line_line(kv), 0.0346410,
                        rel_tol=1e-12)
