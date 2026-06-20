# SPDX-License-Identifier: MIT
"""R2 — the hardware-ID suite: per-component measurements + the fit-to-trace loop.

Define the measurements that pin the sim to a real machine, run them against the sim
model to emit traces, and wire the real->sim->real fit hook. The measurement DEFINITIONS
and the loop are buildable + sim-to-sim verifiable now; the real numbers need real parts
(honestly hardware-gated). Inference reuses `domain_model` (the RS1 posterior) and
`reality_gap.score_trace_mismatch` (the R5 truth gate) — no fork.

Measurements (each maps a world `dp` -> a trace; the analytic models come from
reality_gap so the suite is consistent with what the envs consume):
  torque_speed  : available-torque fraction over a joint-speed sweep (back-EMF droop)
  stall_torque  : torque fraction at zero speed (current limit)
  thermal_rise  : torque loss as the thermal derate engages
  latency_step  : command->response delay (a delayed step)
  friction_slide: tangential force ratio (a slide test)
  step_response : damped 2nd-order settle (mass/damping/stiffness)
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reality_gap import (default_uncertainty, sample_domain_params, actuator_scale,  # noqa: E402
                         score_trace_mismatch, log_parity_trace)
import domain_model as dm  # noqa: E402

SPEEDS = np.linspace(0.0, 160.0, 24)      # wide enough to enter the back-EMF droop region


def torque_speed(dp):   return np.asarray(actuator_scale(SPEEDS, dp), float)
def stall_torque(dp):   return np.asarray(actuator_scale(np.zeros(4), dp), float)


def thermal_rise(dp):
    # torque fraction as the derate ramps from cold (0) to hot (full derate)
    t = np.linspace(0, 1, 20)
    hot = {**dp, "thermal_derate": dp["thermal_derate"] * 1.0}
    cold = {**dp, "thermal_derate": 0.0}
    f_cold = float(actuator_scale(np.array([5.0]), cold)[0])
    f_hot = float(actuator_scale(np.array([5.0]), hot)[0])
    return f_cold + (f_hot - f_cold) * t


def latency_step(dp):
    t = np.linspace(0, 0.05, 25)
    return (t >= dp["latency_s"]).astype(float)


def friction_slide(dp):
    # tangential force vs normal load: slope = mu (friction)
    load = np.linspace(0, 10, 20)
    return dp["friction"] * load


def step_response(dp):
    t = np.linspace(0, 1, 30)
    zeta = 0.3 + 0.5 * dp["joint_damping"]
    return 1 - np.exp(-zeta * 6 * t) * np.cos(6 * t)


MEASUREMENTS = {
    "torque_speed": torque_speed, "stall_torque": stall_torque,
    "thermal_rise": thermal_rise, "latency_step": latency_step,
    "friction_slide": friction_slide, "step_response": step_response,
}


def run_suite(dp) -> dict:
    """Run every measurement against the sim model for world `dp` -> {name: trace}."""
    return {name: fn(dp) for name, fn in MEASUREMENTS.items()}


def _world_to_dp(world: dict):
    """Map a domain_model world (ranking axes) onto a full reality_gap dp for the suite."""
    dp = sample_domain_params(0, default_uncertainty())
    dp = {**dp, "friction": world["friction"], "joint_damping": world["joint_damping"],
          "kt": dp["kt"] * world["kt_scale"], "ke_phase": dp["ke_phase"] * world["kt_scale"],
          "i_limit": dp["i_limit"] * world["i_limit_scale"], "latency_s": world["latency_s"]}
    return dp


def suite_mismatch(world_norm, real_traces) -> float:
    """Pointwise (time-aligned) mismatch of the suite's sim traces vs real traces.
    R2 measurements are CONTROLLED tests (a commanded step, a speed sweep) so they ARE
    time-aligned — pointwise MSE keeps the edge/settle/slope info. (Contrast R5's
    `score_trace_mismatch`, which distribution-matches the UNcontrolled fight where time
    alignment is meaningless. Different gate for a different signal — same idea, by design.)"""
    sim = run_suite(_world_to_dp(dm.denorm(world_norm)))
    return float(np.mean([np.mean((sim[k] - real_traces[k]) ** 2)
                          for k in real_traces if k in sim]))


def fit_to_traces(real_traces, rounds=8, n=64, seed=0):
    """The real->sim->real loop: tighten the world posterior so the suite reproduces the
    measured traces. `real_traces` = bench logs on hardware; here a hidden sim world."""
    rng = np.random.default_rng(seed)
    post = dm.Posterior.prior()
    for _ in range(rounds):
        cand = post.sample_norm(n, rng)
        mism = np.array([suite_mismatch(u, real_traces) for u in cand])
        post = dm.update_posterior(post, cand, mism)
    return post


if __name__ == "__main__":
    # 1. the suite runs against the sim model and emits traces
    dp0 = sample_domain_params(0, default_uncertainty())
    traces = run_suite(dp0)
    print(f"[R2] suite ran {len(traces)} measurements; trace lengths: "
          f"{ {k: len(v) for k, v in traces.items()} }")
    log_parity_trace(traces, HERE / "../build/hwid_traces.npz")
    print(f"[R2] torque-speed droop: stall {traces['torque_speed'][0]:.2f} -> "
          f"{SPEEDS[-1]:.0f}rad/s {traces['torque_speed'][-1]:.2f} (back-EMF, the ranking-relevant effect)")

    # 2. the fit hook is wired: recover a HIDDEN world from its suite traces (sim-to-sim)
    rng = np.random.default_rng(3)
    true_u = rng.uniform(0.2, 0.8, dm.DIM)
    real = run_suite(_world_to_dp(dm.denorm(true_u)))
    post = fit_to_traces(real)
    # Identifiable from THIS suite: friction (slide), damping (step), latency (edge),
    # kt (droop onset). i_limit/restitution/mass are NOT separable from a normalized
    # torque curve (kt and current-limit confound; mass/restitution need a drop test) —
    # an honest identifiability boundary, flagged here, that RS5 would target next.
    probed = ["friction", "joint_damping", "latency_s", "kt_scale"]
    idx = [dm.AXES.index(a) for a in probed]
    err = float(np.linalg.norm(post.mean[idx] - true_u[idx]))
    err0 = float(np.linalg.norm(np.full(len(idx), 0.5) - true_u[idx]))
    print(f"[R2] sim-to-sim fit on identifiable axes {probed}: error {err0:.3f} -> {err:.3f}")
    ok = err < 0.7 * err0
    print(f"PROVEN: R2 hardware-ID suite runs against the sim, emits traces, and the "
          f"fit-to-trace loop recovers the world (sim-to-sim): {ok}. Real bench numbers "
          f"are hardware-gated (swap real_traces for measured logs).")
    sys.exit(0 if ok else 1)
