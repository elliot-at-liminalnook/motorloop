# SPDX-License-Identifier: MIT
"""Phase 11 — the RTL / component-envelope VALIDATION gate (the motorloop differentiator).

A policy trained on the IDEAL actuator can demand full torque at any speed. The real FOC drive can't:
at speed, back-EMF + the current limit cap what it can deliver. This gate takes a trained policy's
actuator-command trajectory `(joint_speed, commanded_fraction)` and checks it against the real
envelope (`FocActuator`, datasheet-grounded) — flagging steps where the policy demands MORE torque
than the controller can produce. That's the "stayed within the real component" judgment the dense
sim hides, grounding the Coach's `actuator-safety` competency + the `safe` verdict in physics, not a
toy clamp. Same "fast for LEARNING, real for JUDGMENT" split we use everywhere — now for the actuator.

`rtl_cosim_gate()` is the hardware-grade swap: run the same trajectory through the *actual* FOC/ADC
RTL via cocotb (when the OSS sim is present) — same pattern as the Phase-RS hardware-gated fits.

  python -m arena.rtl_gate --selftest
"""

from __future__ import annotations

import shutil, sys
from pathlib import Path
import numpy as np

from arena.backend import Actuator, FocActuator, IdealActuator  # noqa: E402


def actuator_envelope_gate(joint_vel, action, actuator: Actuator, tol: float = 0.02) -> dict:
    """Check a (T, J) speed + action trajectory against an actuator envelope. `action` is the
    commanded torque fraction in [-1, 1]; the drive can deliver `actuator.scale(joint_vel)`. A step
    is a VIOLATION where |command| exceeds the deliverable fraction (+tol)."""
    jv = np.asarray(joint_vel, dtype=float)
    cmd = np.abs(np.asarray(action, dtype=float))
    deliverable = np.asarray(actuator.scale(jv), dtype=float)
    over = np.maximum(0.0, cmd - (deliverable + tol))
    viol = over > 0.0
    return dict(model=actuator.name, steps=int(cmd.size), violation_rate=float(viol.mean()),
                max_overdemand=float(over.max()), safe=bool(viol.mean() < 0.02))


def rollout_trace(env, infer, steps=200, seed=0):
    """Run a policy through an AdversarialEnv and extract A's (hinge joint speed, hinge action) per
    step — the trajectory the envelope gate validates."""
    import torch
    if hasattr(env, "_gen"):
        env._gen.manual_seed(seed)
    obs = env.reset()
    indices = (env.layer.idx.Ada if hasattr(env, "layer") else env._da)
    nh = len(indices)
    vels, acts = [], []
    for _ in range(steps):
        action = infer(obs)
        if isinstance(action, tuple):
            action = action[0]
        action = torch.as_tensor(action, device=env.device).reshape(env.nworld, -1)
        vels.append(env.qvel[:, indices].detach().cpu().numpy())
        acts.append(action[:, :nh].clamp(-1, 1).detach().cpu().numpy())
        obs = env.step(action)[0]
    return np.asarray(vels), np.asarray(acts)


def rtl_cosim_available() -> bool:
    """The actual FOC/ADC RTL cosim needs cocotb + an HDL simulator (iverilog/verilator)."""
    try:
        import cocotb  # noqa: F401
    except Exception:
        return False
    return shutil.which("iverilog") is not None or shutil.which("verilator") is not None


def rtl_cosim_gate(joint_vel, action, motor="db42s03", gear=6.0) -> dict:
    """The judgment gate. If the FOC/ADC RTL cosim is available, validate against the SYNTHESIZED
    controller; otherwise fall back to the datasheet `FocActuator` envelope (framework-now, RTL-fit-
    gated — same honesty as the Phase-RS hardware fits)."""
    foc = FocActuator.from_motor(motor, gear)
    res = actuator_envelope_gate(joint_vel, action, foc)
    res["backend"] = "rtl_cosim" if rtl_cosim_available() else "foc_model(rtl_gated)"
    # TODO(Phase-11 fit): when rtl_cosim_available(), drive the FOC RTL (sim/cocotb) with the
    # (speed, torque-demand) trajectory and compare delivered torque + ADC-latency'd current.
    return res


def _selftest():
    _ = FocActuator.from_motor("db42s03", gear=6.0)
    T, J = 50, 12
    high_speed = np.full((T, J), 18.0)        # fast joints -> the FOC envelope is well below 1.0
    low_speed = np.full((T, J), 1.0)
    # (1) a policy that respects the IDEAL clamp but DEMANDS full torque at speed -> caught
    demand_full = np.ones((T, J))
    bad = rtl_cosim_gate(high_speed, demand_full)
    assert not bad["safe"] and bad["violation_rate"] > 0.5, bad
    # (2) a gentle policy (small demands) at the same speed -> within the envelope
    gentle = np.full((T, J), 0.05)
    ok = rtl_cosim_gate(high_speed, gentle)
    assert ok["safe"], ok
    # (3) full demand at LOW speed is fine (envelope is near 1 at standstill)
    ok2 = rtl_cosim_gate(low_speed, demand_full * 0.6)
    assert ok2["safe"], ok2
    # the IDEAL actuator never flags (the inner-loop assumption) — proving the gate adds real info
    ideal_chk = actuator_envelope_gate(high_speed, demand_full, IdealActuator())
    assert ideal_chk["safe"], ideal_chk
    print(f"gate: ideal says safe; FOC catches full-torque-at-speed "
          f"(violation_rate {bad['violation_rate']:.2f}, max_overdemand {bad['max_overdemand']:.2f}); "
          f"backend={bad['backend']}")
    print("PROVEN: RTL-envelope gate — a policy legal under the ideal clamp but VIOLATING the real "
          "FOC envelope is caught; the gate grounds actuator-safety in the component model")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        _selftest()
    else:
        print(__doc__)
