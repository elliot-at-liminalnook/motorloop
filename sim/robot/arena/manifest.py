# SPDX-License-Identifier: MIT
"""Phase 12 — the reproducibility MANIFEST + episode recorder (cheap, foundational).

Every run should be replayable and comparable. A `manifest` pins the exact provenance — code version,
seed, config, robot-model hash, policy checkpoint hash, backend, machine profile — so a result is
EVIDENCE, not a demo. `record_episode` optionally dumps the full per-step obs/action/reward/safety
trace (replayable). `regression` diffs two runs' metrics so a change in behavior is visible.

  python -m arena.manifest --selftest
"""

from __future__ import annotations

import hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def file_hash(path) -> str:
    p = Path(path)
    return _sha(p.read_bytes()) if p.exists() else "absent"


def code_version() -> dict:
    """git commit if available; else a hash of the kernel + body source (repo may not be git)."""
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True,
                           text=True, timeout=5)
        if h.returncode == 0 and h.stdout.strip():
            return {"git": h.stdout.strip()[:12]}
    except Exception:
        pass
    src = HERE.parent.parent
    files = ["train_adversarial.py", "gen_robot_mjcf.py", "robot.toml"]
    return {"src_hash": _sha(b"".join((src / f).read_bytes() for f in files if (src / f).exists()))}


def machine_profile() -> dict:
    prof = {"python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count()}
    try:
        import torch
        import warp
        prof["torch"] = torch.__version__
        prof["warp"] = warp.__version__
        prof["devices"] = ([torch.cuda.get_device_name(i)
                            for i in range(torch.cuda.device_count())] or ["cpu"])
    except Exception:
        pass
    return prof


def write_manifest(path, run_id, seed, config: dict, checkpoint=None, backend="mujoco_warp",
                   robot_toml=None, ts=None) -> dict:
    """Pin a run's full provenance to `path` (manifest.json)."""
    robot_toml = robot_toml or (HERE.parent.parent / "robot.toml")
    m = {
        "run_id": run_id, "ts": ts or time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": seed,
        "code": code_version(), "machine": machine_profile(), "backend": backend,
        "config": config, "robot_model_hash": file_hash(robot_toml),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_hash": file_hash(checkpoint) if checkpoint else None,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(m, indent=2))
    return m


def read_manifest(path) -> dict:
    return json.loads(Path(path).read_text())


def record_episode(path, reset, step, infer, steps, seed=0) -> int:
    """Dump a full per-step trace (obs/action/reward/done/safety) to JSONL — a replayable episode.
    `reset(seed)->state`, `step(state, action)->(state, obs, reward, done, safety)`, `infer(obs)->action`."""
    import numpy as np
    state = reset(seed)
    obs = getattr(state, "obs", None)
    if obs is None:
        state, obs = state
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for t in range(steps):
            a = infer(obs)
            state, obs2, r, done, safety = step(state, a)
            f.write(json.dumps(dict(t=t, obs=np.asarray(obs).round(4).tolist(),
                                    action=np.asarray(a).round(4).tolist(), reward=float(r),
                                    done=float(done), safety=safety)) + "\n")
            obs = obs2; n += 1
            if done:
                break
    return n


def regression(metrics_a: dict, metrics_b: dict, keys=None) -> dict:
    """Diff two runs' metrics (e.g. two manifests' final-eval results) -> {key: (a, b, delta)}."""
    keys = keys or sorted(set(metrics_a) & set(metrics_b))
    out = {}
    for k in keys:
        a, b = metrics_a.get(k), metrics_b.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[k] = dict(a=a, b=b, delta=round(b - a, 6))
        elif a != b:
            out[k] = dict(a=a, b=b, changed=True)
    return out


def _selftest():
    import tempfile
    import numpy as np
    d = Path(tempfile.mkdtemp())
    # (1) manifest round-trips + pins real provenance
    m = write_manifest(d / "manifest.json", run_id="t0", seed=7,
                       config={"steps": 1000, "opponent": "frozen"}, checkpoint=None,
                       backend="mujoco_warp")
    m2 = read_manifest(d / "manifest.json")
    assert m2 == m and m2["seed"] == 7 and m2["run_id"] == "t0"
    assert ("git" in m["code"]) or ("src_hash" in m["code"])      # code version pinned either way
    assert m["robot_model_hash"] != "absent" and "python" in m["machine"]

    # (2) episode records + REPLAYS deterministically (same seed -> identical trace)
    def reset(seed):
        rng = np.random.default_rng(seed); return {"rng": rng}, rng.standard_normal(3)  # (state, obs)
    def infer(obs): return np.tanh(obs[:2])                        # deterministic policy
    def step(state, a):
        obs = state["rng"].standard_normal(3); r = float(np.sum(a))
        done = 0.0; safety = {"sat": float(np.mean(np.abs(a) > 0.95))}
        return {"rng": state["rng"], "obs": obs}, obs, r, done, safety
    n1 = record_episode(d / "ep1.jsonl", reset, step, infer, steps=20, seed=42)
    n2 = record_episode(d / "ep2.jsonl", reset, step, infer, steps=20, seed=42)
    assert n1 == 20 and (d / "ep1.jsonl").read_text() == (d / "ep2.jsonl").read_text()  # replayable
    row = json.loads((d / "ep1.jsonl").read_text().splitlines()[0])
    assert {"t", "obs", "action", "reward", "safety"} <= set(row)

    # (3) regression diff surfaces a metric change
    diff = regression({"win_rate": 0.2, "survival_rate": 0.0}, {"win_rate": 0.35, "survival_rate": 0.1})
    assert diff["win_rate"]["delta"] == 0.15 and diff["survival_rate"]["delta"] == 0.1
    print("PROVEN: manifest pins provenance (code/seed/config/robot/checkpoint/machine) + round-trips; "
          "episode records & replays deterministically; regression diffs two runs -> runs are evidence")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
