#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""T3 mechanism liveness: test that mechanisms OBSERVABLY FIRE, not that their
code exists.

The audit's catalogue of dead mechanisms that were built, reviewed, and
believed: per-episode reset randomization (brax's AutoResetWrapper replays one
cached state per env forever), --max-grad-norm/--lr-schedule (plumbed, never
passed by any script), the pure-`pd` control mode (the DEFAULT mode, never once
rolled out — only cpg_pd was ever exercised). Each test here observes a
mechanism doing something, in the cheapest way that can't be faked:

  1. mode matrix   — every control mode any production script uses gets a
                     10-step smoke rollout (subprocess: the mode is an
                     import-time env-var constant)
  2. flag liveness — every argparse flag a trainer declares is READ somewhere
                     in that trainer (static; the runtime half — what a run
                     actually received — is preflight's resolved-config JSON)
  3. reset diversity — XFAIL(strict): successive auto-reset episodes in the
                     same env slot are IDENTICAL today (the AutoResetWrapper
                     defect, audit item 6). strict=true means the custom reset
                     wrapper (plan B.4) flipping this green breaks CI until the
                     xfail marker is removed — the flip itself is verified.
  4. rng advance   — commands differ across reset keys and resample over time
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SCRATCH_OUT = os.environ.get("CODESIGN_OUT", "/tmp/liveness_out")

SMOKE_SNIPPET = """
import sys; sys.path.insert(0, {here!r})
import jax, jax.numpy as jnp
import commanded_env as C
env = C._build()()
s = env.reset(jax.random.PRNGKey(0))
step = jax.jit(env.step)
for t in range(10):
    s = step(s, jnp.zeros(env.action_size))
    assert jnp.isfinite(s.reward), "non-finite reward"
assert s.obs.shape == (env.observation_size,)
print("MODE-OK", C.CMD_CONTROL_MODE)
"""


@pytest.mark.parametrize("mode", ["pd", "cpg_pd"])
def test_mode_matrix_smoke(mode):
    """Every production control mode rolls out. `pd` is the walking plan's mode
    AND the module default — it had never been executed before 2026-07."""
    env = {**os.environ, "CMD_CONTROL_MODE": mode, "JAX_PLATFORMS": "cpu",
           "CODESIGN_OUT": SCRATCH_OUT}
    r = subprocess.run([sys.executable, "-c", SMOKE_SNIPPET.format(here=str(HERE))],
                       env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"mode={mode} smoke failed:\n{r.stdout}\n{r.stderr}"
    assert f"MODE-OK {mode}" in r.stdout


TRAINERS = ["train_adversarial.py", "train_commanded.py", "pbt_train.py"]


@pytest.mark.parametrize("trainer", TRAINERS)
def test_every_declared_flag_is_read(trainer):
    """A flag that parses but is never read is a lie in --help. (The historical
    twist was the caller side — flags plumbed but never PASSED — which is now
    covered at runtime by preflight's argv echo + resolved-config dump.)"""
    src = (HERE / trainer).read_text()
    flags = re.findall(r'add_argument\(\s*"--([a-z0-9-]+)"', src)
    assert flags, f"{trainer}: no flags found — pattern rot?"
    unread = []
    for flag in flags:
        dest = flag.replace("-", "_")
        # read as args.<dest>, p["<dest>"]-style, or getattr(args, "<dest>")
        if not re.search(rf'args\.{dest}\b|[\"\']{dest}[\"\']', src.replace(f'"--{flag}"', "")):
            unread.append(flag)
    assert not unread, f"{trainer}: declared but never read: {unread}"


def _episode_starts(env, steps=15, ep=5):
    import jax
    import jax.numpy as jnp
    import numpy as np
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    s = env.reset(keys)
    step = jax.jit(env.step)
    starts = []                       # env-0 hinge qpos right after each auto-reset
    for t in range(steps):
        s = step(s, jnp.zeros((2, 12)))
        if t % ep == ep - 1:          # episode boundary -> state was auto-reset
            starts.append(np.asarray(jax.device_get(s.pipeline_state.qpos[0, 7:19])).copy())
    d01 = float(np.abs(starts[0] - starts[1]).max())
    d12 = float(np.abs(starts[1] - starts[2]).max())
    return d01, d12


def test_stock_brax_auto_reset_replays_one_state():
    """Documents WHY reset_bank.py exists: stock brax restarts every episode from
    the same cached state. If this ever FAILS, brax fixed it upstream — retire
    the custom wrapper."""
    from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper, VmapWrapper
    import commanded_env as C
    env = AutoResetWrapper(VmapWrapper(EpisodeWrapper(C._build()(), episode_length=5,
                                                      action_repeat=1)))
    d01, d12 = _episode_starts(env)
    assert d01 < 1e-6 and d12 < 1e-6, "brax fixed AutoReset replay — retire reset_bank.py"


def test_banked_auto_reset_gives_distinct_episodes():
    """B.4's wrapper: successive episodes in the same env slot start DIFFERENT."""
    import jax
    from reset_bank import make_wrap_fn
    import commanded_env as C
    wrap = make_wrap_fn(jax.random.PRNGKey(3), bank_size=8, canonical_frac=0.5,
                        root_dof=0)
    env = wrap(C._build()(), episode_length=5, action_repeat=1)
    d01, d12 = _episode_starts(env)
    assert d01 > 1e-6 and d12 > 1e-6, (
        f"banked resets identical (Δ={d01:.2e}, {d12:.2e}) — bank gather broken")


def test_commands_differ_across_keys_and_resample_over_time():
    import jax
    import jax.numpy as jnp
    import commanded_env as C
    env = C._build()()
    a = env.reset(jax.random.PRNGKey(0))
    b = env.reset(jax.random.PRNGKey(1))
    assert float(jnp.abs(a.info["cmd"] - b.info["cmd"]).max()) > 1e-6, \
        "reset commands identical across keys — command RNG dead"
    # reset noise: joint pose differs across keys too
    assert float(jnp.abs(a.pipeline_state.qpos[7:19] - b.pipeline_state.qpos[7:19]).max()) > 1e-6, \
        "reset joint noise identical across keys"
    # in-episode resample: after CMD_HOLD_STEPS the command changes
    s = a
    step = jax.jit(env.step)
    cmd0 = s.info["cmd"]
    changed = False
    for _ in range(C.CMD_HOLD_STEPS + 2):
        s = step(s, jnp.zeros(env.action_size))
        if float(jnp.abs(s.info["cmd"] - cmd0).max()) > 1e-6:
            changed = True
            break
    assert changed, "command never resampled within an episode — cmd_timer/rng dead"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
