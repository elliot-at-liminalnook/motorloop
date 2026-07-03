# SPDX-License-Identifier: MIT
"""T2 trainer preflight — derived-quantity sanity BEFORE any GPU time is spent.

The audit (notes/training-uplift-audit.md) found the canonical "12M-step" runs
were ~37 PPO iterations under a 0.66 s credit horizon feeding a 4x32 network —
three structural failures visible in SECONDS from the config, discovered only
after weeks of GPU evidence was misread. This module makes those quantities
loud at startup and refuses to launch a run that cannot work:

    env-steps / PPO iteration = batch x minibatches x unroll
    total PPO iterations      = steps / above          FAIL < 200 from-scratch
    credit horizon            = dt / (1 - gamma)       FAIL < 2x task timescale
    episode duration          = episode_length x dt    WARN < 2x time-to-contact
    first-layer fan-in        = obs_dim / hidden[0]    WARN > 4:1

It also echoes argv and dumps the RESOLVED config (post-defaults) as JSON into
the run directory — "what did this run actually train with" must never again
require code archaeology.

Modes: strict (default) raises on red lines; warn prints them; off is for
tooling that reuses trainer mains for non-training purposes. pbt_train passes
warn to its subprocesses (its per-cycle step slices are legitimately short —
the cumulative budget is PBT's own concern).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

# V.3: half the system's config flows through env vars (CMD_CONTROL_MODE trained a
# whole run in `cardinal` mode with yaw commands silently dormant before anyone
# noticed). The registry of legitimate knobs is SCANNED from the code's actual
# os.environ.get() calls — a declared list would drift; the scan cannot.
_ENV_PREFIXES = ("CMD_",)
_ENV_GET_RE = re.compile(
    r'os\.environ\.get\(\s*"(CMD_[A-Z0-9_]+)"\s*(?:,\s*"([^"]*)")?')


def env_var_contract(strict=True, stream=None, source_dir=None):
    """Resolve every CMD_* knob the code can read; FAIL on unknown CMD_* in the
    environment (a typo'd override silently does nothing — the worst failure mode
    a config system can have). Returns {name: (value, is_overridden)}."""
    out = stream or sys.stdout
    src = Path(source_dir or Path(__file__).resolve().parent)
    known = {}
    for f in sorted(src.glob("*.py")):
        try:
            for m in _ENV_GET_RE.finditer(f.read_text()):
                known.setdefault(m.group(1), m.group(2))
        except OSError:
            continue
    present = {k: v for k, v in os.environ.items()
               if any(k.startswith(p) for p in _ENV_PREFIXES)}
    unknown = sorted(set(present) - set(known))
    resolved = {k: dict(value=os.environ.get(k, d), default=d,
                        overridden=(k in present)) for k, d in sorted(known.items())}
    overrides = {k: v["value"] for k, v in resolved.items() if v["overridden"]}
    print(f"[preflight] env contract: {len(known)} CMD_* knobs known, "
          f"{len(overrides)} overridden" + (f": {overrides}" if overrides else ""),
          file=out)
    for u in unknown:
        print(f"[preflight] FAIL: unknown env var {u}={present[u]!r} — no code reads "
              f"this (typo? the knob it meant to set kept its default)", file=out)
    if unknown and strict:
        raise PreflightError(f"[preflight] unknown CMD_* env vars: {unknown}")
    return resolved

# Task timescales for this robot (50 Hz control):
STRIDE_PERIOD_S = 0.5        # ~2x AIRTIME_TARGET (0.2 s swing) + stance
TIME_TO_CONTACT_S = 2.0      # spawn separation / commanded closing speed, roughly


class PreflightError(SystemExit):
    pass


def preflight_check(*, steps, batch, minibatches, unroll, episode_length,
                    discounting, control_dt=0.02, obs_dim=None, hidden0=512,
                    from_scratch=True, mode="strict", tag="run", run_dir=None,
                    resolved=None, stream=None):
    """Compute + print derived training quantities; enforce red lines.

    Returns the derived dict. Raises PreflightError on a red line in strict
    mode. `resolved` (e.g. vars(args)) is dumped to run_dir/<tag>_resolved_config.json
    together with the derived quantities and argv.
    """
    out = stream or sys.stdout
    per_iter = int(batch) * int(minibatches) * int(unroll)
    iters = float(steps) / max(per_iter, 1)
    horizon_s = control_dt / max(1.0 - float(discounting), 1e-9)
    episode_s = float(episode_length) * control_dt
    fan_in = (float(obs_dim) / float(hidden0)) if obs_dim else None

    failures, warnings = [], []
    if from_scratch and iters < 200:
        failures.append(
            f"total PPO iterations = {iters:.0f} (< 200): steps={steps:,} at "
            f"{per_iter:,} env-steps/iteration cannot train from scratch — the "
            f"'12M-step' runs of 2026-H1 were 37 iterations and went nowhere")
    if horizon_s < 2 * STRIDE_PERIOD_S:
        failures.append(
            f"credit horizon dt/(1-γ) = {horizon_s:.2f} s < 2x stride period "
            f"({2 * STRIDE_PERIOD_S:.1f} s): a swing phase cannot see the reward "
            f"of the step it sets up (γ=0.97 gave 0.66 s)")
    if episode_s < 2 * TIME_TO_CONTACT_S:
        warnings.append(
            f"episode = {episode_s:.1f} s < 2x time-to-contact ({2 * TIME_TO_CONTACT_S:.0f} s)")
    if fan_in is not None and fan_in > 4.0:
        warnings.append(
            f"first-layer fan-in {obs_dim}:{hidden0} = {fan_in:.1f}:1 (> 4:1 — "
            f"the 470-dim-obs-into-32-wide bottleneck was this warning)")

    print(f"[preflight:{tag}] argv: {' '.join(sys.argv)}", file=out)
    print(f"[preflight:{tag}] env-steps/iter={per_iter:,}  iterations={iters:,.0f}  "
          f"credit-horizon={horizon_s:.2f}s  episode={episode_s:.1f}s"
          + (f"  fan-in={fan_in:.1f}:1" if fan_in is not None else "")
          + f"  from_scratch={from_scratch}", file=out)
    for w in warnings:
        print(f"[preflight:{tag}] WARN: {w}", file=out)
    for f in failures:
        print(f"[preflight:{tag}] FAIL: {f}", file=out)

    derived = dict(per_iter=per_iter, iterations=iters, credit_horizon_s=horizon_s,
                   episode_s=episode_s, fan_in=fan_in, failures=failures,
                   warnings=warnings, mode=mode)
    env_contract = env_var_contract(strict=(mode == "strict"), stream=out)
    if run_dir is not None:
        p = Path(run_dir) / f"{tag}_resolved_config.json"
        try:
            p.write_text(json.dumps(
                dict(argv=sys.argv, resolved=_jsonable(resolved or {}),
                     env_contract=env_contract,
                     derived=_jsonable(derived), t=time.time()), indent=2))
            print(f"[preflight:{tag}] resolved config -> {p}", file=out)
        except OSError as e:                       # never let logging kill a run
            print(f"[preflight:{tag}] WARN: could not write {p}: {e}", file=out)

    if failures and mode == "strict":
        raise PreflightError(
            f"[preflight:{tag}] refusing to launch: {len(failures)} red line(s) "
            f"(see above; --preflight warn to override deliberately)")
    return derived


def _jsonable(d):
    return {k: (v if isinstance(v, (int, float, str, bool, type(None), list, dict))
                else str(v)) for k, v in dict(d).items()}
