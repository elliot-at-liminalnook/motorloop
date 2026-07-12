# SPDX-License-Identifier: MIT
"""V.5: registry of every `state.info` key with its LIFETIME.

  episodic   — meaningful only within one episode; a reset (banked or stock)
               may freely replace it, and stale values leaking across an
               episode boundary is a bug.
  persistent — learning/streaming state that must SURVIVE episode boundaries;
               a wrapper that resets it silently kills the mechanism it feeds
               (the audit's bank-swap caveat: per-env RND predictor + Adam).

reset_bank.BankedAutoResetWrapper swaps only pipeline_state/obs (mirroring
the old wrapper stack), so persistent keys survive by construction; this registry makes
that contract CHECKABLE: test_info_keys.py fails on any unregistered key, so a
new mechanism can't add per-env state without declaring who owns its lifetime.
"""
from __future__ import annotations

EPISODIC, PERSISTENT = "episodic", "persistent"

# AdversarialEnv (train_adversarial.py: _info(), reset(), step())
FIGHTER_INFO_KEYS = {
    "design": EPISODIC,             # per-episode body sample
    "prev_dist": EPISODIC,
    "prev_dealt": EPISODIC,
    "t": EPISODIC,
    "vel_ema": EPISODIC,            # not_moving gate smoother
    "dp": EPISODIC,                 # reality-gap world draw (per episode)
    "lidar_rng": PERSISTENT,        # advances every step, forever
    "lidar_prev_scans": EPISODIC,   # frame-stack FIFO
    "lidar_scan_history": EPISODIC, # latency FIFO
    "her_goal": EPISODIC,
    "her_achieved": EPISODIC,
    "rnd_predictor": PERSISTENT,    # per-env learner (audit caveat)
    "rnd_opt_state": PERSISTENT,    # its Adam moments
    "phase": EPISODIC,              # CPG gait phase (legacy mode)
    "prop_hist": EPISODIC,          # B.1 proprio history ring
    "prev_act": EPISODIC,
    "prop_hist_B": EPISODIC,
    "prev_act_B": EPISODIC,
    "walker_prev_act": EPISODIC,    # C.2 walker-pursuer's own previous action (its obs needs it)
    "air_time_A": EPISODIC,         # B.3 gait terms
    "prev_feet_A": EPISODIC,
    "loco_drill": EPISODIC,         # B.3 per-episode drill flag
    "dealt_cum": EPISODIC,          # C.1 KO gate accumulator
    # wrapper-owned (not created by the env):
    "bank_idx": PERSISTENT,         # reset_bank cursor — orbits the bank across episodes
    "steps": EPISODIC,
    "truncation": EPISODIC,
    "first_pipeline_state": PERSISTENT,  # legacy autoreset cache (unused with bank)
    "first_obs": PERSISTENT,
}

# CommandedEnv (commanded_env.py: reset(), reset_with_command(), step())
COMMANDED_INFO_KEYS = {
    "cmd": EPISODIC,
    "rng": PERSISTENT,              # command-resample stream, advances every step
    "design": EPISODIC,
    "cmd_timer": EPISODIC,
    "remote": EPISODIC,
    "phase": EPISODIC,
    "prev_cmd": EPISODIC,
    "prior_strength": EPISODIC,
    "transition_amount": EPISODIC,
    "transition_timer": EPISODIC,
    "route_wp": EPISODIC,
    "route_prev_dist": EPISODIC,
    "wp2_residual_step": EPISODIC,
    "wp3_residual_step": EPISODIC,
    "air_time": EPISODIC,
    "prev_action": EPISODIC,
    "prev_feet_xy": EPISODIC,
    "bank_idx": PERSISTENT,
    "steps": EPISODIC,
    "truncation": EPISODIC,
    "first_pipeline_state": PERSISTENT,
    "first_obs": PERSISTENT,
}
