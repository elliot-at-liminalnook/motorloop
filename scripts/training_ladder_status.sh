#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Read-only status for the 31-rung RunPod job launched from training_ladder.py.
set -euo pipefail

POD_FILE="${RUNPOD_POD_FILE:-/tmp/runpod_podid}"
SSH_FILE="${RUNPOD_SSH_FILE:-/tmp/runpod_ssh}"
KEY_FILE="${RUNPOD_KEY_FILE:-$HOME/RUNPOD_API_KEY}"
IDENTITY="${RUNPOD_SSH_KEY:-$HOME/.ssh/runpod_ed25519}"
REMOTE_OUT="${LADDER_REMOTE_OUT:-/root/proj/out/training_ladder}"

[[ -s "$POD_FILE" && -s "$SSH_FILE" && -s "$KEY_FILE" ]] || {
  echo "missing RunPod state: expected $POD_FILE, $SSH_FILE, and $KEY_FILE" >&2
  exit 2
}

POD="$(cat "$POD_FILE")"
read -r IP PORT <<<"$(cat "$SSH_FILE")"
API_KEY="$(tr -d '\r\n' < "$KEY_FILE")"
INFO="$(curl --fail --silent -H "Authorization: Bearer $API_KEY" \
  "https://rest.runpod.io/v1/pods/$POD")"
python3 -c 'import json,sys; p=json.load(sys.stdin); print(
  "POD", p.get("id"), p.get("desiredStatus"),
  "cost/hr", p.get("costPerHr"), "started", p.get("lastStartedAt", ""))' <<<"$INFO"

ssh -i "$IDENTITY" -p "$PORT" -o StrictHostKeyChecking=no \
  -o IdentitiesOnly=yes -o LogLevel=ERROR "root@$IP" bash -s -- "$REMOTE_OUT" <<'REMOTE'
set -u
out="$1"
echo "--- sequence ---"
ps -eo pid,etimes,args | grep -E 'training_ladder.py run|train_mesh_warp.py' \
  | grep -v grep || echo "no ladder process"
grep -ahE '^=== RUNG|^ACCEPT|^STOP|^RETENTION|^PFSP round|^LADDER COMPLETE' \
  "$out/ladder.log" 2>/dev/null | tail -12 || true
echo "--- persisted state ---"
python3 - "$out/ladder_state.json" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
if not p.exists():
    print("state not written yet")
else:
    s=json.loads(p.read_text())
    done=s.get("completed", [])
    print(f"accepted={len(done)}/31 latest={done[-1] if done else 'none'} failed={s.get('failed')}")
    history=s.get("retention_history", [])
    failures=sum(not row.get("pass", False) for row in history)
    print(f"retention_replays={len(history)} retention_failures={failures}")
PY
echo "--- latest trainer metric ---"
latest_log="$(ls -1t "$out/logs"/rung_*.log 2>/dev/null | head -1)"
if [[ -n "$latest_log" ]]; then
  echo "log=$(basename "$latest_log")"
  awk '
    /^COMMAND / { metric = "" }
    /METRIC step=/ { metric = $0 }
    END {
      if (metric != "") print metric
      else print "no metric yet in current attempt"
    }
  ' "$latest_log"
else
  echo "no trainer log yet"
fi
echo "--- learning health ---"
latest_diag="$(ls -1t "$out"/rung_*.diagnostics.json 2>/dev/null | head -1)"
if [[ -n "$latest_diag" ]]; then
  latest_stats="${latest_diag%.diagnostics.json}.stats.json"
  latest_metrics="${latest_diag%.diagnostics.json}.metrics.jsonl"
  python3 - "$latest_diag" "$latest_stats" "$latest_metrics" <<'PY'
import json, math, pathlib, sys

diagnostics_path, stats_path, metrics_path = map(pathlib.Path, sys.argv[1:])
record = json.loads(diagnostics_path.read_text())
diagnostics = record.get("diagnostics", {})
evaluation = record.get("evaluation", {})
robust = diagnostics.get("robust_gates", {})
checks = robust.get("checks", [])
failed = [row for row in checks if not row.get("pass", False)]
print(f"step={record.get('step')} robust_pass={robust.get('all_pass')} "
      f"worst={robust.get('worst_metric')} "
      f"margin={robust.get('worst_relative_margin')}")
if failed:
    print("failing_gates=" + ", ".join(
        f"{row.get('metric')}={row.get('value'):.6g} "
        f"{row.get('comparison')}{row.get('threshold'):.6g} "
        f"margin={row.get('relative_margin'):.3f}"
        for row in failed))

adaptive = diagnostics.get("adaptive_contracts", {})
ceiling = float(adaptive.get("dual_max", 0.0))
rows = adaptive.get("constraints", []) + adaptive.get("competence", [])
if rows:
    def violated(row):
        observed, target = float(row["observed"]), float(row["target"])
        return observed > target if row["comparison"] == "<=" else observed < target
    print("adaptive=" + ", ".join(
        f"{row['name']}:{row['observed']:.4g}{row['comparison']}{row['target']:.4g} "
        f"dual={row['dual']:.3g}"
        + (" SATURATED" if ceiling and violated(row)
           and float(row["dual"]) >= 0.99 * ceiling else "")
        for row in rows))

trust = diagnostics.get("trust_region", {})
controller = diagnostics.get("kl_controller", {})
clipping = diagnostics.get("gradient_clipping", {})
critic = diagnostics.get("critic", {}).get("after_update", {})
print("optimizer="
      f"kl={trust.get('approx_kl', float('nan')):.4g}/"
      f"{controller.get('target_kl', float('nan')):.4g} "
      f"ess={trust.get('effective_sample_fraction', float('nan')):.3f} "
      f"epochs={controller.get('epochs_completed')}/{controller.get('epochs_requested')} "
      f"lr={controller.get('learning_rate_next', float('nan')):.3g} "
      f"actor_clip={clipping.get('actor', {}).get('clipped_fraction', float('nan')):.2f} "
      f"critic_clip={clipping.get('critic', {}).get('clipped_fraction', float('nan')):.2f} "
      f"critic_ev={critic.get('explained_variance', float('nan')):.3f} "
      f"critic_nrmse={critic.get('normalized_rmse', float('nan')):.3f}")

timing = diagnostics.get("timing", {})
throughput = float("nan")
if metrics_path.exists():
    for line in metrics_path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "evaluation":
            throughput = row.get("env_steps_per_second", throughput)
elif stats_path.exists():
    stats = json.loads(stats_path.read_text())
    if stats.get("evals"):
        throughput = stats["evals"][-1].get("env_steps_per_second", throughput)
print("timing="
      f"rollout={timing.get('rollout_seconds', float('nan')):.2f}s "
      f"opt={timing.get('optimization_seconds', float('nan')):.2f}s "
      f"eval={timing.get('evaluation_seconds', float('nan')):.2f}s "
      f"throughput={throughput:.0f} env-step/s")

roles = evaluation.get("reward_role_shares", {})
print("behavior="
      f"foot_air_min={evaluation.get('foot_air_fraction_min', float('nan')):.3f} "
      f"foot_activity={evaluation.get('ladder_foot_activity', float('nan')):.3f} "
      f"contact_entropy={evaluation.get('contact_pattern_entropy', float('nan')):.3f} "
      f"contact_patterns={evaluation.get('contact_pattern_count', 'n/a')} "
      f"clock_diagnostic={evaluation.get('ladder_step_clock', float('nan')):.3f}")
if roles:
    print("reward_roles=" + ", ".join(
        f"{name}={float(value):.3f}" for name, value in roles.items()))

replay = diagnostics.get("checkpoint_replay", {})
print(f"checkpoint_replay_pass={replay.get('pass')} "
      f"tolerance_used={replay.get('max_tolerance_ratio', float('nan')):.3g}")
alerts = diagnostics.get("alerts", [])
print("alerts=" + (", ".join(
    f"{row.get('severity')}:{row.get('code')}" for row in alerts) or "none"))
PY
else
  echo "no structured diagnostics yet"
fi
echo "--- GPU ---"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
  --format=csv,noheader 2>/dev/null || true
echo "--- historical errors (may include recovered retries) ---"
grep -ahiE 'traceback|out of memory|FAILED|subprocess failed|STOP rung' \
  "$out/ladder.log" "$out/logs"/*.log 2>/dev/null | tail -5 || echo "none"
REMOTE
