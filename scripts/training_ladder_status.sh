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
echo "--- GPU ---"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
  --format=csv,noheader 2>/dev/null || true
echo "--- historical errors (may include recovered retries) ---"
grep -ahiE 'traceback|out of memory|FAILED|subprocess failed|STOP rung' \
  "$out/ladder.log" "$out/logs"/*.log 2>/dev/null | tail -5 || echo "none"
REMOTE
