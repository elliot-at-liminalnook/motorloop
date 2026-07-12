#!/bin/bash
# SPDX-License-Identifier: MIT
# Rich live-pod snapshot of a running training run: combat decomposition + system + economics.
# Reads the pod id from /tmp/runpod_podid and ssh from /tmp/runpod_ssh (set during bringup).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEY=$(tr -d '\n' < ~/RUNPOD_API_KEY)
PY="${WARP_PY:-$ROOT/.venv-warp/bin/python}"
POD=$(cat /tmp/runpod_podid 2>/dev/null)
[ -z "$POD" ] && { echo "no /tmp/runpod_podid — is a pod up?"; exit 1; }
echo "===== LIVE RUN STATUS $(date +%H:%M:%S) ====="
INFO=$(curl -s -H "Authorization: Bearer $KEY" https://rest.runpod.io/v1/pods/$POD 2>/dev/null \
  | $PY -c "import sys,json;d=json.load(sys.stdin);print(d.get('desiredStatus'), d.get('costPerHr'), d.get('lastStartedAt',''))" 2>/dev/null)
echo "POD: $INFO  (status costPerHr startedAt) | budget \$25"
/tmp/rp.sh '
echo "RUN: procs=$(ps -eo args|grep -E "curriculum_drive|train_adv|arena.cli"|grep -v grep|wc -l) | $(grep -aE "PHASE |ROUND " /root/proj/out/drive.log 2>/dev/null | tail -1)"
echo "--- held-out benchmark (combat decomposition) ---"
for f in /root/proj/out/*_benchmark.jsonl; do [ -s "$f" ] && echo "$(basename $f .json): $(tail -1 $f)"; done 2>/dev/null | tail -2
echo "--- training (latest eval) ---"; grep -aE "\] step " /root/proj/out/curr_*.log /root/proj/out/sp_*.log 2>/dev/null | tail -1
echo "--- throughput / GPU ---"; grep -aoE "throughput=[0-9]+" /root/proj/out/curr_*.log 2>/dev/null | tail -1
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null
echo "--- errors ---"; grep -aiE "traceback|resource_exhausted|cusolver|FAILED rc=" /root/proj/out/drive.log /root/proj/out/curr_*.log 2>/dev/null | grep -aviE "import warp" | tail -2 || echo none
' 2>&1 | grep -avE "Failed to import warp"
