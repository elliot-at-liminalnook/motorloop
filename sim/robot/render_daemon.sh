#!/bin/bash
# Milestone-video render daemon. Watches CODESIGN_OUT for new {TAG}_ms_*.pkl checkpoints (dropped by
# train_adversarial.py's --milestone-gap) and renders each into a 1v1 self-mirror mp4 (the policy vs a
# clone of itself) under $OUT/videos/, so the evolution of the policy can be watched over training.
# Runs on the GPU pod (MUJOCO_GL=egl, GPU rendering) or locally (osmesa). A puller downloads the mp4s.
#   CODESIGN_OUT=... MUJOCO_GL=egl bash render_daemon.sh <tag>
set -u
OUT="${CODESIGN_OUT:-/root/proj/out}"; TAG="${1:-ac}"; GL="${MUJOCO_GL:-egl}"
PY="${RENDER_PY:-python3}"; HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT/videos"
echo "[render_daemon] watching $OUT for ${TAG}_ms_*.pkl  (GL=$GL) -> $OUT/videos/"
render_one() {
  local ms="$1" base out
  base=$(basename "$ms" .pkl); out="$OUT/videos/$base.mp4"
  [ -f "$out" ] && return 0
  [ -f "$out.lock" ] && return 0
  : > "$out.lock"
  echo "[render_daemon] rendering $base ..."
  if MUJOCO_GL=$GL CODESIGN_OUT="$OUT" "$PY" "$HERE/render_fight_video.py" \
        --a "$ms" --b "$ms" --out "$out" --steps 200 --sep 0.6 --label "$base" >/dev/null 2>&1; then
    echo "[render_daemon] OK   $out ($(du -h "$out" 2>/dev/null | cut -f1))"
  else
    echo "[render_daemon] FAIL $base"; rm -f "$out"
  fi
  rm -f "$out.lock"
}
while true; do
  shopt -s nullglob
  for ms in "$OUT"/*_ms_*.pkl; do render_one "$ms"; done
  shopt -u nullglob
  # NOTE: do NOT auto-exit on ARENA_DONE — a stale ARENA_DONE (left by a killed launcher during a
  # restart) was killing the daemon mid-run, so the league milestones never rendered. Run until killed.
  sleep 45
done
