#!/bin/bash
# SPDX-License-Identifier: MIT
# Verify-gated snapshot of the `arena` framework build (the build's resume/backup property).
# Runs the phase's verify command from the ledger; only on success does it tar arena/ + ledger
# to sim/build/fw-snapshots/ and flip the phase to `verified`.
#   PHASE=N [WARP_PY=...] bash scripts/fw_snapshot.sh
set -u
PHASE="${PHASE:?set PHASE=N}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WARP_PY="${WARP_PY:-$ROOT/.venv-warp/bin/python}"
cd "$ROOT"; export WARP_PY
LP="import sys; sys.path.insert(0,'sim/robot'); from arena import _ledger"
VCMD=$("$WARP_PY" -c "$LP; print(_ledger.verify_cmd($PHASE))")
echo "== verify phase $PHASE: $VCMD =="
if ( eval "$VCMD" ); then            # subshell: a `cd` in the verify cmd must NOT leak to the tar
  mkdir -p sim/build/fw-snapshots
  SNAP="sim/build/fw-snapshots/phase${PHASE}-$(date +%Y%m%d-%H%M%S).tgz"
  if tar czf "$SNAP" -C sim/robot arena; then
    "$WARP_PY" -c "$LP; _ledger.mark($PHASE,'verified','$SNAP')"
    echo "SNAPSHOT_OK phase $PHASE -> $SNAP (ledger: verified)"
  else
    echo "TAR_FAILED phase $PHASE"; rm -f "$SNAP"; exit 1
  fi
else
  echo "VERIFY_FAILED phase $PHASE — NOT snapshotting / NOT marking verified"; exit 1
fi
