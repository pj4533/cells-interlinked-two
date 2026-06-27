#!/usr/bin/env bash
# Supervised backend entrypoint — run by the launchd agent
# com.cellsinterlinked.backend (see ~/Library/LaunchAgents/, installed via
# run_backend.sh install).
#
# WHY launchd instead of the old detached `screen`: the backend kept dying
# overnight. The death was a graceful SIGTERM (the log ends on a
# multiprocessing resource_tracker cleanup warning, which only prints when the
# interpreter shuts down via atexit — i.e. NOT a hard OOM SIGKILL). No kernel /
# jetsam trace pinned the sender. screen does not relaunch a killed child, so
# the box sat idle until noticed. launchd with KeepAlive restarts the process on
# ANY exit cause (SIGTERM, crash, OOM-kill) and RunAtLoad brings it back after a
# login/reboot — so the hunt self-heals regardless of what killed it.
#
# This script runs in the FOREGROUND as launchd's supervised process: the final
# `exec` replaces the shell with caffeinate→python, so when python dies the
# whole job exits and launchd relaunches it. A backgrounded resumer POSTs the
# DMT loop's /start once the model is ready (the loop does NOT auto-resume on a
# fresh boot — it only reloads the atlas), so a restart resumes the hunt too.
set -uo pipefail
cd "$(dirname "$0")"            # server/

export PATH="/Users/pj4533/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"

echo "===== backend start $(date '+%F %T') (launchd-supervised) ====="

# Background resumer: wait for the model to load, then resume the DMT loop ONLY
# IF it was running when the backend went down. The controller writes a
# run-intent sentinel (data/atlas_dmt/.should_run) on start() and removes it on
# stop(). So: crash/reboot while running → sentinel present → auto-resume
# (self-heal); user-stopped or never-started → no sentinel → stay stopped and
# wait for a manual start. Idempotent either way.
SENTINEL="data/atlas_dmt/.should_run"
(
  for _ in $(seq 1 120); do          # up to ~10 min for the ~24 GB model load
    if curl -s --max-time 4 http://localhost:8000/health 2>/dev/null | grep -q '"model_loaded":true'; then
      if [[ -f "$SENTINEL" ]]; then
        sleep 2
        resp=$(curl -s --max-time 10 -X POST http://localhost:8000/autoresearch-dmt/start \
                 -H 'Content-Type: application/json' -d '{}' 2>/dev/null)
        echo "[resumer] was running (sentinel present) → DMT loop start -> ${resp:-<no response>}"
      else
        echo "[resumer] no run-intent sentinel — leaving DMT loop stopped (awaiting manual start)"
      fi
      exit 0
    fi
    sleep 5
  done
  echo "[resumer] gave up waiting for model_loaded"
) &

# caffeinate -is keeps the Mac awake (idle + system sleep) for the run's
# lifetime. exec so launchd supervises the real process tree, not a wrapper.
exec caffeinate -is uv run python -m cells_interlinked
