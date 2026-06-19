#!/usr/bin/env bash
# Backend control for unattended (overnight / multi-hour) runs.
#
# Backed by a launchd LaunchAgent (com.cellsinterlinked.backend) instead of the
# old detached `screen`. WHY: the backend kept dying overnight — a graceful
# SIGTERM with no kernel/jetsam trace (the log ended on a multiprocessing
# resource_tracker cleanup warning, which only prints on a clean interpreter
# shutdown). `screen` does not relaunch a killed child, so the box sat idle.
# launchd KeepAlive restarts on ANY exit cause, and RunAtLoad brings it back
# after login/reboot. The agent runs server/ci_backend_supervised.sh, which
# execs `caffeinate -is uv run python -m cells_interlinked` and (in the
# background) re-POSTs /autoresearch-dmt/start once the model is ready, so the
# DMT hunt resumes on every restart.
#
#   ./run_backend.sh install    install + load the launchd agent (first time)
#   ./run_backend.sh start      load/kick the agent (start it)
#   ./run_backend.sh stop       unload the agent (stops it; no auto-restart)
#   ./run_backend.sh restart    kickstart -k (kill + relaunch)
#   ./run_backend.sh status     is it up? (launchd state + health)
#   ./run_backend.sh attach     tail the live log (Ctrl-c to detach)
#   ./run_backend.sh uninstall  unload + remove the agent
set -uo pipefail
cd "$(dirname "$0")"            # server/

LABEL=com.cellsinterlinked.backend
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
DOMAIN="gui/$(id -u)"
SERVICE="${DOMAIN}/${LABEL}"
LOG=/tmp/ci_backend.log

loaded() { launchctl print "$SERVICE" >/dev/null 2>&1; }

case "${1:-status}" in
  install)
    # Copy the repo-tracked plist into LaunchAgents on a fresh checkout.
    if [[ ! -f "$PLIST" ]]; then
      if [[ -f "${LABEL}.plist" ]]; then
        mkdir -p "$HOME/Library/LaunchAgents"
        cp "${LABEL}.plist" "$PLIST" && echo "copied ${LABEL}.plist -> $PLIST"
      else
        echo "missing plist: $PLIST (and no repo copy ./${LABEL}.plist)"; exit 1
      fi
    fi
    if loaded; then echo "already installed; use 'restart' to bounce it"; exit 0; fi
    launchctl bootstrap "$DOMAIN" "$PLIST" && echo "installed + started ($LABEL)"
    echo "  log: $LOG   status: ./run_backend.sh status"
    ;;
  start)
    if loaded; then
      launchctl kickstart "$SERVICE" && echo "started (was loaded)"
    else
      launchctl bootstrap "$DOMAIN" "$PLIST" && echo "loaded + started"
    fi
    ;;
  stop)
    if loaded; then launchctl bootout "$SERVICE" && echo "stopped (agent unloaded; no auto-restart until 'start')"; else echo "(not loaded)"; fi
    ;;
  restart)
    if loaded; then launchctl kickstart -k "$SERVICE" && echo "restarted"; else launchctl bootstrap "$DOMAIN" "$PLIST" && echo "loaded + started"; fi
    ;;
  status)
    if loaded; then
      state=$(launchctl print "$SERVICE" 2>/dev/null | grep -E "state = |pid = " | tr -s ' ' | sed 's/^ *//')
      echo "agent loaded — ${state:-(no pid yet)}"
      echo "health: $(curl -s --max-time 4 http://localhost:8000/health || echo 'not responding yet')"
    else
      echo "down (agent not loaded) — './run_backend.sh install' or 'start'"
    fi
    ;;
  attach)
    echo "tailing $LOG (Ctrl-c to detach; the backend keeps running)"; tail -f "$LOG"
    ;;
  uninstall)
    if loaded; then launchctl bootout "$SERVICE" 2>/dev/null; fi
    echo "agent unloaded (plist left at $PLIST; rm it to fully remove)"
    ;;
  *)
    echo "usage: $0 {install|start|stop|restart|status|attach|uninstall}"; exit 1 ;;
esac
