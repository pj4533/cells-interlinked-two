#!/usr/bin/env bash
# Durable backend launcher for unattended (overnight / multi-hour) runs.
#
# The backend runs inside a DETACHED `screen` session, so it lives in its own
# process session independent of whatever terminal — or agent — started it.
# That's the fix for the overnight death: the previous launch was a `nohup &`
# child of the launching session's process group, which got SIGTERM'd when that
# session was torn down. A screen server daemonizes away from that group.
#
# `caffeinate -is` keeps the Mac awake (idle + system sleep) for the run's
# lifetime. Does NOT survive a reboot — restart manually after one.
#
#   ./run_backend.sh          start (no-op if already running)
#   ./run_backend.sh stop     stop the session
#   ./run_backend.sh attach   watch it live (detach again with Ctrl-a d)
#   ./run_backend.sh status   is it up?
set -euo pipefail
cd "$(dirname "$0")"            # server/
SESSION=ci-backend

# Note: `screen -list` exits non-zero in normal cases, which under `pipefail`
# would mask a grep match — so match the captured output with bash globbing.
LOG=/tmp/ci_backend.log

running() { [[ "$(screen -list 2>/dev/null || true)" == *".${SESSION}"* ]]; }

case "${1:-start}" in
  start)
    if running; then echo "already running (screen session '${SESSION}')"; exit 0; fi
    # Redirect to a logfile (append, with a start marker) so a death leaves a
    # diagnosable tail — screen's own buffer is lost when the session dies.
    echo "===== backend start $(date '+%F %T') =====" >> "$LOG"
    screen -dmS "$SESSION" bash -c "caffeinate -is uv run python -m cells_interlinked >> '$LOG' 2>&1"
    echo "started backend in detached screen '${SESSION}' (log: $LOG)"
    echo "  attach: ./run_backend.sh attach   stop: ./run_backend.sh stop"
    ;;
  stop)
    if running; then screen -S "$SESSION" -X quit && echo "stopped"; else echo "(not running)"; fi
    ;;
  attach)
    screen -r "$SESSION"
    ;;
  status)
    if running; then echo "up — $(curl -s --max-time 4 http://localhost:8000/health || echo 'health check failed')"; else echo "down"; fi
    ;;
  *)
    echo "usage: $0 {start|stop|attach|status}"; exit 1 ;;
esac
