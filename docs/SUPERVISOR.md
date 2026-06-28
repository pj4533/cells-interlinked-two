# Backend supervisor — durable, self-healing, resume-on-intent

The backend runs under a **launchd LaunchAgent** so unattended (overnight /
multi-day) runs survive crashes and reboots. This replaced the old detached
`screen` launcher, which did not relaunch a killed child — the backend kept dying
overnight and sitting idle until noticed (the death was a graceful SIGTERM with no
kernel/jetsam trace; `screen` just doesn't restart).

## Pieces

- `com.cellsinterlinked.backend.plist` — the LaunchAgent (tracked in `server/`,
  installed into `~/Library/LaunchAgents/`). `KeepAlive=true` (restart on **any**
  exit — SIGTERM, crash, OOM-kill), `RunAtLoad=true` (back after login/reboot),
  `ThrottleInterval=30` (don't thrash during the ~30-60s model load).
- `server/ci_backend_supervised.sh` — the entrypoint the agent runs. `exec`s
  `caffeinate -is uv run python -m cells_interlinked` (so launchd supervises the
  real process and the Mac stays awake), and in the background **conditionally
  resumes the DMT autoresearch loop** (see below).
- `server/run_backend.sh` — thin `launchctl` control surface:
  `{install|start|stop|restart|status|attach|uninstall}`. `attach` tails
  `/tmp/ci_backend.log`.

## Resume-on-intent

The DMT autoresearch loop does **not** auto-start just because the backend booted.
It resumes **only if it was running** when the backend went down — so an explicit
stop stays stopped, but a crash-while-running self-heals.

Mechanism: a **run-intent sentinel** file `data/atlas_dmt/.should_run`.
- `DmtController.start()` writes it; `stop()` removes it (even if not currently
  running, so a stop always means "stay stopped").
- `start()`-while-running re-asserts intent (cancels a pending stop, restores the
  sentinel) so it can't be left stop-pending or sentinel-less.
- The resumer in `ci_backend_supervised.sh` waits for `model_loaded`, then POSTs
  `/autoresearch-dmt/start` **only if the sentinel exists**; otherwise it leaves
  the loop stopped and logs that it's awaiting a manual start.
- First-ever boot has no sentinel → the loop waits for a manual start.

## Common operations

```bash
cd server
./run_backend.sh install     # first time: copy plist + bootstrap
./run_backend.sh status      # up? + health
./run_backend.sh restart     # after a Python change (the loop resumes iff it was running)
./run_backend.sh stop        # stop (stays stopped across restarts)
./run_backend.sh attach      # tail the live log
```

For interactive foreground dev (`uv run python -m cells_interlinked`), stop the
agent first or it holds port 8000.

## Gotchas (institutional memory)

- `screen -S … -X quit` used to leave the `caffeinate→uv→python` child tree
  orphaned/alive — when migrating or cleaning up, `pkill -f "uv run python -m
  cells_interlinked"` to be sure port 8000 frees before bootstrapping.
- The log is `/tmp/ci_backend.log` (appended, with `===== backend start … =====`
  markers). screen's own buffer was lost on death; this isn't.
