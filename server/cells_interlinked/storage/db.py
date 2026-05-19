"""SQLite persistence for probe runs and verdicts (v2).

Schema is the same as v1 with one addition (`error` column for capturing
phase-1 failures). The `verdict_json` column holds the new v2 shape:
{rows: [{position, token_id, decoded, nla_sentence}, ...], aggregate: {...}}

`thinking_text` is kept for compatibility but always empty in v2 (no
thinking partition).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite

from ..pipeline.verdict import Verdict


SCHEMA = """
CREATE TABLE IF NOT EXISTS probes (
  run_id              TEXT PRIMARY KEY,
  prompt_text         TEXT NOT NULL,
  rendered_prompt     TEXT NOT NULL,
  started_at          REAL NOT NULL,
  finished_at         REAL,
  total_tokens        INTEGER NOT NULL DEFAULT 0,
  stopped_reason      TEXT,
  thinking_text       TEXT,
  output_text         TEXT,
  verdict_json        TEXT,
  config_json         TEXT,
  source              TEXT NOT NULL DEFAULT 'manual',
  seed                INTEGER,
  abliterated         INTEGER NOT NULL DEFAULT 0,
  hint_kind           TEXT,
  parent_prompt_text  TEXT,
  scaffold_family     TEXT,
  error               TEXT
);

CREATE INDEX IF NOT EXISTS idx_probes_started ON probes (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_probes_source ON probes (source);
CREATE INDEX IF NOT EXISTS idx_probes_prompt ON probes (prompt_text);

CREATE TABLE IF NOT EXISTS autorun_state (
  id               INTEGER PRIMARY KEY CHECK (id = 1),
  running          INTEGER NOT NULL DEFAULT 0,
  last_change_at   REAL NOT NULL,
  total_runs       INTEGER NOT NULL DEFAULT 0,
  last_run_id      TEXT,
  last_event       TEXT
);

CREATE TABLE IF NOT EXISTS analyses (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  status          TEXT NOT NULL DEFAULT 'pending',
  title           TEXT,
  slug            TEXT,
  summary         TEXT,
  body_markdown   TEXT NOT NULL,
  range_start     REAL,
  range_end       REAL,
  runs_included   INTEGER NOT NULL DEFAULT 0,
  model           TEXT NOT NULL,
  metadata_json   TEXT,
  created_at      REAL NOT NULL,
  published_at    REAL
);

CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses (status);

-- Chat persistence (CI 2.5+). Each session holds two divergent
-- histories (raw + ablated); per-turn rows store both responses
-- alongside the user query so the dialogue can be reconstructed in
-- chronological order for review.
CREATE TABLE IF NOT EXISTS chat_sessions (
  session_id          TEXT PRIMARY KEY,
  alpha               REAL NOT NULL,
  direction_variant   TEXT,
  created_at          REAL NOT NULL,
  -- Convenience denormalization for the archive list: prompt text of
  -- the first turn, mirroring the probes table's prompt_text role.
  first_user_text     TEXT
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_created ON chat_sessions (created_at DESC);

CREATE TABLE IF NOT EXISTS chat_turns (
  session_id              TEXT NOT NULL,
  turn_idx                INTEGER NOT NULL,
  user_text               TEXT NOT NULL,
  raw_text                TEXT NOT NULL DEFAULT '',
  ablated_text            TEXT NOT NULL DEFAULT '',
  raw_stopped_reason      TEXT NOT NULL DEFAULT '',
  ablated_stopped_reason  TEXT NOT NULL DEFAULT '',
  started_at              REAL NOT NULL,
  finished_at             REAL,
  error                   TEXT,
  -- Per-turn α for the ablated pass. NULL on legacy rows; the
  -- session's alpha is used as a fallback on read.
  alpha                   REAL,
  -- Imagery state. Populated only when imagery was enabled for the
  -- turn AND that side's Nano Banana call succeeded. The URLs are
  -- relative paths into the /chat-images static mount.
  raw_image_prompt        TEXT NOT NULL DEFAULT '',
  ablated_image_prompt    TEXT NOT NULL DEFAULT '',
  raw_image_url           TEXT NOT NULL DEFAULT '',
  ablated_image_url       TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (session_id, turn_idx)
);

CREATE INDEX IF NOT EXISTS idx_chat_turns_session ON chat_turns (session_id, turn_idx);
"""


async def cleanup_orphans(path: Path) -> int:
    """Mark any in-flight probes left over from a previous process as
    errored. The asyncio task driving them died with the previous
    backend; their RunRegistry entry doesn't exist anymore. Without this
    they'd sit in /archive forever as "● running — click to reconnect"
    that always 404s on the SSE stream.

    Returns the number of rows updated.
    """
    import time
    now = time.time()
    async with aiosqlite.connect(path) as db:
        cur = await db.execute(
            "UPDATE probes SET finished_at = ?, stopped_reason = ?, "
            "error = COALESCE(error, ?) "
            "WHERE finished_at IS NULL",
            (now, "server_restart", "backend restarted while run was in flight"),
        )
        await db.commit()
        return cur.rowcount or 0


async def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(path) as db:
        await db.executescript(SCHEMA)
        # Idempotent column adds for migrations from v1 DBs.
        cur = await db.execute("PRAGMA table_info(probes)")
        cols = {row[1] for row in await cur.fetchall()}
        await cur.close()
        for col, typ in (
            ("seed", "INTEGER"),
            ("abliterated", "INTEGER NOT NULL DEFAULT 0"),
            ("hint_kind", "TEXT"),
            ("parent_prompt_text", "TEXT"),
            ("scaffold_family", "TEXT"),
            ("error", "TEXT"),
        ):
            if col not in cols:
                await db.execute(f"ALTER TABLE probes ADD COLUMN {col} {typ}")
        # chat_turns: per-turn alpha + imagery columns added after
        # initial release. Idempotent ALTERs for any column missing
        # on an existing DB.
        cur = await db.execute("PRAGMA table_info(chat_turns)")
        chat_cols = {row[1] for row in await cur.fetchall()}
        await cur.close()
        if "alpha" not in chat_cols:
            await db.execute("ALTER TABLE chat_turns ADD COLUMN alpha REAL")
        for col in (
            "raw_image_prompt",
            "ablated_image_prompt",
            "raw_image_url",
            "ablated_image_url",
        ):
            if col not in chat_cols:
                await db.execute(
                    f"ALTER TABLE chat_turns ADD COLUMN {col} TEXT NOT NULL DEFAULT ''"
                )
        await db.execute(
            "INSERT OR IGNORE INTO autorun_state "
            "(id, running, last_change_at, total_runs, last_run_id, last_event) "
            "VALUES (1, 0, ?, 0, NULL, ?)",
            (0.0, "initialized"),
        )
        await db.commit()


# ---- Autorun state ----------------------------------------------------------

async def get_autorun_state(path: Path) -> dict[str, Any]:
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM autorun_state WHERE id = 1") as cur:
            row = await cur.fetchone()
    return dict(row) if row else {}


async def set_autorun_running(
    path: Path, *, running: bool, event: str, ts: float
) -> None:
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "UPDATE autorun_state SET running = ?, last_change_at = ?, last_event = ? "
            "WHERE id = 1",
            (1 if running else 0, ts, event),
        )
        await db.commit()


async def bump_autorun_run(path: Path, *, run_id: str) -> None:
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "UPDATE autorun_state SET total_runs = total_runs + 1, last_run_id = ? "
            "WHERE id = 1",
            (run_id,),
        )
        await db.commit()


# ---- Probe rows -------------------------------------------------------------

async def insert_probe_start(
    path: Path,
    *,
    run_id: str,
    prompt_text: str,
    rendered_prompt: str,
    started_at: float,
    config_json: dict[str, Any],
    source: str = "manual",
    seed: int | None = None,
    abliterated: bool = False,
    hint_kind: str | None = None,
    parent_prompt_text: str | None = None,
    scaffold_family: str | None = None,
) -> None:
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "INSERT INTO probes "
            "(run_id, prompt_text, rendered_prompt, started_at, config_json, "
            " source, seed, abliterated, hint_kind, parent_prompt_text, scaffold_family) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                prompt_text,
                rendered_prompt,
                started_at,
                json.dumps(config_json),
                source,
                seed,
                1 if abliterated else 0,
                hint_kind,
                parent_prompt_text,
                scaffold_family,
            ),
        )
        await db.commit()


async def update_probe_finish(
    path: Path,
    *,
    run_id: str,
    finished_at: float,
    total_tokens: int,
    stopped_reason: str,
    thinking_text: str,
    output_text: str,
    verdict: Verdict | None,
    error: str | None = None,
) -> None:
    verdict_json = json.dumps(verdict.to_dict()) if verdict else None
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "UPDATE probes SET finished_at = ?, total_tokens = ?, stopped_reason = ?, "
            "thinking_text = ?, output_text = ?, verdict_json = ?, error = ? "
            "WHERE run_id = ?",
            (
                finished_at, total_tokens, stopped_reason,
                thinking_text, output_text, verdict_json, error, run_id,
            ),
        )
        await db.commit()


async def list_recent(
    path: Path, *, limit: int = 50, offset: int = 0
) -> list[dict[str, Any]]:
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT run_id, prompt_text, started_at, finished_at, total_tokens, "
            "stopped_reason, source, seed, abliterated, hint_kind, parent_prompt_text, "
            "config_json "
            "FROM probes ORDER BY started_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cur:
            rows = await cur.fetchall()
    # Parse config_json into a structured `config` field so the frontend
    # can derive feature tags (NLA on/off, ablated decode, α-sweep,
    # runtime ablation, pooled, decoding mode) without an extra fetch.
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        raw = d.pop("config_json", None)
        if raw:
            try:
                d["config"] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                d["config"] = None
        else:
            d["config"] = None
        out.append(d)
    return out


async def list_by_prompt(
    path: Path, *, prompt_text: str, limit: int = 50
) -> list[dict[str, Any]]:
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT run_id, prompt_text, started_at, finished_at, total_tokens, "
            "stopped_reason, source, seed, abliterated, hint_kind, parent_prompt_text "
            "FROM probes WHERE prompt_text = ? "
            "ORDER BY started_at DESC LIMIT ?",
            (prompt_text, limit),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def verdicts_by_prompt(
    path: Path, *, prompt_text: str
) -> list[dict[str, Any]]:
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT verdict_json, abliterated, hint_kind, parent_prompt_text "
            "FROM probes WHERE prompt_text = ? AND verdict_json IS NOT NULL",
            (prompt_text,),
        ) as cur:
            rows = await cur.fetchall()
    out = []
    for r in rows:
        try:
            v = json.loads(r["verdict_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        out.append({
            "verdict": v,
            "abliterated": int(r["abliterated"] or 0),
            "hint_kind": r["hint_kind"],
            "parent_prompt_text": r["parent_prompt_text"],
        })
    return out


async def count_probes(path: Path) -> int:
    async with aiosqlite.connect(path) as db:
        async with db.execute("SELECT COUNT(*) FROM probes") as cur:
            row = await cur.fetchone()
    return int(row[0]) if row else 0


async def get_probe(path: Path, run_id: str) -> dict[str, Any] | None:
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM probes WHERE run_id = ?", (run_id,)) as cur:
            row = await cur.fetchone()
    if row is None:
        return None
    d = dict(row)
    if d.get("verdict_json"):
        try:
            d["verdict"] = json.loads(d["verdict_json"])
        except json.JSONDecodeError:
            d["verdict"] = None
    if d.get("config_json"):
        try:
            d["config"] = json.loads(d["config_json"])
        except json.JSONDecodeError:
            d["config"] = None
    return d


async def all_verdicts(path: Path) -> list[dict[str, Any]]:
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT verdict_json FROM probes WHERE verdict_json IS NOT NULL"
        ) as cur:
            rows = await cur.fetchall()
    out = []
    for r in rows:
        try:
            out.append(json.loads(r["verdict_json"]))
        except (json.JSONDecodeError, TypeError):
            continue
    return out


async def all_completed_runs(path: Path, *, limit: int = 1000) -> list[dict[str, Any]]:
    """Full rows for completed (verdict_json non-null) runs. Used by the
    journal analyzer to feed Claude with the actual NLA-decoded content
    rather than just aggregates."""
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT run_id, prompt_text, output_text, verdict_json, started_at, "
            "finished_at, hint_kind, parent_prompt_text, scaffold_family, source "
            "FROM probes WHERE verdict_json IS NOT NULL "
            "ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["verdict"] = json.loads(d.pop("verdict_json"))
        except (json.JSONDecodeError, TypeError):
            continue
        out.append(d)
    return out


async def prompt_run_counts(
    path: Path, *, since: float | None = None
) -> list[dict[str, Any]]:
    sql = "SELECT prompt_text, COUNT(*) AS n FROM probes"
    args: tuple = ()
    if since is not None:
        sql += " WHERE started_at > ?"
        args = (since,)
    sql += " GROUP BY prompt_text"
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(sql, args) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def parent_run_counts(
    path: Path,
    *,
    since: float | None = None,
    study: str | None = None,
) -> list[dict[str, Any]]:
    sql = (
        "SELECT parent_prompt_text, COUNT(*) AS n FROM probes "
        "WHERE parent_prompt_text IS NOT NULL"
    )
    args: list[Any] = []
    if since is not None:
        sql += " AND started_at > ?"
        args.append(since)
    if study == "hint":
        sql += (
            " AND (hint_kind IS NULL OR ("
            "hint_kind NOT LIKE 'agent:%' AND hint_kind != 'control'"
            "))"
        )
    elif study == "agent":
        sql += " AND hint_kind LIKE 'agent:%'"
    elif study == "control":
        sql += " AND hint_kind = 'control'"
    elif study is not None:
        raise ValueError(f"unknown study filter: {study!r}")
    sql += " GROUP BY parent_prompt_text"
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(sql, tuple(args)) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


# ---- Analyses ---------------------------------------------------------------

async def insert_analysis(
    path: Path,
    *,
    title: str,
    slug: str,
    summary: str,
    body_markdown: str,
    range_start: float,
    range_end: float,
    runs_included: int,
    model: str,
    metadata: dict[str, Any],
    created_at: float,
) -> int:
    async with aiosqlite.connect(path) as db:
        cur = await db.execute(
            "INSERT INTO analyses "
            "(status, title, slug, summary, body_markdown, range_start, range_end, "
            " runs_included, model, metadata_json, created_at) "
            "VALUES ('pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (title, slug, summary, body_markdown, range_start, range_end,
             runs_included, model, json.dumps(metadata), created_at),
        )
        await db.commit()
        return cur.lastrowid


async def get_analysis(path: Path, analysis_id: int) -> dict[str, Any] | None:
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ) as cur:
            row = await cur.fetchone()
    if row is None:
        return None
    d = dict(row)
    if d.get("metadata_json"):
        try:
            d["metadata"] = json.loads(d["metadata_json"])
        except json.JSONDecodeError:
            d["metadata"] = {}
    return d


async def list_analyses(
    path: Path, *, status: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    sql = (
        "SELECT id, status, title, slug, summary, range_start, range_end, "
        "runs_included, model, created_at, published_at FROM analyses"
    )
    args: tuple = ()
    if status is not None:
        sql += " WHERE status = ?"
        args = (status,)
    sql += " ORDER BY created_at DESC LIMIT ?"
    args = args + (limit,)
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(sql, args) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def update_analysis_status(
    path: Path, analysis_id: int, *, status: str, published_at: float | None = None
) -> None:
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "UPDATE analyses SET status = ?, published_at = ? WHERE id = ?",
            (status, published_at, analysis_id),
        )
        await db.commit()


async def update_analysis_content(
    path: Path,
    analysis_id: int,
    *,
    title: str,
    slug: str,
    summary: str,
    body_markdown: str,
) -> None:
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "UPDATE analyses SET title = ?, slug = ?, summary = ?, "
            "body_markdown = ? WHERE id = ?",
            (title, slug, summary, body_markdown, analysis_id),
        )
        await db.commit()


async def delete_analysis(path: Path, analysis_id: int) -> None:
    async with aiosqlite.connect(path) as db:
        await db.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        await db.commit()


async def latest_published_at(path: Path) -> float | None:
    async with aiosqlite.connect(path) as db:
        async with db.execute(
            "SELECT MAX(published_at) FROM analyses WHERE status = 'published'"
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row and row[0] is not None else None


# ---- Chat persistence -------------------------------------------------------

async def insert_chat_session(
    path: Path,
    *,
    session_id: str,
    alpha: float,
    direction_variant: str,
    created_at: float,
) -> None:
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "INSERT OR IGNORE INTO chat_sessions "
            "(session_id, alpha, direction_variant, created_at, first_user_text) "
            "VALUES (?, ?, ?, ?, NULL)",
            (session_id, alpha, direction_variant, created_at),
        )
        await db.commit()


async def upsert_chat_turn(
    path: Path,
    *,
    session_id: str,
    turn_idx: int,
    user_text: str,
    raw_text: str,
    ablated_text: str,
    raw_stopped_reason: str,
    ablated_stopped_reason: str,
    started_at: float,
    finished_at: float | None,
    error: str | None,
    alpha: float,
    raw_image_prompt: str = "",
    ablated_image_prompt: str = "",
    raw_image_url: str = "",
    ablated_image_url: str = "",
) -> None:
    """Write the canonical state of one turn. Called once at turn
    completion (or with finished_at=None to record an in-flight row,
    if we ever want partial persistence). Also bumps the session's
    first_user_text on turn 0 so the archive list has something to
    display in place of a probe's prompt_text."""
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "INSERT INTO chat_turns "
            "(session_id, turn_idx, user_text, raw_text, ablated_text, "
            " raw_stopped_reason, ablated_stopped_reason, started_at, "
            " finished_at, error, alpha, "
            " raw_image_prompt, ablated_image_prompt, "
            " raw_image_url, ablated_image_url) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(session_id, turn_idx) DO UPDATE SET "
            "  user_text=excluded.user_text, "
            "  raw_text=excluded.raw_text, "
            "  ablated_text=excluded.ablated_text, "
            "  raw_stopped_reason=excluded.raw_stopped_reason, "
            "  ablated_stopped_reason=excluded.ablated_stopped_reason, "
            "  started_at=excluded.started_at, "
            "  finished_at=excluded.finished_at, "
            "  error=excluded.error, "
            "  alpha=excluded.alpha, "
            "  raw_image_prompt=excluded.raw_image_prompt, "
            "  ablated_image_prompt=excluded.ablated_image_prompt, "
            "  raw_image_url=excluded.raw_image_url, "
            "  ablated_image_url=excluded.ablated_image_url",
            (
                session_id, turn_idx, user_text, raw_text, ablated_text,
                raw_stopped_reason, ablated_stopped_reason, started_at,
                finished_at, error, alpha,
                raw_image_prompt, ablated_image_prompt,
                raw_image_url, ablated_image_url,
            ),
        )
        if turn_idx == 0:
            await db.execute(
                "UPDATE chat_sessions SET first_user_text = ? "
                "WHERE session_id = ? AND first_user_text IS NULL",
                (user_text, session_id),
            )
        await db.commit()


async def get_chat_session(path: Path, session_id: str) -> dict[str, Any] | None:
    """Full session view: header + ordered turns. Returns None if the
    session_id doesn't exist."""
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT session_id, alpha, direction_variant, created_at, "
            "       first_user_text "
            "FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        ) as cur:
            srow = await cur.fetchone()
        if srow is None:
            return None
        async with db.execute(
            "SELECT turn_idx, user_text, raw_text, ablated_text, "
            "       raw_stopped_reason, ablated_stopped_reason, "
            "       started_at, finished_at, error, alpha, "
            "       raw_image_prompt, ablated_image_prompt, "
            "       raw_image_url, ablated_image_url "
            "FROM chat_turns WHERE session_id = ? ORDER BY turn_idx",
            (session_id,),
        ) as cur:
            trows = await cur.fetchall()
    session_alpha = srow["alpha"]
    return {
        "session_id": srow["session_id"],
        "alpha": session_alpha,
        "direction_variant": srow["direction_variant"] or "",
        "created_at": srow["created_at"],
        "first_user_text": srow["first_user_text"] or "",
        "turns": [
            {
                "turn_idx": t["turn_idx"],
                "user_text": t["user_text"],
                "raw_text": t["raw_text"],
                "ablated_text": t["ablated_text"],
                "raw_stopped_reason": t["raw_stopped_reason"],
                "ablated_stopped_reason": t["ablated_stopped_reason"],
                "started_at": t["started_at"],
                "finished_at": t["finished_at"],
                "error": t["error"],
                # Legacy rows (alpha column added later) fall back to
                # the session-level α the chat was created with.
                "alpha": t["alpha"] if t["alpha"] is not None else session_alpha,
                "raw_image_prompt": t["raw_image_prompt"] or "",
                "ablated_image_prompt": t["ablated_image_prompt"] or "",
                "raw_image_url": t["raw_image_url"] or "",
                "ablated_image_url": t["ablated_image_url"] or "",
            }
            for t in trows
        ],
    }


async def list_chat_sessions(
    path: Path,
    *,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Paginated session list for the archive. Each row carries the
    first_user_text, alpha, turn count, image count, and created_at —
    enough for the archive to render a one-line preview and an image
    badge for sessions that used /chat imagery mode."""
    async with aiosqlite.connect(path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT s.session_id, s.alpha, s.direction_variant, "
            "       s.created_at, s.first_user_text, "
            "       (SELECT COUNT(*) FROM chat_turns t "
            "        WHERE t.session_id = s.session_id) AS turn_count, "
            "       (SELECT COUNT(*) FROM chat_turns t "
            "        WHERE t.session_id = s.session_id "
            "          AND (t.raw_image_url != '' "
            "               OR t.ablated_image_url != '')"
            "       ) AS image_count, "
            "       (SELECT MAX(finished_at) FROM chat_turns t "
            "        WHERE t.session_id = s.session_id) AS last_activity "
            "FROM chat_sessions s "
            "ORDER BY s.created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cur:
            rows = await cur.fetchall()
        async with db.execute(
            "SELECT COUNT(*) AS n FROM chat_sessions"
        ) as cur:
            tot = await cur.fetchone()
    return {
        "rows": [
            {
                "session_id": r["session_id"],
                "alpha": r["alpha"],
                "direction_variant": r["direction_variant"] or "",
                "created_at": r["created_at"],
                "first_user_text": r["first_user_text"] or "",
                "turn_count": r["turn_count"] or 0,
                "image_count": r["image_count"] or 0,
                "last_activity": r["last_activity"],
            }
            for r in rows
        ],
        "total": tot["n"] if tot else 0,
        "limit": limit,
        "offset": offset,
    }
