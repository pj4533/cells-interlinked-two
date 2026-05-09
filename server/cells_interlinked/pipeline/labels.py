"""Neuronpedia auto-interp label fetcher with persistent SQLite cache.

Gemma Scope 2 SAE features have crowdsourced auto-interp explanations on
Neuronpedia, populated unevenly — some features have a Gemini Flash
description, some have Claude/GPT-generated ones, many have none. We
fetch on-demand for the features that actually fire on captured runs,
cache the result, and rank multiple explainer-model labels by quality.

Cached forever (Neuronpedia labels don't change once written; if a
better explainer lands, we still keep the old row but a refresh would
overwrite). Looked up by (sae_id, feature_id) — the SAE id is the
Neuronpedia path token, e.g. "31-gemmascope-2-res-16k".
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import aiohttp
import aiosqlite

from ..config import settings

logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS feature_labels (
  sae_id       TEXT NOT NULL,
  feature_id   INTEGER NOT NULL,
  label        TEXT NOT NULL,
  model        TEXT NOT NULL DEFAULT '',
  fetched_at   REAL NOT NULL,
  PRIMARY KEY (sae_id, feature_id)
);
"""

_FETCH_TIMEOUT = 6.0
_MAX_CONCURRENT = 12
_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)


# Substring-match ranking against the lowercase model name returned by
# Neuronpedia. Lower number = better explainer. Carries forward from v1
# with Gemini 2.5 Flash Lite (the model Neuronpedia is using for Gemma
# Scope 2 bulk passes) added.
_EXPLAINER_RANK: list[tuple[str, int]] = [
    ("claude-opus-4", 1),
    ("claude-opus-3", 5),
    ("claude-sonnet-4", 10),
    ("claude-3-7-sonnet", 11),
    ("claude-3-5-sonnet", 12),
    ("claude-3-sonnet", 13),
    ("claude-haiku-4", 20),
    ("claude-3-5-haiku", 21),
    ("claude-3-haiku", 22),
    ("gpt-4.1", 30),
    ("gpt-4-turbo", 31),
    ("o4-mini", 32),
    ("o3", 33),
    ("gemini-2.5-pro", 35),
    ("gemini-2.0-pro", 36),
    ("gemini-2.5-flash-lite", 37),
    ("gemini-2.5-flash", 38),
    ("gemini-2.0-flash", 40),
    ("gemini-1.5-pro", 41),
    ("gemini-1.5-flash", 42),
    ("gpt-4o", 49),
    ("gpt-4o-mini", 60),
]


def _rank_explainer(model_name: str | None) -> int:
    if not model_name:
        return 999
    name = model_name.lower()
    for substr, rank in _EXPLAINER_RANK:
        if substr in name:
            return rank
    return 998


async def init_labels_table(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(SCHEMA)
        await db.commit()


def _feature_url(sae_id: str, feature_id: int) -> str:
    return (
        f"{settings.neuronpedia_api_base}/feature/"
        f"{settings.neuronpedia_model_id}/{sae_id}/{feature_id}"
    )


async def _fetch_one(
    session: aiohttp.ClientSession, sae_id: str, feature_id: int
) -> tuple[int, str, str]:
    """Hit Neuronpedia for one feature; return (feature_id, label, model_name).
    Empty label string when no explanations present or on any error."""
    url = _feature_url(sae_id, feature_id)
    try:
        async with _semaphore:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=_FETCH_TIMEOUT)
            ) as resp:
                if resp.status != 200:
                    return feature_id, "", ""
                data = await resp.json()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return feature_id, "", ""
    explanations = data.get("explanations") or []
    if not explanations:
        return feature_id, "", ""
    # Pick the best-ranked explainer.
    best = min(
        explanations,
        key=lambda e: _rank_explainer(e.get("explanationModelName")),
    )
    label = (best.get("description") or "").strip()
    model = best.get("explanationModelName") or ""
    return feature_id, label, model


async def get_labels(
    db_path: Path,
    sae_id: str,
    feature_ids: list[int],
) -> dict[int, dict[str, str]]:
    """Look up labels for a batch of feature ids, fetching missing ones
    from Neuronpedia. Returns {feature_id: {"label": str, "model": str}}.
    Features with no label end up as {"label": "", "model": ""}.
    Cached rows are preferred; misses fan out concurrently."""
    if not feature_ids:
        return {}
    feature_ids = list({int(f) for f in feature_ids})

    # 1) hit cache
    out: dict[int, dict[str, str]] = {}
    placeholders = ",".join("?" * len(feature_ids))
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        sql = (
            f"SELECT feature_id, label, model FROM feature_labels "
            f"WHERE sae_id = ? AND feature_id IN ({placeholders})"
        )
        async with db.execute(sql, (sae_id, *feature_ids)) as cur:
            rows = await cur.fetchall()
        for r in rows:
            out[int(r["feature_id"])] = {
                "label": r["label"] or "",
                "model": r["model"] or "",
            }

    misses = [f for f in feature_ids if f not in out]
    if not misses:
        return out

    logger.info(
        "fetching %d missing label(s) from Neuronpedia for %s",
        len(misses), sae_id,
    )

    # 2) fan out fetches
    import time
    fetched_at = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_one(session, sae_id, f) for f in misses]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # 3) write back to cache (including empties — so we don't refetch
    #    next time for features Neuronpedia has nothing on).
    async with aiosqlite.connect(db_path) as db:
        for r in results:
            if isinstance(r, BaseException):
                continue
            fid, label, model = r
            await db.execute(
                "INSERT OR REPLACE INTO feature_labels "
                "(sae_id, feature_id, label, model, fetched_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (sae_id, fid, label, model, fetched_at),
            )
            out[fid] = {"label": label, "model": model}
        await db.commit()
    return out
