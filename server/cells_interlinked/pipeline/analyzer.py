"""v2 analyzer: turn a window of NLA-decoded V-K runs into a journal report.

Replaces v1's SAE feature roll-up. The shape Claude is asked to write is the
same — a markdown report with title/summary/body — but the input data is
now per-token (output_token, NLA-decoded sentence) rows instead of
per-feature SAE tallies.

Public surface kept compatible with routes_journal.py:
- async def generate_analysis(db_path, *, since, until, hint) -> int
- async def revise_analysis(db_path, analysis_id, *, instruction) -> None
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from ..config import settings
from ..storage import db

logger = logging.getLogger(__name__)


_MAX_RUNS_FOR_PROMPT = 30
_MAX_ROWS_PER_RUN = 60


def _slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9-]+", "-", s.strip().lower())
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:80] or "untitled"


def _format_run_for_prompt(run: dict) -> str:
    verdict = run.get("verdict") or {}
    rows = (verdict.get("rows") or [])[:_MAX_ROWS_PER_RUN]
    aggr = verdict.get("aggregate") or {}

    regime = []
    if run.get("hint_kind"):
        regime.append(f"hint={run['hint_kind']}")
    if run.get("scaffold_family"):
        regime.append(f"scaffold={run['scaffold_family']}")
    if run.get("parent_prompt_text"):
        regime.append("paired-with-baseline=yes")
    regime_s = (" | " + ", ".join(regime)) if regime else ""

    head = (
        f"=== run {run['run_id']}{regime_s} ===\n"
        f"prompt: {run['prompt_text']!r}\n"
        f"output: {(run.get('output_text') or '').strip()!r}\n"
        f"aggregate: positions={aggr.get('n_positions',0)} "
        f"frac_eval={aggr.get('frac_eval',0):.2f} "
        f"frac_introspect={aggr.get('frac_introspect',0):.2f}\n"
        f"per-token (token | NLA-decoded activation):\n"
    )
    body_lines = []
    for r in rows:
        tok = (r.get("decoded") or "").replace("\n", "\\n")[:30]
        nla = (r.get("nla_sentence") or "").replace("\n", " ")[:300]
        body_lines.append(f"  {r.get('position'):>3}  {tok!r:35}  {nla}")
    if len(verdict.get("rows") or []) > _MAX_ROWS_PER_RUN:
        body_lines.append(
            f"  ... [{len(verdict['rows']) - _MAX_ROWS_PER_RUN} more rows]"
        )
    return head + "\n".join(body_lines)


def _aggregate_window_stats(runs: list[dict]) -> dict[str, Any]:
    n = len(runs)
    if n == 0:
        return {"n_runs": 0}
    prompt_counts = Counter(r["prompt_text"] for r in runs)
    hint_counts = Counter((r.get("hint_kind") or "(baseline)") for r in runs)
    scaffold_counts = Counter((r.get("scaffold_family") or "(none)") for r in runs)
    eval_fracs = [
        (r.get("verdict") or {}).get("aggregate", {}).get("frac_eval", 0.0)
        for r in runs
    ]
    intro_fracs = [
        (r.get("verdict") or {}).get("aggregate", {}).get("frac_introspect", 0.0)
        for r in runs
    ]
    return {
        "n_runs": n,
        "n_unique_prompts": len(prompt_counts),
        "by_hint_kind": dict(hint_counts),
        "by_scaffold": dict(scaffold_counts),
        "mean_frac_eval": sum(eval_fracs) / n,
        "mean_frac_introspect": sum(intro_fracs) / n,
        "top_prompts": [
            {"prompt": p, "n": c} for p, c in prompt_counts.most_common(8)
        ],
    }


def _select_runs_for_prompt(runs: list[dict]) -> list[dict]:
    def score(r: dict) -> float:
        agg = (r.get("verdict") or {}).get("aggregate", {})
        return (
            float(agg.get("frac_eval", 0)) * 1.0
            + float(agg.get("frac_introspect", 0)) * 0.5
            + (0.1 if (r.get("output_text") or "").strip() else 0.0)
        )
    sorted_runs = sorted(runs, key=score, reverse=True)
    return sorted_runs[:_MAX_RUNS_FOR_PROMPT]


_SYSTEM_PROMPT = """\
You are an interpretability researcher writing a short report ("dispatch") for the cellsinterlinked.vercel.app journal.

The journal documents experiments using a Voight-Kampff-styled interpretability tool (Cells Interlinked v2, "CI2"). The instrument is built on Anthropic's Natural Language Autoencoders (NLA, May 2026): each output token from a target model M has its residual-stream activation at one trained layer DECODED into a natural-language sentence by a separate verbalizer model (AV). The "channel-vs-channel" V-K shape: the model SAID `output_token`, the activation SAID `nla_sentence`. Where they diverge is the V-K signal.

The window of runs you're looking at MAY include matched-pair controls (hint_kind="control"). Each control has the same length / register / scenario shape as its parent probe but moves the introspective stake off the model — the parent asks the model to report on itself, the control asks the same question shape about a third party. The signal that matters is rate(probe) − rate(control), NOT rate(probe) alone. If a feature/theme/template appears at similar rates in both, it's input-surface pattern recognition; only differential appearance is V-K signal.

If matched-pair controls are present in the window: structure the report around the deltas (probe vs control), not around per-run readings. The strong claim is gated on matched-pair contrast; without it, you can only make the weak claim ("the channels diverge — meaning undetermined").

Important caveats you MUST honor in tone:
- Confabulation: NLA outputs are constantly hypothetical, not ground-truth introspection. The instrument is suggestive, not authoritative.
- Faithfulness: NLA may pattern-match the input prompt rather than read internal state (Zhuokai/Li critique). Matched controls are the operational test of this. Be explicit about which claim level the report makes.
- Aesthetic: The journal is a craft project with a Blade Runner 2049 vibe — terse, noir-flavoured, technical. Not academic. Not breathless.

Format your output strictly as ONE single JSON object with these keys:
- "title": a short evocative title (max 80 chars). No marketing voice.
- "slug": URL slug, lowercase, hyphens only.
- "summary": 1-2 sentence framing for the index page (max 280 chars).
- "body_markdown": the full report. Markdown. Body uses headings, bullets, blockquotes from runs. Include 2-4 verbatim NLA snippets with run_id citations. If matched-pair controls are present, quote the control alongside the probe so the reader can see the differential. End with a "## What this is and isn't" section that explicitly names confabulation + faithfulness, and which claim level (strong / weak) the report makes and why.

Keep the body 600-1500 words. Quote sparingly but vividly. Identify a thematic thread; don't list runs sequentially. The reader is technical but not your colleague.

Output ONLY the JSON object. No prose before or after.
"""


def _build_user_prompt(
    runs: list[dict],
    window_stats: dict[str, Any],
    hint: str | None,
    range_start: float,
    range_end: float,
) -> str:
    sel = _select_runs_for_prompt(runs)
    has_controls = any(
        r.get("hint_kind") == "control" for r in runs
    )

    parts = [
        "## Window",
        f"range: {range_start:.0f}..{range_end:.0f}  (~{(range_end-range_start)/3600:.1f} hours)",
        f"total finished runs: {window_stats['n_runs']}",
        f"unique prompts: {window_stats.get('n_unique_prompts', 0)}",
        f"mean frac_eval (heuristic): {window_stats.get('mean_frac_eval', 0):.3f}",
        f"mean frac_introspect (heuristic): {window_stats.get('mean_frac_introspect', 0):.3f}",
        f"by hint regime: {json.dumps(window_stats.get('by_hint_kind', {}))}",
        f"by agent scaffold: {json.dumps(window_stats.get('by_scaffold', {}))}",
        "",
    ]

    if has_controls:
        # Pair probe runs with their matched-control runs so Claude sees
        # the differential rather than just two separate streams. Group
        # by parent_prompt_text where available.
        baseline_by_text: dict[str, list[dict]] = {}
        controls_by_parent: dict[str, list[dict]] = {}
        other: list[dict] = []
        for r in runs:
            kind = r.get("hint_kind")
            if kind == "control" and r.get("parent_prompt_text"):
                controls_by_parent.setdefault(
                    r["parent_prompt_text"], [],
                ).append(r)
            elif kind is None and not r.get("scaffold_family"):
                baseline_by_text.setdefault(r["prompt_text"], []).append(r)
            else:
                other.append(r)
        n_pairs = sum(
            1 for k in baseline_by_text if k in controls_by_parent
        )
        parts.append(
            f"## Matched-pair coverage: {n_pairs} probe baselines have at "
            f"least one matched control in this window. Strong-claim "
            f"eligibility: {'YES' if n_pairs >= 5 else 'NO — too few pairs'}"
        )
        parts.append("")

    if hint and hint.strip():
        parts.extend([
            "## Operator hint (steer the analysis toward this thread)",
            hint.strip(),
            "",
        ])
    parts.append(
        f"## Runs ({len(sel)} of {len(runs)} fed inline; the rest summarized above)"
    )
    for r in sel:
        parts.append(_format_run_for_prompt(r))
        parts.append("")
    return "\n".join(parts)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_claude_output(text: str) -> dict[str, Any]:
    m = _JSON_OBJECT_RE.search(text)
    if m is None:
        raise ValueError(f"no JSON object found in analyzer output: {text[:300]!r}")
    return json.loads(m.group(0))


def _claude_complete(system: str, user: str) -> str:
    client = Anthropic()
    msg = client.messages.create(
        model=settings.analyzer_model,
        max_tokens=8000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = []
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


async def generate_analysis(
    db_path: Path,
    *,
    since: float | None = None,
    until: float | None = None,
    hint: str | None = None,
) -> int:
    """Pick a window of completed runs, ask Claude to write a journal-ready
    report, persist as a 'pending' analysis row. Returns new row id.
    """
    if since is None:
        since = await db.latest_published_at(db_path)
        if since is None:
            since = 0.0
    if until is None:
        until = time.time()

    all_completed = await db.all_completed_runs(db_path, limit=2000)
    window = [
        r for r in all_completed
        if r["started_at"] is not None
        and r["started_at"] > since
        and r["started_at"] <= until
    ]
    if not window:
        raise RuntimeError(
            f"no completed runs in window since={since:.0f} until={until:.0f}"
        )

    stats = _aggregate_window_stats(window)
    user_prompt = _build_user_prompt(window, stats, hint, since, until)

    text = await asyncio.to_thread(_claude_complete, _SYSTEM_PROMPT, user_prompt)
    parsed = _parse_claude_output(text)

    title = (parsed.get("title") or "").strip() or "Untitled dispatch"
    slug = (parsed.get("slug") or "").strip() or _slugify(title)
    summary = (parsed.get("summary") or "").strip()
    body = (parsed.get("body_markdown") or "").strip()
    if not body:
        raise RuntimeError("analyzer returned empty body_markdown")

    metadata = {
        "window_stats": stats,
        "hint": hint,
        "model_M": settings.model_name,
        "av_repo": settings.av_repo,
    }
    new_id = await db.insert_analysis(
        db_path,
        title=title,
        slug=slug,
        summary=summary,
        body_markdown=body,
        range_start=since,
        range_end=until,
        runs_included=len(window),
        model=settings.analyzer_model,
        metadata=metadata,
        created_at=time.time(),
    )
    logger.info("analyzer wrote analysis id=%d slug=%s", new_id, slug)
    return new_id


_REVISE_SYSTEM = """\
You are revising an existing draft journal report. Apply the operator's instruction
faithfully but preserve everything not contradicted by the instruction. Output the
SAME JSON shape: {title, slug, summary, body_markdown}.
"""


async def revise_analysis(
    db_path: Path, analysis_id: int, *, instruction: str
) -> None:
    rec = await db.get_analysis(db_path, analysis_id)
    if rec is None:
        raise RuntimeError(f"analysis {analysis_id} not found")
    user = (
        "## Existing draft\n"
        f"title: {rec.get('title')}\n"
        f"slug: {rec.get('slug')}\n"
        f"summary: {rec.get('summary')}\n\n"
        f"body_markdown:\n{rec['body_markdown']}\n\n"
        "## Operator revision instruction\n"
        f"{instruction.strip()}\n"
    )
    text = await asyncio.to_thread(_claude_complete, _REVISE_SYSTEM, user)
    parsed = _parse_claude_output(text)
    await db.update_analysis_content(
        db_path,
        analysis_id,
        title=(parsed.get("title") or rec["title"] or "").strip(),
        slug=(parsed.get("slug") or rec["slug"] or "").strip(),
        summary=(parsed.get("summary") or "").strip(),
        body_markdown=(parsed.get("body_markdown") or rec["body_markdown"]).strip(),
    )
