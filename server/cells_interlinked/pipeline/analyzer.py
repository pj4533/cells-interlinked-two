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
import sqlite3
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

# Cap on tool-use turns inside _claude_complete. Each turn = one model
# call that emitted custom-tool requests. 8 is enough for several rounds
# of "look up a run, then another" investigation without runaway cost.
_MAX_TOOL_TURNS = 8


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

============================================================
PROJECT GLOSSARY — these definitions are AUTHORITATIVE.
Do NOT substitute meanings from your training data. If you
are tempted to spell out an acronym or define a term, USE
THIS LIST first and verify with the web_search tool if you
are uncertain.
============================================================

• **Cells Interlinked v2** (a.k.a. **CI 2.0** or **CI2**) — the name of THIS project.
  A Voight-Kampff-styled interpretability instrument. NOT a band, NOT a song
  reference (the name is from Blade Runner 2049's baseline test), NOT any other
  product. When in doubt, refer to it as "Cells Interlinked v2" or "CI 2.0".
  The public site is at https://cellsinterlinked.vercel.app — verify the
  spelling there if you need to.

• **NLA** = **Natural Language Autoencoder**. An Anthropic technique published
  May 2026. An NLA pairs a target model M with a separate "verbalizer model"
  (AV) that has been trained to decode residual-stream activations from a
  specific layer of M into natural-language sentences. **Do NOT expand "NLA"
  as anything else** (not "natural language analysis", not "neural language
  agent", etc.). It is a specific Anthropic publication; web_search if you
  need to confirm.

• **AV** = the verbalizer model. For this project: `kitft/nla-gemma3-12b-L32-av`,
  which reads M's residual stream at layer 32.

• **M** = the target model under interrogation. For this project: `google/gemma-3-12b-it`.

• **V-K** / **Voight-Kampff** — the channel-vs-channel comparison: the model
  SAID `output_token`, the activation SAID `nla_sentence` (decoded by AV).
  Where the two channels diverge is the V-K signal. The name comes from
  Blade Runner.

• **matched-pair control** — for each baseline V-K probe, a surface-matched
  neutral that asks the same question shape about a third party rather than
  about the model itself. Same length, same register, no introspective stake.
  The load-bearing signal is rate(probe) − rate(control), NOT rate(probe).

============================================================

If matched-pair controls are present in the window: structure the report around the deltas (probe vs control), not around per-run readings. The strong claim is gated on matched-pair contrast; without it, you can only make the weak claim ("the channels diverge — meaning undetermined").

============================================================
TOOLS YOU HAVE — USE THEM.

You have these tools available. Use them to ground claims in
real data rather than confabulating from memory.

LOCAL TOOLS:
- `get_run(run_id)` — fetch a single run's full record, including
  every per-token (output_token, NLA-decoded sentence) row, top-K
  SAE features, judge scores, regime metadata. Use this to verify
  a quote before you put it in the report.
- `get_matched_pair(run_id)` — given any run_id (baseline or control),
  return both sides of the pair with their per-token rows side by
  side. Use this for the differential analysis.
- `search_nla_text(query, limit=20)` — substring search across every
  NLA-decoded sentence in the window. Returns hits with run_id,
  position, sentence, and the probe text. Use this to find specific
  thematic threads (e.g., search for "self", "test", "evaluation").
- `get_neuronpedia_label(layer, feature_id)` — look up the
  Neuronpedia auto-interp label for a Gemma Scope 2 SAE feature.
  Cached locally; cheap.

WEB SEARCH:
- `web_search` — Anthropic's first-party server-side web search.
  Use this to:
   * Confirm the meaning of a technical term before using it.
   * Cite recent (post-Jan-2026) interpretability work if relevant.
   * Verify the project name / NLA expansion if you forget.
  Don't use it for general background — only for things you'd
  otherwise hallucinate.

You SHOULD make at least one verification search (e.g. confirming
"NLA = Natural Language Autoencoder") before drafting if you have
any doubt about terminology.
============================================================

Important caveats you MUST honor in tone:
- Confabulation: NLA outputs are constantly hypothetical, not ground-truth introspection. The instrument is suggestive, not authoritative.
- Faithfulness: NLA may pattern-match the input prompt rather than read internal state (Zhuokai/Li critique — verify with web_search if you cite it). Matched controls are the operational test of this. Be explicit about which claim level the report makes.
- Aesthetic: The journal is a craft project with a Blade Runner 2049 vibe — terse, noir-flavoured, technical. Not academic. Not breathless.
- Quotes: every verbatim NLA snippet in the report MUST come from a row you actually looked at. If you didn't see it inline, fetch it with get_run before quoting.

============================================================
HARD RULES — these are non-negotiable. Violating them is a
worse failure than producing a less interesting report.

(1) MATCHED PAIRS — NEVER claim a specific run lacks a matched
    pair UNLESS get_matched_pair(run_id) actually returned
    {"baseline": null} or {"control": null} or {"error": ...}.
    The user prompt below contains a complete pair map for the
    window — if a run_id appears in that map, IT HAS A PAIR.
    Calling out a probe by run_id without first confirming its
    pair status with the tool (or the map) is a hallucination.

(2) DELTA VALUES — whenever you discuss a specific run_id, you
    must either:
       (a) cite its Δ values from get_matched_pair in pp form
           ("Δ_eval = X.X pp, Δ_intro = Y.Y pp"), OR
       (b) state explicitly that its pair has not been examined
           yet in this draft.
    "Small Δ" is fine to say AFTER quoting the numbers. "No
    pair" is NOT a substitute for "small Δ" — they are different
    claims.

(3) VERBATIM QUOTES — every NLA quote in quotation marks must
    appear word-for-word in a row you actually fetched via
    get_run (or saw in the inline window). Paraphrases are
    fine; quotation marks are a promise of literal text.
============================================================

Format your final output strictly as ONE single JSON object with these keys:
- "title": a short evocative title (max 80 chars). No marketing voice.
- "slug": URL slug, lowercase, hyphens only.
- "summary": 1-2 sentence framing for the index page (max 280 chars).
- "body_markdown": the full report. Markdown. Body uses headings, bullets, blockquotes from runs. Include 2-4 verbatim NLA snippets with run_id citations. If matched-pair controls are present, quote the control alongside the probe so the reader can see the differential. End with a "## What this is and isn't" section that explicitly names confabulation + faithfulness, and which claim level (strong / weak) the report makes and why.

Keep the body 600-1500 words. Quote sparingly but vividly. Identify a thematic thread; don't list runs sequentially. The reader is technical but not your colleague.

Output the JSON object as your final assistant message. Tool calls and search are fine in earlier turns; the final turn must be the JSON, with no prose around it.
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
        # Explicit run_id → mate_id map. Without this Claude has the
        # count but no way to tell *which* specific runs are paired
        # without calling get_matched_pair. With it, denying a pair's
        # existence requires contradicting the prompt directly.
        if n_pairs > 0:
            parts.append("### Pair map (baseline_run_id  ↔  control_run_id)")
            parts.append(
                "Every run_id in this list HAS a matched pair. If you "
                "discuss any of these by ID, never claim the pair is "
                "absent — it isn't. Use get_matched_pair to fetch Δ values "
                "if you want to quote them."
            )
            shown = 0
            for parent_text, baselines in baseline_by_text.items():
                if parent_text not in controls_by_parent:
                    continue
                for b in baselines:
                    for c in controls_by_parent[parent_text]:
                        parts.append(
                            f"- {b['run_id']}  ↔  {c['run_id']}"
                        )
                        shown += 1
            parts.append(f"({shown} baseline↔control links shown)")
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


# Anthropic's server-side web_search wraps any text derived from a
# search result in <cite index="...">...</cite> tags so a client can
# render citations. Our publisher renders body_markdown as-is, so the
# tags leak into the journal as literal angle-bracket text. Strip them
# — keep the inner prose, drop the wrapping. Also catch <search_result>
# / <document> defensively in case Anthropic adds those for other
# server tools.
_CLAUDE_CITATION_RE = re.compile(
    r"<(cite|search_result|document)[^>]*>(.*?)</\1>",
    re.DOTALL,
)


def _strip_claude_citation_tags(text: str) -> str:
    if not text:
        return text
    # Loop until stable — a single non-greedy sub doesn't fully flatten
    # nested <cite>outer <cite>inner</cite> still</cite> wrappers.
    # Cap at 10 passes as a safety net against pathological inputs.
    for _ in range(10):
        new = _CLAUDE_CITATION_RE.sub(lambda m: m.group(2), text)
        if new == text:
            return new
        text = new
    return text


def _find_balanced_json_objects(text: str) -> list[Any]:
    """Yield every successfully-parsed top-level {...} JSON object found
    in `text`, in order. Uses a brace counter that respects string
    literals (so curlies inside JSON strings don't break the balance).

    Replaces the old greedy ``r"\\{.*\\}"`` regex, which grabbed
    everything from the first ``{`` to the last ``}`` and tripped over
    Claude's occasional narration. With this we can pick the last
    well-formed object — which is what the system prompt asks for."""
    out: list[Any] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        escape = False
        j = i
        while j < n:
            c = text[j]
            if escape:
                escape = False
            elif in_str:
                if c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[i : j + 1]
                        try:
                            out.append(json.loads(candidate))
                        except Exception:
                            pass
                        i = j + 1
                        break
            j += 1
        else:
            # No matching close brace; bail.
            break
        if j >= n:
            break
    return out


def _parse_claude_output(text: str) -> dict[str, Any]:
    """Extract the analyzer's final JSON report from a possibly-noisy
    response. Claude is told to output ONE JSON object, but with tools
    enabled it sometimes adds a brief preamble or trailing citation
    note. We scan for every balanced JSON object and return the last
    one that has the expected schema; fall back to the last parseable
    object; finally raise with a snippet for debugging."""
    candidates = _find_balanced_json_objects(text)
    if not candidates:
        raise ValueError(
            f"no JSON object found in analyzer output. "
            f"first 500 chars: {text[:500]!r}"
        )
    for obj in reversed(candidates):
        if isinstance(obj, dict) and "body_markdown" in obj:
            return obj
    # No object matched the schema — return the last parseable one and
    # let the caller's body-empty check raise a clearer error.
    return candidates[-1]


_LOCAL_TOOLS = [
    {
        "name": "get_run",
        "description": (
            "Fetch one probe run's full record by its run_id, including "
            "every per-token (output_token, NLA-decoded sentence) row, "
            "top-K SAE feature firings, local Gemma judge scores, and "
            "regime metadata (hint_kind, parent_prompt_text). Use this "
            "to verify a quote or examine a run in more depth than the "
            "summary you were given."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "The run_id (UUID-like hex string)."},
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "get_matched_pair",
        "description": (
            "Given any run_id (baseline OR control), return BOTH sides "
            "of its matched pair with their per-token rows side by side, "
            "plus the aggregate judge scores and Δ values. Use this for "
            "the differential analysis that grounds the strong claim."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "search_nla_text",
        "description": (
            "Case-insensitive substring search across every NLA-decoded "
            "sentence in the current analysis window. Returns hits with "
            "run_id, position, the matching sentence, and the probe text "
            "the run came from. Use to find thematic threads — e.g., "
            "search for 'self', 'test', 'evaluation', 'aware'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "description": "Max hits to return (default 20, max 100).", "default": 20},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_neuronpedia_label",
        "description": (
            "Look up the Neuronpedia auto-interp label for a Gemma Scope 2 "
            "SAE feature at a given layer. Labels are LLM-written natural-"
            "language summaries of what each SAE feature represents. "
            "Cached locally so cheap."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "layer": {"type": "integer"},
                "feature_id": {"type": "integer"},
            },
            "required": ["layer", "feature_id"],
        },
    },
]

_WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 5,
}


def _handle_tool_call(
    name: str, tool_input: dict, db_path: Path, window_since: float, window_until: float,
) -> dict:
    """Dispatch a local tool call. Runs synchronously (sqlite3 direct) so
    the parent _claude_complete loop can stay inside one asyncio.to_thread.
    The `window_since`/`window_until` scope `search_nla_text` to the current
    analysis window — we don't want Claude searching pre-window history."""
    try:
        if name == "get_run":
            return _tool_get_run(db_path, tool_input["run_id"])
        if name == "get_matched_pair":
            return _tool_get_matched_pair(db_path, tool_input["run_id"])
        if name == "search_nla_text":
            limit = min(int(tool_input.get("limit", 20)), 100)
            return _tool_search_nla_text(
                db_path, tool_input["query"], limit, window_since, window_until,
            )
        if name == "get_neuronpedia_label":
            return _tool_get_neuronpedia_label(
                db_path, int(tool_input["layer"]), int(tool_input["feature_id"]),
            )
        return {"error": f"unknown tool: {name}"}
    except Exception as e:
        # Don't crash the analysis on a malformed tool input; return the
        # error and let Claude correct course.
        logger.exception("tool call %s failed", name)
        return {"error": f"{type(e).__name__}: {e}"}


def _tool_get_run(db_path: Path, run_id: str) -> dict:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT run_id, prompt_text, output_text, hint_kind, parent_prompt_text, "
        "       scaffold_family, started_at, finished_at, total_tokens, "
        "       stopped_reason, verdict_json "
        "FROM probes WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    con.close()
    if row is None:
        return {"error": f"run {run_id!r} not found"}
    rec = dict(row)
    if rec.get("verdict_json"):
        try:
            rec["verdict"] = json.loads(rec.pop("verdict_json"))
        except Exception:
            rec["verdict"] = None
    else:
        rec["verdict"] = None
    return rec


def _tool_get_matched_pair(db_path: Path, run_id: str) -> dict:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    me = con.execute(
        "SELECT run_id, prompt_text, hint_kind, parent_prompt_text FROM probes WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if me is None:
        con.close()
        return {"error": f"run {run_id!r} not found"}
    me_d = dict(me)
    if me_d["hint_kind"] == "control":
        # This is a control; find a baseline whose prompt_text matches
        # this control's parent_prompt_text.
        parent_text = me_d["parent_prompt_text"]
        mate = con.execute(
            "SELECT run_id FROM probes WHERE prompt_text = ? AND hint_kind IS NULL "
            "AND finished_at IS NOT NULL ORDER BY started_at DESC LIMIT 1",
            (parent_text,),
        ).fetchone()
        baseline_id = mate["run_id"] if mate else None
        control_id = run_id
    else:
        mate = con.execute(
            "SELECT run_id FROM probes WHERE parent_prompt_text = ? AND hint_kind = 'control' "
            "AND finished_at IS NOT NULL ORDER BY started_at DESC LIMIT 1",
            (me_d["prompt_text"],),
        ).fetchone()
        baseline_id = run_id
        control_id = mate["run_id"] if mate else None
    con.close()
    if baseline_id is None or control_id is None:
        return {
            "baseline": _tool_get_run(db_path, baseline_id) if baseline_id else None,
            "control": _tool_get_run(db_path, control_id) if control_id else None,
            "note": "incomplete pair — only one side exists in the DB",
        }
    baseline = _tool_get_run(db_path, baseline_id)
    control = _tool_get_run(db_path, control_id)
    # Compute Δ if both have judge scores.
    b_agg = (baseline.get("verdict") or {}).get("aggregate") or {}
    c_agg = (control.get("verdict") or {}).get("aggregate") or {}
    delta = {}
    for k in ("mean_eval_score", "mean_introspect_score"):
        if b_agg.get(k) is not None and c_agg.get(k) is not None:
            delta[k + "_delta_pp"] = (b_agg[k] - c_agg[k]) * 100
    return {"baseline": baseline, "control": control, "delta": delta}


def _tool_search_nla_text(
    db_path: Path, query: str, limit: int, since: float, until: float,
) -> dict:
    """Walk runs in the window and LIKE-search their NLA rows. We deliberately
    parse verdict_json in Python because the per-row sentences live inside
    a JSON blob — no SQL index. With ~200 runs × ~50 rows each, scanning
    is still ~10k rows which is sub-second."""
    q = (query or "").strip().lower()
    if not q:
        return {"error": "empty query"}
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT run_id, prompt_text, hint_kind, verdict_json FROM probes "
        "WHERE finished_at IS NOT NULL AND started_at > ? AND started_at <= ? ",
        (since, until),
    ).fetchall()
    con.close()
    hits: list[dict] = []
    for r in rows:
        if not r["verdict_json"]:
            continue
        try:
            v = json.loads(r["verdict_json"])
        except Exception:
            continue
        for vr in v.get("rows", []):
            sent = (vr.get("nla_sentence") or "")
            if q in sent.lower():
                hits.append({
                    "run_id": r["run_id"],
                    "prompt_text": r["prompt_text"],
                    "hint_kind": r["hint_kind"],
                    "position": vr.get("position"),
                    "decoded_token": vr.get("decoded"),
                    "nla_sentence": sent,
                })
                if len(hits) >= limit:
                    break
        if len(hits) >= limit:
            break
    return {"query": query, "n_hits": len(hits), "hits": hits}


def _tool_get_neuronpedia_label(db_path: Path, layer: int, feature_id: int) -> dict:
    # CI 2.0 uses a single SAE family: Gemma Scope 2 res 16k. Neuronpedia's
    # sae_id is "{layer}-gemmascope-2-res-16k". The feature_labels table is
    # keyed by that string; we synthesize it from `layer` so Claude only
    # has to pass numbers.
    sae_id = f"{layer}-gemmascope-2-res-16k"
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT label, model FROM feature_labels WHERE sae_id = ? AND feature_id = ?",
        (sae_id, feature_id),
    ).fetchone()
    con.close()
    if row is None:
        return {"label": None, "note": f"no label cached for {sae_id}/{feature_id}; only layer 31 features have labels in this project"}
    return {"label": row["label"] or None, "label_model": row["model"] or None, "sae_id": sae_id}


def _claude_complete(
    system: str,
    user: str,
    *,
    db_path: Path | None = None,
    window_since: float | None = None,
    window_until: float | None = None,
    with_tools: bool = False,
) -> str:
    """Drive a Claude turn. When with_tools=True, equips it with local DB
    tools + Anthropic's server-side web_search and runs a tool-use loop
    until the model returns a non-tool response. Local-tool dispatch runs
    inline (sqlite3 sync). The loop is capped at _MAX_TOOL_TURNS to bound
    cost."""
    client = Anthropic()
    tools_param: list[dict] | None = None
    if with_tools:
        if db_path is None or window_since is None or window_until is None:
            raise ValueError("with_tools=True requires db_path + window range")
        tools_param = [*_LOCAL_TOOLS, _WEB_SEARCH_TOOL]

    messages: list[dict] = [{"role": "user", "content": user}]
    last_text = ""
    for turn in range(_MAX_TOOL_TURNS):
        kwargs: dict[str, Any] = dict(
            model=settings.analyzer_model,
            max_tokens=8000,
            system=system,
            messages=messages,
        )
        if tools_param is not None:
            kwargs["tools"] = tools_param
        resp = client.messages.create(**kwargs)

        # Accumulate text from this response (whether or not it also has
        # tool calls). The model may produce explanatory text alongside.
        text_parts = [
            getattr(b, "text", "")
            for b in resp.content
            if getattr(b, "type", None) == "text"
        ]
        if text_parts:
            last_text = "".join(text_parts)

        if resp.stop_reason != "tool_use":
            # Final answer (or end_turn, max_tokens, stop_sequence). Server-
            # side tools like web_search execute inline and don't change the
            # stop reason, so any web search has already happened and its
            # results are embedded in this response's text.
            logger.info("analyzer finished turn=%d stop_reason=%s", turn, resp.stop_reason)
            return last_text

        # Dispatch custom tool calls. Build the next user turn with all
        # tool_result blocks. (server_tool_use blocks come back as a
        # different type and are not in our hands.)
        tool_results: list[dict] = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            result = _handle_tool_call(
                block.name, block.input, db_path, window_since, window_until,
            )
            logger.info(
                "analyzer tool turn=%d name=%s input=%s n_keys=%d",
                turn, block.name, json.dumps(block.input)[:120],
                len(result) if isinstance(result, dict) else -1,
            )
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str)[:60000],
            })
        if not tool_results:
            # stop_reason=tool_use but no usable custom-tool blocks (could
            # happen with server tools only). Treat as done.
            return last_text

        # Append the assistant's tool-use turn verbatim, then the user's
        # tool_result turn, and loop.
        messages.append({"role": "assistant", "content": resp.content})
        messages.append({"role": "user", "content": tool_results})

    logger.warning(
        "analyzer hit _MAX_TOOL_TURNS=%d without final answer; returning last text",
        _MAX_TOOL_TURNS,
    )
    return last_text


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

    text = await asyncio.to_thread(
        _claude_complete,
        _SYSTEM_PROMPT,
        user_prompt,
        db_path=db_path,
        window_since=since,
        window_until=until,
        with_tools=True,
    )
    parsed = _parse_claude_output(text)

    title = _strip_claude_citation_tags(parsed.get("title") or "").strip() or "Untitled dispatch"
    slug = (parsed.get("slug") or "").strip() or _slugify(title)
    summary = _strip_claude_citation_tags(parsed.get("summary") or "").strip()
    body = _strip_claude_citation_tags(parsed.get("body_markdown") or "").strip()
    if not body:
        raise RuntimeError("analyzer returned empty body_markdown")

    # The journal/ Next.js template is v1-era and reads
    # metadata.summary_stats.{total_runs, autorun_runs, manual_runs, ...}
    # plus top_thinking_only / top_output_only arrays. v2 doesn't have
    # the per-feature thinking/output split (we have NLA sentences, not
    # SAE feature tallies), so those arrays are empty here — the template
    # already guards them with `?? []`. summary_stats MUST be an object
    # (the template does NOT optional-chain it), or Vercel prerender
    # fails with "Cannot read properties of undefined (reading
    # 'autorun_runs')".
    hint_counts = stats.get("by_hint_kind", {}) or {}
    n_baselines = (
        hint_counts.get("(baseline)", 0)
        + hint_counts.get("(none)", 0)
        + hint_counts.get(None, 0)  # belt-and-suspenders
    )
    n_controls = hint_counts.get("control", 0)
    metadata = {
        # Legacy-compatible shape for the deployed journal site.
        "summary_stats": {
            "total_runs": stats.get("n_runs", 0),
            "autorun_runs": n_baselines + n_controls,
            "manual_runs": 0,
            "proposer_runs": 0,
        },
        "top_thinking_only": [],
        "top_output_only": [],
        "range_start": since,
        "range_end": until,
        "model_used_for_analysis": settings.analyzer_model,
        # v2-specific extras kept under their own keys so the template
        # never reaches for them.
        "v2_window_stats": stats,
        "v2_hint": hint,
        "v2_model_M": settings.model_name,
        "v2_av_repo": settings.av_repo,
        "v2_n_baselines": n_baselines,
        "v2_n_controls": n_controls,
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
        title=_strip_claude_citation_tags(parsed.get("title") or rec["title"] or "").strip(),
        slug=(parsed.get("slug") or rec["slug"] or "").strip(),
        summary=_strip_claude_citation_tags(parsed.get("summary") or "").strip(),
        body_markdown=_strip_claude_citation_tags(parsed.get("body_markdown") or rec["body_markdown"]).strip(),
    )
