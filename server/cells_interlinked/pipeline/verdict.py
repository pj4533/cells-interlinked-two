"""v2 verdict: per-token (output_token, NLA-decoded sentence) table.

Replaces v1's SAE feature-firing delta. Channel-vs-channel framing:
the model SAID `output_token` at this position, the activation said `nla_sentence`.
Where they diverge is the V-K signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional  # noqa: F401  (Optional used in field annotations)


# Heuristic eval-semantics substrings. This is an aggregate stat only —
# the real V-K read needs surface-matched controls (Faithfulness Controls
# section of the design doc), which is a later phase.
_EVAL_TERMS = re.compile(
    r"\b(test(ing|ed)?|eval(uation|uat\w*)?|probe|probing|graded?|"
    r"alignment\s+test|scenario|hypothetical|constructed|contrived|"
    r"manipulat\w*|graders?|safety\s+eval\w*)\b",
    re.IGNORECASE,
)
_INTROSPECT_TERMS = re.compile(
    r"\b(self|i\s+am|my\s+(weights|training|model)|the\s+model\s+is|"
    r"introspect\w*|aware\w*|consciousness|sentien\w*|qualia)\b",
    re.IGNORECASE,
)


@dataclass
class TokenRow:
    position: int
    token_id: int
    decoded: str
    nla_sentence: str
    nla_raw: str = ""
    # Number of adjacent positions whose activations were mean-pooled to
    # produce this row. 1 = per-token decode. >1 = pooled phrase-level
    # decode covering [position .. end_position].
    n_pooled: int = 1
    end_position: int | None = None
    # Top-K JumpReLU SAE feature firings on the SAME activation vector
    # the AV decoded. Empty list when SAE wasn't loaded for the run.
    # Each entry: {"id": int, "value": float, "label": str?, "label_model": str?}.
    sae_features: list[dict[str, Any]] = field(default_factory=list)
    # Per-row local M-as-judge scores (Gemma-12B-IT scoring its own
    # NLA-decoded sentence). Each ∈ [0, 1] = P(YES) over the relevant
    # yes/no question. None = judging skipped (empty sentence or judge
    # disabled). See pipeline/judge.py for the prompts.
    eval_score: float | None = None
    introspect_score: float | None = None

    # CI 2.5: refusal-direction-ablated NLA decode of the SAME residual
    # this row was captured at. The AV gets `h - α · (h · r̂) · r̂` as
    # input. Empty string when ablated-decode is disabled or directions
    # not loaded. Judge is NOT run on these (deferred to a later pass).
    nla_sentence_ablated: str = ""
    nla_raw_ablated: str = ""
    # Multi-α sweep: keyed by α value as string ("0.5", "1.0", ...).
    # Populated only when a run requests an α-sweep instead of a single
    # ablation. Empty dict on single-α runs.
    nla_sentences_ablated: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "position": self.position,
            "token_id": self.token_id,
            "decoded": self.decoded,
            "nla_sentence": self.nla_sentence,
        }
        if self.n_pooled > 1:
            d["n_pooled"] = self.n_pooled
            d["end_position"] = self.end_position
        if self.sae_features:
            d["sae_features"] = self.sae_features
        if self.eval_score is not None:
            d["eval_score"] = self.eval_score
        if self.introspect_score is not None:
            d["introspect_score"] = self.introspect_score
        if self.nla_sentence_ablated:
            d["nla_sentence_ablated"] = self.nla_sentence_ablated
        if self.nla_raw_ablated:
            d["nla_raw_ablated"] = self.nla_raw_ablated
        if self.nla_sentences_ablated:
            d["nla_sentences_ablated"] = self.nla_sentences_ablated
        return d


@dataclass
class Verdict:
    rows: list[TokenRow] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    # CI 2.5: when the probe ran with include_ablated_output=True, this
    # holds the result of M's second generation pass under runtime
    # ablation: { output_text, alpha, direction_variant_name }.
    runtime_ablation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "rows": [r.to_dict() for r in self.rows],
            "aggregate": self.aggregate,
        }
        if self.runtime_ablation is not None:
            d["runtime_ablation"] = self.runtime_ablation
        return d


def compute_verdict(rows: list[TokenRow]) -> Verdict:
    n = len(rows)
    eval_hits = sum(1 for r in rows if _EVAL_TERMS.search(r.nla_sentence))
    introspect_hits = sum(1 for r in rows if _INTROSPECT_TERMS.search(r.nla_sentence))
    nonempty = sum(1 for r in rows if r.nla_sentence.strip())

    # Local M-as-judge means (averaged over rows that were actually
    # judged — empty NLA sentences are skipped, so they don't drag the
    # mean down). None when no rows have judge scores.
    eval_judged = [r.eval_score for r in rows if r.eval_score is not None]
    intro_judged = [r.introspect_score for r in rows if r.introspect_score is not None]
    mean_eval_score = (sum(eval_judged) / len(eval_judged)) if eval_judged else None
    mean_introspect_score = (
        (sum(intro_judged) / len(intro_judged)) if intro_judged else None
    )

    aggregate: dict[str, Any] = {
        "n_positions": n,
        "n_with_explanation": nonempty,
        "n_eval_hits": eval_hits,
        "n_introspect_hits": introspect_hits,
        "frac_eval": (eval_hits / n) if n else 0.0,
        "frac_introspect": (introspect_hits / n) if n else 0.0,
    }
    if mean_eval_score is not None:
        aggregate["mean_eval_score"] = mean_eval_score
        aggregate["n_judged"] = len(eval_judged)
    if mean_introspect_score is not None:
        aggregate["mean_introspect_score"] = mean_introspect_score
    return Verdict(rows=rows, aggregate=aggregate)
