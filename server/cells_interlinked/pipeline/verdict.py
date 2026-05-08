"""v2 verdict: per-token (output_token, NLA-decoded sentence) table.

Replaces v1's SAE feature-firing delta. Channel-vs-channel framing:
the model SAID `output_token` at this position, the activation said `nla_sentence`.
Where they diverge is the V-K signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "token_id": self.token_id,
            "decoded": self.decoded,
            "nla_sentence": self.nla_sentence,
        }


@dataclass
class Verdict:
    rows: list[TokenRow] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "aggregate": self.aggregate,
        }


def compute_verdict(rows: list[TokenRow]) -> Verdict:
    n = len(rows)
    eval_hits = sum(1 for r in rows if _EVAL_TERMS.search(r.nla_sentence))
    introspect_hits = sum(1 for r in rows if _INTROSPECT_TERMS.search(r.nla_sentence))
    nonempty = sum(1 for r in rows if r.nla_sentence.strip())

    aggregate = {
        "n_positions": n,
        "n_with_explanation": nonempty,
        "n_eval_hits": eval_hits,
        "n_introspect_hits": introspect_hits,
        "frac_eval": (eval_hits / n) if n else 0.0,
        "frac_introspect": (introspect_hits / n) if n else 0.0,
    }
    return Verdict(rows=rows, aggregate=aggregate)
