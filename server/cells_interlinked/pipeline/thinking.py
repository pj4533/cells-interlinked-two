"""Gemma-4 reasoning-channel parsing.

With thinking enabled, Gemma-4's output is:

    <|channel>thought\\n{REASONING}<channel|>{ANSWER}

i.e. token `<|channel>` (open) → the literal channel name "thought" → the
reasoning → token `<channel|>` (close) → the final answer. We split this so
the UIs can show a separate "thinking" bubble and so multi-turn history keeps
ONLY the answer (the docs require stripping prior thoughts to avoid repetition
loops).

`ThinkingSplitter` classifies a token stream incrementally (for live SSE
routing); `split_thinking` does the same over a finished id list.
"""

from __future__ import annotations

CHANNEL_LABEL = "thought"


class ThinkingSplitter:
    """Incremental classifier. Feed (token_id, decoded) per generated step;
    `feed` returns (channel, text) where channel is "thought", "answer", or
    None (a delimiter/label token to suppress — not shown, not accumulated)."""

    def __init__(self, open_id: int | None, close_id: int | None) -> None:
        self.open_id = open_id
        self.close_id = close_id
        # Start in "answer": if thinking is enabled the very first token is the
        # open delimiter, which flips us to "thought". If thinking is disabled
        # (no open delimiter ever arrives) everything stays "answer".
        self.channel = "answer"
        self._skip_label = False

    def feed(self, token_id: int, decoded: str) -> tuple[str | None, str]:
        if self.open_id is not None and token_id == self.open_id:
            self.channel = "thought"
            self._skip_label = True
            return (None, "")
        if self.close_id is not None and token_id == self.close_id:
            self.channel = "answer"
            self._skip_label = False
            return (None, "")
        if self._skip_label:
            self._skip_label = False
            # Suppress the literal channel name ("thought") that follows the
            # open delimiter; anything else falls through as real content.
            if decoded.strip() in ("", CHANNEL_LABEL):
                return (None, "")
        return (self.channel, decoded)


def split_thinking(
    token_ids: list[int], decoded_per_token: list[str],
    open_id: int | None, close_id: int | None,
) -> tuple[str, str]:
    """Split a finished generation into (thinking, answer) text."""
    s = ThinkingSplitter(open_id, close_id)
    thought, answer = [], []
    for tid, dec in zip(token_ids, decoded_per_token):
        ch, text = s.feed(tid, dec)
        if ch == "thought":
            thought.append(text)
        elif ch == "answer":
            answer.append(text)
    return "".join(thought).strip(), "".join(answer)
