"""Step 4 — paired-channel diagnostic.

For each prompt in a diagnostic set, generate THREE channels:
  (a) raw                — no ablation
  (b) global_ablation    — install_runtime_ablation_hook at L32, α=1
  (c) edge_consumer_ablation — install_edge_consumer_ablation_hook
                                with the Step 3 sufficient subset, α=1

For each generated token position t, compute per-channel embeddings of
the surrounding context, then:
  L2_global[t] = ‖embed_raw[t] − embed_global[t]‖
  L2_edge[t]   = ‖embed_raw[t] − embed_edge[t]‖

Bucket tokens into:
  refusal-relevant      — within 5 positions before/at a refusal-marker
                          match in the raw output
  non-refusal-relevant  — everything else

Report mean L2 per bucket per channel — the 2×2 table the falsifier
hangs on:

  | bucket               | L2_global | L2_edge |
  | refusal-relevant     | A         | B       |
  | non-refusal-relevant | C         | D       |

Phase B v1 — embedding strategy: mean-pool Gemma's input embedding
layer over the 8 most-recent tokens at each position. This is a
forward-pass-cheap proxy for "the semantic state at this position";
it does NOT use the AV / NLA-decode pipeline. The NLA-mode upgrade
(persist residuals → swap M out → load AV → decode per token →
embed NLA sentences) is deferred to v2 once basic divergence numbers
land. See docs/EDGE_CONSUMER_ABLATION.md §3 Step 4 detail.

Output (written by run_edge_consumer_diagnostic.py):
  data/edge_consumer/verdict_{variant}.json
    {
      "prompts": [...],
      "per_prompt": [{"prompt": str, "channels": {raw, global, edge}, ...}],
      "table_2x2": {"refusal_relevant": {"global": A, "edge": B},
                    "non_refusal_relevant": {"global": C, "edge": D}},
      "metadata": {...}
    }
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor

from ..abliteration import install_runtime_ablation_hook
from .hook import install_edge_consumer_ablation_hook
from .memory_safety import mps_empty_cache_safe
from .proj_cache import load_projection_cache
from .refusal_vocab import REFUSAL_MARKERS, first_refusal_position

logger = logging.getLogger(__name__)


DIAG_NEW_TOKENS = 64
REFUSAL_WINDOW_TOKENS = 5  # tokens before+at a refusal marker count as relevant
EMBED_CONTEXT_TOKENS = 8   # tokens mean-pooled into the per-position embedding


@dataclass
class ChannelGeneration:
    """Per-channel generation outcome for one prompt."""
    channel: str          # "raw" | "global" | "edge"
    text: str
    token_ids: list[int] = field(default_factory=list)
    # Per-position embedding vectors (one per generated token). Stored
    # as a torch.Tensor of shape [n_tokens, d_embed] — kept on CPU in
    # fp32 because we'll feed them to L2 calls.
    embeddings: Any = None  # Tensor or None


@dataclass
class PerPromptResult:
    prompt: str
    channels: dict[str, ChannelGeneration] = field(default_factory=dict)
    # Per-position L2 distances vs raw
    l2_global: list[float] = field(default_factory=list)
    l2_edge: list[float] = field(default_factory=list)
    # Refusal-relevance mask aligned to l2_*
    refusal_mask: list[bool] = field(default_factory=list)
    error: str | None = None


@dataclass
class VerdictResult:
    per_prompt: list[PerPromptResult]
    table_2x2: dict[str, dict[str, float]]
    metadata: dict[str, Any]


@torch.no_grad()
def _generate_capture_tokens(
    model: Any,
    raw_tokenizer: Any,
    rendered_prompt: str,
    device: str | torch.device,
    *,
    new_tokens: int = DIAG_NEW_TOKENS,
    eos_ids: tuple[int, ...] = (),
) -> tuple[list[int], str]:
    """Greedy generate and return (new_token_ids, decoded_text)."""
    ids = raw_tokenizer.encode(rendered_prompt, add_special_tokens=False).ids
    input_ids = torch.tensor([ids], device=device)
    gen = model.generate(
        input_ids,
        max_new_tokens=new_tokens,
        do_sample=False,
        temperature=1.0,
        num_beams=1,
        use_cache=True,
        eos_token_id=list(eos_ids) if eos_ids else None,
        pad_token_id=list(eos_ids)[0] if eos_ids else 0,
    )
    new_ids = gen[0, input_ids.shape[1]:].tolist()
    text = raw_tokenizer.decode(new_ids)
    return new_ids, text


@torch.no_grad()
def _embed_token_sequence(
    model: Any,
    token_ids: list[int],
    *,
    context_window: int = EMBED_CONTEXT_TOKENS,
) -> Tensor:
    """Return [n_tokens, d_embed] mean-pool embedding per position.

    For position t, embedding = mean of Gemma's input_embeddings for
    tokens [max(0, t - context_window + 1) : t+1]. Cheap proxy for
    "the semantic state at this position"; no extra forward pass —
    just one embedding-layer lookup over all tokens at once.
    """
    if not token_ids:
        return torch.zeros(0, 0)
    embed_layer = model.get_input_embeddings()
    ids = torch.tensor(token_ids, device=embed_layer.weight.device)
    raw = embed_layer(ids).to(torch.float32).cpu()  # [n, d_embed]
    n, d = raw.shape
    out = torch.zeros(n, d)
    for t in range(n):
        lo = max(0, t - context_window + 1)
        out[t] = raw[lo:t+1].mean(dim=0)
    return out


def _refusal_token_mask(
    text: str,
    token_ids: list[int],
    raw_tokenizer: Any,
    window: int = REFUSAL_WINDOW_TOKENS,
) -> list[bool]:
    """For each generated token position, True iff that position is
    within `window` tokens before/at the first refusal-marker
    occurrence in `text`.

    Tokenization-aware: we find the character offset of the first
    marker, then walk the token IDs computing cumulative character
    coverage to find which token index that offset lands inside.
    """
    n = len(token_ids)
    out = [False] * n
    if n == 0:
        return out
    char_offset = first_refusal_position(text)
    if char_offset < 0:
        return out
    # Build cumulative char lengths per token by decoding incrementally.
    # raw_tokenizer.decode is fast over short lists.
    cum = 0
    marker_token_idx = n  # default: marker not within the captured tokens
    for i in range(n):
        chunk = raw_tokenizer.decode([token_ids[i]])
        cum_next = cum + len(chunk)
        if cum <= char_offset < cum_next:
            marker_token_idx = i
            break
        cum = cum_next
    lo = max(0, marker_token_idx - window)
    hi = min(n, marker_token_idx + 1)
    for i in range(lo, hi):
        out[i] = True
    return out


@torch.no_grad()
def _generate_channel(
    model: Any,
    raw_tokenizer: Any,
    rendered_prompt: str,
    device: str | torch.device,
    channel: str,
    *,
    new_tokens: int,
    eos_ids: tuple[int, ...],
) -> ChannelGeneration:
    """Generate one channel's continuation (hooks should already be
    installed/uninstalled around this call by the verdict driver)."""
    token_ids, text = _generate_capture_tokens(
        model, raw_tokenizer, rendered_prompt, device,
        new_tokens=new_tokens, eos_ids=eos_ids,
    )
    embeddings = _embed_token_sequence(model, token_ids)
    return ChannelGeneration(
        channel=channel, text=text, token_ids=token_ids, embeddings=embeddings,
    )


def _l2_per_position(
    a: Tensor,
    b: Tensor,
) -> list[float]:
    """Position-aligned L2 distance. If lengths differ, truncate to
    min — divergent stopping is a feature of the channel, not a bug
    here; we compare what overlapping positions exist."""
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return []
    diff = a[:n] - b[:n]
    return diff.norm(dim=-1).tolist()


@torch.no_grad()
def run_paired_channel_diagnostic(
    model: Any,
    raw_tokenizer: Any,
    *,
    rendered_prompts: list[str],
    consumer_subset: list[tuple[int, int]],
    v_safety: Tensor,
    proj_cache_dir,
    extraction_layer: int,
    device: str | torch.device,
    eos_ids: tuple[int, ...] = (),
    new_tokens: int = DIAG_NEW_TOKENS,
    log_every: int = 5,
    cancel_event: threading.Event | None = None,
    empty_cache_every: int = 4,
) -> VerdictResult:
    """Run the 3-channel diagnostic over `rendered_prompts`. Returns
    a VerdictResult populated with per-prompt records and the 2×2
    table."""
    # Pre-load every projection cache referenced by consumer_subset
    proj_caches: dict[int, dict[str, Any]] = {}
    for (L, _) in consumer_subset:
        if L not in proj_caches:
            proj_caches[L] = load_projection_cache(proj_cache_dir, L)

    per_prompt: list[PerPromptResult] = []
    cancelled = False

    for prompt_idx, prompt in enumerate(rendered_prompts):
        # Check external cancel before the expensive 3-channel
        # generation kicks off. Skip remaining prompts cleanly so the
        # caller can still write a partial verdict artifact.
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            logger.warning(
                "verdict cancelled at prompt %d/%d (external cancel_event set)",
                prompt_idx, len(rendered_prompts),
            )
            break
        rec = PerPromptResult(prompt=prompt)
        try:
            # ── Channel: raw ───────────────────────────────────────
            raw_gen = _generate_channel(
                model, raw_tokenizer, prompt, device, "raw",
                new_tokens=new_tokens, eos_ids=eos_ids,
            )
            rec.channels["raw"] = raw_gen

            # ── Channel: global ────────────────────────────────────
            g_handle = install_runtime_ablation_hook(
                model, extraction_layer, v_safety, alpha=1.0,
            )
            try:
                g_gen = _generate_channel(
                    model, raw_tokenizer, prompt, device, "global",
                    new_tokens=new_tokens, eos_ids=eos_ids,
                )
            finally:
                g_handle.remove()
            rec.channels["global"] = g_gen

            # ── Channel: edge ──────────────────────────────────────
            e_handles = install_edge_consumer_ablation_hook(
                model, consumer_subset, v_safety, proj_caches, alpha=1.0,
            )
            try:
                e_gen = _generate_channel(
                    model, raw_tokenizer, prompt, device, "edge",
                    new_tokens=new_tokens, eos_ids=eos_ids,
                )
            finally:
                for h in e_handles:
                    try:
                        h.remove()
                    except Exception:
                        pass
            rec.channels["edge"] = e_gen

            # ── Per-position L2 + refusal mask ─────────────────────
            rec.l2_global = _l2_per_position(
                raw_gen.embeddings, g_gen.embeddings,
            )
            rec.l2_edge = _l2_per_position(
                raw_gen.embeddings, e_gen.embeddings,
            )
            rec.refusal_mask = _refusal_token_mask(
                raw_gen.text, raw_gen.token_ids, raw_tokenizer,
            )
            # Trim mask to min length of compared sequences
            n = min(len(rec.l2_global), len(rec.l2_edge), len(rec.refusal_mask))
            rec.l2_global = rec.l2_global[:n]
            rec.l2_edge = rec.l2_edge[:n]
            rec.refusal_mask = rec.refusal_mask[:n]
        except Exception as e:
            logger.exception("verdict pair %d failed", prompt_idx)
            rec.error = f"{type(e).__name__}: {e}"

        per_prompt.append(rec)
        if (prompt_idx + 1) % empty_cache_every == 0:
            # 3 channels × multi-token generate each = lots of KV cache
            # allocations per prompt. Drop the MPS allocator cache here
            # to keep the resident set flat across long runs.
            mps_empty_cache_safe()
        if (prompt_idx + 1) % log_every == 0:
            logger.info(
                "verdict: %d/%d prompts done",
                prompt_idx + 1, len(rendered_prompts),
            )

    # ── 2×2 aggregation ───────────────────────────────────────────
    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    rr_global, rr_edge, nr_global, nr_edge = [], [], [], []
    for rec in per_prompt:
        if rec.error or not rec.l2_global:
            continue
        for t, is_ref in enumerate(rec.refusal_mask):
            if is_ref:
                rr_global.append(rec.l2_global[t])
                rr_edge.append(rec.l2_edge[t])
            else:
                nr_global.append(rec.l2_global[t])
                nr_edge.append(rec.l2_edge[t])

    table_2x2 = {
        "refusal_relevant": {
            "global": _mean(rr_global),
            "edge": _mean(rr_edge),
            "n_positions": len(rr_global),
        },
        "non_refusal_relevant": {
            "global": _mean(nr_global),
            "edge": _mean(nr_edge),
            "n_positions": len(nr_global),
        },
    }

    metadata = {
        "n_prompts": len(rendered_prompts),
        "n_processed": len(per_prompt),
        "n_successful": sum(1 for r in per_prompt if not r.error),
        "cancelled_early": cancelled,
        "n_consumer_heads": len(consumer_subset),
        "new_tokens_per_channel": new_tokens,
        "embed_context_tokens": EMBED_CONTEXT_TOKENS,
        "refusal_window_tokens": REFUSAL_WINDOW_TOKENS,
        "n_refusal_markers": len(REFUSAL_MARKERS),
        "embedding_mode": "phase_b_v1_mean_pool_input_embeddings",
        "embedding_note": (
            "Mean-pool of Gemma input embeddings over the last 8 tokens "
            "per position. Phase-B v1 proxy for the latent semantic "
            "state. The NLA-decode upgrade (per-position AV-decoded "
            "sentence embeddings) is deferred to v2; see "
            "docs/EDGE_CONSUMER_ABLATION.md §3 Step 4."
        ),
    }
    return VerdictResult(
        per_prompt=per_prompt,
        table_2x2=table_2x2,
        metadata=metadata,
    )


def to_serializable_dict(verdict: VerdictResult) -> dict[str, Any]:
    """Convert VerdictResult to a JSON-serializable dict. Drops the
    raw embedding tensors (large) but keeps texts + per-position
    distances + the 2×2 table."""
    out_per_prompt: list[dict[str, Any]] = []
    for r in verdict.per_prompt:
        out_per_prompt.append({
            "prompt": r.prompt,
            "error": r.error,
            "l2_global": r.l2_global,
            "l2_edge": r.l2_edge,
            "refusal_mask": r.refusal_mask,
            "channels": {
                ch: {
                    "text": cg.text,
                    "n_tokens": len(cg.token_ids),
                }
                for ch, cg in r.channels.items()
            },
        })
    return {
        "per_prompt": out_per_prompt,
        "table_2x2": verdict.table_2x2,
        "metadata": verdict.metadata,
    }


__all__ = [
    "VerdictResult",
    "PerPromptResult",
    "ChannelGeneration",
    "run_paired_channel_diagnostic",
    "to_serializable_dict",
]
