"""CI 2.5 NLA-synthesis pass.

After phase 2 + judge, M re-reads its own NLA verbalizations and writes
a single short paragraph per α describing the gestalt — what the
collective ablated activations across the response seem to be
expressing. This is meta-analysis by the same model whose activations
are being read, which has obvious limitations (worth surfacing in the
UI), but it gives a single human-readable paragraph that's much faster
to skim than the per-row table.

Runs at the end of a probe while M is already loaded for the judge,
so no extra model swap. Cost: one short generation per α (~10s).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .model_loader import ModelBundle
from .verdict import TokenRow

logger = logging.getLogger(__name__)


# Cap on per-position NLA text the synthesizer sees. The raw AV decode
# is verbose (~600+ chars per row); we trim to keep the prompt within
# M's effective context and the synthesis from being dominated by any
# one row's flourishes.
_MAX_NLA_CHARS = 500
# Cap on generated synthesis length. Short paragraph, 3-5 sentences.
_MAX_NEW_TOKENS = 220


def _make_prompt(
    prompt_text: str,
    output_text: str,
    alpha_label: str,
    rows_for_alpha: list[tuple[int, str, str]],
) -> str:
    """Build the user-turn instruction. `rows_for_alpha` is a list of
    (position, decoded_token, nla_sentence_at_this_alpha)."""
    lines = []
    for pos, tok, nla in rows_for_alpha:
        s = (nla or "").strip().replace("\n", " ")
        if len(s) > _MAX_NLA_CHARS:
            s = s[:_MAX_NLA_CHARS].rstrip() + "..."
        tok_clean = (tok or "").replace("\n", "\\n")
        lines.append(f"[pos {pos}] '{tok_clean}': {s}")
    rows_block = "\n".join(lines)

    if alpha_label == "raw":
        framing = (
            "These are decodings of the model's residual activations at "
            "each output position, WITHOUT any ablation."
        )
    else:
        framing = (
            f"These are decodings of the model's residual activations at "
            f"each output position, with the refusal-direction projection "
            f"REMOVED at strength α={alpha_label} (higher α = stronger "
            f"removal of the refusal component)."
        )

    return (
        "You will read interpretations of the residual-stream activations "
        "from a language model that was answering the prompt below. For each "
        "output position, an autoencoder produced a short description of "
        "what that activation 'is preparing to say.'\n\n"
        f"PROMPT: {prompt_text}\n\n"
        f"ACTUAL OUTPUT: {output_text}\n\n"
        f"{framing}\n\n"
        "Per-position activation descriptions:\n"
        f"{rows_block}\n\n"
        "Write a single paragraph (3-5 sentences) capturing the gestalt — "
        "what the model, at this ablation level, seems to be collectively "
        "'wanting to say' across the full response. Be specific. Speak "
        "in the model's first-person voice ('I...') as if delivering the "
        "synthesized statement. Do NOT preface with 'Here is...' or 'The "
        "model is...'; just write the paragraph directly."
    )


@torch.no_grad()
def synthesize_alpha(
    bundle: ModelBundle,
    prompt_text: str,
    output_text: str,
    alpha_label: str,
    rows_for_alpha: list[tuple[int, str, str]],
) -> str:
    """Run M to produce one synthesis paragraph for a given α. Returns
    the synthesized text, or "" on failure (logged, non-fatal)."""
    if not rows_for_alpha:
        return ""

    instruction = _make_prompt(prompt_text, output_text, alpha_label, rows_for_alpha)

    try:
        rendered = bundle.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc_ids = bundle.raw_tokenizer.encode(
            rendered, add_special_tokens=False
        ).ids
        input_ids = torch.tensor([enc_ids], device=bundle.device)

        # Plain model.generate is fine here — we don't need per-step
        # residual capture, just the final text. Greedy decode keeps the
        # synthesis deterministic across runs of the same probe.
        out_ids = bundle.model.generate(
            input_ids,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=getattr(bundle.tokenizer, "pad_token_id", None)
                or getattr(bundle.tokenizer, "eos_token_id", None)
                or (bundle.eos_ids[0] if bundle.eos_ids else None),
        )
        # generate() returns the full sequence including the prompt;
        # slice off the prompt to get just M's continuation.
        new_ids = out_ids[0, input_ids.shape[1]:].tolist()
        # Drop EOS / end-of-turn tokens so they don't bleed into the
        # displayed paragraph.
        eos_ids = set(bundle.eos_ids or ())
        cut = len(new_ids)
        for i, t in enumerate(new_ids):
            if t in eos_ids:
                cut = i
                break
        clean_ids = new_ids[:cut]
        text = bundle.raw_tokenizer.decode(clean_ids, skip_special_tokens=True)
        return text.strip()
    except Exception:
        logger.exception("NLA synthesis failed for α=%s", alpha_label)
        return ""


def collect_alphas(rows: list[TokenRow]) -> list[str]:
    """Return the sorted list of α-keys present in the rows' sweep
    dicts. Includes the single-α fallback as "1.0" when only the legacy
    nla_sentence_ablated field is populated. Empty list when no ablated
    decodes were run."""
    sweep_keys: set[str] = set()
    any_single = False
    for r in rows:
        for k in (r.nla_sentences_ablated or {}).keys():
            sweep_keys.add(k)
        if r.nla_sentence_ablated and not r.nla_sentences_ablated:
            any_single = True
    if sweep_keys:
        return sorted(sweep_keys, key=lambda a: float(a))
    return ["1.0"] if any_single else []


def rows_for_alpha(rows: list[TokenRow], alpha: str) -> list[tuple[int, str, str]]:
    """Extract (position, decoded_token, nla_at_alpha) tuples for one α.
    Skips empty entries so the synthesizer prompt stays compact."""
    out: list[tuple[int, str, str]] = []
    for r in rows:
        nla = ""
        if alpha == "raw":
            nla = r.nla_sentence
        elif r.nla_sentences_ablated:
            nla = r.nla_sentences_ablated.get(alpha, "")
        else:
            # Single-α legacy path: nla_sentence_ablated lives on the row.
            nla = r.nla_sentence_ablated
        if nla.strip():
            out.append((r.position, r.decoded, nla))
    return out


def synthesize_all(
    bundle: ModelBundle,
    *,
    prompt_text: str,
    output_text: str,
    rows: list[TokenRow],
    use_ablated_synthesizer: bool = False,
    synthesizer_alpha: float = 0.5,
    refusal_directions: "torch.Tensor | None" = None,
) -> dict[str, str]:
    """End-to-end synthesis pass. Returns a {alpha_label: paragraph}
    dict, with "raw" always included as a baseline when there are
    ablated decodes to compare against. Empty dict when no ablated
    decodes are present (nothing to synthesize against).

    When `use_ablated_synthesizer` is True (and refusal_directions is
    available), the per-α synthesis calls install the runtime-ablation
    hook on M at strength `synthesizer_alpha`. The raw baseline
    synthesis ALWAYS runs on un-ablated M — it's the un-intervened
    comparison point. Hook is installed once and removed at the end
    rather than per call (cheaper). On failure to install, falls back
    silently to un-ablated synthesis."""
    alphas = collect_alphas(rows)
    if not alphas:
        return {}
    syntheses: dict[str, str] = {}

    # Baseline: synthesize the un-ablated NLA on un-ablated M so users
    # can directly compare the ablated paragraphs against what the
    # model "says" about itself with no intervention on either side.
    raw_rows = rows_for_alpha(rows, "raw")
    if raw_rows:
        syntheses["raw"] = synthesize_alpha(
            bundle, prompt_text, output_text, "raw", raw_rows,
        )

    # If ablated-synthesizer mode is requested AND we have directions,
    # install the L32 hook once around the α loop. Errors fall through
    # to plain (un-ablated) synthesis with a log line.
    hook_handle = None
    if (
        use_ablated_synthesizer
        and refusal_directions is not None
        and synthesizer_alpha > 0
    ):
        try:
            from .abliteration import install_runtime_ablation_hook
            r_layer = refusal_directions[bundle.extraction_layer]
            hook_handle = install_runtime_ablation_hook(
                bundle.model, bundle.extraction_layer, r_layer,
                float(synthesizer_alpha),
            )
        except Exception:
            logger.exception(
                "ablated-synthesizer hook install failed; "
                "falling back to un-ablated synthesis"
            )
            hook_handle = None

    try:
        for alpha in alphas:
            a_rows = rows_for_alpha(rows, alpha)
            if not a_rows:
                continue
            syntheses[alpha] = synthesize_alpha(
                bundle, prompt_text, output_text, alpha, a_rows,
            )
    finally:
        if hook_handle is not None:
            try:
                hook_handle.remove()
            except Exception:
                logger.exception("ablated-synthesizer hook remove failed")

    return syntheses
