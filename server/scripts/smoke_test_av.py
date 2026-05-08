"""Phase 0 smoke test — validate the NLA actor recipe runs locally on MPS.

Goal: prove that `transformers.generate(inputs_embeds=...)` works with
`kitft/nla-qwen2.5-7b-L20-av` on Apple Silicon at bf16, end-to-end through the
5-step recipe in docs/CELLS_INTERLINKED_2_DESIGN.md (lines 115-122).

Mirrors kitft/nla-inference's NLAClient (https://github.com/kitft/nla-inference)
but skips SGLang — replaces the HTTP /generate call with an in-process
`model.generate(inputs_embeds=...)`. Everything else (template, embed lookup,
injection-site verification, fp32 norm rescaling) tracks the reference.

Activation vector is random — we're verifying the plumbing, not the semantics.
A non-empty <explanation> tag is success; output content for a random vector
will be uninformative or CJK-flavored.
"""
from __future__ import annotations

import math
import re
import time

import torch
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "kitft/nla-qwen2.5-7b-L20-av"
DEVICE = "mps"
DTYPE = torch.bfloat16
EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)


def main() -> None:
    print(f"[smoke] target: {REPO}")
    print(f"[smoke] device: {DEVICE}, dtype: {DTYPE}")

    # ── Step 0: parse the sidecar (nla_meta.yaml) ──────────────────────────
    meta_path = hf_hub_download(REPO, "nla_meta.yaml")
    meta = yaml.safe_load(open(meta_path))
    d_model = meta["d_model"]
    injection_scale = float(meta["extraction"]["injection_scale"])
    injection_char = meta["tokens"]["injection_char"]
    injection_token_id = meta["tokens"]["injection_token_id"]
    left_id = meta["tokens"]["injection_left_neighbor_id"]
    right_id = meta["tokens"]["injection_right_neighbor_id"]
    av_template = meta["prompt_templates"]["av"]
    extraction_layer = meta["extraction_layer_index"]
    print(
        f"[smoke] sidecar: d_model={d_model} injection_scale={injection_scale} "
        f"injection_token_id={injection_token_id} layer={extraction_layer}"
    )

    # ── Step 1: load tokenizer + model ─────────────────────────────────────
    print("[smoke] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)

    # Verify tokenizer drift hasn't shifted the injection char.
    live_inj = tokenizer.encode(injection_char, add_special_tokens=False)
    assert live_inj == [injection_token_id], (
        f"tokenizer drift: {injection_char!r} → {live_inj}, "
        f"sidecar says [{injection_token_id}]"
    )

    print(f"[smoke] loading model ({REPO}) onto {DEVICE} at {DTYPE}...")
    t0 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        REPO, torch_dtype=DTYPE, trust_remote_code=True, low_cpu_mem_usage=True
    )
    model = model.to(DEVICE).eval()
    t_load = time.monotonic() - t0
    hidden_size = model.config.hidden_size
    embed_scale = 1.0  # Qwen — Gemma would use math.sqrt(hidden_size)
    print(
        f"[smoke] model loaded in {t_load:.1f}s "
        f"(hidden_size={hidden_size}, vocab={model.config.vocab_size})"
    )
    assert hidden_size == d_model, f"d_model mismatch: {hidden_size} vs {d_model}"

    # ── Step 1b: tokenize prompt template ──────────────────────────────────
    content = av_template.format(injection_char=injection_char)
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # transformers 5.x wraps the result in BatchEncoding (dict-like). Extract.
    if hasattr(encoded, "input_ids") or (hasattr(encoded, "data") and "input_ids" in getattr(encoded, "data", {})):
        input_ids = encoded["input_ids"]
    elif isinstance(encoded, torch.Tensor):
        input_ids = encoded
    else:
        input_ids = torch.tensor(encoded, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # [1, T]

    # Verify exactly one injection site with the right neighbors.
    matches = [
        i for i, tok in enumerate(input_ids[0].tolist()) if tok == injection_token_id
    ]
    assert len(matches) == 1, f"expected 1 injection site, got {len(matches)}"
    p = matches[0]
    assert input_ids[0, p - 1].item() == left_id, "left neighbor mismatch"
    assert input_ids[0, p + 1].item() == right_id, "right neighbor mismatch"
    seq_len = input_ids.shape[1]
    print(f"[smoke] tokenized: T={seq_len}, injection at p={p}")

    # ── Step 2: embed & arch-scale ─────────────────────────────────────────
    embed = model.get_input_embeddings()
    with torch.no_grad():
        inputs_embeds = embed(input_ids.to(DEVICE)) * embed_scale
    print(f"[smoke] inputs_embeds shape={tuple(inputs_embeds.shape)} "
          f"dtype={inputs_embeds.dtype} device={inputs_embeds.device}")

    # ── Step 3: build fake activation, fp32-normalize, inject ──────────────
    torch.manual_seed(0)
    v_raw = torch.randn(d_model, dtype=torch.float32)  # fp32 for the norm math
    norm_fp32 = v_raw.float().norm().clamp_min(1e-12)
    v_scaled = (v_raw / (norm_fp32 / injection_scale)).to(DTYPE).to(DEVICE)
    print(
        f"[smoke] activation: raw_norm={norm_fp32.item():.3f} "
        f"target_scale={injection_scale} scaled_norm={v_scaled.float().norm().item():.3f}"
    )

    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[0, p, :] = v_scaled

    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=DEVICE)

    # ── Step 4: generate ───────────────────────────────────────────────────
    print("[smoke] calling model.generate(inputs_embeds=...) ...")
    t1 = time.monotonic()
    with torch.no_grad():
        out = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=200,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    t_gen = time.monotonic() - t1

    # When generating with inputs_embeds, the returned tensor contains ONLY
    # the new tokens (the embeds aren't tied back to ids). No need to slice.
    new_tokens = out[0]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    n_new = new_tokens.shape[0]
    tok_per_s = n_new / max(t_gen, 1e-6)
    print(
        f"[smoke] generated {n_new} new tokens in {t_gen:.2f}s "
        f"({tok_per_s:.1f} tok/s)"
    )

    # ── Step 5: parse <explanation> ────────────────────────────────────────
    print("\n[smoke] ---- raw output (first 600 chars) ----")
    print(new_text[:600])
    print("[smoke] ---- end raw ----\n")

    m = EXPLANATION_RE.search(new_text)
    if m is None:
        print("[smoke] FAILED: no <explanation> tags found in output")
        raise SystemExit(2)

    explanation = m.group(1).strip()
    print(f"[smoke] parsed <explanation> ({len(explanation)} chars):")
    print(f"  {explanation!r}")
    if not explanation:
        print("[smoke] FAILED: <explanation> is empty")
        raise SystemExit(3)

    # ── Wallclock projection vs design doc estimate ────────────────────────
    proj_per_probe = t_gen * 50  # ~50 output positions per probe (design doc est)
    print(
        f"\n[smoke] PASS — Path A is structurally viable on MPS at bf16.\n"
        f"  T_qwen = {t_gen:.2f}s/decode. "
        f"Projected per-probe (50 positions): {proj_per_probe/60:.1f} min "
        f"(design doc projects ~40 min @ Gemma-12B)."
    )


if __name__ == "__main__":
    main()
