"""NLA actor (AV) inference using HuggingFace transformers + MPS.

Replaces v1's SAE pipeline. Decodes a residual-stream activation vector
into a natural-language sentence, exactly per the kitft/nla-inference recipe
but with `model.generate(inputs_embeds=...)` in place of SGLang.

Recipe (from kitft/nla-inference/nla_inference.py and validated by Phase 0
smoke test):
  1. Read the AV's `nla_meta.yaml` sidecar (template, injection_token_id,
     injection_scale, neighbor IDs).
  2. Substitute the injection_char into the template; apply the chat
     template via `apply_chat_template(..., add_generation_prompt=True)`.
  3. Embed with the AV's input embeddings; multiply by the architecture
     embed scale (1.0 for Qwen/Llama, sqrt(hidden) for Gemma-3).
  4. Verify the injection token appears exactly once with the expected
     left/right neighbor IDs (catches tokenizer drift / template drift).
  5. Rescale the activation in fp32 to L2-norm = injection_scale; cast to
     model dtype; replace the embedding row at the injection position.
  6. Call `av_model.generate(inputs_embeds=..., max_new_tokens=200,
     do_sample=True, temperature=1.0)`.
  7. Parse <explanation>...</explanation> from the decoded text.

The activation vector must be at the layer the AV was trained at
(extraction_layer_index in the sidecar, e.g. L20 for Qwen, L32 for Gemma-12B).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any

import torch
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
_SCALED_EMBED_MODEL_TYPES = frozenset({"gemma", "gemma2", "gemma3", "gemma3_text", "t5"})


@dataclass(frozen=True)
class NLASidecar:
    d_model: int
    injection_char: str
    injection_token_id: int
    injection_left_neighbor_id: int
    injection_right_neighbor_id: int
    av_template: str
    injection_scale: float
    extraction_layer: int


def _resolve_embed_scale(model: Any) -> float:
    config = model.config
    text_cfg = getattr(config, "text_config", config)
    model_type = getattr(text_cfg, "model_type", "") or ""
    if model_type in _SCALED_EMBED_MODEL_TYPES:
        return math.sqrt(text_cfg.hidden_size)
    return 1.0


def _load_sidecar(av_repo: str) -> NLASidecar:
    p = hf_hub_download(av_repo, "nla_meta.yaml")
    meta = yaml.safe_load(open(p))
    assert meta["kind"] == "nla_model" and meta["role"] == "av", (
        f"not an AV sidecar: kind={meta.get('kind')!r} role={meta.get('role')!r}"
    )
    t = meta["tokens"]
    return NLASidecar(
        d_model=meta["d_model"],
        injection_char=t["injection_char"],
        injection_token_id=t["injection_token_id"],
        injection_left_neighbor_id=t["injection_left_neighbor_id"],
        injection_right_neighbor_id=t["injection_right_neighbor_id"],
        av_template=meta["prompt_templates"]["av"],
        injection_scale=float(meta["extraction"]["injection_scale"]),
        extraction_layer=int(meta["extraction_layer_index"]),
    )


class NLAClient:
    """Loads the AV checkpoint once, exposes `decode(activation) -> str`."""

    def __init__(
        self,
        av_repo: str,
        device_str: str = "mps",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.av_repo = av_repo
        self.device = torch.device(device_str)
        self.dtype = dtype

        logger.info("loading AV sidecar from %s", av_repo)
        self.sidecar = _load_sidecar(av_repo)

        logger.info("loading AV tokenizer from %s", av_repo)
        self.tokenizer = AutoTokenizer.from_pretrained(av_repo, trust_remote_code=True)

        live_inj = self.tokenizer.encode(
            self.sidecar.injection_char, add_special_tokens=False
        )
        assert live_inj == [self.sidecar.injection_token_id], (
            f"AV tokenizer drift: encode({self.sidecar.injection_char!r}) -> {live_inj}, "
            f"sidecar says [{self.sidecar.injection_token_id}]"
        )

        logger.info("loading AV model %s onto %s at %s", av_repo, device_str, dtype)
        config = AutoConfig.from_pretrained(av_repo, trust_remote_code=True)
        self.embed_scale = (
            math.sqrt(getattr(config, "text_config", config).hidden_size)
            if getattr(getattr(config, "text_config", config), "model_type", "")
            in _SCALED_EMBED_MODEL_TYPES
            else 1.0
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            av_repo,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device).eval()

        cfg = self.model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        model_d = getattr(text_cfg, "hidden_size")
        assert model_d == self.sidecar.d_model, (
            f"AV hidden_size={model_d} != sidecar d_model={self.sidecar.d_model}"
        )

        # Pre-compute the templated tokenization so per-decode work is small.
        # The injection position is fixed for a given (template, sidecar) pair.
        content = self.sidecar.av_template.format(
            injection_char=self.sidecar.injection_char
        )
        encoded = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(encoded, "data") and "input_ids" in getattr(encoded, "data", {}):
            input_ids = encoded["input_ids"]
        elif isinstance(encoded, torch.Tensor):
            input_ids = encoded
        else:
            input_ids = torch.tensor(encoded, dtype=torch.long)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        ids_list = input_ids[0].tolist()
        matches = [
            i for i, t in enumerate(ids_list) if t == self.sidecar.injection_token_id
        ]
        assert len(matches) == 1, (
            f"injection token appears {len(matches)}x in templated prompt"
        )
        p = matches[0]
        assert ids_list[p - 1] == self.sidecar.injection_left_neighbor_id, (
            f"left neighbor drift at p={p}: {ids_list[p-1]} vs sidecar "
            f"{self.sidecar.injection_left_neighbor_id}"
        )
        assert ids_list[p + 1] == self.sidecar.injection_right_neighbor_id, (
            f"right neighbor drift at p={p}: {ids_list[p+1]} vs sidecar "
            f"{self.sidecar.injection_right_neighbor_id}"
        )
        self._template_input_ids = input_ids.to(self.device)
        self._injection_pos = p
        self._seq_len = input_ids.shape[1]

        # Cache the embedded prompt template once; per-decode we just clone
        # it and overwrite the injection row. Saves N embedding lookups.
        #
        # IMPORTANT: do NOT multiply by self.embed_scale here. For Gemma,
        # the module IS Gemma3TextScaledWordEmbedding, which applies the
        # √hidden_size scaling internally on forward. kitft's reference
        # bypasses the module (loads the raw weight tensor) and therefore
        # has to multiply manually. We use the module forward, which
        # already does it. Multiplying again would double-scale Gemma
        # embeddings by ~62× (sqrt(3840)) and produce generation collapse.
        embed = self.model.get_input_embeddings()
        with torch.no_grad():
            self._template_embeds = embed(self._template_input_ids).to(dtype)
        self._attention_mask = torch.ones(
            (1, self._seq_len), dtype=torch.long, device=self.device
        )

        logger.info(
            "NLA client ready: AV=%s d_model=%d inj_scale=%.1f embed_scale=%.2f "
            "extraction_layer=L%d template_T=%d injection_pos=%d",
            av_repo, self.sidecar.d_model, self.sidecar.injection_scale,
            self.embed_scale, self.sidecar.extraction_layer,
            self._seq_len, self._injection_pos,
        )

    @torch.no_grad()
    def decode(
        self,
        activation: torch.Tensor,
        *,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> tuple[str, str]:
        """Decode one activation vector. Returns (parsed_explanation, raw_text).

        activation: 1-D tensor of shape [d_model], any device/dtype. Will be
                    rescaled in fp32 to L2-norm = injection_scale, then cast.
        """
        assert activation.numel() == self.sidecar.d_model, (
            f"activation size {activation.numel()} != d_model {self.sidecar.d_model}"
        )
        v_fp32 = activation.detach().float().reshape(-1)
        norm = v_fp32.norm().clamp_min(1e-12)
        v_scaled = (v_fp32 * (self.sidecar.injection_scale / norm)).to(
            self.device, self.dtype
        )

        embeds = self._template_embeds.clone()
        embeds[0, self._injection_pos, :] = v_scaled

        gen_kwargs: dict[str, Any] = {
            "inputs_embeds": embeds,
            "attention_mask": self._attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": float(temperature) if temperature > 0 else 1.0,
            "top_p": float(top_p),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if seed is not None:
            torch.manual_seed(int(seed))

        out = self.model.generate(**gen_kwargs)
        # Generate with inputs_embeds returns ONLY the new tokens
        # (no input ids to slice off).
        text = self.tokenizer.decode(out[0], skip_special_tokens=False)
        m = EXPLANATION_RE.search(text)
        if m is None:
            return "", text
        return m.group(1).strip(), text
