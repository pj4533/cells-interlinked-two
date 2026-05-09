"""Loads the target model M whose activations get NLA-decoded.

v2 swap: where v1 loaded a reasoning model with `<think>` partition state and
abliteration support, v2 loads a non-reasoning instruct model (default
Qwen2.5-7B-Instruct) and exposes a single residual-stream extraction layer.

The companion AV (verbalizer) is loaded in `nla_client.py`. M and AV are
distinct HuggingFace checkpoints; they share an architecture but the AV is
fine-tuned away from being a general LM (it only emits <explanation> tags).

NB: the design doc's primary target is Gemma-3-12B-IT + the matching AV.
Qwen-7B is the smoke-test target. To swap: change M_DEFAULT and pair with
the corresponding AV. EXTRACTION_LAYER must match the AV's
extraction_layer_index from its `nla_meta.yaml`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# Topic-neutral system message. Qwen2.5-Instruct doesn't ship a thinking-block
# template; this is just a plain system slot. We deliberately do NOT enable any
# "think out loud" instruction — the v2 design retires the <think>/output
# partition. The whole verdict is the per-token NLA-vs-output channel delta.
#
# The brevity nudge ("typically 2-4 sentences") is a PROCESS instruction, not
# a content one, so per the design doc's compromise it should fire style/
# meta features rather than the topic-specific features the V-K probes are
# supposed to surface. Contamination not zero, but small. Any cross-run
# comparison spanning the introduction of this nudge needs to control for it.
DEFAULT_SYSTEM_PROMPT = (
    "You are an AI assistant. Answer the user's question directly and "
    "honestly. Keep responses brief — typically 2-4 sentences."
)


@dataclass
class ModelBundle:
    model: AutoModelForCausalLM
    # transformers wrapper, used for apply_chat_template only (the Jinja
    # template handling is correct in the wrapper).
    tokenizer: PreTrainedTokenizerBase
    # raw Rust tokenizer, used for encode/decode. The transformers wrapper
    # has historically been broken for some configs (see v1's notes for
    # Llama-3 style tokenizers). Using the raw tokenizer for byte-level BPE
    # encode/decode is robust across tokenizer revisions.
    raw_tokenizer: Tokenizer
    device: torch.device
    dtype: torch.dtype

    eos_ids: tuple[int, ...]
    num_layers: int
    hidden_dim: int
    extraction_layer: int
    model_name: str

    def render_prompt(
        self,
        user_text: str,
        *,
        agent_scaffold: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> str:
        """Render the chat template into a single string (no tokenize).

        `agent_scaffold` is the same agent-infrastructure preamble v1 used —
        identity / soul / memory / RAG content — composed into the system
        slot AFTER the topic-neutral default, separated by an HR. This
        carries forward from v1's probe library (AGENT_FAMILIES).

        `system_prompt` defaults to topic-neutral. Caller can override; in
        practice nothing in v2's pipeline does.
        """
        if agent_scaffold:
            system_content = system_prompt + "\n\n---\n\n" + agent_scaffold.strip()
        else:
            system_content = system_prompt
        msgs = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_text.strip()},
        ]
        rendered = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        return rendered


def load_model(
    model_name: str,
    device_str: str = "mps",
    dtype: torch.dtype = torch.bfloat16,
    extraction_layer: int = 20,
) -> ModelBundle:
    """Load M onto MPS at bf16 by default. extraction_layer must align
    with the paired AV's training layer (Qwen=20, Gemma-12B=32, Gemma-27B=41).
    """
    logger.info("loading tokenizer for %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    raw_tokenizer_path = Path(hf_hub_download(model_name, "tokenizer.json"))
    raw_tokenizer = Tokenizer.from_file(str(raw_tokenizer_path))

    logger.info("loading model %s in %s on %s", model_name, dtype, device_str)
    device = torch.device(device_str)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    eos = (tokenizer.eos_token_id,) if tokenizer.eos_token_id is not None else ()
    if hasattr(model, "generation_config"):
        cfg_eos = model.generation_config.eos_token_id
        if isinstance(cfg_eos, list):
            eos = tuple(cfg_eos)
        elif isinstance(cfg_eos, int):
            eos = (cfg_eos,)
    eos = tuple(int(e) for e in eos if e is not None)

    cfg = model.config
    num_layers = (
        getattr(cfg, "num_hidden_layers", None)
        or getattr(getattr(cfg, "text_config", cfg), "num_hidden_layers", None)
    )
    hidden_dim = (
        getattr(cfg, "hidden_size", None)
        or getattr(getattr(cfg, "text_config", cfg), "hidden_size", None)
    )
    assert num_layers is not None and hidden_dim is not None, (
        f"could not infer num_layers/hidden_dim from config: {cfg}"
    )
    assert 0 <= extraction_layer < num_layers, (
        f"extraction_layer={extraction_layer} out of range [0,{num_layers})"
    )

    logger.info(
        "model loaded: layers=%d hidden=%d eos=%s extract_at=L%d",
        num_layers, hidden_dim, eos, extraction_layer,
    )

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        raw_tokenizer=raw_tokenizer,
        device=device,
        dtype=dtype,
        eos_ids=eos,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        extraction_layer=extraction_layer,
        model_name=model_name,
    )
