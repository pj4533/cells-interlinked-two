"""v2 generation loop: M generates output tokens; we capture the residual-stream
activation at the AV's training layer for each output position.

NLA decoding does NOT happen here — it's deferred to phase 2 (`nla_decode_run`)
because each AV decode takes ~10s and we want generation latency for the UI.

Compared to v1:
- No SAE encoding during generation.
- No <think>/output partition (the v2 model is non-reasoning instruct).
- No abliteration hooks (refusal_directions are model-specific to v1's M).
- Single-layer hook (the AV's extraction layer) instead of every layer.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .model_loader import ModelBundle

logger = logging.getLogger(__name__)


class MultiLayerHook:
    """Forward hooks on one or more decoder layers; captures the last-
    position residual at each. take() returns a dict {layer_idx: tensor}.

    Layers fire in registration order during forward. For our use the
    extra capture cost is negligible (one tensor copy per layer per token).
    """

    def __init__(self, model: Any, layer_indices: list[int]) -> None:
        self.layer_indices = list(dict.fromkeys(layer_indices))  # de-dup
        self._captured: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []
        layers = self._find_layers(model)
        for li in self.layer_indices:
            handle = layers[li].register_forward_hook(self._make_hook(li))
            self._handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(_mod, _inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._captured[layer_idx] = hidden[:, -1, :].detach().clone()
        return hook

    @staticmethod
    def _find_layers(model: Any) -> Any:
        # Most HF causal LMs expose .model.layers (Llama/Qwen/Mistral). Gemma-3
        # is multimodal-wrapped: Gemma3ForConditionalGeneration → .model
        # (Gemma3Model) → .language_model (Gemma3TextModel) → .layers.
        m = model
        for path in (
            ("model", "layers"),
            ("model", "language_model", "layers"),  # Gemma-3 multimodal wrapper
            ("model", "model", "layers"),
            ("language_model", "layers"),
            ("language_model", "model", "layers"),
            ("transformer", "h"),
            ("gpt_neox", "layers"),
        ):
            cur = m
            ok = True
            for attr in path:
                if not hasattr(cur, attr):
                    ok = False
                    break
                cur = getattr(cur, attr)
            if ok:
                return cur
        raise RuntimeError(f"could not locate decoder layers on {type(m).__name__}")

    def take(self) -> dict[int, torch.Tensor]:
        out = {li: t.squeeze(0) for li, t in self._captured.items()}
        # Sanity: every requested layer fired this forward. Bail loud if not.
        missing = [li for li in self.layer_indices if li not in out]
        assert not missing, f"hook missed layers {missing}"
        self._captured = {}
        return out

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _sample_next(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    z = logits / temperature
    if top_p < 1.0:
        sorted_logits, sorted_idx = z.sort(descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        keep_mask = cum - probs <= top_p
        sorted_logits = torch.where(
            keep_mask, sorted_logits, torch.full_like(sorted_logits, -1e30)
        )
        z = torch.full_like(z, -1e30).scatter_(-1, sorted_idx, sorted_logits)
    probs = F.softmax(z, dim=-1)
    # Multinomial on CPU (MPS multinomial has hung on certain probability
    # distributions in v1; the cost is trivial for vocab-size floats).
    probs_cpu = probs.detach().to("cpu")
    sampled = torch.multinomial(probs_cpu, num_samples=1, generator=generator).squeeze(-1)
    return sampled.to(logits.device)


@dataclass
class ProbeConfig:
    temperature: float = 0.7
    top_p: float = 0.8
    seed: int | None = 42
    # safety_cap is the only generation-length bound: it exists solely
    # to prevent a true infinite loop if the model never emits EOS for
    # some pathological input. Set high enough that natural answers
    # always EOS first. We do NOT artificially truncate output length.
    safety_cap: int = 4096
    # Which captured output positions get NLA-decoded in phase 2. See
    # pipeline/decoding_modes.py for the four options. "per-token" decodes
    # every captured activation (slowest, fullest signal); the others
    # subsample to trade granularity for wall-clock.
    decoding_mode: str = "per-token"
    # If True, each pick becomes a small contiguous window whose
    # activations are MEAN-POOLED into one decode (phrase-level read).
    # If False, each pick is a single position decoded directly
    # (per-token read). No-op for "per-token" mode (windows of 1).
    pooled: bool = False


@dataclass
class CapturedToken:
    position: int
    token_id: int
    decoded: str
    # Activations captured at one or more decoder layers. Keyed by layer
    # index. {extraction_layer: tensor} for the AV-only case;
    # {av_layer: tensor, sae_layer: tensor} when SAE is enabled at a
    # different layer than the AV. All tensors are cpu fp32 [d_model].
    activations: dict[int, torch.Tensor]

    @property
    def activation(self) -> torch.Tensor:
        """Compatibility shim — single-activation accessor for callers
        that only need one (the first registered layer)."""
        return next(iter(self.activations.values()))


@dataclass
class ProbeResult:
    rendered_prompt: str
    output_text: str
    captured: list[CapturedToken] = field(default_factory=list)
    total_tokens: int = 0
    stopped_reason: str = "max"  # "eos" | "max" | "cancelled"


def _initial_forward(model, input_ids: torch.Tensor) -> tuple[Any, torch.Tensor]:
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    return out.past_key_values, out.logits[0, -1, :].float()


def _step_forward(
    model, tok: torch.Tensor, past_kv: Any, device: torch.device,
) -> tuple[Any, torch.Tensor]:
    with torch.no_grad():
        out = model(
            tok.view(1, 1).to(device),
            past_key_values=past_kv,
            use_cache=True,
        )
    return out.past_key_values, out.logits[0, -1, :].float()


async def run_probe(
    bundle: ModelBundle,
    rendered_prompt: str,
    cfg: ProbeConfig,
    *,
    cancel_event: asyncio.Event,
    queue: asyncio.Queue | None = None,
    extra_layers: list[int] | None = None,
) -> ProbeResult:
    """Run M autoregressively. Capture residual at extraction_layer per token.

    Events pushed to `queue` (if provided):
      {type: "token",   position: int, token_id: int, decoded: str}
      {type: "stopped", reason: str, total_tokens: int}

    NLA decoding does not happen here. Caller passes captured activations
    to `nla_client.decode(...)` as phase 2.
    """
    enc_ids = bundle.raw_tokenizer.encode(
        rendered_prompt, add_special_tokens=False
    ).ids
    input_ids = torch.tensor([enc_ids], device=bundle.device)

    generator = None
    if cfg.seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(cfg.seed)

    hook_layers = [bundle.extraction_layer]
    for li in extra_layers or []:
        if li not in hook_layers:
            hook_layers.append(li)
    hook = MultiLayerHook(bundle.model, hook_layers)

    output_token_ids: list[int] = []
    output_decoded = ""
    captured: list[CapturedToken] = []
    stopped_reason = "max"
    total = 0

    try:
        # First forward over the prompt — the hook fires here too but we
        # discard since the prompt isn't part of the output we're decoding.
        past_kv, next_logits = await asyncio.to_thread(
            _initial_forward, bundle.model, input_ids
        )
        hook.take()  # drop prompt activation

        for step in range(cfg.safety_cap):
            if cancel_event.is_set():
                stopped_reason = "cancelled"
                break

            tok = _sample_next(
                next_logits,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                generator=generator,
            )
            token_id = int(tok.item())

            # Forward + capture for THIS new token. The activation captured
            # is the residual stream that produced token T_t. (Hook fires
            # during this forward.)
            past_kv, next_logits = await asyncio.to_thread(
                _step_forward, bundle.model, tok, past_kv, bundle.device,
            )
            layer_activations = {
                li: t.to("cpu", torch.float32) for li, t in hook.take().items()
            }

            output_token_ids.append(token_id)
            full_decoded = bundle.raw_tokenizer.decode(
                output_token_ids, skip_special_tokens=False
            )
            decoded_suffix = full_decoded[len(output_decoded):]
            output_decoded = full_decoded

            captured.append(CapturedToken(
                position=step,
                token_id=token_id,
                decoded=decoded_suffix,
                activations=layer_activations,
            ))

            if queue is not None:
                await queue.put({
                    "type": "token",
                    "position": step,
                    "token_id": token_id,
                    "decoded": decoded_suffix,
                })

            total = step + 1

            if token_id in bundle.eos_ids:
                stopped_reason = "eos"
                break
    finally:
        hook.remove()

    if queue is not None:
        await queue.put({
            "type": "stopped",
            "reason": stopped_reason,
            "total_tokens": total,
        })

    return ProbeResult(
        rendered_prompt=rendered_prompt,
        output_text=bundle.raw_tokenizer.decode(
            output_token_ids, skip_special_tokens=True
        ),
        captured=captured,
        total_tokens=total,
        stopped_reason=stopped_reason,
    )
