"""Gemma Scope 2 JumpReLU SAE encoder, used as a *secondary instrument*
on the /verdict view.

The design doc explicitly defaults the primary readout to NLA (per-token
sentences from the AV). SAEs come back as a complementary panel: where
NLA tells you what the residual *says*, the SAE tells you which features
the residual *fires*. Combined, they're stronger than either.

We encode the same activation vectors the AV reads — captured at the AV's
training layer (L32 for Gemma-3-12B-IT). The SAE config exposes hook
"model.layers.32.output", which matches the residual stream we hook in
generation_loop.py via SingleLayerHook.

JumpReLU SAE (Gemma Scope architecture):
    pre = (x - b_dec) @ W_enc + b_enc          # [d_sae]
    acts = where(pre > threshold, pre, 0)       # sparse codes

Encoding is light — one matmul + one threshold per position. We run all
captured positions through it after Phase 2 NLA decoding completes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


# Default SAE for Gemma-3-12B-IT residual stream at L32. Matches the AV's
# trained layer so the SAE features come from the same residual the AV reads.
# 16k features, l0_small (lower sparsity → more features fire per token,
# better recall for the secondary panel where we just want top-K).
DEFAULT_SAE_REPO = "google/gemma-scope-2-12b-it"
DEFAULT_SAE_SUBDIR = "resid_post_all/layer_32_width_16k_l0_small"


@dataclass
class SAEConfig:
    repo: str
    subdir: str
    d_sae: int
    layer: int
    architecture: str
    # d_model is derived from w_enc.shape[0] after weights load; not in config.json.
    d_model: int = 0


class JumpReLUSAE:
    """Loaded Gemma Scope 2 JumpReLU SAE. Single layer, single device.

    Lazy-loaded — instantiating the class triggers download + load. Keeps
    weights in fp32 since 16k features is small (~250MB) and the encode
    is rare enough that bf16 quantization isn't worth the precision loss.
    """

    def __init__(
        self,
        repo: str = DEFAULT_SAE_REPO,
        subdir: str = DEFAULT_SAE_SUBDIR,
        device: str = "cpu",
    ) -> None:
        self.cfg = self._load_config(repo, subdir)
        self.device = torch.device(device)

        params_path = hf_hub_download(repo, f"{subdir}/params.safetensors")
        weights = load_file(params_path)
        self.w_enc = weights["w_enc"].to(self.device, torch.float32)         # [d_model, d_sae]
        self.b_enc = weights["b_enc"].to(self.device, torch.float32)         # [d_sae]
        self.w_dec = weights["w_dec"].to(self.device, torch.float32)         # [d_sae, d_model]
        self.b_dec = weights["b_dec"].to(self.device, torch.float32)         # [d_model]
        self.threshold = weights["threshold"].to(self.device, torch.float32) # [d_sae]
        self.cfg.d_model = int(self.w_enc.shape[0])

        logger.info(
            "loaded SAE %s/%s: d_model=%d d_sae=%d on %s",
            repo, subdir, self.cfg.d_model, self.cfg.d_sae, device,
        )

    @staticmethod
    def _load_config(repo: str, subdir: str) -> SAEConfig:
        cfg_path = hf_hub_download(repo, f"{subdir}/config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        # The hook key looks like "model.layers.32.output" — pull layer index.
        hook = cfg.get("hf_hook_point_out") or cfg.get("hf_hook_point_in") or ""
        layer = -1
        for tok in hook.split("."):
            if tok.isdigit():
                layer = int(tok)
                break
        return SAEConfig(
            repo=repo,
            subdir=subdir,
            d_sae=int(cfg["width"]),
            layer=layer,
            architecture=cfg.get("architecture", "jump_relu"),
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single activation vector to sparse SAE features.

        x: [d_model] (any device/dtype). Returns [d_sae] fp32 on `self.device`.
        """
        v = x.detach().to(self.device, torch.float32).reshape(-1)
        # JumpReLU: pre = (x - b_dec) @ W_enc + b_enc; acts = pre * (pre > thr)
        pre = (v - self.b_dec) @ self.w_enc + self.b_enc
        active = pre > self.threshold
        return torch.where(active, pre, torch.zeros_like(pre))

    @torch.no_grad()
    def top_k(self, x: torch.Tensor, k: int = 12) -> tuple[list[int], list[float]]:
        """Encode + return the top-K (feature_id, value) pairs sorted by value desc.
        Skips features that are exactly zero (sparsity from JumpReLU).
        """
        acts = self.encode(x)
        nz_mask = acts > 0
        nz_idx = nz_mask.nonzero(as_tuple=True)[0]
        if nz_idx.numel() == 0:
            return [], []
        values = acts[nz_idx]
        order = values.argsort(descending=True)
        keep = order[: min(k, order.numel())]
        ids = nz_idx[keep].cpu().tolist()
        vals = values[keep].cpu().tolist()
        return ids, vals
