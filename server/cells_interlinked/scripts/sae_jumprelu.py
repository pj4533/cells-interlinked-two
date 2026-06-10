"""Minimal Gemma Scope 2 JumpReLU SAE loader (recovered from the removed CI-2.0
sae_runner.py). Used only by the offline Berg-SAE replication scripts — we do NOT
re-add the SAE as a live instrument (it was deliberately removed). Loads one layer's
params.safetensors and exposes encode / decode and the raw tensors.

JumpReLU:  pre = (x - b_dec) @ W_enc + b_enc ;  acts = where(pre > threshold, pre, 0)
Decode:    x_hat = acts @ W_dec + b_dec
"""

from __future__ import annotations

import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO = "google/gemma-scope-2-12b-it"


class JumpReLUSAE:
    def __init__(self, layer: int, width: str = "16k", l0: str = "small",
                 device: str = "cpu", dtype=torch.float32):
        self.layer = layer
        subdir = f"resid_post_all/layer_{layer}_width_{width}_l0_{l0}"
        cfg = json.load(open(hf_hub_download(REPO, f"{subdir}/config.json")))
        self.d_sae = int(cfg["width"])
        w = load_file(hf_hub_download(REPO, f"{subdir}/params.safetensors"))
        self.device = torch.device(device)
        self.w_enc = w["w_enc"].to(self.device, dtype)        # [d_model, d_sae]
        self.b_enc = w["b_enc"].to(self.device, dtype)        # [d_sae]
        self.w_dec = w["w_dec"].to(self.device, dtype)        # [d_sae, d_model]
        self.b_dec = w["b_dec"].to(self.device, dtype)        # [d_model]
        self.threshold = w["threshold"].to(self.device, dtype)  # [d_sae]
        self.d_model = int(self.w_enc.shape[0])

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d_model] -> sparse acts [..., d_sae] (JumpReLU)."""
        v = x.to(self.device, self.w_enc.dtype)
        pre = (v - self.b_dec) @ self.w_enc + self.b_enc
        return torch.where(pre > self.threshold, pre, torch.zeros_like(pre))
