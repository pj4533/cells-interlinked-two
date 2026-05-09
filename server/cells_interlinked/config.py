"""Env-driven configuration for v2.

v2 swaps the SAE-based readout for an NLA-based readout, so the SAE-related
settings retire. M and AV are two separate HF checkpoints; their hidden
sizes must match (the AV is a fine-tune of the same architecture as M).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    server_host: str = os.getenv("SERVER_HOST", "127.0.0.1")
    server_port: int = int(os.getenv("SERVER_PORT", "8000"))

    # Default to Qwen2.5-7B-Instruct + the matching kitft AV. This pair
    # was validated end-to-end by the Phase 0 smoke test on this hardware
    # and fits comfortably on 64GB MPS at bf16 (~30GB total). Override
    # via env to swap to Gemma-3-12B-IT + nla-gemma3-12b-L32-av (~48GB
    # total — still fits, just slower per probe).
    model_name: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    av_repo: str = os.getenv("AV_REPO", "kitft/nla-qwen2.5-7b-L20-av")
    extraction_layer: int = int(os.getenv("EXTRACTION_LAYER", "20"))

    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))
    seed: int = int(os.getenv("SEED", "42"))

    # Per-position NLA decode tuning (passed to AV.generate).
    nla_max_new_tokens: int = int(os.getenv("NLA_MAX_NEW_TOKENS", "180"))
    nla_temperature: float = float(os.getenv("NLA_TEMPERATURE", "1.0"))

    db_path: Path = Path(os.getenv("DB_PATH", "./data/probes.sqlite"))
    dtype: str = os.getenv("DTYPE", "bfloat16")
    device: str = os.getenv("DEVICE", "mps")

    # Autorun pacing — gap between probes inside the worker loop.
    autorun_interval_sec: float = float(os.getenv("AUTORUN_INTERVAL_SEC", "5"))

    # SAE secondary instrument — Gemma Scope 2 JumpReLU SAE on the same
    # residual layer the AV reads. Loaded only when M is a Gemma model
    # at the matching layer (currently L32 for Gemma-3-12B-IT). NLA stays
    # the primary readout; SAE features appear as an additional panel.
    sae_enabled: bool = os.getenv("SAE_ENABLED", "1") == "1"
    sae_repo: str = os.getenv("SAE_REPO", "google/gemma-scope-2-12b-it")
    # Default switched from layer_32 (no Neuronpedia coverage) to layer_31:
    # Neuronpedia hosts auto-interp labels only for the four canonical
    # layers (12, 24, 31, 41), not the dense per-layer `_all` variants.
    # L31 is the closest canonical layer to the AV's L32; adjacent layers
    # in a 48-layer model are strongly correlated computational states,
    # so cross-reference is meaningful even though the SAE and AV read
    # different exact layers.
    sae_subdir: str = os.getenv(
        "SAE_SUBDIR", "resid_post/layer_31_width_16k_l0_small"
    )
    # The layer M's residual stream is captured at to feed into the SAE.
    # Should match the layer index in `sae_subdir`.
    sae_extraction_layer: int = int(os.getenv("SAE_EXTRACTION_LAYER", "31"))
    # Encode is fast; CPU is fine and saves MPS memory pressure during
    # generation. Override to "mps" if you want it on-device.
    sae_device: str = os.getenv("SAE_DEVICE", "cpu")
    sae_top_k: int = int(os.getenv("SAE_TOP_K", "12"))

    # Neuronpedia auto-interp label lookups. Keyed by the SAE id pattern
    # Neuronpedia uses on its feature URLs (e.g.
    # "31-gemmascope-2-res-16k").
    neuronpedia_api_base: str = os.getenv(
        "NEURONPEDIA_API_BASE", "https://www.neuronpedia.org/api"
    )
    neuronpedia_model_id: str = os.getenv(
        "NEURONPEDIA_MODEL_ID", "gemma-3-12b-it"
    )
    neuronpedia_sae_id: str = os.getenv(
        "NEURONPEDIA_SAE_ID", "31-gemmascope-2-res-16k"
    )

    # Analyzer for the journal publish flow. Reads ANTHROPIC_API_KEY from env.
    analyzer_model: str = os.getenv("ANALYZER_MODEL", "claude-opus-4-7")


settings = Settings()
