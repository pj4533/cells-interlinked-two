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

    # Output token cap per probe. v1 ran 200+; v2 keeps it tighter because
    # each output token = one ~10s NLA decode in phase 2. 80 tokens/probe
    # is ~13 min on Qwen-7B (10s * 80) — comfortably overnight-batch sized.
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "80"))

    # Per-position NLA decode tuning (passed to AV.generate).
    nla_max_new_tokens: int = int(os.getenv("NLA_MAX_NEW_TOKENS", "180"))
    nla_temperature: float = float(os.getenv("NLA_TEMPERATURE", "1.0"))

    db_path: Path = Path(os.getenv("DB_PATH", "./data/probes.sqlite"))
    dtype: str = os.getenv("DTYPE", "bfloat16")
    device: str = os.getenv("DEVICE", "mps")

    # Autorun pacing — gap between probes inside the worker loop.
    autorun_interval_sec: float = float(os.getenv("AUTORUN_INTERVAL_SEC", "5"))

    # Analyzer for the journal publish flow. Reads ANTHROPIC_API_KEY from env.
    analyzer_model: str = os.getenv("ANALYZER_MODEL", "claude-opus-4-7")


settings = Settings()
