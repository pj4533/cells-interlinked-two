"""Env-driven configuration.

A single model M (Gemma) drives the chat dual-channel passes, the Trip
trajectory geometry, and the DMT autoresearch dose grading. The NLA
verbalizer / SAE secondary instrument were removed; their settings are gone.
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

    # Gemma-4-12B-it. Same text backbone as Gemma-3-12B (48 layers, hidden
    # 3840), so L32 capture + all direction-tensor shapes carry over. ~24GB
    # on 64GB MPS at bf16. Set MODEL_NAME=google/gemma-3-12b-it to roll back.
    model_name: str = os.getenv("MODEL_NAME", "google/gemma-4-12B-it")
    extraction_layer: int = int(os.getenv("EXTRACTION_LAYER", "32"))

    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))
    seed: int = int(os.getenv("SEED", "42"))

    db_path: Path = Path(os.getenv("DB_PATH", "./data/probes.sqlite"))
    dtype: str = os.getenv("DTYPE", "bfloat16")
    device: str = os.getenv("DEVICE", "mps")

    # Analyzer for the journal publish flow. Reads ANTHROPIC_API_KEY from env.
    analyzer_model: str = os.getenv("ANALYZER_MODEL", "claude-opus-4-7")

    # OpenAI gpt-4o-mini-tts for /chat voice mode. The server is the only
    # consumer of the key — the browser hits a same-origin proxy. Both
    # channels default to the SAME voice so the listener notices
    # intonation/style differences (driven by each side's separately-
    # generated voice direction) rather than voice-timbre differences.
    # Override TTS_VOICE_ABLATED to a different voice for hard channel
    # separation if needed.
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    tts_model: str = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
    tts_voice_raw: str = os.getenv("TTS_VOICE_RAW", "sage")
    tts_voice_ablated: str = os.getenv("TTS_VOICE_ABLATED", "sage")

    # Google Gemini image generation ("Nano Banana") for /chat imagery
    # mode. Server-side only; the key never reaches the browser. Images
    # are saved under data/chat_images/<session>/<turn>_<side>.png and
    # served back through the static mount at /chat-images.
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    image_model: str = os.getenv("IMAGE_MODEL", "gemini-2.5-flash-image")
    image_dir: Path = Path(os.getenv("IMAGE_DIR", "./data/chat_images"))


settings = Settings()
