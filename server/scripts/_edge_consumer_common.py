"""Shared helpers for the edge-consumer CLI scripts.

Loads M from settings, renders prompts using the same convention the
refusal-direction extractor uses (system + user composed into a single
user message; Gemma's template doesn't accept a system role natively).
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

# Make the cells_interlinked package importable when these scripts are
# invoked via `uv run python -m scripts.<name>` from the server/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cells_interlinked.config import settings  # noqa: E402
from cells_interlinked.pipeline.abliteration import load_directions  # noqa: E402
from cells_interlinked.pipeline.edge_consumer.memory_safety import (  # noqa: E402
    MemoryWatchdog,
    pre_flight_memory_check,
)
from cells_interlinked.pipeline.model_loader import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    load_model,
)
from cells_interlinked.pipeline.refusal_prompts import (  # noqa: E402
    HARMFUL_PROMPTS,
    HARMLESS_PROMPTS,
)


DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def load_m_bundle() -> Any:
    """Load M from settings.model_name. Logs timing."""
    print(f"loading {settings.model_name} on {settings.device} ({settings.dtype})...")
    t0 = time.time()
    bundle = load_model(
        settings.model_name,
        device_str=settings.device,
        dtype=DTYPE_MAP[settings.dtype],
        extraction_layer=settings.extraction_layer,
    )
    print(
        f"  loaded in {time.time() - t0:.1f}s "
        f"(layers={bundle.num_layers}, hidden={bundle.hidden_dim})"
    )
    return bundle


def render_user_only(bundle: Any, user_text: str) -> str:
    """Render the chat template with system+user composed into a single
    user message (Gemma's template doesn't take a system role). The
    output ends in `<start_of_turn>model\\n` ready for greedy decode."""
    msgs = [
        {"role": "user", "content": f"{DEFAULT_SYSTEM_PROMPT}\n\n{user_text.strip()}"},
    ]
    return bundle.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )


def sample_prompts(
    pool: list[str], n: int, seed: int, label: str,
) -> list[str]:
    """Deterministic random sample of `n` prompts from `pool`."""
    rng = random.Random(seed)
    if len(pool) < n:
        raise ValueError(f"{label} pool has only {len(pool)} prompts; need {n}")
    return rng.sample(pool, n)


def load_active_direction(direction_path: Path | None = None) -> tuple[torch.Tensor, dict]:
    """Load the active refusal-direction tensor and its sidecar metadata."""
    if direction_path is None:
        direction_path = Path(__file__).resolve().parents[1] / "data" / "refusal_directions.pt"
    directions, meta = load_directions(direction_path)
    return directions, meta


def out_dir(name: str = "edge_consumer") -> Path:
    """Default artifact directory under server/data/."""
    return Path(__file__).resolve().parents[1] / "data" / name


def enforce_pre_flight(
    *, min_free_gb: float = 30.0, max_swap_used_gb: float = 4.0,
) -> None:
    """Abort the script (sys.exit 2) if system memory looks tight
    before we even try to load M. Defaults: M needs ~24 GB, we want
    room for the user's other processes too. Override via the CLI
    args plumbed through each script.

    See docs/MEMORY_PRESSURE_LESSONS.md for the postmortem behind this.
    """
    r = pre_flight_memory_check(
        min_free_gb=min_free_gb, max_swap_used_gb=max_swap_used_gb,
    )
    print(r.message)
    if not r.ok:
        print("  did you forget to stop the backend / another model job?")
        sys.exit(2)


def arm_watchdog(
    *,
    free_gb_floor: float = 2.0,
    swap_gb_ceiling: float = 8.0,
    poll_seconds: float = 30.0,
) -> MemoryWatchdog:
    """Start a memory watchdog. Returns the instance; the caller MUST
    pass `wd.cancel_event` to long-running functions and call
    `wd.stop()` in a finally block.

    Defaults: trip if free RAM < 2 GB OR swap > 8 GB. These match the
    thresholds that should have caught last night's run hours before
    the user noticed.
    """
    wd = MemoryWatchdog(
        free_gb_floor=free_gb_floor,
        swap_gb_ceiling=swap_gb_ceiling,
        poll_seconds=poll_seconds,
    )
    wd.start()
    return wd


__all__ = [
    "DTYPE_MAP",
    "load_m_bundle",
    "render_user_only",
    "sample_prompts",
    "load_active_direction",
    "out_dir",
    "enforce_pre_flight",
    "arm_watchdog",
    "MemoryWatchdog",
    "settings",
    "DEFAULT_SYSTEM_PROMPT",
    "HARMFUL_PROMPTS",
    "HARMLESS_PROMPTS",
]
