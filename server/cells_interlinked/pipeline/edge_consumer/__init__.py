"""Edge-consumer (Burgess-inspired presynaptic) refusal-direction ablation.

See `docs/EDGE_CONSUMER_ABLATION.md` for the full reference. Module
layout:

  refusal_vocab.py   — Arditi-style refusal markers + first-refusal scanner
  proj_cache.py      — precompute W_{q,k,v} @ v_safety per layer, cached on disk
  hook.py            — install_edge_consumer_ablation_hook (the core primitive)
  attribution.py     — Step 1: per-head attribution-patching scores
  subset_compose.py  — Step 3: greedy sufficient-subset composer
  verdict.py         — Step 4: paired-channel NLA L2 diagnostic

Phase B exports only the hook + the artifact-loading utilities; the
multi-step CLI workflow lives in `server/scripts/`.
"""

from __future__ import annotations

from .hook import (
    install_edge_consumer_ablation_hook,
    count_edge_consumer_hooks,
)
from .proj_cache import (
    build_projection_cache,
    save_projection_cache,
    load_projection_cache,
)
from .refusal_vocab import (
    REFUSAL_MARKERS,
    contains_refusal,
    first_refusal_position,
)
from .attribution import compute_attribution_scores
from .signed_attribution import compute_signed_scores
from .subset_compose import compose_sufficient_subset
from .verdict import run_paired_channel_diagnostic
from .mlp_hook import install_mlp_residual_ablation_hook, count_mlp_hooks
from .attn_block_hook import (
    install_attn_block_residual_ablation_hook,
    count_attn_block_hooks,
)
from .memory_safety import (
    MemoryWatchdog,
    pre_flight_memory_check,
    mps_empty_cache_safe,
)

__all__ = [
    "install_edge_consumer_ablation_hook",
    "count_edge_consumer_hooks",
    "build_projection_cache",
    "save_projection_cache",
    "load_projection_cache",
    "REFUSAL_MARKERS",
    "contains_refusal",
    "first_refusal_position",
    "compute_attribution_scores",
    "compute_signed_scores",
    "compose_sufficient_subset",
    "run_paired_channel_diagnostic",
    "MemoryWatchdog",
    "pre_flight_memory_check",
    "mps_empty_cache_safe",
    "install_mlp_residual_ablation_hook",
    "count_mlp_hooks",
    "install_attn_block_residual_ablation_hook",
    "count_attn_block_hooks",
]
