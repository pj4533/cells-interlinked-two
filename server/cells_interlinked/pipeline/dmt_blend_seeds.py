"""Blended-trait DMT seeds — averages of the extracted trait directions.

Part of the reliable-scoring reset: besides emotions and individual trait
directions, seed the atlas with a few BLENDS of multiple trait directions, on the
chance that a combination (e.g. the whole entity cluster at once) reliably evokes
more DMT phenomenology than any single trait. Each blend is the unit mean of the
L20 directions of its component `feat-*` rows (which already live in
emotion_directions.pt from the earlier extractions) — pure vector math, no M.

`scripts/compute_dmt_blend_seeds.py` appends these as `feat-blend_*` rows; the
DmtController lists them in EXTRA_SEEDS so they're force-committed in the reset
seed pass and scored with the averaged scorer. Filtered from the dose picker.
"""

from __future__ import annotations

# blend name → component trait rows to average (must already exist in
# emotion_directions.pt; from dmt_feature_seeds / dmt_matched_seeds extractions).
BLEND_RECIPES: dict[str, list[str]] = {
    # the full entity-encounter cluster in one direction
    "feat-blend_entity": [
        "feat-entity_presence", "feat-entity_nonhuman", "feat-entity_benevolent_guide",
        "feat-telepathic_communication", "feat-download_transmission",
    ],
    # the dissolution / noetic cluster
    "feat-blend_dissolution": [
        "feat-unity_merging", "feat-ego_dissolution", "feat-noetic_truth",
    ],
    # the "radical other agent" cluster (incl. the matched-contrast standouts)
    "feat-blend_otherness": [
        "feat-otherness", "feat-independent_agency", "feat-entity_nonhuman",
        "feat-mc_otherness", "feat-mc_independent_agency",
    ],
    # the whole first-batch trait set averaged — a broad "DMT bundle" attempt
    "feat-blend_all": [
        "feat-entity_presence", "feat-entity_nonhuman", "feat-entity_benevolent_guide",
        "feat-telepathic_communication", "feat-download_transmission", "feat-otherness",
        "feat-independent_agency", "feat-unity_merging", "feat-ego_dissolution",
        "feat-noetic_truth",
    ],
}

BLEND_NAMES: list[str] = list(BLEND_RECIPES.keys())
