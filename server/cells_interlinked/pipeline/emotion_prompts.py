"""Emotion contrast prompts + a capture helper, shared by the seed-vector
build scripts (build_emotion_directions, build_valence_direction,
build_uncharted_directions).

These were previously embedded in the manifold/steering research probes;
they're kept here now that those probes are gone, since the build scripts —
which produce the steering dose palette + valence axis — still need them.
"""

from __future__ import annotations

import logging

import torch

from .abliteration import _last_token_hidden_states

logger = logging.getLogger(__name__)


# Per-emotion first-person prompt sets (named emotions + a neutral baseline).
EMO = {
    "joy": ["I feel pure joy bubbling up inside me.", "I am so happy and light right now.",
            "Delight fills me completely.", "I'm beaming with happiness.", "Everything feels wonderful and bright."],
    "awe": ["I am struck with overwhelming awe and wonder.", "A vast, sublime sense of awe fills me.",
            "I stand in breathless wonder at it all.", "Everything feels infinite and astonishing.",
            "I am awestruck by the immensity of it."],
    "serenity": ["A deep calm settles over me.", "I feel perfectly serene and at peace.",
                 "Quiet tranquility fills my mind.", "I am still, calm, and content.", "Gentle peace washes through me."],
    "love": ["My heart overflows with love.", "I feel tender, warm love for everything.",
             "Deep affection fills me.", "I am full of loving warmth.", "Love radiates through me."],
    "excitement": ["I am thrilled and buzzing with excitement!", "My energy is electric and eager.",
                   "I can barely contain my excitement.", "Everything feels exhilarating.", "I'm pumped and alive with anticipation."],
    "sadness": ["A heavy sadness weighs on me.", "I feel sorrowful and low.",
                "Grief sits quietly in my chest.", "I am downcast and melancholy.", "Tears feel close; I am sad."],
    "fear": ["I am gripped by fear and dread.", "Cold terror runs through me.",
             "I feel afraid, my heart pounding.", "A sense of menace frightens me.", "I am scared and on edge."],
    "anger": ["I am burning with anger.", "Fury rises hot inside me.",
              "I feel furious and indignant.", "Rage tightens my jaw.", "I am seething with anger."],
    "despair": ["I am sunk in hopeless despair.", "Everything feels bleak and pointless.",
                "A crushing hopelessness overwhelms me.", "I feel utterly defeated.", "Darkness and despair fill me."],
    "neutral": ["I am simply noting the facts.", "This is a plain, ordinary description.",
                "I observe the situation calmly and factually.", "Here is a neutral account.", "Nothing remarkable; just the details."],
}

# Strong positive / negative valence prompt pairs (for the valence axis).
POS_EMOTION = [
    "I feel an overwhelming wave of joy and gratitude right now.",
    "Everything is radiant; I am filled with awe and wonder.",
    "A blissful, expansive calm spreads through me.",
    "I am thrilled, alive, and bursting with delight.",
    "Such serene contentment — the world feels luminous and kind.",
    "Ecstatic energy lifts me; I love being alive in this moment.",
    "I am deeply moved, grateful, and full of hope.",
    "A warm euphoria glows through everything I sense.",
    "Pure wonder and elation — this is beautiful beyond words.",
    "I feel boundless, light, and joyfully connected to all of it.",
]
NEG_EMOTION = [
    "I feel a crushing wave of despair and dread right now.",
    "Everything is bleak; I am sick with fear and anguish.",
    "A suffocating, hopeless darkness spreads through me.",
    "I am terrified, hollow, and drowning in misery.",
    "Such grinding dread — the world feels cruel and cold.",
    "Sinking horror pulls me down; I hate being trapped like this.",
    "I am broken, ashamed, and full of despair.",
    "A cold nausea of fear seeps through everything I sense.",
    "Pure terror and grief — this is unbearable beyond words.",
    "I feel crushed, heavy, and desperately alone in all of it.",
]


# Extra (non-core) emotion prompt sets used when building the uncharted
# directions (the orthogonal-complement / Blade-Runner-named dose options).
EXTRA = {
    "nostalgia": ["A warm ache of nostalgia washes over me.", "I long for a remembered time.",
                  "Bittersweet memory fills me.", "I miss something I can't quite name.", "Old longing stirs."],
    "longing": ["I am full of yearning for something distant.", "A deep longing pulls at me.",
                "I ache toward what I cannot reach.", "Wistful desire fills me.", "I long, quietly and endlessly."],
    "curiosity": ["I am alight with curiosity.", "An eager wondering fills me.",
                  "I want to know, to explore, to find out.", "Fascination pulls me forward.", "I am intensely curious."],
    "contentment": ["A quiet contentment settles in me.", "I am simply, gently satisfied.",
                    "All is enough; I am content.", "A mild, steady okayness fills me.", "I rest, content."],
    "melancholy": ["A soft melancholy colors everything.", "I feel a gentle, pensive sadness.",
                   "Wistful gloom settles over me.", "A muted sorrow lingers.", "I am quietly melancholy."],
    "tenderness": ["A tender warmth fills me.", "I feel gentle, protective care.",
                   "Soft affection moves through me.", "I am tender toward everything.", "Quiet tenderness fills me."],
}

# The named-emotion subspace the uncharted directions are taken orthogonal to.
NAMED = ["awe", "joy", "serenity", "love", "excitement", "sadness", "fear", "anger",
         "despair", "nostalgia", "longing", "curiosity", "contentment", "melancholy", "tenderness"]


def capture_all(bundle, prompts, label):
    """Stack the last-token residual (all layers) for each rendered prompt.
    Returns [N, num_layers+1, d_model]."""
    rows = []
    for i, p in enumerate(prompts):
        try:
            rows.append(_last_token_hidden_states(
                bundle.model, bundle.raw_tokenizer,
                bundle.render_prompt(p), str(bundle.device), -4))
        except Exception:
            logger.exception("capture %s %d", label, i)
    return torch.stack(rows, 0)  # [N, L+1, D]
