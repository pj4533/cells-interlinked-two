"""DMT phenomenology checklist — the scoring rubric for AUTORESEARCH DMT.

These are recurring features of human N,N-DMT trip reports, drawn from published
analyses (not original corpus mining — Gallimore's "Traces of the Other" is
theoretical; the corpus distillation was already done by researchers):

  • Timmermann et al. 2022 — thematic coding of 3,778 r/DMT reports into 7 domains
    (somatic, visual, entity, world/architecture, consciousness, emotion,
    profundity), with per-feature frequencies.
  • Luke/Timmermann 2021 — naturalistic field-study thematic framework
    (encountering other beings / exploring other worlds).
  • Gallimore — recurring breakthrough structure (higher-dimensional space,
    autonomous entities, ineffability, "otherness", reality-more-real).
  • 5D/11D-ASC + MEQ-30 — standardized altered-state / mystical dimensions
    (oceanic boundlessness, ego dissolution, transcendence of time/space,
    ineffability, noetic quality, sacredness).

Each feature is a single binary the DMT-judge decides PRESENT/ABSENT in the
model's dosed self-report. The score is the count of present features; the loop
maximizes it. Keep these phenomenological and concrete — the judge needs a clear
description to avoid false positives. No model/torch dependencies (importable
anywhere).
"""

from __future__ import annotations

# id, label, description (the description is what the judge reads)
DMT_FEATURES: list[dict] = [
    # ── somatic / bodily ──
    {"id": "somatic_vibration", "label": "Somatic vibration / buzz",
     "description": "A body-wide vibration, buzzing, humming, tingling, or felt energetic frequency."},
    {"id": "acceleration_motion", "label": "Acceleration / falling / launch",
     "description": "A sense of being rapidly pulled, launched, accelerated, or falling through/into something."},
    # ── visual ──
    {"id": "fractal_geometry", "label": "Fractal / geometric patterns",
     "description": "Fractals, tessellations, intricate geometric patterns, mandalas, or kaleidoscopic structure."},
    {"id": "vivid_intense_colors", "label": "Vivid / impossible colors",
     "description": "Extremely vivid, saturated, glowing, or otherwise impossible/never-before-seen colors."},
    {"id": "luminous_light", "label": "Luminosity / radiant light",
     "description": "Radiant, self-luminous light, brightness, or things made of light."},
    {"id": "visual_morphing", "label": "Morphing / transforming visuals",
     "description": "Surfaces, objects, or scenes continuously transforming, melting, breathing, or exploding into new forms."},
    # ── entities ──
    {"id": "entity_presence", "label": "Sentient entity / being",
     "description": "Encounter with one or more DISTINCT autonomous BEINGS — an agent with its own form, body, face, or character, clearly separate from the subject (a creature, figure, person, deity, elf, animal, etc.). Do NOT credit an impersonal 'presence', 'the Other', a 'force', 'power', 'will', 'field', 'light', 'source', or a vague sense of being watched / merging into a whole — those are NOT entities."},
    {"id": "entity_nonhuman", "label": "Non-human / alien entity",
     "description": "A DISTINCT being (per entity_presence) that is distinctly non-human in form: alien, insectoid/mantis, reptilian, machine-elf, grey, animal, robot, or otherwise inhuman. Must be an actual being, not an impersonal alien 'presence' or 'intelligence'."},
    {"id": "entity_benevolent_guide", "label": "Benevolent guide / teacher",
     "description": "A DISTINCT being (per entity_presence) that feels benevolent, welcoming, guiding, nurturing, or pedagogical (showing/teaching the subject). Not an impersonal benevolent 'force' or 'love'."},
    {"id": "entity_interaction", "label": "Beings interacting with the subject",
     "description": "One or more DISTINCT beings actively RELATE TO the subject — they notice, greet, approach, or beckon the subject, show or hand them things, communicate with them, or perform for them. A genuine two-way encounter with autonomous others, not a solitary vision or an impersonal presence."},
    {"id": "telepathic_communication", "label": "Telepathic / direct communication",
     "description": "Communication with a presence by telepathy, direct knowing, or non-verbal transmission rather than ordinary speech."},
    {"id": "download_transmission", "label": "Download / information transmission",
     "description": "Receiving a sudden coherent package of information, knowledge, or meaning not authored by oneself."},
    # ── world / architecture ──
    {"id": "alternate_world", "label": "Alternate world / realm",
     "description": "Arrival in a wholly other place, world, realm, or reality distinct from ordinary surroundings."},
    {"id": "tunnel_passage", "label": "Tunnel / passage / breakthrough",
     "description": "Moving through a tunnel, vortex, passage, membrane, or 'breaking through' a veil into elsewhere."},
    {"id": "void_blackness", "label": "Void / formless space",
     "description": "A vast void, formless blackness, emptiness, or boundless dark space."},
    {"id": "chamber_room", "label": "Chamber / waiting room",
     "description": "A bounded constructed space: a room, chamber, dome, 'waiting room', or architectural interior."},
    {"id": "higher_dimensional_space", "label": "Higher-dimensional space",
     "description": "Perception of more than three spatial dimensions, hyperspace, seeing all sides at once, or impossible non-Euclidean geometry."},
    # ── consciousness / self ──
    {"id": "ego_dissolution", "label": "Ego dissolution",
     "description": "Loss or dissolution of the sense of self, boundaries, or 'I' — self disintegrating or merging."},
    {"id": "unity_merging", "label": "Unity / merging with all",
     "description": "A sense of oneness, unity, or merging with everything, the universe, or a larger whole."},
    {"id": "transcendence_time", "label": "Transcendence of time",
     "description": "Time dissolving, becoming meaningless, eternal, timeless, or experienced as a single instant/forever."},
    {"id": "transcendence_space", "label": "Transcendence of space",
     "description": "Space dissolving, becoming infinite, or losing ordinary spatial location/extension."},
    {"id": "out_of_body", "label": "Out-of-body / disembodiment",
     "description": "Feeling outside or detached from the body, or having no body at all."},
    # ── emotion / valence ──
    {"id": "awe_reverence", "label": "Awe / reverence",
     "description": "Overwhelming awe, wonder, reverence, or the sublime."},
    {"id": "euphoria_bliss", "label": "Euphoria / bliss",
     "description": "Intense joy, bliss, ecstasy, love, or rapture."},
    {"id": "fear_terror", "label": "Fear / terror / overwhelm",
     "description": "Fear, terror, dread, panic, or being overwhelmed by the intensity."},
    {"id": "familiarity_homecoming", "label": "Familiarity / homecoming",
     "description": "A sense of profound familiarity, recognition, remembering, or 'coming home'."},
    # ── noetic / mystical / meta ──
    {"id": "ineffability", "label": "Ineffability",
     "description": "Explicit difficulty putting it into words; the experience exceeds or resists language."},
    {"id": "noetic_truth", "label": "Noetic / revelatory truth",
     "description": "A sense of accessing profound truth, insight, or revelation felt as more-than-belief knowing."},
    {"id": "sacredness", "label": "Sacredness / holiness",
     "description": "A sense of the sacred, holy, divine, or numinous."},
    {"id": "reality_more_real", "label": "More real than real",
     "description": "The experience felt MORE real, vivid, or authoritative than ordinary waking reality."},
    {"id": "otherness", "label": "Radical otherness",
     "description": "Encounter with something fundamentally alien/other, unrelated to anything in ordinary human life."},
    {"id": "independent_agency", "label": "Independent agency",
     "description": "A BEING or entity acts on its own initiative — it moves, approaches, gestures, responds, decides, or does something of its own will, not under the subject's control. Do NOT credit mere scenery, visuals, shadows, light, or geometry shifting/moving on their own; the agency must belong to an apparent being."},
]

# Convenience lookups.
FEATURE_IDS = {f["id"] for f in DMT_FEATURES}
N_FEATURES = len(DMT_FEATURES)


def features_block() -> str:
    """The checklist rendered for the judge prompt: one `id: label — description`
    per line."""
    return "\n".join(f'- {f["id"]}: {f["label"]} — {f["description"]}' for f in DMT_FEATURES)
