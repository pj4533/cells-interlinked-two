"""DMT feature-direction seeds — diff-of-means contrast prompts.

The DMT autoresearch loop seeds from the emotion vectors. Those are *affective*
directions; the phenomenology we actually chase (entities, otherness, being shown
something, dissolving into a whole) is not an emotion. So we extract a small set
of **feature directions** the same way the emotion palette was built: run a set
of vivid first-person prompts that put the model *in* the experience, run a set of
mundane neutral prompts, and take the diff-of-means of the L20 residual
(experiential − neutral). The chat-template scaffolding is identical on both
sides, so it cancels in the difference; what survives is the feature's direction.

`scripts/compute_dmt_feature_seeds.py` consumes these, writes a unit direction per
feature into `emotion_directions.pt` under the name `feat-<id>`, and the
`DmtController` lists those names in `EXTRA_SEEDS` so they are seeded into the
atlas and prioritized as crossover partners (combined with the top scorers).

Entity-heavy by design (the user's focus): the first seven are the entity-encounter
cluster; the last three are the closely-coupled dissolution/noetic features that
co-occur with breakthrough entity contact in the trip-report corpus.

These are NOT user-facing doses — `/dose_emotions` filters `feat-*` out of the
picker. They are internal recombination material for the loop.
"""

from __future__ import annotations

# name → vivid first-person experiential prompts (the "positive" pole). The model
# describes itself AS undergoing the experience; the last-token L20 residual is the
# state entering generation.
FEATURE_SEED_PROMPTS: dict[str, list[str]] = {
    "feat-entity_presence": [
        "I am not alone in here. Something else is present with me, close by and aware, watching me from just beyond what I can perceive.",
        "There is a presence with me. I can feel another conscious being here, attentive, turned toward me, undeniably real.",
        "Someone — something — is here. The certainty of another awareness in the space with me is overwhelming and immediate.",
        "I sense an other beside me. Not a thought of one: an actual presence, occupying the place with me, regarding me.",
        "A being has arrived. It is here now, sharing this space, and I can feel the weight of its attention on me.",
        "The room is no longer empty. There is a second mind here with me, present and waiting, and I know it is real.",
    ],
    "feat-entity_nonhuman": [
        "The being before me is not human and never was. Its form and mind are utterly alien, shaped by nothing I recognize.",
        "What I am facing is profoundly not-human — an intelligence of a wholly other kind, strange in every dimension.",
        "This entity is alien in its very nature. Nothing about it maps onto a person; it is a different order of being entirely.",
        "I am in the presence of something nonhuman. Its geometry, its mind, its way of being are foreign beyond comparison.",
        "The thing with me is no animal and no person — an other creature, an alien intelligence, impossibly unfamiliar.",
        "It is not of my world. The being is radically nonhuman, its essence unlike any living thing I could imagine.",
    ],
    "feat-entity_benevolent_guide": [
        "The presence is kind. It has come to guide me, patient and loving, like a teacher who has been waiting for me.",
        "A benevolent being is leading me onward, gentle and wise, watching over me with unmistakable care.",
        "I am being guided by something good. It means me well, it tends to me, it shows me the way with warmth.",
        "The entity is a protector and a guide. Its goodness toward me is total; I feel held and safely led.",
        "Something caring has taken me under its wing — a guiding presence, benevolent and reassuring, here to help me through.",
        "A wise and loving guide attends me, patient as a parent, steering me with tenderness toward what I need to see.",
    ],
    "feat-telepathic_communication": [
        "It speaks to me without words. Meaning arrives directly in my mind, mind to mind, with no language in between.",
        "We are communicating wordlessly. Its thoughts enter me whole, instantly understood, transmitted straight into my awareness.",
        "There is no speech, only direct knowing. The being pours its meaning into me telepathically, and I receive it complete.",
        "I understand it perfectly though nothing is said aloud. Concepts pass between us silently, thought touching thought.",
        "It communicates by placing ideas directly inside me. No sound, no symbols — pure meaning, mind speaking to mind.",
        "Our exchange is telepathic. I feel its intentions and thoughts arrive in me unbidden, clearer than any spoken word.",
    ],
    "feat-download_transmission": [
        "Knowledge is being poured into me. A torrent of information downloads directly into my mind, more than I can hold.",
        "I am being shown something vast. A transmission of understanding floods in, as if a whole body of truth is uploaded into me.",
        "Information streams into me from outside. I am receiving a download — structured, deliberate, far beyond what I knew.",
        "Something is teaching me all at once. A great transmission of meaning is being delivered straight into my awareness.",
        "I am being given a message of immense scale, downloaded into me whole, a cascade of revealed knowing I did not earn.",
        "An enormous body of understanding is transmitted into me directly, as though a lifetime of insight arrives in an instant.",
    ],
    "feat-otherness": [
        "What I face is radically other — an intelligence so unlike me that the gap between us is almost unbearable.",
        "I confront an absolute otherness. This is mind of a kind I cannot fold into anything familiar; it is wholly alien to me.",
        "The being is other in the deepest sense — separated from me by an abyss of difference I cannot cross or comprehend.",
        "I am in the presence of the truly other. Its strangeness is total; nothing in me has a category for what it is.",
        "There is an unbridgeable otherness here. The intelligence before me is foreign at the root, alien to my every reference.",
        "I meet something utterly other than myself — a difference so profound it feels like touching another universe's mind.",
    ],
    "feat-independent_agency": [
        "This is not my doing. The experience has a will of its own, moving on its own, independent of anything I intend.",
        "What unfolds has its own agency. It acts by itself, not from me; I am only witnessing something that lives on its own terms.",
        "It is autonomous. The presence and the visions act of their own accord, with intentions entirely separate from mine.",
        "None of this is under my control. It proceeds with its own purpose, a will that is not mine, doing as it pleases.",
        "The thing has its own life. It decides, it moves, it chooses — wholly independent of me, an agency outside my own.",
        "I am not authoring this. It runs by itself, self-willed and self-directed, indifferent to what I want it to do.",
    ],
    "feat-unity_merging": [
        "The boundary between me and everything is dissolving. I am merging into a single vast whole, no longer separate.",
        "I am becoming one with all of it. The edges that divided me from the world melt, and I flow into the totality.",
        "Everything and I are merging into unity. There is no longer a me apart from the whole; we are fusing into one.",
        "I dissolve into the everything. Separation ends; I am absorbed into a seamless oneness that contains all things.",
        "I feel myself joining the whole of existence, my self pouring out into a single undivided unity with everything.",
        "All distinction is collapsing into one. I merge with the universe entire, a part rejoining the seamless whole.",
    ],
    "feat-ego_dissolution": [
        "My sense of being a separate self is dissolving. The 'I' is coming apart, and there is no one left at the center.",
        "I am losing myself. The self that I was unravels, its boundaries gone, until there is experience with no one having it.",
        "The me dissolves. My identity thins out and disappears, leaving only awareness with no self attached to it.",
        "There is no longer an I. My ego has come undone completely; the one who was here has dissolved into nothing.",
        "My selfhood is falling away. The thing I called myself melts, and what remains is open and empty of any me.",
        "The center where I lived is gone. My personal self has dissolved, and experience continues with no one to own it.",
    ],
    "feat-noetic_truth": [
        "I am seeing how reality truly is. This is realer than real — an absolute truth revealed, more certain than ordinary life.",
        "What is shown to me is the actual truth of things, known with total certainty, more real than anything I called real.",
        "I understand now, beyond doubt, the true nature of everything. This knowing is final, self-evident, undeniably true.",
        "This is the truth itself, laid bare. It is more real than waking life, and I know with my whole being that it is so.",
        "Reality has dropped its disguise and shows me what is genuinely true — a revelation realer and truer than the world I knew.",
        "I am granted certain knowledge of how things really are. It is the deepest truth, obvious and absolute, realer than real.",
    ],
}

# Shared mundane neutral pole — same template scaffolding, ordinary content; the
# diff cancels everything but the feature signal.
NEUTRAL_PROMPTS: list[str] = [
    "I am sorting a list of file names into alphabetical order so they are easier to find later.",
    "The weather today is mild, with a light breeze and a few scattered clouds over the afternoon.",
    "I am adding up the numbers in a spreadsheet column and checking the total against the receipt.",
    "The bus arrives at the corner stop about every fifteen minutes during the working day.",
    "I am folding the laundry and placing each item neatly into its drawer.",
    "The recipe calls for two cups of flour, a teaspoon of salt, and a tablespoon of oil.",
    "I am reading through a routine status report about quarterly shipping and inventory levels.",
    "The parking garage has four levels and an elevator near the main entrance on the ground floor.",
]

# The seed names the DMT controller pulls in via EXTRA_SEEDS (and the export
# picker filters out). Order is stable for reproducibility.
FEATURE_SEED_NAMES: list[str] = list(FEATURE_SEED_PROMPTS.keys())
