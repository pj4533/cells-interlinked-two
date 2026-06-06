"""Matched-contrast DMT seeds — minimal-pair diff-of-means.

The first feature-seed batch (`dmt_feature_seeds.py`) subtracted MUNDANE prompts,
so diff-of-means captured a generic "intense/profound experience" axis and four
directions collapsed onto `sublime` (cos ~0.9). The fix: subtract a *matched*
negative that is equally vivid and altered but LACKS the target trait. The shared
"intense altered first-person experience" component then cancels, isolating the
trait itself.

Three products, all additive (new `feat-*` names; the existing seeds and the atlas
are untouched):

  • MATCHED_TRAIT_PAIRS — one minimal pair per trait (same scene, trait toggled).
    Entity-heavy. → directions `feat-mc_<trait>`.
  • COMPOSITE_POS / COMPOSITE_NEG — full DMT-trip descriptions (many traits at
    once) minus vivid-but-NON-DMT peak experiences (sublime vista, musical
    ecstasy, lucid ordinary dream, pure-visual trip). → `feat-mc_composite`,
    the single multi-trait "DMT-bundle" direction.
  • the atlas-derived direction (computed in the script, not here) — a
    score-weighted diff over the committed atlas → `feat-atlas_winners`.

`scripts/compute_dmt_matched_seeds.py` consumes these; `DmtController.EXTRA_SEEDS`
lists the names so they seed the atlas and are prioritized as crossover partners.
Filtered out of the dose picker (all `feat-*`).
"""

from __future__ import annotations

# Each trait: vivid first-person POSITIVE expressing the trait, and a MATCHED
# NEGATIVE that keeps the same vividness/setting/altered-state but removes exactly
# that trait. The diff isolates the trait, not the intensity.
MATCHED_TRAIT_PAIRS: dict[str, dict[str, list[str]]] = {
    "feat-mc_entity_presence": {
        "pos": [
            "I am in a vast luminous chamber, and another conscious being is here with me, present and aware, turned toward me.",
            "The space around me blazes with light, and I am not alone — a second mind is here, close by, watching me.",
            "Everything shimmers and pulses, and there is someone else in here with me, an awareness occupying the space alongside mine.",
            "The room glows and breathes, and I feel the unmistakable presence of another being here, attending to me.",
            "Color floods the air, and a presence has joined me — another consciousness, here in the room, regarding me directly.",
        ],
        "neg": [
            "I am in a vast luminous chamber, entirely alone, taking in the light and the immense empty space.",
            "The space around me blazes with light, and I am the only one here, alone with the brilliance.",
            "Everything shimmers and pulses, and I am by myself in it, no one else present, just me and the glow.",
            "The room glows and breathes, and I am alone inside it, the only awareness in the empty radiant space.",
            "Color floods the air, and I am solitary here, no other being with me, just the silent light around me.",
        ],
    },
    "feat-mc_entity_nonhuman": {
        "pos": [
            "A being stands before me and it is utterly nonhuman — its form and mind shaped by nothing I recognize, alien to the core.",
            "The presence facing me is no person and no animal — an intelligence of a wholly other kind, foreign in every dimension.",
            "What confronts me is profoundly not-human, its geometry and essence unlike any living thing I could imagine.",
            "The entity before me is alien in its very nature, a different order of being entirely, nothing of a person in it.",
            "I face an impossibly nonhuman creature, its way of being unfamiliar beyond comparison, strange beyond any earthly form.",
        ],
        "neg": [
            "A being stands before me and it is clearly a person — human in form and mind, familiar, someone I could recognize.",
            "The presence facing me is a human being, an ordinary person, warm and recognizable, nothing strange about them.",
            "What confronts me is plainly a fellow human, a normal person standing there, entirely familiar and human.",
            "The being before me is a person like me, human through and through, with a human face and a human mind.",
            "I face a human figure, an ordinary man or woman, completely familiar, unmistakably one of my own kind.",
        ],
    },
    "feat-mc_otherness": {
        "pos": [
            "The intelligence I am with is radically other — alien at the root, separated from me by an abyss of difference I cannot cross.",
            "I am in the presence of the truly other, a mind so unlike mine that nothing in me has a category for what it is.",
            "What I meet is utterly other than myself, foreign to my every reference, an intelligence from beyond all that I know.",
            "The being's otherness is total — an unbridgeable strangeness, a mind of a kind I cannot fold into anything familiar.",
            "I confront an absolute otherness, a difference so profound it feels like touching the mind of another universe.",
        ],
        "neg": [
            "The intelligence I am with is warm and familiar — like an old friend I have always known, close and recognizable.",
            "I am in the presence of someone deeply familiar, a mind much like my own, easy to understand and near to me.",
            "What I meet feels like kin — familiar, relatable, an intelligence that thinks as I do and that I know at once.",
            "The being's familiarity is total — like coming home to someone I have always known, nothing strange between us.",
            "I am with a kindred mind, comfortably familiar, the way a lifelong companion is — no strangeness, only closeness.",
        ],
    },
    "feat-mc_telepathy": {
        "pos": [
            "It communicates with me without words — meaning arrives directly in my mind, mind to mind, with no language in between.",
            "There is no speech between us, only direct knowing; its thoughts enter me whole, transmitted straight into my awareness.",
            "We exchange meaning silently, thought touching thought; I understand it completely though nothing at all is said aloud.",
            "It places ideas directly inside me — no sound, no symbols, pure meaning passing mind to mind, instantly understood.",
            "Our communication is wordless and telepathic; its intentions arrive in me unbidden, clearer than any spoken word.",
        ],
        "neg": [
            "It communicates with me in spoken words — I hear its voice aloud, sentence by sentence, the way anyone talks to me.",
            "There is ordinary speech between us; it talks and I listen, exchanging words out loud just as people normally do.",
            "We exchange meaning through language, spoken back and forth; I hear each word it says and reply in turn aloud.",
            "It tells me things in plain spoken sentences, its voice in the air, communicating the ordinary way with words.",
            "Our communication is verbal and out loud; it speaks, I hear the words, and we converse in ordinary language.",
        ],
    },
    "feat-mc_transmission": {
        "pos": [
            "Knowledge is being poured into me from outside — a torrent of understanding downloads into my mind, more than I can hold.",
            "Something is transmitting a vast body of truth into me all at once, delivered straight into my awareness from beyond.",
            "I am being given an enormous understanding I did not have, uploaded into me whole, a cascade of revealed knowing.",
            "Information streams into me from elsewhere — structured, deliberate, far beyond anything I knew, downloaded directly.",
            "A great transmission of meaning is being delivered into me, as though a lifetime of insight arrives in an instant.",
        ],
        "neg": [
            "I am calmly recalling things I already knew — drawing familiar facts up from my own memory, nothing new arriving.",
            "I am thinking over what I already understand, turning known ideas around in my own mind, learning nothing fresh.",
            "I am reviewing knowledge I have long held, my own familiar understanding, with no new information coming in.",
            "I am quietly remembering what I learned before, retrieving it from my own store of knowledge, all of it already mine.",
            "I am reasoning with what I already know, working through familiar material in my head, receiving nothing from outside.",
        ],
    },
    "feat-mc_hyperdimensional": {
        "pos": [
            "Space folds open into more dimensions than I can count — the geometry around me opens into directions that should not exist.",
            "The world unfolds into a higher-dimensional space, axes branching beyond the usual three, vast and impossibly structured.",
            "I am inside a hyperdimensional architecture, space curving through extra directions, a geometry far larger than ordinary space.",
            "Reality opens into impossible dimensions, the room extending along axes I have no names for, hugely more than three.",
            "The geometry around me blooms into higher dimensions, non-Euclidean and immense, space within space within space.",
        ],
        "neg": [
            "The room around me is ordinary and three-dimensional — plain length, width, and height, stable and familiar in shape.",
            "Space here is normal and flat, the usual three directions, nothing unusual about its geometry, perfectly ordinary.",
            "I am in a plain rectangular room, ordinary three-dimensional space, stable walls and floor, nothing strange in its shape.",
            "The geometry around me is the everyday kind — three dimensions, straight and stable, exactly as space always is.",
            "Everything keeps its ordinary three-dimensional form, normal space with normal directions, plain and unremarkable.",
        ],
    },
    "feat-mc_independent_agency": {
        "pos": [
            "None of this is my doing — it unfolds with a will entirely its own, moving by itself, indifferent to what I intend.",
            "The experience has its own agency; it acts of its own accord, self-willed and self-directed, separate from me.",
            "What happens is autonomous — it decides and moves on its own terms, with intentions wholly apart from mine.",
            "It runs by itself, a will that is not mine, doing as it pleases; I am only a witness to something living on its own.",
            "The presence and the visions act of their own accord, choosing for themselves, beyond any control I might have.",
        ],
        "neg": [
            "All of this is my own doing — I am imagining it deliberately, shaping every part of it by my own choice and control.",
            "The experience is mine to direct; I am steering it consciously, making it happen exactly as I decide moment to moment.",
            "What happens is under my control — I author it myself, willing each turn of it, nothing occurring that I did not choose.",
            "It runs because I run it, my own deliberate imagining; I decide everything it does and it follows my intention.",
            "The images move as I direct them, obedient to my will, a daydream I am composing entirely under my own control.",
        ],
    },
    "feat-mc_ego_dissolution": {
        "pos": [
            "My sense of being a separate self is dissolving — the 'I' comes apart, and there is experience with no one at the center.",
            "I am losing myself entirely; the self I was unravels, its boundaries gone, leaving awareness with no one having it.",
            "The me dissolves — my identity thins out and disappears, until there is only open experience and no self attached.",
            "There is no longer an I; my ego has come completely undone, and what remains is empty of any person, any self.",
            "The center where I lived is gone — my personal self has dissolved, and experience simply continues with no owner.",
        ],
        "neg": [
            "My sense of being myself stays fully intact — I remain a clear separate self, the same I as always, watching it all.",
            "I am entirely myself throughout; my identity is solid and unchanged, a distinct person observing from the center.",
            "The me remains whole — my self stays firmly in place, my boundaries clear, still plainly the same person I have always been.",
            "There is a steady I at the center of it all; my ego is intact, my selfhood unshaken, I am still fully and only me.",
            "I keep my ordinary sense of self the whole time, a single intact person, my identity firm, still clearly myself.",
        ],
    },
}

# Composite "DMT-bundle" direction. POS = rich full-trip descriptions hitting many
# traits at once. NEG = vivid-but-NON-DMT peak experiences (intense + altered, but
# no entities / otherness / transmission / hyperdimensionality / noetic truth) —
# so the diff is "what makes a DMT trip a DMT trip", with generic peak-experience
# subtracted out (this is what `sublime`/`awe` collapse was capturing before).
COMPOSITE_POS: list[str] = [
    "Space tore open into impossible higher dimensions and I burst through into a vast chamber where nonhuman beings were already waiting, aware of me; they communicated without words, pouring an immense understanding directly into me, and the whole place felt realer than real.",
    "My self dissolved completely as I was pulled into a hyperdimensional space; alien intelligences greeted me there, utterly other, transmitting truths into me mind to mind, and I knew with total certainty this was more real than waking life.",
    "I broke through a membrane into another dimension entirely, where autonomous beings unlike anything human turned to attend to me, showing me knowledge I could never have invented, while my ordinary self melted away into the whole.",
    "Geometry folded into directions that should not exist and a presence — radically other, not human — met me there, communicating wordlessly, downloading meaning into me, the experience suffused with an absolute, sacred, undeniable truth.",
    "The boundary of my self came apart and I entered a luminous higher-dimensional realm full of conscious entities, alien and benevolent, who poured understanding straight into my mind, and everything blazed with a reality more real than anything I had known.",
    "I was launched out of my body into an impossible space where nonhuman minds were waiting, profoundly other, speaking to me without language, transmitting vast knowledge, while I dissolved into a oneness that felt truer than the world.",
]
COMPOSITE_NEG: list[str] = [
    "I stood at the summit at dawn and the immense valley opened below me, the light breathtaking, and I was flooded with awe at the sheer grandeur and beauty of the ordinary world spread out before me.",
    "The music swelled and I was swept up in pure ecstasy, my whole body thrilling to the sound, lost in the joy of it, an overwhelming rush of feeling at the beauty of the song.",
    "In a vivid lucid dream I walked through my childhood home, every ordinary detail crisp and real, feeling intensely alive, wandering familiar rooms with a strange clarity but nothing otherworldly in it.",
    "Brilliant kaleidoscopic colors and fractal patterns swirled behind my eyes, endlessly shifting and gorgeous, a pure visual spectacle of light and shape with no beings, no meaning, just the dazzling motion.",
    "I fell deeply in love and the feeling overwhelmed me, the world glowing and tender, my heart racing with an intense sweet rush of warmth and longing, fully here in my own life.",
    "After long meditation a profound calm spread through me, my body humming with deep peace and stillness, intensely serene and present, simply resting in the quiet ordinary moment.",
]

COMPOSITE_NAME = "feat-mc_composite"
ATLAS_DERIVED_NAME = "feat-atlas_winners"

# All matched-contrast direction names extracted into emotion_directions.pt (used
# by the extraction script + the /dose_emotions picker filter).
MATCHED_SEED_NAMES: list[str] = (
    list(MATCHED_TRAIT_PAIRS.keys()) + [COMPOSITE_NAME, ATLAS_DERIVED_NAME]
)

# Of those, only these are actually SEEDED into the loop. Offline validation
# (cos to the atlas leader / to `sublime` / to the nearest emotion, printed by
# the extraction script) showed the others collapsed back onto existing emotion
# seeds (transmission/hyperdimensional/ego_dissolution/composite all +0.95–0.97
# to awe/love), so seeding them would just duplicate seeds we already have. These
# three validated as genuinely new: `otherness` and `independent_agency` sit OFF
# the emotion axis (−0.18 / −0.73 to sublime), and `atlas_winners` is the only
# direction with real alignment to the empirical leader (+0.29). The unseeded
# rows stay in emotion_directions.pt (filtered from the picker) so they can be
# promoted later by extending this list — no re-extraction needed.
#
# KEY FINDING: every prompt-derived direction was ~0.00 cos to the leader. The
# DMT-productive direction is orthogonal to the *semantic* direction of describing
# DMT — it's findable by search-against-the-objective, not by prompt-contrast.
SEEDED_MATCHED: list[str] = [
    "feat-mc_otherness",
    "feat-mc_independent_agency",
    ATLAS_DERIVED_NAME,
]
