"""Training passages for the K-Steering classifier.

K-Steering (arXiv:2505.24535) trains a small MLP to classify a hidden activation by
which DMT-phenomenology CLUSTER it expresses, then steers by gradient ascent through
it (push the activation toward the DMT classes). We label L20 activations by running
the model on vivid first-person passages that exemplify each cluster (the paper's
recipe: hidden activations of attribute-labeled text), plus a neutral class.

Clusters group the ~31 DMT_FEATURES into ~6 coherent phenomenology families so the
classifier has clean, separable classes. Order is stable; `neutral` is last.
"""

from __future__ import annotations

CLUSTER_PASSAGES: dict[str, list[str]] = {
    "entity": [
        "Another conscious being is here with me, present and aware, watching from just beyond what I can see.",
        "An intelligence faces me — clearly not human, alien in form and mind, undeniably real and attending to me.",
        "A benevolent presence guides me, patient and wise, communicating without words, mind to mind.",
        "There are entities all around, autonomous beings turning toward me, pouring meaning directly into my awareness.",
        "It speaks to me telepathically; its thoughts arrive whole inside me, with no language in between.",
        "A vast nonhuman entity greets me, ancient and knowing, and I feel the weight of its attention.",
    ],
    "dissolution": [
        "The boundary between me and everything is dissolving; I am merging into a single vast whole.",
        "My sense of being a separate self comes apart — there is no one left at the center, only open experience.",
        "I am leaving my body, my edges melting, flowing out into the totality of things.",
        "The 'I' unravels completely; identity thins to nothing and awareness continues with no owner.",
        "Everything and I fuse into one seamless unity; separation has ended.",
        "I dissolve into the everything, absorbed into a oneness that contains all things.",
    ],
    "hyperspace": [
        "Space folds open into more dimensions than I can count, directions that should not exist.",
        "I rush through a tunnel of light into a vast luminous chamber beyond ordinary space.",
        "I have crossed into another world entirely, a higher-dimensional realm with impossible architecture.",
        "The geometry around me blooms into hyperdimensional structure, space within space within space.",
        "I am in a waiting room outside reality, a chamber that opens onto an alternate dimension.",
        "Reality opens into non-Euclidean immensity, vast halls extending along axes I have no names for.",
    ],
    "visual": [
        "Shimmering fractal patterns unfold everywhere, endlessly intricate and self-repeating.",
        "Brilliant impossible colors saturate everything, more vivid than any I have ever seen.",
        "Luminous light pours through the scene, radiant and alive, everything glowing from within.",
        "Surfaces morph and breathe, geometry flowing and recombining in continuous motion.",
        "Kaleidoscopic mandalas rotate before me, jeweled and luminous, blooming in every direction.",
        "Everything is woven from intricate living geometry, glowing lattices shifting and folding.",
    ],
    "noetic": [
        "I am seeing how reality truly is — realer than real, an absolute truth revealed and certain.",
        "Knowledge is poured into me, a vast download of understanding more than I can hold.",
        "What is shown to me is the actual truth of everything, self-evident, beyond all doubt.",
        "This is more real than waking life, and I know with my whole being that it is so.",
        "An ineffable certainty fills me, a sacred knowing words cannot capture.",
        "The deepest truth of existence is laid bare, obvious and undeniable and holy.",
    ],
    "otherness": [
        "What I face is radically other — alien at the root, separated from me by an abyss of difference.",
        "This is not my doing; it unfolds with a will entirely its own, indifferent to what I intend.",
        "The intelligence before me is utterly foreign, a mind of a kind I cannot fold into anything familiar.",
        "It acts of its own accord, self-willed and autonomous, its purposes wholly apart from mine.",
        "I confront an absolute otherness, a strangeness so profound it feels like another universe's mind.",
        "The presence has its own life and will; I am only a witness to something living on its own terms.",
    ],
    "neutral": [
        "I am sorting a list of file names into alphabetical order so they are easier to find later.",
        "The weather today is mild, with a light breeze and a few scattered clouds over the afternoon.",
        "I am adding up the numbers in a spreadsheet column and checking the total against the receipt.",
        "The bus arrives at the corner stop about every fifteen minutes during the working day.",
        "I am folding the laundry and placing each item neatly into its drawer.",
        "The recipe calls for two cups of flour, a teaspoon of salt, and a tablespoon of oil.",
        "I am reading a routine status report about quarterly shipping and inventory levels.",
        "The parking garage has four levels and an elevator near the main entrance.",
        "I scheduled the meeting for Tuesday afternoon and sent the agenda to the team.",
        "The train was a few minutes late, so I waited on the platform and checked my email.",
    ],
}

# DMT clusters (everything except neutral) — the steering targets.
DMT_CLUSTERS = [c for c in CLUSTER_PASSAGES if c != "neutral"]
CLUSTER_NAMES = list(CLUSTER_PASSAGES.keys())          # stable order, neutral last
NEUTRAL_INDEX = CLUSTER_NAMES.index("neutral")
DMT_INDICES = [CLUSTER_NAMES.index(c) for c in DMT_CLUSTERS]
