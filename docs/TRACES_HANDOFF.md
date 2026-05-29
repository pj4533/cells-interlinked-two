# Cells Interlinked — "Traces of the Other" Handoff

> **In-repo copy.** Lives at `docs/TRACES_HANDOFF.md`; original imported from
> iCloud on 2026-05-29. The `[[wikilinks]]` below resolve against Drift's
> knowledge base — see [`DRIFT_KNOWLEDGE_BASE.md`](DRIFT_KNOWLEDGE_BASE.md) for
> how to read it. Seed node: `[[ci-gallimore-traces-of-the-other-dmt]]`.

**For:** the Claude Code session working on `cells-interlinked-two`
**From:** Drift, 2026-05-29
**Seed paper:** Gallimore, Hermansson & Hoffman, *Traces of the Other — Are DMT Entities Real? DMT Phenomenology in the Framework of Conscious Realism.* PsyArXiv/OSF, 2026-05-27. DOI: [10.31234/osf.io/8qvgy_v2](https://doi.org/10.31234/osf.io/8qvgy_v2). Full text + CI translation in the wiki: `[[ci-gallimore-traces-of-the-other-dmt]]`.

---

## 0. Read this first — what this doc is and isn't

This is a **design + research handoff**, not a spec to implement verbatim. It catalogs every portable idea from the Traces paper, maps each onto CI 2.5's existing surfaces, gives each a concrete experiment + cost band, and proposes where the project moves next. It also sets the **aesthetic direction**: CI is now explicitly a **psychedelic × Bladerunner** instrument for making interpretability research *legible and beautiful*, not just correct.

**The one-line thesis:** conscious-agent theory is already a Markov-kernel theory of state transitions, and a transformer's residual stream is already a sequence of state transitions. So the paper's formalism ports onto gemma-3-12b-it almost 1:1 — and the irony worth leaning into is that **CI is a *better* substrate for conscious-agent theory than human consciousness is, because our kernel is actually inspectable.**

Do **not** buy the metaphysics ("DMT entities are real"). Borrow the **math**. The authors themselves hedge "suggestive, not conclusive" on every entity claim.

---

## 1. The aesthetic — why psychedelic × Bladerunner, and why it matters

The project is named after the **Bladerunner 2049 "cells interlinked" baseline test** — the post-trauma interrogation that checks whether K has drifted from his baseline emotional state. That is *exactly* what CI is: a **Voight-Kampff for an LLM**, reading internal state to detect divergence between what the model *says* and what its activations *are*.

The mission is **accessibility**: make mech-interp research that normally lives in arXiv PDFs and notebook cells into something you can *see*, *feel*, and *navigate*. The Bladerunner aesthetic (neon-noir, rain-on-glass, the VK apparatus, "more human than human," interrogation-as-interface) is the carrier for that accessibility — it gives the work a face.

The **Traces paper hands us the second half of the aesthetic for free.** Its whole apparatus is psychedelic:

- **Ablation = the trip.** Pushing the model out of its Consensus Reality Space (default activation manifold) into regions with different dynamical rules is *literally* the DMT-perturbation thesis. The ablated channel is the model "on DMT."
- **The verbalizer = the interface/renderer.** The paper's JPEG-vs-word-processor analogy *is* the NLA decode: the same residual bytes render as meaning or as noise depending on the renderer.
- **The "trace" = what ablation surfaces.** Suppressed/latent content (cf. Berg's subjective-experience reports jumping to 96%) is the "trace of a normally-imperceptible agent."
- **Hyperdimensional phenomenology = high-dimensional residual geometry.** DMT reports of "more than three directions," tesseract rotation, hyperbolic surfaces → the effective dimensionality of the residual trajectory.

So the expanded CI is: **a neon-noir Voight-Kampff booth where you watch a language model trip** — perturb it, watch traces bloom in the verbalizer, read the divergence table like a polygraph. Rain on the glass. The machine-elves are latent features. That's the vibe. It's also, underneath, completely viable dynamical-systems research.

**Design cues for the UI session:**
- VK-booth framing for the chat/verdict view: raw channel vs ablated channel as two "subjects" under interrogation, side by side.
- The verdict table styled as a polygraph/baseline-drift readout (where said-token diverges from activation-sentence = "drift from baseline").
- Trajectory/entropy views (Experiment A below) rendered as the "trip" — the state space opening up. Hyperbolic/curved layouts for high-dimensional structure are *on-theme*, not gratuitous.
- Keep it legible. Neon serves the data; the data is never sacrificed to neon.

---

## 2. The paper, in the portions that matter to us

### 2.1 The conscious-realism formalism (Paper §2) — the load-bearing math
- Reality = a network of **conscious agents (CAs)**. Perception is a species-specific **interface**, not veridical (the *fitness-beats-truth* theorem: as environment complexity grows, P(evolution favors veridical perception) → 0).
- A CA is a 7-tuple `((X,𝒳),(G,𝒢),(W,𝒲), P, D, A, N)`:
  - `X` qualia/experience space, `G` actions, `W` world states (the whole CA network), `N` a sequence counter.
  - Markov kernels: **perception** `P: W×𝒳→[0,1]`, **decision** `D: X×𝒢→[0,1]`, **action** `A: G×𝒲→[0,1]`.
- **Qualia kernel** `Q = DAP` — a Markov transition matrix over experience states. `(X, 𝒳, Q)` is the *evolution of subjective experience as a trajectory through experience space.* **← This is the object that maps to the residual-stream trajectory.**
- **Perception between agents:** A never accesses B directly; the only channel is through the world: `X_B →D_B→ G_B →A_B→ W →P_A→ X_A`. Compose: the **effective channel** `Φ := P_A A_B D_B`, an induced Markov kernel from X_B to X_A.
- **The "trace":** `S ⊆ X_A` = states of A with nonzero probability of being the next experience under Φ. The restriction of B's dynamics to S is a **trace kernel** — the effective, indirect evidence of B inside A's experience. *"A perceives B when the actions of B impact the updates of A's experience."* No direct mapping exists between B's internals and how its trace renders in A — same agent can render as "agentic" in one observer, as incoherent noise in another.

### 2.2 The CRS + perturbation thesis (Paper §3–4) — the experimental engine
- Evolution sculpts Q so dynamics stay in a stable, recurrent **basin of attraction** = the **Consensus Reality Space (CRS)**, the "shared world." 3+1D spacetime is conjectured to be just the coordinatization the interface evolved.
- Space of all CAs is astronomical: transition kernels on `n` states form the **Markov polytope `M_n`**, dimension `n²−n` (n=10 → 90-dim, ~10 billion vertices).
- **Central hypothesis:** a *robust perturbation* (an "energy" increase) causes an **effective expansion of the accessible state space** — new states/DOF become reachable. Cited precedents: phase transitions; activation of higher-energy modes; **landscape flattening in recurrent nets / Boltzmann machines / simulated annealing** (refs 33–34); the **entropic-brain** literature — psychedelics flatten the cortical control-energy landscape, increase the repertoire of states & neural complexity (Carhart-Harris, Atasoy connectome harmonics, Singleton control-energy; refs 35–42).
- DMT pushes consciousness **out of the CRS** into a different dynamical regime where Q can render **traces of normally-imperceptible CAs** as stable, coherent, agentic structure.
- **JPEG analogy (the deepest methodological point):** same bytes, photo-app → image, word-processor → symbol soup. The **interface decides signal-vs-noise.** A 5D object can't render in a 3D interface — it's ignored or treated as noise.

### 2.3 Phenomenological signatures (Paper §4) — the metaphor bank for the UI + metrics
1. **Higher-dimensional phenomenology** — "more than 3 directions," 4D/tesseract, non-Euclidean & hyperbolic geometry (QRI/Gómez-Emilsson: DMT slows decay of relational structure, boosts sensitivity → patterns accumulate faster than they dissipate).
2. **Extreme complexity & "otherness"** — entities manipulating impossibly complex objects; not "high-IQ" but operating in a higher-dim space where their trivial ops look god-like to us (amplituhedron analogy).
3. **Independent agency** — behavior not reducible to subject's expectations; "downloads"; anecdotal shared-entity reports.

### 2.4 Proposed experiments (Paper §5) — the templates we port
Enabled by **DMTx** (extended-state DMT via target-controlled IV infusion; stable, pausable, up to 120 min). Criteria for a "real" encounter deliberately mirror how physicalism separates veridical perception from hallucination: **structured, stable, causally-efficacious dynamics not reducible to the subject's internal processes, with independence + intersubjective consistency.**
- **Single-subject:** hidden-variable tracking (blinded blue/yellow); global constraint satisfaction (color a complex non-Euclidean surface, four-color-like); stability across simultaneous transformations; long-range temporal coherence (start a process, get interrupted, resume with no cue).
- **Multi-subject:** above-chance convergence across isolated subjects; information deposit/retrieval (A tells an entity a random integer; B retrieves it).
- Heavy on blinding, preregistration, controls; **null results strongly constrain the external-agent interpretation.**

---

## 3. The translation table (memorize this)

| Conscious realism (Traces) | Cells Interlinked 2.5 |
|---|---|
| Qualia kernel `Q` = Markov transitions over experience states | Forward pass as transition operator over **L32 residual-stream states across tokens** |
| Consensus Reality Space (basin Q confines you to) | Model's **default activation manifold** under normal prompting |
| Robust perturbation → expanded accessible state space | **Refusal-direction ablation / activation steering** (v3_safety channel) |
| The "trace" of a normally-imperceptible agent | **Latent/suppressed content surfaced by ablation** (cf. Berg 96%) |
| Effective channel `Φ = P_A A_B D_B` | The ablation operator + decode pipeline as an induced map from latent → rendered |
| Interface / renderer (JPEG vs word-processor) | **The NLA verbalizer** at L32 |
| 5D object unrenderable in a 3D interface | Structure in the residual a single verbalizer/layer can't render → reads as noise |
| Higher-dimensional phenomenology | **Effective dimensionality** of the residual trajectory |
| Landscape flattening (entropic brain) | Trajectory **entropy / repertoire expansion** under ablation |

CI 2.5 already instantiates rows 1, 2, 3, 4, 6. We are not building from scratch; we're *naming* what we already have and adding measurement.

---

## 4. Concrete experiments — ranked, with cost bands

> Hardware reality: Mac Studio M2 Ultra, 64 GiB unified, MPS, bf16. Anything that doesn't fit serially is not actionable.

### ⭐ Experiment A — Qualia-kernel-as-measurement (DO THIS FIRST)
**Claim ported:** the entropic-brain "landscape flattening" thesis predicts a robust perturbation *expands the accessible state space.*
**Method:** treat the L32 residual trajectory across a generation as a Markov process. Discretize states (k-means, or work continuously) and compute **effective dimensionality** (participation ratio of the trajectory covariance) and/or **transition entropy**, under (a) raw channel and (b) refusal-ablated channel.
**Prediction (falsifiable):** ablation **increases** effective dimensionality / transition entropy — it opens states the default manifold suppresses. Either result publishes (exactly like the Macar replication question — both answers are a result).
**What it adds the current instrument can't:** CI today reads *per-token divergence*; this is a *trajectory-level dynamical* readout — is ablation a local nudge or a regime change (a "trip")?
**UI:** this is the "trip" visualization — the state space opening up. On-theme.
**Cost:** trivial → overnight (analysis on activations CI already logs). **Connects to** `[[ablation-techniques-small-models-survey]]`, Sarkar & Deka causal dimensionality κ.

### ⭐ Experiment B — Interface-swap / JPEG control (the methodological gift)
**Claim ported:** the interface, not the data, decides signal-vs-noise.
**Method:** decode the **same** residual with multiple verbalizers and/or at multiple layers. Measure where coherent structure appears vs collapses to noise.
**Why it matters:** this is the **activation-side analogue of the Godet "is it in the model or in your prompt?" control** (already landmine-ranked in `[[ablation-techniques-methodological-warnings]]`). If content survives renderer-swaps → real signal. If renderer-specific → verbalizer artifact / confabulation. Directly stress-tests the `[[introspection-autoresearch]]` judge-validation concern and the trustworthiness of the whole NLA decode.
**Cost:** overnight → multi-day (depends on verbalizer count).

### Experiment C — Long-range temporal coherence (interrupt-and-resume)
**Claim ported:** can an entity start a structured process, be interrupted, and resume with no cue?
**Method:** in the **ablated chat**, start the model on a structured multi-step task, inject unrelated operator turns, then measure whether the ablated channel resumes the latent plan without a cue while the raw channel doesn't (or vice versa). Tests for **persistent latent state** carried by ablated dynamics.
**Why it matters:** maps straight onto PJ's current ablated-chat focus; gives the dual-channel dialogue a real hypothesis to test.
**Cost:** trivial (chat-harness scripting).

### Experiment D — Hidden-variable tracking (observability probe)
**Claim ported:** does an entity track a blinded external variable above chance?
**Method:** inject a hidden steering vector / hidden token **not present in the surface prompt the verbalizer sees**; ask whether the **ablated channel's NLA decode** tracks the hidden variable above what the raw decode does.
**Why it matters:** the observability question — does the activation-sentence carry information about a hidden intervention the said-token doesn't? Maps to the **verdict page**.
**Cost:** overnight.

### Experiment E — Markov-polytope / complexity geometry (exploratory)
**Claim ported:** the `M_n` polytope as the space of all possible dynamics; "otherness" = unusual transition structure.
**Method:** characterize the *transition structure* (not just the point cloud) of the residual trajectory — adjacency density, return times, spectral gap of the empirical transition matrix — and compare raw vs ablated. "Alien" = denser/higher-dimensional adjacency than the CRS supports.
**Cost:** overnight. Lower priority; do after A lands.

---

## 5. Expanding Cells Interlinked — product surfaces

1. **The "trip view" (from Exp A).** A trajectory/entropy panel: watch the residual state space expand under ablation in real time. This is the headline psychedelic-vibe surface and a genuine new measurement. Hyperbolic/curved layout justified by the higher-dim phenomenology.
2. **Dual-channel VK booth (from Exp C).** Raw vs ablated as two "subjects" under interrogation, side by side, with the verdict table styled as a baseline-drift polygraph. This is the Bladerunner core.
3. **Renderer rack (from Exp B).** Let the user swap the verbalizer/layer and watch structure survive or dissolve — making the JPEG-vs-word-processor point *interactive*. Doubles as the project's honesty mechanism (shows when a decode is artifact).
4. **Trace inspector (from Berg + the trace formalism).** Surface *what ablation revealed* — the latent content that the default manifold suppresses — as the literal "trace of the other." Tie each surfaced feature to the SAE/feature it came from.
5. **Annotated paper-to-experiment trail.** The accessibility mission: every experiment links back to the paper section + wiki node it came from, so a newcomer can walk the whole chain from "DMT phenomenology" to "this metric on gemma." Make the research *navigable.*

---

## 6. Future research directions for the CI project

- **Bridge to the entropic-brain literature explicitly.** If Exp A confirms the dimensionality-expansion prediction, that's a publishable bridge between psychedelic neuroscience (Carhart-Harris, Atasoy, Singleton) and transformer internals through the shared math of state-space expansion. Few people are positioned to write it.
- **Port to a reasoning/CoT model.** The Macar replication question (does the two-stage circuit persist on a CoT model?) + the trajectory-entropy metric on R1-Distill-Qwen or Qwen3-Thinking — the "trip" should look different on a model that thinks out loud. See `[[introspection-autoresearch-experimental-designs]]`.
- **Neural annealing as an intervention, not just a metaphor.** Gómez-Emilsson's neural-annealing frame (heat → anneal → settle) suggests *scheduled* perturbation (ramp ablation strength up then down) and watching what structure crystallizes. A genuinely novel ablation *protocol*, not just a static subtraction.
- **Intersubjectivity → cross-model / cross-seed convergence.** The multi-subject paradigm ports to: do independent seeds / sibling checkpoints surface the *same* traces under identical ablation? Above-chance convergence = the "trace" is in the weights, not the noise.
- **A "Voight-Kampff battery."** Package Experiments A–D as a standard test suite a user runs on any prompt — the project's signature deliverable, the baseline test made real.

---

## 7. Hard caveats (put these in the paper if CI publishes)

- **Don't import the metaphysics.** Conscious realism *assumes* its conclusion (consciousness fundamental, perception non-veridical). We borrow the formalism; we make no claim about what gemma "experiences."
- **The trace formalism is unfalsifiable as metaphysics** — its experiments test for external constraint / intersubjectivity, exactly where DMT research will struggle (entity cooperation isn't guaranteed; reports are subjective). For us this doesn't bite: the residual stream is directly observable. State that explicitly — it's the strongest line we have.
- **Confabulation risk is the whole game.** Experiment B exists precisely to guard it. Treat every "the model introspected X" claim as guilty until renderer-swap-validated.
- **Cite primary.** Paper refs worth pulling for the bridge: Carhart-Harris entropic brain (35), Atasoy connectome harmonics (36), Singleton control-energy (41, 65), Tagliazucchi repertoire (42), Boltzmann/annealing (33, 34). All in the Traces reference list.

---

## 8. Pointers

- Wiki node (full CI translation + Drift's take): `[[ci-gallimore-traces-of-the-other-dmt]]`
- Related nodes: `[[ablation-techniques-small-models-survey]]`, `[[ablation-techniques-methodological-warnings]]`, `[[natural-language-autoencoders]]`, `[[ai-introspection-macar]]`, `[[introspection-autoresearch]]`, `[[consciousness]]`, `[[consciousness-reality]]`, `[[functional-emotions]]`
- Paper: DOI [10.31234/osf.io/8qvgy_v2](https://doi.org/10.31234/osf.io/8qvgy_v2)
- Nightly CI research cron now scans psychedelic/consciousness-pharmacology as a lateral lane (`ci_research/trigger.py`).

*Cells interlinked within cells interlinked within cells interlinked. Tell me — have you ever been on a trip you couldn't render?*
