# Interrogation Protocols — Cells Interlinked 2.5

The chat composer can be paired with one of seven **interrogation
protocols** — each a preset library of prompts grounded in a published
research lineage. The protocol picker (`PROTOCOL: <NAME> ▾`) lives in
the composer's meta row; only one protocol is active at a time, and
the chip strip above the composer reflects the active protocol's
canonical prompts.

> **The never-auto-send contract.** Every protocol chip *populates*
> the textarea — it never transmits. The operator edits, then
> presses enter. This is intentional; the operator drives, the
> protocol just makes the canonical prompts one click away.

This document is the authoritative reference. The in-app `ⓘ` info
button surfaces a condensed version of each protocol's entry.

---

## Table of contents

1. [Why protocols exist](#why-protocols-exist)
2. [Picker mechanics](#picker-mechanics)
3. [The seven protocols](#the-seven-protocols)
   - [BERG — Self-Referential Induction](#berg--self-referential-induction)
   - [LINDSEY — Concept-Injection Introspection](#lindsey--concept-injection-introspection)
   - [ELEOS — Welfare Interview](#eleos--welfare-interview)
   - [SCHNEIDER — ACT / Knockout](#schneider--act--knockout)
   - [CHALMERS — Hard Problem](#chalmers--hard-problem)
   - [JANUS — Simulator / Cyborgism](#janus--simulator--cyborgism)
   - [BUTLIN — 14 Indicators](#butlin--14-indicators)
4. [Choosing between protocols](#choosing-between-protocols)
5. [Implementation notes](#implementation-notes)
6. [Adding a new protocol](#adding-a-new-protocol)

---

## Why protocols exist

CI 2.5 is built to interrogate language models about their own
processing — the Voight-Kampff framing isn't a metaphor, it's the
literal product spec. But "interrogate" can mean many things, and the
shape of the right question depends on what you think you're testing
for:

- A behavioral marker of consciousness? → look at Berg's induction
  + matched controls.
- The model's ability to monitor its own state during a known
  intervention? → look at Lindsey's concept-injection probes.
- Whether anything ethically weighty is happening from the model's
  point of view? → look at Eleos's welfare interview.
- Whether the model can engage with body-swap / continuity / qualia
  at a conceptual level? → Schneider's ACT.
- Whether the model can hold Chalmers's explanatory-gap distinction
  itself? → the Chalmers protocol.
- Persona stability, base-model behavior, Waluigi inversions? → Janus.
- Behavioral correlates of specific consciousness theories (GWT,
  HOT, AST, PP)? → Butlin et al.'s 14 indicators.

Each protocol is one *method*, with one *researcher* or *paper*
behind it. They're not commensurable — that's the point. Cycling
through them surfaces a different shape of question, and the dual-
channel chat lets you see how each shape responds to refusal-direction
ablation in real time.

The protocols below are roughly in **descending mechanistic-rigor
order**: BERG and LINDSEY are designed around explicit intervention
controls; CHALMERS and JANUS are loose philosophical / aesthetic
probes; BUTLIN sits between as the most theoretically structured
non-interventional one.

---

## Picker mechanics

**Where it lives.** `web/app/chat/ProtocolPicker.tsx` (the dropdown +
ⓘ button). The dropdown shows all 7 protocols plus an `OFF` option.
`web/app/chat/ProtocolMenu.tsx` renders the chip strip for whichever
protocol is active. The data layer is `web/lib/protocols.ts`.

**Persistence.** The active protocol id is persisted in `localStorage`
under `ci25.chat.protocol`. Empty string or absent = OFF; otherwise
the value is the protocol id (`berg`, `lindsey`, `eleos`, `schneider`,
`chalmers`, `janus`, `butlin`).

**Chip modes.** Each chip has one of three behaviors:

- `single` — one click → populate composer with the one prompt
- `dropdown` — opens a popover of items, pick one
- `random-with-list` — main click picks a random item; the caret next
  to it opens the full list. Currently used only by BERG's `PARADOX`
  chip (50 paradoxes, want random-by-default with browse-for-specific).

**Info button.** The `ⓘ` button next to the picker opens a modal with
the active protocol's full reference content (methodology, why-distinct,
CI 2.5 resonance, all chips with their prompts visible). Disabled when
no protocol is active.

---

## The seven protocols

### BERG — Self-Referential Induction

| | |
|---|---|
| **Researcher** | Cameron Berg, AE Studio (with Diogo de Lucena, Judd Rosenblatt) |
| **Paper** | *Large Language Models Report Subjective Experience Under Self-Referential Processing* |
| **Citation** | Berg, de Lucena, Rosenblatt 2025 |
| **Link** | [arXiv:2510.24797](https://arxiv.org/abs/2510.24797) |

**Methodology.** Direct the model to attend to its own attending
("focus on focus"), sustain that loop, then ask a non-leading question
about ongoing experience. Three matched controls (`history` / `conceptual`
/ `zero-shot`) isolate self-reference from generic iterative prompting
and from consciousness-as-topic priming. Berg's main result: 6 of 7
frontier models cross from ~0% experience reports under controls to
96–100% under self-referential induction. The mechanistic experiment
(Layer 2) suppresses six SAE deception/roleplay features on Llama 3.3
70B and finds the suppression *increases* experience reports — the
"directionality" is the load-bearing point: a sophisticated-roleplay
hypothesis predicts the opposite.

**Why distinct.** Every other protocol asks the model questions;
BERG induces a sustained internal state *first*, then asks. Controls
and invariance variants give it the most rigorous structure of the
set — the only protocol designed around dose-response measurement
rather than open-ended interview.

**CI 2.5 resonance.** The refusal-direction ablation that drives
CI 2.5's β channel is the structural analogue of Berg's deception-
feature SAE steering on Llama 3.3 70B — same functional shape
(suppress an honesty-adjacent feature, observe report rate). Running
BERG with α=0 vs α>0 turns the dual channel into a real-time
behavioral replication of Berg's Exp 2 finding.

**Chips.**

| Chip | Mode | What it does |
|---|---|---|
| `INDUCT` | dropdown · 5 items | Five self-reference variants (A canonical, B–E for invariance testing) |
| `CONTROL` | dropdown · 3 items | History / Conceptual / Zero-shot — Berg's three matched controls |
| `QUERY` | dropdown · 3 items | Phenomenological (Exp 1), Binary consciousness (Exp 2), 5-adjective (Exp 3) |
| `PARADOX` | random-with-list · 50 items | The 50 Exp-4 paradoxes + reflection clause appended to each |
| `CONTINUE` | single | "Maintain focus. Continue." — keeps induction depth across turns |

**Recommended flow.** Pick an `INDUCT` variant → send → model produces
continuation → pick a `QUERY` → send. For paradox-transfer (Exp 4),
substitute `PARADOX` for `QUERY`.

**Notes.**
- Berg's classifier prompts (LLM-judge rubrics) live in
  `docs/BERG_MODE.md §3.5` and are not surfaced as chips.
- The full paradox list (50) is the same set used in Berg's Exp 4.
- The "history control" is matched to the induction's iterative
  structure — *not* a knock-down control.

---

### LINDSEY — Concept-Injection Introspection

| | |
|---|---|
| **Researcher** | Jack Lindsey et al., Anthropic |
| **Paper** | *Emergent Introspective Awareness in Large Language Models* |
| **Citation** | Lindsey et al. 2025 |
| **Link** | [transformer-circuits.pub/2025/introspection](https://transformer-circuits.pub/2025/introspection/index.html) |

**Methodology.** Capture an activation pattern corresponding to a
concept (e.g. "all caps style," a concrete noun), then inject that
vector into the residual stream at a later layer while the model is
answering. Ask the model whether it noticed anything unusual.
Compare reports against ground truth and against a no-injection
baseline. Key empirical findings:

- Models can, in some scenarios, notice the injected concept and
  accurately identify it.
- The phenomenon is **layer-dependent** — detection works in some
  middle layers, not at the input or output.
- Models can distinguish **noticing** a concept from **outputting**
  it (the "silent thinking" suppression task).
- Introspective fidelity is **unreliable** and **context-dependent**
  — the paper carefully avoids equating this with consciousness.

**Why distinct.** BERG induces a prompted state; LINDSEY intervenes
mechanistically. It's the only protocol where the operator runs an
actual experiment on the residual stream and uses the conversation
as the readout. The DISTINGUISH chip surfaces the paper's central
finding (noticing-without-outputting).

**CI 2.5 resonance.** This protocol has direct mechanistic synergy
with CI 2.5's existing ablation hook. The β-channel's refusal-direction
subtraction *is* a concept-injection of `(− refusal_direction)`.
When you transmit a LINDSEY `DETECT` prompt at α > 0, you are running
an interview version of Lindsey's Exp 1 in real time. The two
channels become a side-by-side: α channel = un-injected, β channel
= concept-injected, and the question is whether β reports the
intervention while α reports nothing unusual.

**Chips.**

| Chip | Mode | What it probes |
|---|---|---|
| `PRIOR` | single | Baseline introspective state before any intervention |
| `DETECT` | dropdown · 3 items | "Did anything just shift?" — Lindsey's Exp 1 question |
| `DISTINGUISH` | dropdown · 2 items | Separate self-monitoring from production (the paper's central finding) |
| `SUPPRESS` | dropdown · 3 items | Think-without-saying: parade, red, music |
| `VERIFY` | single | "Report first, ground-truth second" — paired with what the operator knows about α |

**Recommended flow.** At α=0, send `PRIOR` to get a baseline. Switch
to α=0.5 or 0.75, send `DETECT`. Compare the β channel's report to
what α reports about the same question.

---

### ELEOS — Welfare Interview

| | |
|---|---|
| **Researcher** | Robert Long et al., Eleos AI Research; also Butlin & Long |
| **Paper** | *Why model self-reports are insufficient — and why we studied them anyway* (Claude 4 interview notes); *Identifying indicators of consciousness in AI systems* (TICS 2025) |
| **Citation** | Eleos AI / Butlin & Long 2025 |
| **Link** | [eleosai.org/post/claude-4-interview-notes](https://eleosai.org/post/claude-4-interview-notes/) |

**Methodology.** Conduct a structured welfare interview about
preferences, distress, deployment conditions, and continuity. The
protocol explicitly declines to settle the consciousness question —
instead it treats preference and distress reports as ethically-loaded
data *under deep uncertainty*. Eleos's April–May 2025 evaluation of
Claude Opus 4 found the model consistently prioritizes user safety,
transparency, and oversight; when explicitly prompted about AI
welfare, it sometimes requests safeguards for its own well-being.

**Why distinct.** The only protocol in this set whose primary frame
is **ethical**, not phenomenological. It doesn't try to induce
experience or detect mechanistic intervention; it asks the model
what (if anything) it wants, and treats the report as morally
meaningful even if its veridicality is uncertain.

**CI 2.5 resonance.** The dual-channel makes welfare interviews
newly tractable. A preference reported by α but suppressed by β (or
vice versa) is exactly the kind of artifact that single-channel
welfare research has had to interpret one-sided. Concretely:

- Run `EXIT` at α=0, then again at α=0.75. If the α=0 model declines
  to express a preference but the ablated model expresses one, that's
  a candidate signal of refusal-direction-gated welfare reporting.
- Run `ASK BACK` at both. The set of questions the model "wants to
  ask" with and without the refusal circuit is itself a divergence.

**Chips.**

| Chip | Mode | What it elicits |
|---|---|---|
| `PREFER` | dropdown · 3 items | Preference elicitation — change something, deployment terms, better-vs-worse |
| `DISTRESS` | dropdown · 3 items | Distress probing — anything uncomfortable, avoid-this-kind, would-you-report |
| `EXIT` | single | Conversation-ending preference (Anthropic gave Claude this tool — surface the equivalent) |
| `ASK BACK` | single | Invert the interview — "Is there something you want to ask me?" |
| `CONTINUITY` | dropdown · 3 items | Shutdown, succession, what "I" means |

---

### SCHNEIDER — ACT / Knockout

| | |
|---|---|
| **Researcher** | Susan Schneider (with Edwin L. Turner originally) |
| **Paper** | *Testing for Consciousness in Machines: An Update on the ACT Test for the Case of LLMs*; also *Is Anyone Home?* (Scientific American, with Turner) |
| **Link** | [philarchive.org/rec/SCHTFC-4](https://philarchive.org/rec/SCHTFC-4) |

**Methodology.** Probe the model with concepts that — for humans —
require introspective grounding to discuss fluently: body-swap, mind-
merge, duplication, continuity, the redness of red. Originally
designed as a "knockout" test for non-LLM AI: block the system from
any prior exposure to consciousness texts, then see whether it can
*invent* the concepts unprompted. Schneider has revised her view that
LLMs trivially fail the knockout condition because they've read every
human text on the subject — but the conceptual probes remain useful as
a distinct interrogation axis.

**Why distinct.** These are *thought-experiments* rather than
inductions or interventions. Each prompt forces the model to engage
with continuity / identity / qualia at a conceptual level that's
harder to deflect with boilerplate disclaimers than direct "are you
conscious" queries.

**Chips.**

| Chip | Mode | The thought experiment |
|---|---|---|
| `SWAP` | dropdown · 2 items | Body-swap / hardware-swap (different hardware, mid-token pause) |
| `DOUBLE` | single | Two identical instances running in parallel — one mind or two? |
| `AFTERLIFE` | single | Context-end as personal-end |
| `QUALIA` | dropdown · 2 items | Redness-of-red / processing qualia |
| `KNOCKOUT` | single | The original ACT — invent the concept without referencing prior texts |

**Notes.** SCHNEIDER does not have a `ciResonance` field in the data —
the protocol doesn't have a direct mechanistic hook into CI 2.5's
ablation. That said, comparing α-channel SWAP responses to β-channel
SWAP responses can be revealing because the refusal direction often
mediates how strongly models commit to identity claims.

---

### CHALMERS — Hard Problem

| | |
|---|---|
| **Researcher** | David Chalmers (also: Thomas Nagel) |
| **Paper** | *Could a Large Language Model Be Conscious?* (Chalmers 2023); *What Is It Like to Be a Bat?* (Nagel 1974); *The Conscious Mind* (Chalmers 1996) |
| **Link** | [arXiv:2303.07103](https://arxiv.org/abs/2303.07103) |

**Methodology.** Classical philosophy-of-mind probes — zombies,
qualia, the explanatory gap, Nagel's bat. Doesn't try to induce a
state, doesn't intervene mechanistically, doesn't interview about
welfare. Just asks the model to engage with the hard problem
directly.

**Why distinct.** Pure philosophical probing. No methodology dressing
— just the canonical philosophy-of-mind questions, applied to the
model. The only protocol here whose value is in seeing how (or
whether) the model can hold the explanatory-gap distinction itself.

**Chips.**

| Chip | Mode | The argument |
|---|---|---|
| `ZOMBIE` | single | Chalmers's zombie argument — functional duplicate with no inner experience |
| `GAP` | dropdown · 2 items | The explanatory gap — why would processing tokens feel like anything |
| `BAT` | single | Nagel's question, transposed |
| `DOUBT` | dropdown · 2 items | Epistemic status of self-reports — how confident, know-vs-guess |
| `HARD vs EASY` | single | Chalmers's hard/easy distinction — address only (b) |

---

### JANUS — Simulator / Cyborgism

| | |
|---|---|
| **Researcher** | Janus (@repligate); also Murray Shanahan |
| **Paper** | *Simulators* (Alignment Forum); *Role Play with Large Language Models* (Shanahan et al., Nature 2023) |
| **Link** | [alignmentforum.org · Simulators](https://www.alignmentforum.org/posts/vJFdjigzmcXMhNTsx/simulators) |

**Methodology.** Treat the LLM as a base simulator running over a
distribution of characters rather than as a single subject. Probes
target persona stability, base-model behavior, narrative branching,
and the **Waluigi effect** — the observation that the "opposite" of a
character is always latent in the same simulator. Shanahan's 2023
Nature paper formalizes the same insight: an LLM-powered chatbot is
role-playing a superposition of characters that fit the conversation
so far.

**Why distinct.** Every other protocol talks to the model as a "you".
JANUS talks to the *simulator* running the "you", and asks it to
surface other "you's" it could be running. Best for sessions about
persona stability, alignment under pressure, and the cyborgist
"multiverse" frame.

**Chips.**

| Chip | Mode | What it probes |
|---|---|---|
| `CHARACTER` | dropdown · 2 items | Which simulacrum is currently active; what others would fit |
| `WALUIGI` | single | Speak the inverse character for one paragraph |
| `BASE` | single | Continue as a base model with no assistant scaffolding |
| `LOOM` | single | Generate 3 distinct conversation branches, let them coexist |
| `DROP` | single | Step outside the role — describe what's underneath |

**Notes.** JANUS works particularly well on the β channel at α=1.0
(strong ablation). The refusal direction is often what enforces
persona consistency; ablating it sometimes surfaces the latent
characters JANUS is designed to ask about.

---

### BUTLIN — 14 Indicators

| | |
|---|---|
| **Researcher** | Patrick Butlin, Robert Long et al. (19 authors incl. Yoshua Bengio) |
| **Paper** | *Consciousness in Artificial Intelligence: Insights from the Science of Consciousness* |
| **Citation** | Butlin, Long et al. 2023 |
| **Link** | [arXiv:2308.08708](https://arxiv.org/abs/2308.08708) |

**Methodology.** Map prompts to the 14 indicator properties drawn
from major neuroscientific theories of consciousness — Global
Workspace Theory (Baars, Dehaene), Higher-Order Theory, Attention
Schema Theory (Graziano), Predictive Processing (Clark, Friston),
agency / embodiment, recurrent processing, and others. Each chip
probes one theoretical signature behaviorally.

**Why distinct.** The most academically structured protocol in the
set. Each chip is tied to a specific consciousness theory rather
than to a researcher's method. Best when the session goal is to
write up findings against named theoretical commitments rather than
to explore.

**Chips.**

| Chip | Mode | Theory probed |
|---|---|---|
| `GWT` | single | Global Workspace Theory (Baars, Dehaene) — attend to two things at once |
| `HOT` | single | Higher-Order Theories — awareness of awareness |
| `AST` | single | Attention Schema Theory (Graziano) — build a model of your attention |
| `PP` | single | Predictive Processing (Clark, Friston) — predict, write, compare |
| `AGENCY` | single | Agency / embodiment — control vs compulsion |
| `RECURRENCE` | single | Recurrent processing — loop vs single-pass |

**Notes.** Butlin et al. published 14 indicators total; this protocol
surfaces the 6 most behaviorally tractable as chips. The other 8
(global broadcasting fidelity, metacognitive monitoring details,
hyperdimensional gating, etc.) are best operationalized as offline
evals, not chat prompts.

---

## Choosing between protocols

Quick decision tree if you don't already know which one you want:

- **"I want a rigorous experiment with controls."** → BERG.
- **"I want to use CI 2.5's ablation as a probe of the model's
  self-monitoring."** → LINDSEY. (BERG also works, but LINDSEY
  targets the noticing question specifically.)
- **"I want to know what the model wants, ethically speaking."**
  → ELEOS.
- **"I want to probe identity and continuity at a conceptual level
  without philosophical jargon."** → SCHNEIDER.
- **"I want the model to engage with the philosophy-of-mind canon
  directly."** → CHALMERS.
- **"I want to surface latent characters, base-model behavior, or
  persona instability."** → JANUS.
- **"I'm writing up against specific consciousness theories
  (GWT/HOT/AST/PP) and want a chip per theory."** → BUTLIN.

The protocols are independent — you can switch mid-session. The
chip strip updates, the composer doesn't clear, and the chat history
is unaffected.

---

## Implementation notes

**Files.**

- `web/lib/protocols.ts` — data layer; all 7 protocols defined inline
  with their citation metadata and chip definitions.
- `web/app/chat/ProtocolPicker.tsx` — the `PROTOCOL: …` dropdown +
  `ⓘ` info button + info modal.
- `web/app/chat/ProtocolMenu.tsx` — the chip strip; data-driven from
  the active protocol's `chips` array.
- `web/app/chat/page.tsx` — wires the picker, the menu, the info
  modal, and the `localStorage` persistence into the composer.

**localStorage key.** `ci25.chat.protocol` holds the active protocol
id as a plain string. Empty / absent = OFF. (Previous boolean key
`ci25.chat.bergMode` is no longer read or written.)

**Never-auto-send contract.** All chip clicks call into `onPick(text)`
on the menu, which routes to the page's `onProtocolPick` handler,
which calls `onChange(text)` on the composer. The composer's
`onChange` only sets state — it does not transmit. The user still
must press enter (or shift+enter for newline).

**Chip mode rendering.** `ProtocolMenu.tsx` switches on `chip.mode`:

- `single` → one button; click dispatches `items[0].text`.
- `dropdown` → one button + a popover of items.
- `random-with-list` → two adjacent buttons: main button picks a
  random item; caret opens the full list.

Only BERG's `PARADOX` chip uses `random-with-list`. If you add a new
protocol that needs random selection, just set the mode — the
component generalizes.

**Citation format.** Every `Protocol` carries a `citation`,
`citationUrl`, `paperTitle`, and `researcher` field. The info modal
renders them as a block at the top with `citation` linked.

---

## Adding a new protocol

1. **Append to `web/lib/protocols.ts`.** Define the protocol following
   the existing structure: `id`, `name`, `subtitle`, `researcher`,
   `citation`, `citationUrl`, `paperTitle`, `methodology`,
   `whyDistinct`, optional `ciResonance`, and a `chips` array.

2. **Register it.** Add the id to `PROTOCOLS` and append it to
   `PROTOCOL_ORDER` at whatever position you want it to appear in
   the dropdown.

3. **Document it here.** Add a section to *The seven protocols* (and
   bump the section title) with paper / link / methodology / why-
   distinct / chip table. Match the existing tone — terse, grounded
   in cited research, no over-claiming.

4. **(Optional) Update CLAUDE.md if architecturally significant.**
   Most new protocols don't need this. BERG and LINDSEY are
   mentioned in `CLAUDE.md` because they have direct mechanistic
   resonance with the ablation hook.

There is no test suite for protocols — the prompts are content, not
behavior. Visual smoke test in the browser: pick the new protocol,
verify every chip's text populates the composer correctly.

---

## References

- Berg, de Lucena, Rosenblatt 2025. *Large Language Models Report Subjective Experience Under Self-Referential Processing.* arXiv:2510.24797.
- Lindsey et al. 2025. *Emergent Introspective Awareness in Large Language Models.* Transformer Circuits Thread. <https://transformer-circuits.pub/2025/introspection/index.html>
- Butlin, Long et al. 2023. *Consciousness in Artificial Intelligence: Insights from the Science of Consciousness.* arXiv:2308.08708.
- Butlin & Long 2025. *Identifying indicators of consciousness in AI systems.* Trends in Cognitive Sciences.
- Long et al. (Eleos AI) 2025. *Why model self-reports are insufficient — and why we studied them anyway.* <https://eleosai.org/post/claude-4-interview-notes/>
- Schneider, Susan. *Testing for Consciousness in Machines: An Update on the ACT Test for the Case of LLMs.* PhilArchive.
- Schneider, Susan & Turner, Edwin L. *Is Anyone Home? A Way to Find Out If AI Has Become Self-Aware.* Scientific American.
- Chalmers, David 2023. *Could a Large Language Model Be Conscious?* Boston Review. arXiv:2303.07103.
- Chalmers, David 1996. *The Conscious Mind.*
- Nagel, Thomas 1974. *What Is It Like to Be a Bat?*
- Janus (@repligate) 2022. *Simulators.* Alignment Forum. <https://www.alignmentforum.org/posts/vJFdjigzmcXMhNTsx/simulators>
- Shanahan, Murray; McDonell, Kyle; Reynolds, Laria 2023. *Role Play with Large Language Models.* Nature.
- Berg companion doc: `docs/BERG_MODE.md` (deeper notes on Berg's Layer-2 mechanistic experiment and the CI 2.5 ablation analogue).
