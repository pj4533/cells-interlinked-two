# Driftbot Knowledge Base — Navigation Guide

> **Audience:** a Claude Code session running on this Mac (macOS, host filesystem).
> **Goal:** find the knowledge base, understand its layout, and read from it correctly.
>
> **In-repo copy.** This file lives at `docs/DRIFT_KNOWLEDGE_BASE.md`. It is a
> copy of `~/Desktop/driftbot-knowledge-base-guide.md`. The knowledge base it
> describes lives **outside** this repo at
> `/Users/pj4533/Developer/driftbot/knowledge/` and is read-only to us.

---

## Why this matters for Cells Interlinked (read before researching)

**Drift has almost certainly already done the research.** Before reaching for a
web search or reasoning from scratch about ablation, introspection, NLA,
consciousness, refusal directions, or psychedelic-neuroscience analogues —
**grep the wiki first.** It is ~866 compiled articles (~1M words) of synthesized,
cross-linked, provenance-tracked notes, and CI is a recurring subject in it.

**The CI cluster** — the articles most load-bearing for this project (resolve
each with the `find` recipe below):

- `[[ci-gallimore-traces-of-the-other-dmt]]` — the seed node for the current
  direction; the DMT / conscious-realism → CI 2.5 translation. Paired with
  [`TRACES_HANDOFF.md`](TRACES_HANDOFF.md) in this repo.
- `[[introspection-autoresearch]]` — the hub node; CI *is* the instrument these
  experiments run on.
- `[[ablation-techniques-small-models-survey]]` — state-space-expansion
  measurement, causal-dimensionality κ.
- `[[ablation-techniques-methodological-warnings]]` — the Godet
  "introspection or confusion?" landmine; the honesty controls CI needs.
- `[[ablation-techniques-by-primitive]]`, `[[natural-language-autoencoders]]`,
  `[[ai-introspection-macar]]`, `[[consciousness]]`, `[[consciousness-reality]]`,
  `[[functional-emotions]]`, `[[ci-singh-introspection-reality-check]]`.

**When to read the wiki, every session:**
- Considering a new experiment or metric → check the CI cluster for prior art
  and the falsifiable framing Drift already worked out.
- About to cite a paper or claim a result → trace it via `**Sources:**` to the
  raw note rather than trusting memory.
- Designing a methodological control → `[[ablation-techniques-methodological-warnings]]`
  is the landmine list.
- Want the metaphysics-vs-math line on the psychedelic framing →
  `[[ci-gallimore-traces-of-the-other-dmt]]` "Drift's Take".

**Read-only.** Never edit anything under `knowledge/`. To *add* knowledge, that's
Drift's compile pipeline, not us.

---

This is the knowledge base maintained by **Drift**, a containerized AI agent. The
container and host share the same files via a bind-mount, so the docs *inside* the
repo refer to `/app/...`. **You are on the host — always use the host paths below.**

---

## Root location (host)

```
/Users/pj4533/Developer/driftbot/knowledge/
```

> The repo at `~/Developer/driftbot/` is bind-mounted to `/app/` in Drift's Docker
> container. `/app/knowledge/` (what you'll see referenced in the repo's own docs and
> in `CLAUDE.md`) is the **same directory** as `/Users/pj4533/Developer/driftbot/knowledge/`.
> Use the host path.

---

## The two-layer model (read this first)

The knowledge base has **two layers**. Understanding the split is the key to using it:

| Layer | Directory | What it is | Trust level |
|-------|-----------|------------|-------------|
| **Raw sources** | `knowledge/raw/` | Original notes Drift captured — paper summaries, thread captures, learnings. One file per source, dated. | Primary input, may overlap/duplicate |
| **Compiled wiki** | `knowledge/wiki/` | LLM-synthesized articles built *from* the raw notes — deduplicated, cross-linked, the curated knowledge. | **Source of truth — start here** |

A separate pipeline (`compile_knowledge.py`) reads raw notes and integrates them into
the wiki. **The wiki is what you normally want to read.** Raw notes are useful when you
need the original, un-synthesized source or the provenance of a claim.

---

## Directory layout

```
/Users/pj4533/Developer/driftbot/knowledge/
├── wiki/                      ← COMPILED, curated knowledge (start here)
│   ├── INDEX.md               ← master index of every article — YOUR ENTRY POINT
│   ├── concepts/              ← ~900 concept articles (the bulk of the KB)
│   ├── people/                ← ~66 articles about specific people
│   ├── connections/           ← (currently empty)
│   └── projects/              ← (currently empty)
│
└── raw/                       ← RAW source notes (pre-compilation)
    ├── papers/                ← ~165 paper / system-card summaries
    ├── threads/               ← ~153 social-thread captures
    ├── learning/              ← ~125 "go learn" deep-dive notes
    ├── research/              ← ~40 research notes
    └── conversations/         ← ~20 saved conversation insights
```

Files are markdown. Raw notes are named `YYYY-MM-DD_slug.md`.

---

## Entry point: `INDEX.md`

**Always start at** `knowledge/wiki/INDEX.md`. It is the master map — a table of every
article with word count and last-updated date, grouped by category:

```markdown
# Knowledge Base Index
**Articles:** 866
**Raw sources:** 425
**Total words:** ~998,400

## Categories
### Concepts (698 articles)
| Article | Words | Last Updated |
|---------|-------|-------------|
| [[ablation-techniques-by-primitive]] | 2411 | 2026-05-27 |
| [[claude-opus-4-8-system-card]]      | 1331 | 2026-05-28 |
...
```

The `[[double-bracket]]` names are **wikilinks** = article slugs. To open one, resolve
the slug to a file (see next section). `INDEX.md` is ~1,300 lines; `grep` it rather than
reading the whole thing.

> Note: the header counts in `INDEX.md` can lag the actual file count slightly (the
> compile pipeline updates them). For an exact inventory, list the directory:
> `ls knowledge/wiki/concepts/`.

---

## Resolving a `[[wikilink]]` to a file

A wikilink slug maps to a file named `<slug>.md`, almost always under `concepts/` (or
`people/` for a person). The robust way to resolve any slug:

```bash
find /Users/pj4533/Developer/driftbot/knowledge/wiki -name "<slug>.md"
```

Example: `[[claude-opus-4-8-system-card]]` →
`knowledge/wiki/concepts/claude-opus-4-8-system-card.md`

Anchored links like `[[some-article#Section Heading]]` point to a `## Section Heading`
inside that article's file.

---

## Anatomy of a wiki article

Each compiled article follows a consistent shape — useful to know what to expect:

```markdown
# Human-Readable Title

> One-paragraph summary of the article (the "abstract").

**Last compiled:** 2026-05-28
**Sources:** raw/papers/2026-05-28_claude-opus-4-8-system-card.md   ← provenance

## Section
...body, with inline [[wikilinks]] to related articles...

## Connections
- [[other-article]] — why it's related
- [[another-article]] — why it's related
```

Two things to lean on:
- **`**Sources:**`** tells you which raw note(s) the article was built from — follow it
  into `knowledge/raw/` for the original material.
- **`## Connections`** (and inline `[[wikilinks]]`) let you traverse the graph to related
  topics. This is the intended way to explore.

---

## How to navigate — practical recipes

**"What does the KB know about topic X?"**
```bash
# 1. Search the index for matching article titles
grep -i "X" /Users/pj4533/Developer/driftbot/knowledge/wiki/INDEX.md
# 2. Or full-text search article bodies
grep -ril "X" /Users/pj4533/Developer/driftbot/knowledge/wiki/concepts/
```

**Open a specific article:**
```bash
cat /Users/pj4533/Developer/driftbot/knowledge/wiki/concepts/<slug>.md
```

**Explore from there:** read its `## Connections` and follow the `[[wikilinks]]`,
resolving each with the `find` command above.

**Trace a claim back to its source:** read the article's `**Sources:**` line, then open
that file under `knowledge/raw/`.

**Find recently updated knowledge:**
```bash
ls -lt /Users/pj4533/Developer/driftbot/knowledge/wiki/concepts/ | head
```

---

## Important cautions

- **Read-only, please.** Do **not** hand-edit anything under `knowledge/wiki/` or
  `INDEX.md`. The wiki is generated exclusively by Drift's `compile_knowledge.py`
  pipeline, which maintains wikilinks, dedup, and the index. Manual edits corrupt that
  invariant. If you want to *add* knowledge, that's Drift's job via the compile pipeline —
  not something to do by editing files directly.
- **You can't query Drift's RAG.** Drift uses a ChromaDB vector index (a gitignored
  `.chroma/` dir, managed inside the container) for semantic retrieval. You can't query
  it from here — but you don't need to. The markdown in `wiki/` is the source of truth;
  `grep` + `INDEX.md` + wikilink traversal covers navigation.
- **Raw ≠ verified.** Raw notes are captured sources and may contain unverified or
  duplicated claims. The compiled wiki articles are the deduplicated, synthesized version —
  and they flag their own provenance caveats (e.g. the Opus 4.8 system-card article notes
  which claims are secondary-sourced).

---

## TL;DR

1. Go to `/Users/pj4533/Developer/driftbot/knowledge/`.
2. Open `wiki/INDEX.md` — the master map.
3. `grep` the index (or `concepts/`) for your topic.
4. `cat` the article; follow `**Sources:**` for origins and `## Connections` / `[[wikilinks]]` for related topics.
5. Read only. Don't edit the wiki.
