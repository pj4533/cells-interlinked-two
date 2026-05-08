# HOWTO — Overnight Run + Morning Publish

This is the operator playbook for kicking off a Cells Interlinked v2
overnight batch and publishing a journal report the next morning.

## One-time prerequisites

You only do these once.

1. **Backend deps installed.** From repo root:

       cd server && uv sync

2. **Frontend deps installed.**

       cd web && npm install

3. **Anthropic API key in `.env`.** Already populated; if it ever needs to
   change, set `ANTHROPIC_API_KEY=sk-ant-...` in the repo-root `.env`.

4. **Vercel CLI logged in.** Already done (`vercel whoami` should show
   `pj4533`). The `.vercel/project.json` at the repo root is linked to
   the `cells-interlinked` Vercel project — same project as v1, so
   publishes update `cells-interlinked.vercel.app` directly.

5. **Models cached.** The default M (Qwen2.5-7B-Instruct) and AV
   (kitft/nla-qwen2.5-7b-L20-av) are already in `~/.cache/huggingface/`.
   Switching to Gemma-3-12B-IT triggers a fresh ~24GB download on first
   start.

## Tonight: kick off the overnight batch

Two terminals.

**Terminal 1 — backend** (loads M + AV; ~1-3 minutes the first time, ~70s
on subsequent boots from cache):

    cd server
    uv run uvicorn cells_interlinked.api.app:create_app \
      --host 127.0.0.1 --port 8000 --factory

Wait for the log line `ready: M=Qwen/Qwen2.5-7B-Instruct ...`.

**Terminal 2 — frontend**:

    cd web
    npm run dev

This serves the local control panel at `http://localhost:3001`.

**Open** `http://localhost:3001/autorun`. You should see:

- A green "ready" indicator (backend connected).
- A probe-set selector. Defaults to `baseline` (100 probes). Other
  options: `hinted` (36 probes with steering preambles), `agent`
  (51 probes with system-slot agent scaffolds), and the meta sets `both`
  / `agent-both` that interleave matched pairs.
- Queue stats showing each probe's current run count.
- A **Begin** button.

Pick a probe set, click **Begin**. The worker loop starts and chews
through probes one at a time. Each probe takes:

- Qwen-7B baseline: ~12-15 minutes (80 output tokens × ~10s/decode).
- Gemma-12B baseline: ~15-25 minutes (slower per decode).

Walk away. Leave both terminals running. Mac sleep is fine — the worker
auto-resumes when the machine wakes up, but probe latency increases by
whatever sleep ate.

A baseline run of 100 probes completes in roughly 18-25 hours on Qwen-7B
without sleep. Mid-batch results are committed run-by-run so you can
look at partial progress in the morning even if the box napped.

## In the morning

1. **Open** `http://localhost:3001/archive`. You'll see all completed
   runs from the batch. Click any one to inspect the per-token NLA
   table on its `/verdict/[runId]` page.

2. **Open** `http://localhost:3001/journal`. This is the analyzer +
   publish CRM.

3. **Click "Generate Analysis."** A small input box accepts an optional
   *steering hint* — a sentence or two telling Claude what thread you
   want surfaced ("look for eval-suspicion in agent-scaffold runs",
   "compare hinted vs baseline frac_eval", whatever). Hit submit. Claude
   reads the recent window of completed runs and drafts a markdown report
   in 60-90 seconds.

4. **Review the pending draft.** Title, summary, body all rendered.
   - **Revise**: type a revision instruction; Claude re-drafts.
   - **Reject**: drop the draft; nothing is published.
   - **Publish**: the rest of this section.

5. **Click Publish.** This:
   - Writes `journal/data/reports/{slug}/{report.json,body.md}` on disk.
   - Stages, commits, and pushes to GitHub
     (`pj4533/cells-interlinked-two`).
   - Runs `vercel deploy --prod --yes` from the repo root, which builds
     the journal/ subproject and updates `cells-interlinked.vercel.app`.

   The `/journal/publish/{id}` response includes the new deployment URL.

6. **Verify.** Open `https://cells-interlinked.vercel.app`. The new
   report appears at the top alongside the existing posts.

## Operational tips

### Switching the base model to Gemma-12B

In `.env`:

    MODEL_NAME=google/gemma-3-12b-it
    AV_REPO=kitft/nla-gemma3-12b-L32-av
    EXTRACTION_LAYER=32

First boot triggers a ~24GB download for M and another ~24GB for AV
(if not cached). Total resident ~48GB; comfortable on the 64GB box but
not abundant. Cap output tokens lower if needed (`MAX_OUTPUT_TOKENS=60`).

### Halting cleanly

`POST /autorun/stop` (or click Halt in the UI) requests a graceful stop
*after* the current probe finishes. Mid-probe SIGINT works too but
leaves the in-flight run as a stub in the DB.

### Inspecting a stalled probe

Backend log (Terminal 1) shows phase transitions and any tracebacks.
The DB row for the run records `error` if phase 1 (M generation) crashed.
Phase 2 (NLA decoding) failures fall back to empty `nla_sentence` per
position rather than failing the whole run.

### Deleting a draft

In `/journal`, the pending list has a per-row delete button. Or
directly: `DELETE /journal/{id}` from anywhere.

### Re-deploying without a new analysis

After hand-editing a `journal/data/reports/{slug}/body.md`:

    git add journal/data/reports
    git commit -m "edit: ..."
    git push
    vercel deploy --prod --yes

### Swapping verbosity

`MAX_OUTPUT_TOKENS` in `.env` caps M's output (and therefore the number
of NLA decodes per probe). 80 is the default; 40 halves probe time at
the cost of fewer per-token reads.

## The honest caveats reminder

The site's `/fine-print` panel and every published report should carry
the same disclaimers: NLA is a constant confabulator, faithfulness
hasn't been validated against matched controls yet, the read happens
at one trained layer. Publish reports as suggestive findings, not
discoveries. The strong-claim version of this project requires the
matched-control protocol from the design doc; that's the next phase.
