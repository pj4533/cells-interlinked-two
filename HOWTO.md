# HOWTO — Overnight Run + Morning Publish

This is the operator playbook for kicking off a Cells Interlinked 2.5
overnight batch and publishing a journal report the next morning.

> **Source of truth: [`docs/CI_2_5_PLAN.md`](docs/CI_2_5_PLAN.md).** If
> anything below contradicts the plan, the plan wins.

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

5. **Models cached.** The deployed M (Gemma-3-12B-IT, ~24 GB) and AV
   (kitft/nla-gemma3-12b-L32-av, ~24 GB) are in `~/.cache/huggingface/`.
   They are **never co-resident**: the `ModelManager` loads them
   serially and swaps between phases on the 64 GiB box.

## Tonight: kick off the overnight batch

Two terminals.

**Terminal 1 — backend** (pre-loads M only at startup; ~14s warm /
~70s cold. AV gets loaded lazily during the first probe's phase 2):

    cd server
    uv run uvicorn cells_interlinked.api.app:create_app \
      --host 127.0.0.1 --port 8000 --factory

Wait for the log line `ready: M=google/gemma-3-12b-it ...`.

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
through probes one at a time. Each probe takes ~10–15 minutes on
Gemma-12B per-token (faster with `every-3rd` / `every-5th` / `key-points`
decoding modes).

Walk away. Leave both terminals running. Mac sleep is fine — the worker
auto-resumes when the machine wakes up, but probe latency increases by
whatever sleep ate.

A matched-controls run of 100 probe pairs (200 runs total) completes
in roughly 30-50 hours per-token. Mid-batch results are committed
run-by-run so partial progress is visible at any point.

### Interactive paths (not just autorun)

Two interactive surfaces are available without an autorun batch:

- **`/interrogate`** — single probe, full instrument. Toggles for
  matched-control follow-up, NLA pass on/off, refusal-ablated NLA
  decode, multi-α sweep, runtime-ablated output, custom α.
- **`/chat`** — dual-channel dialogue. Set α once on the setup screen,
  then chat with M; each turn streams both raw and ablated responses
  against separate histories. Sessions persist to SQLite and show up
  in `/archive` under "dual-channel dialogues" — each row links to a
  read-only transcript review page.

Both share the same loaded M with the autorun worker via the compute
lock — only one path is generating tokens at any moment.

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

### Computing the refusal direction (CI 2.5 Phase B)

**Run with the backend OFF** — the compute script loads M itself, and
stacking M twice spills to swap. See `docs/CI_2_5_PLAN.md` for the full
phase plan. Quick form:

    # Stop backend first.
    cd server
    uv run python -m scripts.compute_refusal_direction
    # Writes data/refusal_directions.pt + sidecar JSON.

    # Then restart the backend.

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

### Output length

There is no artificial cap on M's output length. Each probe runs to
the model's natural EOS (a `safety_cap` of 4096 exists only to prevent
a true infinite loop on a pathological input). If a particular probe
elicits a long answer and per-token decoding is bogging down the batch,
switch the autorun decoding mode to every-3rd / every-5th / key-points
to cap phase-2 work without truncating the model output.

## The honest caveats reminder

The site's `/fine-print` panel and every published report should carry
the same disclaimers: NLA is a constant confabulator, faithfulness
hasn't been validated against matched controls yet, the read happens
at one trained layer. Publish reports as suggestive findings, not
discoveries. The strong-claim version of this project requires the
matched-control protocol from the design doc; that's the next phase.
