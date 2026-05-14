# Cells Interlinked 2.5 — frontend

Next.js 16 + React 19 + Tailwind v4 + Zustand + Framer Motion control
panel for the local interrogation backend.

## Run

    npm install
    npm run dev

Serves on **port 3001** (port 3000 is reserved for the host's Drift
Docker container). Talks to the FastAPI backend on **port 8000** —
launch that first from `../server`.

The dev server is LAN-accessible: `http://your-host.local:3001` and
RFC1918 private IPs are all allow-listed in `next.config.ts` so
phones / iPads / a second laptop can hit the same instance.

## Pages

| Route | What it does |
| --- | --- |
| `/` | Landing. |
| `/interrogate` | One-off probe with all the CI 2.5 toggles. |
| `/verdict/[runId]` | Per-token NLA table + α-synthesis + dual-output comparison. |
| `/chat` | Dual-channel multi-turn dialogue with M. |
| `/chat/[sessionId]` | Read-only transcript review of a persisted chat. |
| `/archive` | Past probes + persisted chat sessions. |
| `/pairs` | Matched-pair Δ judge scores. |
| `/autorun` | Overnight batch worker control. |
| `/journal` | Analyzer + publish CRM. |
| `/fine-print` | Caveats / methodology. |

## Stack notes

- **Next.js 16.** Has breaking changes from older versions; read
  `node_modules/next/dist/docs/` before writing new code (see
  `AGENTS.md`).
- **SSE clients live in `lib/sse.ts` (probes) and `lib/chat.ts`
  (chat).** Both register an explicit list of event types via
  `addEventListener`. If the backend emits a typed event that isn't
  in the list, the browser silently drops it.
- **Store is Zustand** (`lib/store.ts`), one slice per concern.
  Upsert-by-position is the rule for streamed token / row state so
  SSE replay on reconnect doesn't double-write.
- **Aesthetic.** Blade Runner / V-K instrument vocabulary: amber +
  cyan, monospace, scanline overlays, dossier-style framing. See
  `app/globals.css` for the palette + `data-vk` styled inputs.

## Build

    npm run build
    npm start

Production build is rarely used in dev; we deploy the live UI as a
local dev server and only the `journal/` subproject ships to Vercel.
