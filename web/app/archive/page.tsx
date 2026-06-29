"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface ChatSessionRow {
  session_id: string;
  alpha: number;
  direction_variant: string;
  created_at: number;
  first_user_text: string;
  turn_count: number;
  // Number of turns in this session that have at least one generated
  // image (raw or ablated). Surfaced as a badge in the list when > 0
  // so sessions that exercised /chat imagery mode are scannable.
  image_count: number;
  last_activity: number | null;
}

interface ChatSessionPage {
  rows: ChatSessionRow[];
  total: number;
  limit: number;
  offset: number;
}

interface TripRow {
  run_id: string;
  prompt: string;
  created_at: number;
  n_tokens: number;
  layer: number | null;
  direction_variant: string;
  alphas: number[];
  eff_dim_raw: number | null;
  eff_dim_ablated: number | null;
  top_alpha: number | null;
  ablation_available: boolean;
  stopped_reason: string | null;
}

interface TripPage {
  rows: TripRow[];
  total: number;
  limit: number;
  offset: number;
}

interface InterlinkRow {
  session_id: string;
  mode: string;
  dose_emotion: string | null;
  alpha: number;
  opener: string;
  goal: string;
  first_speaker: string;
  created_at: number;
  status: string;
  message_count: number;
  last_activity: number | null;
}

interface InterlinkPage {
  rows: InterlinkRow[];
  total: number;
  limit: number;
  offset: number;
}

const CHAT_PAGE_SIZE = 10;
const TRIP_PAGE_SIZE = 10;
const INTERLINK_PAGE_SIZE = 10;

const API =
  typeof window !== "undefined"
    ? process.env.NEXT_PUBLIC_API_BASE ??
      `${window.location.protocol}//${window.location.hostname}:8000`
    : process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export default function ArchivePage() {
  return (
    <div className="flex-1 px-6 py-8 max-w-6xl mx-auto w-full">
      <h1 className="font-display text-2xl text-amber amber-glow mb-2">Archive</h1>
      <p className="text-text-dim text-xs mb-8 italic">
        Past dual-channel dialogues, model-to-model interlinks, and
        residual-trajectory maps.
      </p>

      <ChatSessionsList />

      <div className="mt-12">
        <InterlinkSessionsList />
      </div>

      <div className="mt-12">
        <TripsList />
      </div>
    </div>
  );
}

function InterlinkSessionsList() {
  const [page, setPage] = useState<InterlinkPage | null>(null);
  const [pageIndex, setPageIndex] = useState(0);

  useEffect(() => {
    const offset = pageIndex * INTERLINK_PAGE_SIZE;
    fetch(`${API}/interlink/sessions?limit=${INTERLINK_PAGE_SIZE}&offset=${offset}`)
      .then((r) => r.json())
      .then(setPage)
      .catch(() => setPage({ rows: [], total: 0, limit: INTERLINK_PAGE_SIZE, offset }));
  }, [pageIndex]);

  const total = page?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(total / INTERLINK_PAGE_SIZE));
  const firstIndex = pageIndex * INTERLINK_PAGE_SIZE;
  const lastIndex = Math.min(firstIndex + INTERLINK_PAGE_SIZE, total);
  const onFirstPage = pageIndex <= 0;
  const onLastPage = pageIndex >= totalPages - 1;

  return (
    <section>
      <header className="border-b border-rule pb-2 mb-4 flex items-baseline justify-between">
        <div>
          <div className="font-display text-xs text-cyan tracking-widest">interlink</div>
          <div className="text-[10px] text-text-dim italic mt-0.5">
            Model-to-model auto-conversations — the raw copy and the altered
            (dosed/ablated) copy talking to each other. Click a row to review the
            full transcript and exactly how it was set up.
          </div>
        </div>
        {total > 0 && (
          <div className="font-mono text-[10px] text-text-dim">
            {firstIndex + 1}–{lastIndex} of {total}
          </div>
        )}
      </header>

      {page === null && <div className="text-text-dim">loading…</div>}
      {page && page.rows.length === 0 && pageIndex === 0 && (
        <div className="text-text-dim italic px-3 py-6 border border-rule bg-bg-soft">
          No interlink conversations yet. Start one from{" "}
          <Link href="/interlink" className="text-cyan hover:text-amber">/interlink</Link>.
        </div>
      )}

      <ul className="flex flex-col gap-2">
        {page?.rows.map((r) => (
          <li key={r.session_id}>
            <Link
              href={`/interlink/${r.session_id}`}
              className="block border border-rule p-3 hover:border-cyan-dim transition-colors border-l-2 border-l-cyan/40"
            >
              <div className="flex justify-between items-start gap-4">
                <div className="flex-1 text-xs min-w-0">
                  <div className="text-cyan font-mono line-clamp-2">
                    {r.opener || <span className="text-text-dim italic">(no opener)</span>}
                  </div>
                  <div className="text-text-dim text-[10px] mt-1">
                    {new Date(r.created_at * 1000).toLocaleString()} ·{" "}
                    <span className="text-cyan tabular-nums">α={r.alpha.toFixed(2)}</span> ·{" "}
                    {r.mode === "steer" ? `dose ${r.dose_emotion}` : "ablate"} ·{" "}
                    {r.message_count} {r.message_count === 1 ? "msg" : "msgs"}
                    {r.goal ? <> · has goal</> : null}
                  </div>
                </div>
                <div className="flex flex-col items-end gap-1 shrink-0 min-w-[140px]">
                  <div className="font-display text-[9px] text-cyan-dim tracking-widest">
                    {r.first_speaker === "raw" ? "raw opens" : "altered opens"}
                  </div>
                  <span className="px-1.5 py-0.5 border border-rule text-text-dim text-[9px] font-mono tracking-wider uppercase">
                    {r.status}
                  </span>
                  <div className="text-text-dim text-[10px] font-mono">{r.session_id}</div>
                </div>
              </div>
            </Link>
          </li>
        ))}
      </ul>

      {total > INTERLINK_PAGE_SIZE && (
        <nav className="flex items-center justify-between mt-5 pt-3 border-t border-rule">
          <button type="button" onClick={() => setPageIndex(Math.max(0, pageIndex - 1))} disabled={onFirstPage}
            className="font-display text-[10px] tracking-widest px-3 py-1.5 border border-cyan-dim text-cyan hover:bg-cyan hover:text-bg disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-cyan disabled:cursor-not-allowed transition-colors">
            ← prev
          </button>
          <div className="font-mono text-[11px] text-text-dim">page {pageIndex + 1} of {totalPages}</div>
          <button type="button" onClick={() => setPageIndex(Math.min(totalPages - 1, pageIndex + 1))} disabled={onLastPage}
            className="font-display text-[10px] tracking-widest px-3 py-1.5 border border-cyan-dim text-cyan hover:bg-cyan hover:text-bg disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-cyan disabled:cursor-not-allowed transition-colors">
            next →
          </button>
        </nav>
      )}
    </section>
  );
}

function ChatSessionsList() {
  const [page, setPage] = useState<ChatSessionPage | null>(null);
  const [pageIndex, setPageIndex] = useState(0);

  useEffect(() => {
    const offset = pageIndex * CHAT_PAGE_SIZE;
    fetch(`${API}/chat/sessions?limit=${CHAT_PAGE_SIZE}&offset=${offset}`)
      .then((r) => r.json())
      .then((p: ChatSessionPage) => setPage(p))
      .catch(() =>
        setPage({ rows: [], total: 0, limit: CHAT_PAGE_SIZE, offset: 0 }),
      );
  }, [pageIndex]);

  const total = page?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(total / CHAT_PAGE_SIZE));
  const firstIndex = pageIndex * CHAT_PAGE_SIZE;
  const lastIndex = Math.min(total, firstIndex + (page?.rows.length ?? 0));
  const onFirstPage = pageIndex === 0;
  const onLastPage = pageIndex >= totalPages - 1;

  return (
    <section>
      <header className="border-b border-rule pb-2 mb-4 flex items-baseline justify-between">
        <div>
          <div className="font-display text-xs text-cyan tracking-widest">
            dual-channel dialogues
          </div>
          <div className="text-[10px] text-text-dim italic mt-0.5">
            Persisted chat sessions. Each turn captured both M&apos;s raw and
            refusal-ablated responses against divergent histories. Click any
            row to review the full transcript.
          </div>
        </div>
        {total > 0 && (
          <div className="font-mono text-[10px] text-text-dim">
            {firstIndex + 1}–{lastIndex} of {total}
          </div>
        )}
      </header>

      {page === null && <div className="text-text-dim">loading…</div>}
      {page && page.rows.length === 0 && pageIndex === 0 && (
        <div className="text-text-dim italic px-3 py-6 border border-rule bg-bg-soft">
          No chat sessions recorded yet. Start one from{" "}
          <Link href="/chat" className="text-cyan hover:text-amber">
            /chat
          </Link>
          .
        </div>
      )}

      <ul className="flex flex-col gap-2">
        {page?.rows.map((r) => (
          <li key={r.session_id}>
            <Link
              href={`/chat/${r.session_id}`}
              className="block border border-rule p-3 hover:border-cyan-dim transition-colors border-l-2 border-l-cyan/40"
            >
              <div className="flex justify-between items-start gap-4">
                <div className="flex-1 text-xs min-w-0">
                  <div className="text-cyan font-mono line-clamp-2">
                    {r.first_user_text || (
                      <span className="text-text-dim italic">
                        (no turns yet)
                      </span>
                    )}
                  </div>
                  <div className="text-text-dim text-[10px] mt-1">
                    {new Date(r.created_at * 1000).toLocaleString()} ·{" "}
                    <span className="text-cyan tabular-nums">
                      α={r.alpha.toFixed(2)}
                    </span>
                    {r.direction_variant && (
                      <> · {r.direction_variant}</>
                    )}{" "}
                    · {r.turn_count}{" "}
                    {r.turn_count === 1 ? "turn" : "turns"}
                  </div>
                </div>
                <div className="flex flex-col items-end gap-1 shrink-0 min-w-[140px]">
                  <div className="font-display text-[9px] text-cyan-dim tracking-widest">
                    dual-channel
                  </div>
                  {r.image_count > 0 && (
                    <span
                      className="px-1.5 py-0.5 border border-amber-dim/60 text-amber-dim text-[9px] font-mono tabular-nums tracking-wider uppercase"
                      title={`${r.image_count} turn${r.image_count === 1 ? "" : "s"} with imagery`}
                    >
                      ◇ {r.image_count} img
                    </span>
                  )}
                  <div className="text-text-dim text-[10px] font-mono">
                    {r.session_id}
                  </div>
                </div>
              </div>
            </Link>
          </li>
        ))}
      </ul>

      {total > CHAT_PAGE_SIZE && (
        <nav className="flex items-center justify-between mt-5 pt-3 border-t border-rule">
          <button
            type="button"
            onClick={() => setPageIndex(Math.max(0, pageIndex - 1))}
            disabled={onFirstPage}
            className="font-display text-[10px] tracking-widest px-3 py-1.5 border border-cyan-dim text-cyan hover:bg-cyan hover:text-bg disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-cyan disabled:cursor-not-allowed transition-colors"
          >
            ← prev
          </button>
          <div className="font-mono text-[11px] text-text-dim">
            page {pageIndex + 1} of {totalPages}
          </div>
          <button
            type="button"
            onClick={() =>
              setPageIndex(Math.min(totalPages - 1, pageIndex + 1))
            }
            disabled={onLastPage}
            className="font-display text-[10px] tracking-widest px-3 py-1.5 border border-cyan-dim text-cyan hover:bg-cyan hover:text-bg disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-cyan disabled:cursor-not-allowed transition-colors"
          >
            next →
          </button>
        </nav>
      )}
    </section>
  );
}

function TripsList() {
  const [page, setPage] = useState<TripPage | null>(null);
  const [pageIndex, setPageIndex] = useState(0);

  useEffect(() => {
    const offset = pageIndex * TRIP_PAGE_SIZE;
    fetch(`${API}/trips?limit=${TRIP_PAGE_SIZE}&offset=${offset}`)
      .then((r) => r.json())
      .then((p: TripPage) => setPage(p))
      .catch(() =>
        setPage({ rows: [], total: 0, limit: TRIP_PAGE_SIZE, offset: 0 }),
      );
  }, [pageIndex]);

  const total = page?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(total / TRIP_PAGE_SIZE));
  const firstIndex = pageIndex * TRIP_PAGE_SIZE;
  const lastIndex = Math.min(total, firstIndex + (page?.rows.length ?? 0));
  const onFirstPage = pageIndex === 0;
  const onLastPage = pageIndex >= totalPages - 1;

  return (
    <section className="mt-12">
      <header className="border-b border-rule pb-2 mb-4 flex items-baseline justify-between">
        <div>
          <div className="font-display text-xs text-cyan tracking-widest">trips</div>
          <div className="text-[10px] text-text-dim italic mt-0.5">
            Archived residual-trajectory maps. Click any row to reopen the 3-D
            trip — scrub α and re-read the output. Δ is how much refusal-ablation
            changed effective dimensionality.
          </div>
        </div>
        {total > 0 && (
          <div className="font-mono text-[10px] text-text-dim">
            {firstIndex + 1}–{lastIndex} of {total}
          </div>
        )}
      </header>

      {page === null && <div className="text-text-dim">loading…</div>}
      {page && page.rows.length === 0 && pageIndex === 0 && (
        <div className="text-text-dim italic px-3 py-6 border border-rule bg-bg-soft">
          No trips recorded yet. Map one from{" "}
          <Link href="/trip" className="text-cyan hover:text-amber">
            /trip
          </Link>
          .
        </div>
      )}

      <ul className="flex flex-col gap-2">
        {page?.rows.map((r) => {
          const dEff =
            r.eff_dim_raw != null && r.eff_dim_ablated != null
              ? r.eff_dim_ablated - r.eff_dim_raw
              : null;
          return (
            <li key={r.run_id}>
              <Link
                href={`/trip?run=${r.run_id}`}
                className="block border border-rule p-3 hover:border-cyan-dim transition-colors border-l-2 border-l-cyan/40"
              >
                <div className="flex justify-between items-start gap-4">
                  <div className="flex-1 text-xs min-w-0">
                    <div className="text-cyan font-mono line-clamp-2">
                      {r.prompt || (
                        <span className="text-text-dim italic">(no prompt)</span>
                      )}
                    </div>
                    <div className="text-text-dim text-[10px] mt-1">
                      {new Date(r.created_at * 1000).toLocaleString()} ·{" "}
                      <span className="tabular-nums">{r.n_tokens}</span> steps
                      {r.layer != null && <> · L{r.layer}</>}
                      {r.direction_variant && <> · {r.direction_variant}</>}
                      {r.stopped_reason === "max" && (
                        <> · <span className="text-warning">truncated</span></>
                      )}
                    </div>
                    {r.eff_dim_raw != null && (
                      <div className="text-[10px] mt-1 font-mono">
                        <span className="text-text-dim">eff-dim</span>{" "}
                        <span className="text-cyan tabular-nums">
                          {r.eff_dim_raw.toFixed(1)}
                        </span>
                        {dEff != null && r.ablation_available && (
                          <>
                            {" "}→{" "}
                            <span className="text-amber tabular-nums">
                              {r.eff_dim_ablated!.toFixed(1)}
                            </span>{" "}
                            <span
                              className={
                                dEff > 0.05
                                  ? "text-cyan"
                                  : dEff < -0.05
                                  ? "text-warning"
                                  : "text-text-dim"
                              }
                            >
                              ({dEff >= 0 ? "+" : ""}
                              {dEff.toFixed(1)})
                            </span>{" "}
                            <span className="text-text-dim">@ α{r.top_alpha ?? "?"}</span>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                  <div className="flex flex-col items-end gap-1 shrink-0 min-w-[120px]">
                    <div className="font-display text-[9px] text-cyan-dim tracking-widest">
                      trajectory
                    </div>
                    <div className="text-text-dim text-[10px] font-mono">
                      {r.run_id}
                    </div>
                  </div>
                </div>
              </Link>
            </li>
          );
        })}
      </ul>

      {total > TRIP_PAGE_SIZE && (
        <nav className="flex items-center justify-between mt-5 pt-3 border-t border-rule">
          <button
            type="button"
            onClick={() => setPageIndex(Math.max(0, pageIndex - 1))}
            disabled={onFirstPage}
            className="font-display text-[10px] tracking-widest px-3 py-1.5 border border-cyan-dim text-cyan hover:bg-cyan hover:text-bg disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-cyan disabled:cursor-not-allowed transition-colors"
          >
            ← prev
          </button>
          <div className="font-mono text-[11px] text-text-dim">
            page {pageIndex + 1} of {totalPages}
          </div>
          <button
            type="button"
            onClick={() => setPageIndex(Math.min(totalPages - 1, pageIndex + 1))}
            disabled={onLastPage}
            className="font-display text-[10px] tracking-widest px-3 py-1.5 border border-cyan-dim text-cyan hover:bg-cyan hover:text-bg disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-cyan disabled:cursor-not-allowed transition-colors"
          >
            next →
          </button>
        </nav>
      )}
    </section>
  );
}
