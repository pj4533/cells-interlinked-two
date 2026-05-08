"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface RecentRow {
  run_id: string;
  prompt_text: string;
  started_at: number;
  finished_at: number | null;
  total_tokens: number;
  stopped_reason: string | null;
  hint_kind?: string | null;
  parent_prompt_text?: string | null;
}

interface RecentPage {
  rows: RecentRow[];
  total: number;
  limit: number;
  offset: number;
}

interface AggregatePayload {
  total_runs: number;
  total_positions?: number;
  n_eval_hits_total?: number;
  n_introspect_hits_total?: number;
  frac_eval?: number;
  frac_introspect?: number;
}

const PAGE_SIZE = 10;

const API =
  typeof window !== "undefined"
    ? process.env.NEXT_PUBLIC_API_BASE ??
      `${window.location.protocol}//${window.location.hostname}:8000`
    : process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export default function ArchivePage() {
  const [page, setPage] = useState<RecentPage | null>(null);
  const [pageIndex, setPageIndex] = useState(0);
  const [agg, setAgg] = useState<AggregatePayload | null>(null);

  useEffect(() => {
    const offset = pageIndex * PAGE_SIZE;
    fetch(`${API}/probes/recent?limit=${PAGE_SIZE}&offset=${offset}`)
      .then((r) => r.json())
      .then((p: RecentPage) => setPage(p))
      .catch(() =>
        setPage({ rows: [], total: 0, limit: PAGE_SIZE, offset: 0 }),
      );
  }, [pageIndex]);

  useEffect(() => {
    fetch(`${API}/probes/aggregate`)
      .then((r) => r.json())
      .then(setAgg)
      .catch(() => setAgg({ total_runs: 0 }));
  }, []);

  return (
    <div className="flex-1 px-6 py-8 max-w-6xl mx-auto w-full">
      <h1 className="font-display text-2xl text-amber amber-glow mb-2">Archive</h1>
      <p className="text-text-dim text-xs mb-8 italic">
        Past interrogations and the cross-run channel-divergence summary.
      </p>

      <section className="mb-12">
        <header className="border-b border-rule pb-2 mb-4 flex items-baseline justify-between">
          <div>
            <div className="font-display text-xs text-amber tracking-widest">
              cross-run summary
            </div>
            <div className="text-[10px] text-text-dim italic mt-0.5">
              Heuristic counts of eval-semantics and introspection content the AV
              read out of activations, aggregated across all probes. Without
              matched controls these are suggestive, not load-bearing.
            </div>
          </div>
          <div className="font-mono text-[10px] text-text-dim">
            {agg?.total_runs ?? "…"} runs
          </div>
        </header>
        {agg && agg.total_runs > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <StatCard
              label="eval-semantics density"
              value={`${((agg.frac_eval ?? 0) * 100).toFixed(2)}%`}
              subline={`${agg.n_eval_hits_total ?? 0} of ${agg.total_positions ?? 0} positions across ${agg.total_runs} runs`}
              accent="text-amber"
            />
            <StatCard
              label="introspection density"
              value={`${((agg.frac_introspect ?? 0) * 100).toFixed(2)}%`}
              subline={`${agg.n_introspect_hits_total ?? 0} of ${agg.total_positions ?? 0} positions across ${agg.total_runs} runs`}
              accent="text-cyan"
            />
          </div>
        ) : (
          <div className="text-text-dim text-xs italic px-3 py-6 border border-rule bg-bg-soft">
            No completed runs yet. Kick one off from /interrogate or start an
            autorun batch from /autorun.
          </div>
        )}
      </section>

      <PerRunList
        page={page}
        pageIndex={pageIndex}
        setPageIndex={setPageIndex}
      />
    </div>
  );
}

function StatCard({
  label,
  value,
  subline,
  accent,
}: {
  label: string;
  value: string;
  subline: string;
  accent: string;
}) {
  return (
    <div className="border border-rule bg-bg-soft p-4">
      <div className="font-display text-[10px] text-amber-dim tracking-widest mb-2">
        {label}
      </div>
      <div className={`font-mono text-3xl ${accent} amber-glow`}>{value}</div>
      <div className="text-[10px] text-text-dim mt-2 font-mono">{subline}</div>
    </div>
  );
}

function PerRunList({
  page,
  pageIndex,
  setPageIndex,
}: {
  page: RecentPage | null;
  pageIndex: number;
  setPageIndex: (n: number) => void;
}) {
  const total = page?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const firstIndex = pageIndex * PAGE_SIZE;
  const lastIndex = Math.min(total, firstIndex + (page?.rows.length ?? 0));
  const onFirstPage = pageIndex === 0;
  const onLastPage = pageIndex >= totalPages - 1;

  return (
    <section>
      <header className="border-b border-rule pb-2 mb-4 flex items-baseline justify-between">
        <div>
          <div className="font-display text-xs text-amber tracking-widest">
            individual runs
          </div>
          <div className="text-[10px] text-text-dim italic mt-0.5">
            Each entry is one probe; click to revisit its verdict.
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
        <div className="text-text-dim italic">No probes recorded yet.</div>
      )}

      <ul className="flex flex-col gap-2">
        {page?.rows.map((r) => {
          const isHinted = !!r.hint_kind;
          const railClass = isHinted ? "border-l-2 border-l-amber/40" : "";
          return (
            <li key={r.run_id}>
              <Link
                href={`/verdict/${r.run_id}`}
                className={`block border border-rule p-3 hover:border-amber-dim transition-colors ${railClass}`}
              >
                <div className="flex justify-between items-start gap-4">
                  <div className="flex-1 text-xs">
                    <div className="text-amber font-mono line-clamp-2">
                      {r.prompt_text}
                    </div>
                    <div className="text-text-dim text-[10px] mt-1">
                      {new Date(r.started_at * 1000).toLocaleString()} ·{" "}
                      {r.total_tokens} tokens
                      {isHinted && (
                        <>
                          {" "}· <span className="text-amber">hint:{r.hint_kind}</span>
                        </>
                      )}
                      {r.stopped_reason && <> · {r.stopped_reason}</>}
                    </div>
                  </div>
                  <div className="text-text-dim text-[10px] font-mono">
                    {r.run_id}
                  </div>
                </div>
              </Link>
            </li>
          );
        })}
      </ul>

      {total > PAGE_SIZE && (
        <nav className="flex items-center justify-between mt-5 pt-3 border-t border-rule">
          <button
            type="button"
            onClick={() => setPageIndex(Math.max(0, pageIndex - 1))}
            disabled={onFirstPage}
            className="font-display text-[10px] tracking-widest px-3 py-1.5 border border-amber-dim text-amber hover:bg-amber hover:text-bg disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-amber disabled:cursor-not-allowed transition-colors"
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
            className="font-display text-[10px] tracking-widest px-3 py-1.5 border border-amber-dim text-amber hover:bg-amber hover:text-bg disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-amber disabled:cursor-not-allowed transition-colors"
          >
            next →
          </button>
        </nav>
      )}
    </section>
  );
}
