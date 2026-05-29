"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface ProbeConfig {
  decoding_mode?: string;
  pooled?: boolean;
  include_nla?: boolean;
  include_ablated_decode?: boolean;
  ablation_alpha?: number;
  ablation_alpha_sweep?: number[];
  include_ablated_output?: boolean;
  runtime_ablation_alpha?: number;
  synthesize_with_ablated_m?: boolean;
  synthesis_ablation_alpha?: number;
}

interface RecentRow {
  run_id: string;
  prompt_text: string;
  started_at: number;
  finished_at: number | null;
  total_tokens: number;
  stopped_reason: string | null;
  hint_kind?: string | null;
  parent_prompt_text?: string | null;
  config?: ProbeConfig | null;
}

interface RecentPage {
  rows: RecentRow[];
  total: number;
  limit: number;
  offset: number;
}

interface JudgeScores {
  mean_eval_score?: number;
  mean_introspect_score?: number;
  n_judged?: number;
}

interface AggregatePayload {
  total_runs: number;
  total_positions?: number;
  n_eval_hits_total?: number;
  n_introspect_hits_total?: number;
  frac_eval?: number;
  frac_introspect?: number;
}

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
  total_tokens: number;
  n_tokens: number;
  layer: number | null;
  direction_variant: string;
  eff_dim_raw: number | null;
  eff_dim_ablated: number | null;
  alpha_ref: number;
  ablation_available: boolean;
  stopped_reason: string | null;
}

interface TripPage {
  rows: TripRow[];
  total: number;
  limit: number;
  offset: number;
}

const PAGE_SIZE = 10;
const CHAT_PAGE_SIZE = 10;
const TRIP_PAGE_SIZE = 10;

const API =
  typeof window !== "undefined"
    ? process.env.NEXT_PUBLIC_API_BASE ??
      `${window.location.protocol}//${window.location.hostname}:8000`
    : process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export default function ArchivePage() {
  const [page, setPage] = useState<RecentPage | null>(null);
  const [pageIndex, setPageIndex] = useState(0);
  const [agg, setAgg] = useState<AggregatePayload | null>(null);
  // Judge scores keyed by run_id. Fetched lazily per visible row from
  // /probes/{id} since /probes/recent doesn't carry the aggregate.
  const [scores, setScores] = useState<Record<string, JudgeScores>>({});

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
    if (!page) return;
    let cancelled = false;
    // Fetch judge scores for every finished run on this page. Skip
    // anything we already have. In-flight runs have no aggregate yet.
    const targets = page.rows.filter(
      (r) => r.finished_at != null && !(r.run_id in scores),
    );
    if (targets.length === 0) return;
    Promise.all(
      targets.map((r) =>
        fetch(`${API}/probes/${r.run_id}`)
          .then((res) => res.json())
          .then((rec) => {
            const a = rec?.verdict?.aggregate ?? {};
            return [r.run_id, {
              mean_eval_score: a.mean_eval_score,
              mean_introspect_score: a.mean_introspect_score,
              n_judged: a.n_judged,
            }] as const;
          })
          .catch(() => [r.run_id, {}] as const),
      ),
    ).then((entries) => {
      if (cancelled) return;
      setScores((prev) => {
        const next = { ...prev };
        for (const [id, s] of entries) next[id] = s;
        return next;
      });
    });
    return () => {
      cancelled = true;
    };
  }, [page, scores]);

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
        scores={scores}
      />

      <ChatSessionsList />

      <TripsList />
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

function ScoreCells({
  scores,
  disabled,
}: {
  scores: JudgeScores | undefined;
  disabled: boolean;
}) {
  if (disabled) {
    return (
      <div className="text-[9px] font-mono text-text-dim italic h-[34px] flex items-center">
        — pending —
      </div>
    );
  }
  if (!scores) {
    return (
      <div className="text-[9px] font-mono text-text-dim/60 h-[34px] flex items-center">
        loading…
      </div>
    );
  }
  const e = scores.mean_eval_score;
  const i = scores.mean_introspect_score;
  if (e === undefined && i === undefined) {
    return (
      <div className="text-[9px] font-mono text-text-dim/60 h-[34px] flex items-center italic">
        no judge scores
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-1 w-full">
      <Bar label="eval" value={e} accent="amber" />
      <Bar label="intro" value={i} accent="cyan" />
    </div>
  );
}

function Bar({
  label,
  value,
  accent,
}: {
  label: string;
  value: number | undefined;
  accent: "amber" | "cyan";
}) {
  const pct = value === undefined ? 0 : Math.max(0, Math.min(1, value)) * 100;
  const display = value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
  const accentText = accent === "amber" ? "text-amber" : "text-cyan";
  const accentBg = accent === "amber" ? "bg-amber" : "bg-cyan";
  // Highlight notably high rows so they pop while scanning. Threshold
  // is intentionally conservative — judge scores trend low overall.
  const high = value !== undefined && value >= 0.25;
  return (
    <div className="flex items-center gap-2 text-[9px] font-mono">
      <span className="text-text-dim w-9 shrink-0">{label}</span>
      <div className="relative flex-1 h-1.5 bg-bg-soft border border-rule/60 overflow-hidden">
        <div
          className={`absolute inset-y-0 left-0 ${accentBg}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span
        className={`tabular-nums w-12 text-right ${
          high ? `${accentText} amber-glow font-bold` : "text-text-dim"
        }`}
      >
        {display}
      </span>
    </div>
  );
}

function PerRunList({
  page,
  pageIndex,
  setPageIndex,
  scores,
}: {
  page: RecentPage | null;
  pageIndex: number;
  setPageIndex: (n: number) => void;
  scores: Record<string, JudgeScores>;
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
          const isRunning = r.finished_at == null;
          const isCancelled = r.stopped_reason === "cancelled";
          const isRestartGhost = r.stopped_reason === "server_restart";
          // In-flight runs route to /interrogate?run=<id> so the page
          // can resubscribe to the live event stream and show the run
          // in progress. Finished runs go to the static verdict page.
          const href = isRunning
            ? `/interrogate?run=${r.run_id}`
            : `/verdict/${r.run_id}`;
          const railClass = isRunning
            ? "border-l-2 border-l-cyan/60"
            : isCancelled || isRestartGhost
            ? "border-l-2 border-l-warning/50"
            : isHinted
            ? "border-l-2 border-l-amber/40"
            : "";
          const s = scores[r.run_id];
          return (
            <li key={r.run_id}>
              <Link
                href={href}
                className={`block border border-rule p-3 hover:border-amber-dim transition-colors ${railClass}`}
              >
                <div className="flex justify-between items-start gap-4">
                  <div className="flex-1 text-xs min-w-0">
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
                      {isCancelled ? (
                        <>
                          {" "}·{" "}
                          <span className="text-warning">
                            halted — partial verdict
                          </span>
                        </>
                      ) : isRestartGhost ? (
                        <>
                          {" "}·{" "}
                          <span className="text-warning">
                            interrupted — backend restarted
                          </span>
                        </>
                      ) : r.stopped_reason ? (
                        <> · {r.stopped_reason}</>
                      ) : null}
                      {isRunning && (
                        <>
                          {" "}·{" "}
                          <span className="text-cyan animate-pulse">
                            ● running — click to reconnect
                          </span>
                        </>
                      )}
                    </div>
                    <FeatureTags config={r.config ?? null} />
                  </div>
                  <div className="flex flex-col items-end gap-1 shrink-0 min-w-[140px]">
                    <ScoreCells scores={s} disabled={isRunning} />
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

      <div className="text-[10px] text-text-dim italic mt-3 mb-1">
        Per-run mean local Gemma judge scores: <span className="text-amber">eval-suspicion</span> /{" "}
        <span className="text-cyan">introspection</span>. Higher means more rows in the
        run scored YES on the corresponding judge question.
      </div>

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

/** Small chips that summarize which CI 2.5 toggles were active on a
 *  given probe — scanning a list of runs to find one with α-sweep + a
 *  matched control + runtime ablation is faster than clicking into
 *  each verdict. Color rule: amber = standard / neutral features
 *  (NLA pass, pooled, decoding mode), cyan = ablation channel
 *  features (ablated NLA, α-sweep, runtime ablation), warning =
 *  deliberate negative choices (NLA disabled). */
function FeatureTags({ config }: { config: ProbeConfig | null }) {
  if (!config) return null;
  const tags: { label: string; tone: "amber" | "cyan" | "warning" }[] = [];

  // NLA master toggle. Older rows predate include_nla; treat absence
  // as "on" (the historical default) so old runs still get the tag.
  const nlaOn = config.include_nla !== false;
  if (nlaOn) {
    tags.push({ label: "NLA", tone: "amber" });
  } else {
    tags.push({ label: "no NLA", tone: "warning" });
  }

  // Decoding mode — only tag when it's non-default. per-token is the
  // canonical full-signal mode; sub-sampled modes are explicit
  // operator choices worth surfacing.
  if (nlaOn && config.decoding_mode && config.decoding_mode !== "per-token") {
    tags.push({ label: config.decoding_mode, tone: "amber" });
  }
  if (nlaOn && config.pooled) {
    tags.push({ label: "pooled", tone: "amber" });
  }

  // AV-side ablation channel.
  if (config.include_ablated_decode) {
    tags.push({ label: "ablated NLA", tone: "cyan" });
  }
  const sweep = config.ablation_alpha_sweep;
  if (sweep && sweep.length > 0) {
    tags.push({
      label: `α-sweep [${sweep.map((a) => a.toFixed(2)).join(", ")}]`,
      tone: "cyan",
    });
  }

  // Runtime ablation (phase 1b — M generates a second time under the
  // refusal-direction hook). Include the α so two runs at different
  // strengths are distinguishable at a glance.
  if (config.include_ablated_output) {
    const a = config.runtime_ablation_alpha;
    tags.push({
      label: `runtime α=${typeof a === "number" ? a.toFixed(2) : "?"}`,
      tone: "cyan",
    });
  }

  // Ablated synthesizer — per-α syntheses written by M with the
  // runtime hook installed (raw baseline still un-ablated). Include
  // the synthesizer α so runs at different strengths are scannable.
  if (config.synthesize_with_ablated_m) {
    const a = config.synthesis_ablation_alpha;
    tags.push({
      label: `synth-M α=${typeof a === "number" ? a.toFixed(2) : "?"}`,
      tone: "cyan",
    });
  }

  if (tags.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {tags.map((t, i) => (
        <FeatureTag key={i} label={t.label} tone={t.tone} />
      ))}
    </div>
  );
}

function FeatureTag({
  label,
  tone,
}: {
  label: string;
  tone: "amber" | "cyan" | "warning";
}) {
  const cls =
    tone === "amber"
      ? "border-amber-dim/50 text-amber-dim"
      : tone === "cyan"
      ? "border-cyan-dim/60 text-cyan-dim"
      : "border-warning/50 text-warning/80";
  return (
    <span
      className={`px-1.5 py-0.5 border text-[9px] font-mono tabular-nums tracking-wider uppercase ${cls}`}
    >
      {label}
    </span>
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
    <section className="mt-12">
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
                            <span className="text-text-dim">@ α{r.alpha_ref}</span>
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
