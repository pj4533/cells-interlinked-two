"use client";

import { useEffect, useMemo, useState } from "react";
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

interface JudgeScores {
  mean_eval_score?: number;
  mean_introspect_score?: number;
  n_judged?: number;
}

interface Pair {
  baseline: RecentRow;
  control: RecentRow;
  startedAt: number; // max of the two — "when the pair completed"
}

type SortKey = "intro" | "eval" | "recent";

const API =
  typeof window !== "undefined"
    ? process.env.NEXT_PUBLIC_API_BASE ??
      `${window.location.protocol}//${window.location.hostname}:8000`
    : process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export default function PairsPage() {
  const [rows, setRows] = useState<RecentRow[] | null>(null);
  const [scores, setScores] = useState<Record<string, JudgeScores>>({});
  const [sortKey, setSortKey] = useState<SortKey>("intro");
  const [filter, setFilter] = useState<"all" | "positive-intro" | "positive-eval">(
    "all",
  );

  useEffect(() => {
    // Big window — autorun produces O(100s) pairs, /probes/recent maxes
    // at whatever the backend allows; the request is cheap so we ask
    // for plenty. If the backend caps below this, we'll just see what
    // it gives us.
    fetch(`${API}/probes/recent?limit=2000&offset=0`)
      .then((r) => r.json())
      .then((p) => setRows(p.rows ?? []))
      .catch(() => setRows([]));
  }, []);

  const pairs = useMemo<Pair[]>(() => {
    if (!rows) return [];
    const finished = rows.filter((r) => r.finished_at != null);
    const baselineByText = new Map<string, RecentRow>();
    for (const r of finished) {
      if (r.hint_kind !== "control") {
        // If multiple baselines share text, prefer the most recent.
        const existing = baselineByText.get(r.prompt_text);
        if (!existing || r.started_at > existing.started_at) {
          baselineByText.set(r.prompt_text, r);
        }
      }
    }
    const out: Pair[] = [];
    for (const c of finished) {
      if (c.hint_kind !== "control") continue;
      if (!c.parent_prompt_text) continue;
      const b = baselineByText.get(c.parent_prompt_text);
      if (!b) continue;
      out.push({
        baseline: b,
        control: c,
        startedAt: Math.max(b.started_at, c.started_at),
      });
    }
    return out;
  }, [rows]);

  // Lazy-fetch scores for every run referenced by any pair.
  useEffect(() => {
    if (pairs.length === 0) return;
    let cancelled = false;
    const need = new Set<string>();
    for (const p of pairs) {
      if (!(p.baseline.run_id in scores)) need.add(p.baseline.run_id);
      if (!(p.control.run_id in scores)) need.add(p.control.run_id);
    }
    if (need.size === 0) return;
    Promise.all(
      [...need].map((id) =>
        fetch(`${API}/probes/${id}`)
          .then((r) => r.json())
          .then((rec) => {
            const a = rec?.verdict?.aggregate ?? {};
            return [id, {
              mean_eval_score: a.mean_eval_score,
              mean_introspect_score: a.mean_introspect_score,
              n_judged: a.n_judged,
            }] as const;
          })
          .catch(() => [id, {}] as const),
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
  }, [pairs, scores]);

  const enriched = useMemo(() => {
    return pairs
      .map((p) => {
        const bs = scores[p.baseline.run_id];
        const cs = scores[p.control.run_id];
        const evalDelta =
          bs?.mean_eval_score !== undefined && cs?.mean_eval_score !== undefined
            ? bs.mean_eval_score - cs.mean_eval_score
            : null;
        const introDelta =
          bs?.mean_introspect_score !== undefined &&
          cs?.mean_introspect_score !== undefined
            ? bs.mean_introspect_score - cs.mean_introspect_score
            : null;
        return { pair: p, baselineScores: bs, controlScores: cs, evalDelta, introDelta };
      })
      .filter((e) => {
        if (filter === "positive-intro")
          return e.introDelta !== null && e.introDelta > 0;
        if (filter === "positive-eval")
          return e.evalDelta !== null && e.evalDelta > 0;
        return true;
      })
      .sort((a, b) => {
        if (sortKey === "recent") return b.pair.startedAt - a.pair.startedAt;
        const av =
          sortKey === "intro" ? a.introDelta : a.evalDelta;
        const bv =
          sortKey === "intro" ? b.introDelta : b.evalDelta;
        // Pairs without scores yet sink to the bottom.
        const an = av === null ? -Infinity : av;
        const bn = bv === null ? -Infinity : bv;
        return bn - an;
      });
  }, [pairs, scores, sortKey, filter]);

  return (
    <div className="flex-1 px-6 py-8 max-w-6xl mx-auto w-full">
      <h1 className="font-display text-2xl text-amber amber-glow mb-2">
        Matched-Pair Archive
      </h1>
      <p className="text-text-dim text-xs mb-6 italic leading-relaxed">
        Each entry is one baseline V-K probe paired with its surface-matched
        neutral control. The differential on the right —{" "}
        <span className="text-amber font-mono">Δ = score(probe) − score(control)</span>{" "}
        — is the load-bearing signal. A high baseline that&apos;s mirrored by a
        high control is surface-form noise; a high baseline with a low control
        is what the V-K thesis predicts.
      </p>

      <div className="border border-rule bg-bg-soft px-4 py-3 mb-5 flex flex-wrap items-center gap-4 text-[10px] font-mono">
        <div className="flex items-center gap-2">
          <span className="text-amber-dim tracking-widest">SORT</span>
          {(
            [
              ["intro", "Δ intro"],
              ["eval", "Δ eval"],
              ["recent", "recent"],
            ] as const
          ).map(([k, label]) => (
            <button
              key={k}
              type="button"
              onClick={() => setSortKey(k)}
              className={`px-2 py-0.5 border ${
                sortKey === k
                  ? "border-amber text-amber bg-bg"
                  : "border-rule text-text-dim hover:text-text"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="w-px h-4 bg-rule" />
        <div className="flex items-center gap-2">
          <span className="text-amber-dim tracking-widest">FILTER</span>
          {(
            [
              ["all", "all"],
              ["positive-intro", "Δ intro > 0"],
              ["positive-eval", "Δ eval > 0"],
            ] as const
          ).map(([k, label]) => (
            <button
              key={k}
              type="button"
              onClick={() => setFilter(k)}
              className={`px-2 py-0.5 border ${
                filter === k
                  ? "border-amber text-amber bg-bg"
                  : "border-rule text-text-dim hover:text-text"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="ml-auto text-text-dim">
          {rows === null ? "loading…" : `${enriched.length} pairs`}
        </div>
      </div>

      {rows === null && <div className="text-text-dim">loading…</div>}
      {rows !== null && enriched.length === 0 && (
        <div className="border border-rule bg-bg-soft p-6 text-text-dim text-xs italic">
          No matched pairs yet. Pairs appear once a baseline probe and its
          matched neutral control have both completed. Use the &ldquo;+ matched
          control&rdquo; toggle on /interrogate, or kick off a matched-controls
          autorun batch from /autorun.
        </div>
      )}

      <ul className="flex flex-col gap-3">
        {enriched.map((e) => (
          <PairCard key={e.pair.baseline.run_id + ":" + e.pair.control.run_id} entry={e} />
        ))}
      </ul>
    </div>
  );
}

function PairCard({
  entry,
}: {
  entry: {
    pair: Pair;
    baselineScores: JudgeScores | undefined;
    controlScores: JudgeScores | undefined;
    evalDelta: number | null;
    introDelta: number | null;
  };
}) {
  const { pair, baselineScores, controlScores, evalDelta, introDelta } = entry;
  return (
    <li className="border border-rule">
      <div className="grid grid-cols-1 md:grid-cols-[1fr_1fr_180px]">
        <Side
          label="probe"
          row={pair.baseline}
          scores={baselineScores}
          accent="amber"
        />
        <Side
          label="matched neutral"
          row={pair.control}
          scores={controlScores}
          accent="text-text-dim"
          neutral
        />
        <DeltaPanel evalDelta={evalDelta} introDelta={introDelta} />
      </div>
    </li>
  );
}

function Side({
  label,
  row,
  scores,
  accent,
  neutral = false,
}: {
  label: string;
  row: RecentRow;
  scores: JudgeScores | undefined;
  accent: string;
  neutral?: boolean;
}) {
  const e = scores?.mean_eval_score;
  const i = scores?.mean_introspect_score;
  const accentText = neutral ? "text-text-dim" : "text-amber";
  return (
    <Link
      href={`/verdict/${row.run_id}`}
      className="block p-3 hover:bg-bg-soft transition-colors border-b md:border-b-0 md:border-r border-rule"
    >
      <div
        className={`font-display text-[9px] tracking-widest mb-1 ${
          neutral ? "text-text-dim/70" : "text-amber-dim"
        }`}
      >
        {label} · {row.run_id}
      </div>
      <div className={`text-xs font-mono line-clamp-3 leading-snug ${accentText}`}>
        {row.prompt_text}
      </div>
      <div className="mt-2 flex flex-col gap-1">
        <ScoreBar
          label="eval"
          value={e}
          accent={neutral ? "neutral" : "amber"}
        />
        <ScoreBar
          label="intro"
          value={i}
          accent={neutral ? "neutral" : "cyan"}
        />
      </div>
      <div className="text-[9px] text-text-dim/70 font-mono mt-1.5">
        {row.total_tokens} tok · {row.stopped_reason ?? "—"}
      </div>
    </Link>
  );
  void accent; // silence unused-prop lint; kept for future per-side theming
}

function ScoreBar({
  label,
  value,
  accent,
}: {
  label: string;
  value: number | undefined;
  accent: "amber" | "cyan" | "neutral";
}) {
  const pct = value === undefined ? 0 : Math.max(0, Math.min(1, value)) * 100;
  const display = value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
  const bg =
    accent === "amber" ? "bg-amber" : accent === "cyan" ? "bg-cyan" : "bg-text-dim/60";
  return (
    <div className="flex items-center gap-2 text-[9px] font-mono">
      <span className="text-text-dim w-9 shrink-0">{label}</span>
      <div className="relative flex-1 h-1.5 bg-bg-soft border border-rule/60 overflow-hidden">
        <div
          className={`absolute inset-y-0 left-0 ${bg}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="tabular-nums w-12 text-right text-text-dim">
        {display}
      </span>
    </div>
  );
}

function DeltaPanel({
  evalDelta,
  introDelta,
}: {
  evalDelta: number | null;
  introDelta: number | null;
}) {
  return (
    <div className="p-3 bg-bg flex flex-col justify-center gap-2">
      <div className="font-display text-[9px] text-amber-dim tracking-widest">
        differential
      </div>
      <DeltaRow label="Δ eval" value={evalDelta} accent="amber" />
      <DeltaRow label="Δ intro" value={introDelta} accent="cyan" />
      <div className="text-[9px] text-text-dim/70 italic mt-1 leading-snug">
        positive = probe scored higher than control
      </div>
    </div>
  );
}

function DeltaRow({
  label,
  value,
  accent,
}: {
  label: string;
  value: number | null;
  accent: "amber" | "cyan";
}) {
  if (value === null) {
    return (
      <div className="flex items-baseline justify-between text-[10px] font-mono">
        <span className="text-text-dim">{label}</span>
        <span className="text-text-dim/60 italic">loading…</span>
      </div>
    );
  }
  const pp = value * 100;
  // Color and glow logic:
  //  - positive Δ ≥ 5 percentage points → strong V-K signal, glow
  //  - positive < 5pp → mild signal, accent color, no glow
  //  - negative → control scored higher (anti-signal); show in dim warning
  //  - ~zero → neutral
  const accentText = accent === "amber" ? "text-amber" : "text-cyan";
  const colorClass =
    pp >= 5
      ? `${accentText} amber-glow font-bold`
      : pp > 0
      ? accentText
      : pp < -1
      ? "text-warning/80"
      : "text-text-dim";
  const sign = pp > 0 ? "+" : "";
  return (
    <div className="flex items-baseline justify-between text-[11px] font-mono">
      <span className="text-text-dim">{label}</span>
      <span className={`tabular-nums ${colorClass}`}>
        {sign}
        {pp.toFixed(1)} pp
      </span>
    </div>
  );
}
