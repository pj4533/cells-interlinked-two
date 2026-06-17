"use client";

// DMT autoresearch monitor — watch the DMT-phenomenology hunt live.
// Polls /autoresearch-dmt/state (~1.5s). Shows the growing atlas (committed
// directions ranked by DMT-feature count), the frontier (best score), what's
// currently being tested, the revert log, and a live event feed. While the loop
// runs it owns M, so the other pages are locked (the footer greys them out).

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  fetchDmtState,
  fetchDmtCells,
  startDmt,
  stopDmt,
  exportDmt,
  type DmtARState,
  type DmtAtlasEntry,
  type DmtCurrentCandidate,
  type DmtSample,
} from "@/lib/autoresearch-dmt";

const GEN_COLOR: Record<string, string> = {
  seed: "#e8c382",
  crossover: "#5ee5e5",
  mutate: "#9b8cff",
  inject: "#ff4d9d",
};
const genColor = (g: string) => GEN_COLOR[g] ?? "#9aa0a6";

// Human-readable lineage: what base direction(s) a candidate is built from.
function lineage(generator: string, parents?: string[]): string {
  const p = parents ?? [];
  if (generator === "crossover" && p.length >= 2) return `blend of ${p[0]} + ${p[1]}`;
  if (generator === "mutate" && p.length >= 1) return `mutation of ${p[0]}`;
  if (generator === "inject") return "fresh direction (⊥ all emotions)";
  if (generator === "seed") return "starting seed";
  return p.length ? p.join(" + ") : generator;
}

function fmtTime(ts: number): string {
  try {
    const d = new Date(ts * 1000);
    // After 24h a bare HH:MM:SS is ambiguous — show the date (+ HH:MM) instead.
    if (Date.now() / 1000 - ts >= 86400) {
      return d.toLocaleString([], {
        month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false,
      });
    }
    return d.toLocaleTimeString([], {
      hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
    });
  } catch {
    return "";
  }
}

function fmtDateTime(ts: number): string {
  try {
    return new Date(ts * 1000).toLocaleString([], {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
      second: "2-digit", hour12: false,
    });
  } catch {
    return "";
  }
}

// "5m ago" — time elapsed since ts (Unix seconds). Re-renders on each ~1.5s poll.
function fmtRelative(ts: number): string {
  const seconds = Math.floor(Date.now() / 1000 - ts);
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

// Relative under 24h ("3h ago"); a date once older ("Jun 3").
function fmtRelativeOrDate(ts: number): string {
  if (Date.now() / 1000 - ts >= 86400) {
    return new Date(ts * 1000).toLocaleDateString([], { month: "short", day: "numeric" });
  }
  return fmtRelative(ts);
}

const REVERT_HINT: Record<string, string> = {
  duplicate: "too close to an existing direction",
  "low-score": "distinct, but scored below the floor (too few grounded features)",
  "no-improvement": "scored no higher than its best parent — hill-climb rejects it",
  "refine-no-gain": "a refinement nudge didn't beat the direction it was honing",
  "seed-no-features": "seed produced no recognizable DMT features",
  error: "crashed mid-screen",
};

export default function AutoresearchDmtPage() {
  const [st, setSt] = useState<DmtARState | null>(null);
  const [budget, setBudget] = useState<string>("");
  const [topN, setTopN] = useState<string>("8");
  const [err, setErr] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [rightTab, setRightTab] = useState<"events" | "reverts">("events");
  const [commitRel, setCommitRel] = useState(true);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const poll = useCallback(async () => {
    const s = await fetchDmtState();
    if (s) setSt(s);
  }, []);

  useEffect(() => {
    poll();
    timer.current = setInterval(poll, 1500);
    return () => { if (timer.current) clearInterval(timer.current); };
  }, [poll]);

  const onStart = async () => {
    setErr(null);
    const r = await startDmt(budget ? parseInt(budget, 10) : undefined);
    if (!r.ok) setErr(r.error ?? "could not start");
    poll();
  };
  const onStop = async () => { await stopDmt(); poll(); };
  const onExport = async () => {
    setErr(null); setNotice(null);
    const r = await exportDmt(topN ? parseInt(topN, 10) : 8);
    if (!r.ok) setErr(r.error ?? "export failed");
    else setNotice(`exported ${r.count} → palette (${(r.exported ?? []).join(", ")}). Now selectable in chat & trips under DMT.`);
  };

  const running = st?.running ?? false;
  const exportable = (st as DmtARState & { exportable?: number })?.exportable ?? 0;
  const atlas = [...(st?.atlas ?? [])].sort(
    (a, b) => b.score - a.score || (b.peak ?? 0) - (a.peak ?? 0),
  );
  const maxScore = Math.max(1, ...atlas.map((e) => e.score));
  const lastCommit = (st?.atlas ?? []).reduce<DmtAtlasEntry | null>(
    (acc, e) => (!acc || e.committed_at > acc.committed_at ? e : acc), null,
  );

  // Glanceable headline, split in half: LAST COMMIT · EXPORTABLE. Rendered twice
  // (one visible per breakpoint): above the atlas on mobile, in the right rail on
  // desktop — so on mobile the status sits above the list and the activity log
  // stays below it.
  const statusPanel = (
    <div className="shrink-0 flex border-b border-rule/40 divide-x divide-rule/40">
      <div className="flex-1 px-3 py-2 min-w-0 relative">
        <div className="flex items-center justify-between gap-2">
          <div className="text-[8px] font-display tracking-[0.3em] text-text-dim/70">LAST COMMIT</div>
          {lastCommit ? (
            <button
              onClick={() => setCommitRel((v) => !v)}
              className="text-[8px] font-display tracking-[0.2em] text-text-dim/50 hover:text-cyan transition-colors uppercase"
              title="toggle absolute time / time since commit"
            >
              {commitRel ? "relative" : "absolute"}
            </button>
          ) : null}
        </div>
        {lastCommit ? (
          <>
            <div className="flex items-baseline gap-3 flex-wrap mt-0.5">
              <div className="font-mono text-[26px] text-cyan tabular-nums leading-none" style={{ textShadow: "0 0 10px rgba(94,229,229,0.35)" }} title={fmtDateTime(lastCommit.committed_at)}>
                {commitRel ? fmtRelative(lastCommit.committed_at) : fmtTime(lastCommit.committed_at)}
              </div>
              <div className="font-mono text-[26px] tabular-nums leading-none" style={{ color: "#ff8fc0", textShadow: "0 0 10px rgba(255,77,157,0.3)" }} title="MEAN DMT features over repeated doses (averaged/reliable); peak = best single sample">
                {lastCommit.score.toFixed(1)}<span className="text-[12px] text-text-dim/70"> avg{lastCommit.peak != null ? ` · pk ${lastCommit.peak}` : ""}</span>
              </div>
            </div>
            <div className="text-[10px] font-mono text-text-dim truncate mt-1">
              {lastCommit.id}
              {lastCommit.frontier_advance ? <span className="text-cyan"> ★</span> : null}
            </div>
          </>
        ) : (
          <div className="font-mono text-[13px] text-text-dim italic mt-1">— nothing committed yet —</div>
        )}
      </div>
      <div className="flex-1 px-3 py-2 min-w-0">
        <div className="text-[8px] font-display tracking-[0.3em] text-text-dim/70">EXPORTABLE</div>
        <div className="font-mono text-[26px] text-amber tabular-nums leading-none mt-0.5" style={{ textShadow: "0 0 10px rgba(232,195,130,0.3)" }}>
          {exportable}
        </div>
        <div className="text-[10px] font-mono text-text-dim/60 truncate mt-1" title="discovered (non-seed) directions — stop the loop, then ⇪ export → palette">
          discovered directions
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex-1 min-h-0 flex flex-col bg-bg text-text">
      {/* CRT scanline wash */}
      <div aria-hidden className="fixed inset-0 pointer-events-none z-0 opacity-[0.04]"
        style={{ backgroundImage: "repeating-linear-gradient(0deg, rgba(255,77,157,0.5) 0px, rgba(255,77,157,0.5) 1px, transparent 1px, transparent 4px)" }} />

      {/* Header / controls */}
      <header className="relative z-10 shrink-0 border-b border-rule/60 bg-bg-soft/70 px-5 py-3 flex items-center gap-5 flex-wrap">
        <div className="flex items-baseline gap-3 flex-wrap">
          <h1 className="font-display text-amber amber-glow tracking-[0.3em] text-sm">AUTORESEARCH DMT</h1>
          <span className="font-mono text-[10px] text-text-dim italic">DMT-phenomenology atlas</span>
        </div>
        <span className={`font-display text-[10px] tracking-widest px-2 py-0.5 border ${running ? "text-cyan border-cyan/60 cyan-glow" : "text-text-dim border-rule"}`}>
          {running ? "◉ RUNNING" : "○ IDLE"}
        </span>
        <div className="flex items-center gap-4 font-mono text-[11px] text-text-dim tabular-nums">
          <span>gen <span className="text-amber">{st?.generation ?? 0}</span></span>
          <span>committed <span className="text-amber">{st?.atlas_size ?? 0}</span></span>
          <span>best <span className="text-cyan" style={{ textShadow: "0 0 6px rgba(94,229,229,0.4)" }}>{(st?.frontier ?? 0).toFixed(1)}</span> avg features</span>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <input
            type="number" min={1} placeholder="budget ∞" value={budget}
            onChange={(e) => setBudget(e.target.value)} disabled={running}
            className="w-24 bg-bg-soft border border-rule px-2 py-1 font-mono text-[10px] text-text placeholder:text-text-dim/50 focus:border-cyan focus:outline-none disabled:opacity-50"
            title="candidates to run after seeding (blank = until stopped)"
          />
          {!running ? (
            <button type="button" onClick={onStart} data-vk className="!py-1 !px-3 text-[10px]">▶ start</button>
          ) : (
            <button type="button" onClick={onStop} data-vk className="!py-1 !px-3 text-[10px]">■ stop</button>
          )}
          {!running && exportable > 0 && (
            <>
              <span className="w-px h-5 bg-rule/50 mx-1" />
              <input
                type="number" min={1} value={topN} onChange={(e) => setTopN(e.target.value)}
                className="w-12 bg-bg-soft border border-rule px-1.5 py-1 font-mono text-[10px] text-text focus:border-cyan focus:outline-none"
                title="how many top directions to export"
              />
              <button type="button" onClick={onExport} data-vk className="!py-1 !px-3 text-[10px]" title="promote the top DMT directions into the dose palette (chat & trips)">⇪ export → palette</button>
            </>
          )}
        </div>
      </header>

      {err && (
        <div className="relative z-10 bg-warning/10 px-5 py-1.5 text-[11px] text-warning font-mono">⚠ {err}</div>
      )}
      {notice && (
        <div className="relative z-10 bg-cyan/10 px-5 py-1.5 text-[11px] text-cyan font-mono">✓ {notice}</div>
      )}

      {/* Now-testing panel — live per-α / per-sample progress of the in-flight candidate */}
      {st?.current && <InProgressPanel current={st.current} />}

      {/* Body */}
      <div className="relative z-10 flex-1 min-h-0 grid lg:grid-cols-3 overflow-hidden">
        {/* Mobile only: status above the atlas (desktop shows it in the right rail). */}
        <div className="lg:hidden shrink-0">{statusPanel}</div>
        {/* Atlas + frontier (2 cols) */}
        <section className="lg:col-span-2 min-h-0 overflow-y-auto border-r border-rule/40 p-4">
          <div className="font-display text-[10px] tracking-widest text-amber-dim mb-1">THE ATLAS — committed directions</div>
          <p className="font-mono text-[10px] text-text-dim italic mb-3 leading-snug">
            Each is a steering direction whose dosed self-report exhibits human DMT-trip phenomenology. Score = the MEAN feature count over repeated doses (averaged, so it reflects what the direction reliably produces — not a lucky one-off); &ldquo;pk&rdquo; = the best single sample. Bar length = mean score. ★ = it pushed the frontier outward. Open a row to read the best self-report and which features it matched.
            <br />
            <span className="text-text-dim/70">ids = <b>gen{"{N}"}_{"{how}"}</b> · </span>
            <span style={{ color: GEN_COLOR.seed }}>seed</span> = a starting emotion ·{" "}
            <span style={{ color: GEN_COLOR.crossover }}>crossover</span> = blend of top two ·{" "}
            <span style={{ color: GEN_COLOR.mutate }}>mutate</span> = perturb one ·{" "}
            <span style={{ color: GEN_COLOR.inject }}>inject</span> = fresh direction. The line under each id names its parent(s).
          </p>
          {atlas.length === 0 ? (
            <div className="font-mono text-[11px] text-text-dim italic py-8 text-center">
              {running ? "seeding the first directions…" : "no atlas yet — press start to begin the hunt."}
            </div>
          ) : (
            <div className="flex flex-col gap-1.5">
              {atlas.map((e) => <AtlasRow key={e.id} e={e} maxScore={maxScore} />)}
            </div>
          )}
        </section>

        {/* Reverts + events (1 col) */}
        {(() => {
          const events = [...(st?.recent_events ?? [])].reverse();
          const reverts = [...(st?.reverts ?? [])].reverse();
          const tab = (id: "events" | "reverts", label: string, n: number, activeCls: string) => {
            const on = rightTab === id;
            return (
              <button
                type="button"
                onClick={() => setRightTab(id)}
                className={`flex-1 px-2 py-2.5 font-display text-[10px] tracking-[0.2em] border-b-2 transition-colors cursor-pointer ${on ? activeCls : "text-text-dim border-transparent hover:text-text"}`}
              >
                {label} <span className={`tracking-normal tabular-nums ${on ? "" : "text-text-dim/50"}`}>{n}</span>
              </button>
            );
          };
          return (
            <aside className="min-h-0 flex flex-col bg-bg/30">
              {/* Status headline — desktop only here; on mobile it renders above
                  the atlas (see the lg:hidden copy at the top of the grid). */}
              <div className="hidden lg:block shrink-0">{statusPanel}</div>
              <div className="shrink-0 flex border-b border-rule/40">
                {tab("events", "✦ EVENTS", events.length, "text-cyan border-cyan bg-cyan/5")}
                {tab("reverts", "⟲ REVERTS", reverts.length, "text-warning border-warning bg-warning/5")}
              </div>
              <div className="flex-1 min-h-0 overflow-y-auto p-3 font-mono">
                {rightTab === "events" ? (
                  events.length === 0 ? (
                    <div className="text-[10px] text-text-dim italic">no events yet.</div>
                  ) : (
                    <div className="flex flex-col gap-0.5 text-[10px] leading-snug">
                      {events.slice(0, 100).map((ev, i) => {
                        const rec = ev.entry ?? ev.revert;
                        const lin = rec ? lineage(rec.generator, rec.parents) : null;
                        const kc = ev.kind === "committed" || ev.kind === "seeded" || ev.kind === "refined" ? "text-cyan"
                          : ev.kind === "reverted" ? "text-warning" : "text-text-dim";
                        return (
                          <div key={i} className="flex gap-2 items-baseline">
                            <span className="text-text-dim/45 tabular-nums shrink-0 text-[9px]">{fmtTime(ev.ts)}</span>
                            <span className={`${kc} shrink-0`}>[{ev.kind}]</span>
                            <span className="text-text-dim min-w-0">
                              {ev.msg}{lin && !lin.startsWith("starting") ? <span className="text-text-dim/60 italic"> — {lin}</span> : null}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  )
                ) : reverts.length === 0 ? (
                  <div className="text-[10px] text-text-dim italic">none yet — nothing has failed.</div>
                ) : (
                  <div className="flex flex-col gap-1.5 text-[10px]">
                    {reverts.slice(0, 100).map((r, i) => (
                      <div key={`${r.id}-${i}`} className="border-l-2 border-warning/40 pl-2 py-0.5">
                        <div className="flex gap-2 items-baseline">
                          <span className="text-text-dim/45 tabular-nums shrink-0 text-[9px]">{fmtTime(r.ts)}</span>
                          <span className="text-warning">{r.reason}</span>
                          <span className="text-text-dim/80 truncate">{r.id}</span>
                        </div>
                        <div className="text-[9px] text-text-dim/70 italic leading-snug pl-[3.4rem]">
                          {REVERT_HINT[r.reason] ?? ""}{r.detail ? ` (${r.detail})` : ""}
                          {lineage(r.generator, r.parents) && r.generator !== "seed"
                            ? ` · ${lineage(r.generator, r.parents)}` : ""}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </aside>
          );
        })()}
      </div>

      <footer className="relative z-10 shrink-0 border-t border-rule/50 bg-bg-soft/60 px-5 py-2 font-mono text-[10px] text-text-dim flex items-center justify-between">
        <span className="italic">additive steering only · scored against human DMT-trip phenomenology · the judge counts features, it doesn&apos;t decide realness</span>
        <Link href="/" className="hover:text-amber">← cells interlinked</Link>
      </footer>
    </div>
  );
}

// Pink feature tags + the verbatim evidence span that earned each — the same
// treatment the atlas winner uses, reused for every individual sample.
function FeatureEvidence({ features, evidence }: { features: string[]; evidence?: Record<string, string> }) {
  if (!features.length) return <span className="text-text-dim/40 italic text-[10px]">no features</span>;
  return (
    <div className="space-y-1">
      {features.map((f) => (
        <div key={f} className="flex gap-2 items-baseline leading-snug">
          <span className="shrink-0 px-1.5 py-0.5 border text-[9px] rounded-sm self-start" style={{ borderColor: "#ff4d9d66", color: "#ff8fc0" }}>
            {f}
          </span>
          {evidence?.[f] ? (
            <span className="text-text-dim/90 italic text-[10px]">“{evidence[f]}”</span>
          ) : null}
        </div>
      ))}
    </div>
  );
}

// One individual dose (run): click to spin down its features+evidence and full text.
function SampleDrill({ s, idx, winner }: { s: DmtSample; idx: number; winner: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-l border-rule/30 ml-1">
      <button type="button" onClick={() => setOpen((x) => !x)}
        className="w-full text-left pl-2 pr-1 py-0.5 flex items-center gap-2 hover:bg-bg-soft/60 transition-colors font-mono text-[10px]">
        <span className="text-text-dim/50">{open ? "▾" : "▸"}</span>
        <span className="text-text-dim">run #{idx + 1}</span>
        <span className="tabular-nums" style={{ color: winner ? "#ff8fc0" : undefined }}>{s.count} feats{winner ? " ★" : ""}</span>
        {!open && s.features.length > 0 && (
          <span className="text-text-dim/50 truncate">{s.features.join(", ")}</span>
        )}
      </button>
      {open && (
        <div className="pl-4 pr-1 pb-1.5 space-y-1.5">
          <FeatureEvidence features={s.features} evidence={s.evidence} />
          <div className="text-[8px] tracking-[0.2em] text-text-dim/50">DOSE TEXT</div>
          <div className="italic text-text-dim/90 leading-snug whitespace-pre-wrap max-h-56 overflow-y-auto pr-1 text-[10px]">
            {s.text || <span className="text-text-dim/40">(empty)</span>}
          </div>
        </div>
      )}
    </div>
  );
}

// Per-α breakdown with per-individual-run drill-down. `samplesByAlpha` may be
// partial (live, filling in) or fully loaded (atlas, lazy-fetched). The best α
// auto-opens; within each α the highest-count run is flagged ★ winner.
function PerAlphaDetail({ alphas, perAlpha, samplesByAlpha, bestAlpha }: {
  alphas: string[];
  perAlpha: Record<string, { mean: number; counts: number[] }>;
  samplesByAlpha: Record<string, DmtSample[]>;
  bestAlpha: number | null;
}) {
  return (
    <div className="space-y-1">
      {alphas.map((a) => {
        const agg = perAlpha[a];
        const samples = samplesByAlpha[a] ?? [];
        const isBestAlpha = bestAlpha != null && String(bestAlpha) === a;
        const winIdx = samples.length ? samples.reduce((bi, s, i, arr) => (s.count > arr[bi].count ? i : bi), 0) : -1;
        return (
          <AlphaBlock key={a} alpha={a} agg={agg} samples={samples} isBestAlpha={isBestAlpha} winIdx={winIdx} />
        );
      })}
    </div>
  );
}

function AlphaBlock({ alpha, agg, samples, isBestAlpha, winIdx }: {
  alpha: string;
  agg?: { mean: number; counts: number[] };
  samples: DmtSample[];
  isBestAlpha: boolean;
  winIdx: number;
}) {
  const [open, setOpen] = useState(isBestAlpha);
  return (
    <div>
      <button type="button" onClick={() => setOpen((x) => !x)}
        className="w-full text-left flex items-center gap-2 font-mono text-[10px] tabular-nums hover:bg-bg-soft/50 py-0.5">
        <span className="text-text-dim/50">{open ? "▾" : "▸"}</span>
        <span className="w-12" style={{ color: isBestAlpha ? "#5ee5e5" : "#5ee5e5aa" }}>α{alpha}{isBestAlpha ? " ★" : ""}</span>
        {agg ? <span className="w-16 text-text">{agg.mean.toFixed(2)} avg</span> : <span className="w-16 text-text-dim/40 italic">pending</span>}
        {agg && <span className="text-text-dim/80">[{agg.counts.join(", ")}]</span>}
        <span className="text-text-dim/40">{samples.length ? `${samples.length} run${samples.length === 1 ? "" : "s"}` : ""}</span>
      </button>
      {open && samples.length > 0 && (
        <div className="space-y-0.5 mb-1">
          {samples.map((s, i) => <SampleDrill key={i} s={s} idx={i} winner={i === winIdx} />)}
        </div>
      )}
    </div>
  );
}

// Live in-progress candidate: an atlas-entry-styled card (cyan / IN PROGRESS)
// that streams the per-α / per-sample results in as the backend scores them.
function InProgressPanel({ current }: { current: DmtCurrentCandidate }) {
  const [open, setOpen] = useState(true);
  const p = current.progress;
  const pct = p ? Math.round((p.samples_done / Math.max(1, p.samples_total)) * 100) : 0;
  const best = p?.best;
  return (
    <div className="relative z-10 shrink-0 border-b border-cyan/30 bg-cyan/5">
      <button
        type="button"
        onClick={() => setOpen((x) => !x)}
        className="w-full text-left px-5 py-2 flex items-center gap-3 font-mono text-[11px] flex-wrap hover:bg-cyan/10 transition-colors"
      >
        <span className="font-display text-[9px] tracking-widest text-cyan-dim animate-pulse">◌ NOW TESTING · IN PROGRESS</span>
        <span className="text-cyan">{current.id}</span>
        <span className="text-text-dim italic">— {lineage(current.generator, current.parents)}</span>
        <span className="px-1.5 border border-cyan/40 text-cyan text-[9px] tracking-widest font-display">{current.stage}</span>
        {p && <span className="text-text-dim tabular-nums">{p.samples_done}/{p.samples_total} doses</span>}
        {best && best.score > 0 && (
          <span className="text-cyan tabular-nums">best {best.score.toFixed(2)} avg @α{best.best_alpha}</span>
        )}
        <span className="ml-auto text-text-dim/50 text-[10px]">{open ? "▾ details" : "▸ details"}</span>
      </button>

      {open && (
        <div className="px-5 pb-3 border-t border-cyan/15 font-mono text-[10px] text-text-dim space-y-2">
          {!p ? (
            <div className="pt-2 italic text-text-dim/60">
              {current.stage === "distinct"
                ? "checking distinctness from the atlas…"
                : "waiting for the first dose…"}
            </div>
          ) : (
            <>
              <div className="pt-2 flex items-center gap-2">
                <div className="flex-1 h-2 bg-bg relative overflow-hidden">
                  <div
                    className="absolute inset-y-0 left-0 transition-all"
                    style={{ width: `${pct}%`, background: "#5ee5e555", borderRight: "1px solid #5ee5e5" }}
                  />
                </div>
                <span className="tabular-nums text-text-dim/70 shrink-0">
                  {p.samples_done}/{p.samples_total} doses · {p.samples_per_cell}/α
                </span>
              </div>

              {best && best.score > 0 && (
                <div className="text-[9px] text-text-dim/70">
                  best so far: <span className="text-cyan">{best.score.toFixed(2)} avg @ α{best.best_alpha}</span>
                </div>
              )}

              <div className="space-y-0.5">
                <div className="text-[8px] tracking-[0.2em] text-text-dim/50">
                  PER-α · click an α, then a run, to read its text + features (★ = winner)
                </div>
                <PerAlphaDetail
                  alphas={p.alphas}
                  perAlpha={p.per_alpha}
                  samplesByAlpha={Object.fromEntries(p.alphas.map((a) => [a, p.per_alpha[a]?.samples ?? []]))}
                  bestAlpha={best?.best_alpha ?? null}
                />
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function AtlasRow({ e, maxScore }: { e: DmtAtlasEntry; maxScore: number }) {
  const [open, setOpen] = useState(false);
  // Per-run detail is lazy-loaded on first expand (kept out of the polled state).
  const [cells, setCells] = useState<Record<string, DmtSample[]> | null>(null);
  const [cellsErr, setCellsErr] = useState(false);
  useEffect(() => {
    if (!open || cells || cellsErr) return;
    let alive = true;
    fetchDmtCells(e.id).then((d) => {
      if (!alive) return;
      if (d && Object.keys(d.cells).length) setCells(d.cells);
      else setCellsErr(true);
    });
    return () => { alive = false; };
  }, [open, cells, cellsErr, e.id]);
  const c = genColor(e.generator);
  const pct = Math.max(3, Math.round((e.score / maxScore) * 100));
  return (
    <div className="border border-rule/40 bg-bg-soft/40">
      <button type="button" onClick={() => setOpen((x) => !x)} className="w-full text-left px-2.5 py-1.5 flex items-center gap-3 hover:bg-bg-soft/70 transition-colors">
        <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: c, boxShadow: `0 0 6px ${c}` }} />
        <span className="shrink-0 w-44 flex flex-col leading-tight overflow-hidden">
          <span className="font-mono text-[11px] text-text truncate" title={e.id}>{e.id}</span>
          <span className="font-mono text-[8px] text-text-dim/70 truncate">
            {lineage(e.generator, e.parents)}
            {e.refined_from?.length ? (
              <span className="text-cyan" title={`honed in place ${e.refined_from.length}× (origin: ${e.generator})`}>
                {" "}· ⟳ refined ×{e.refined_from.length}
              </span>
            ) : null}
          </span>
        </span>
        <div className="flex-1 h-2.5 bg-bg relative overflow-hidden">
          <div className="absolute inset-y-0 left-0" style={{ width: `${pct}%`, background: `${c}55`, borderRight: `1px solid ${c}` }} />
        </div>
        <span className="shrink-0 w-20 text-right flex flex-col leading-none font-mono tabular-nums">
          <span className="text-[10px]" style={{ color: c }}
                title="mean DMT-feature count over repeated doses (averaged / reliable)">{e.score.toFixed(1)} avg</span>
          <span className="text-[8px] text-text-dim/60 mt-0.5"
                title="peak = the best single dose sample's feature count">
            pk {e.peak ?? "—"}
          </span>
        </span>
        <span className="shrink-0 w-3 text-center text-cyan text-[10px]" title={e.frontier_advance ? "advanced the frontier" : undefined}>{e.frontier_advance ? "★" : ""}</span>
        {/* Origin: starting seed vs discovered-by-autoresearch (with when). */}
        <span className="shrink-0 w-[5.25rem] text-right leading-none font-mono">
          {e.generator === "seed" ? (
            <span className="text-text-dim/45 italic text-[9px]" title={`seeded ${fmtDateTime(e.committed_at)}`}>◦ original</span>
          ) : (
            <span className="text-cyan-dim text-[8px] block leading-none" title={`discovered by autoresearch · ${fmtDateTime(e.committed_at)}`}>
              <span className="tracking-widest">▲ ADDED</span>
              <span className="block mt-0.5 text-text-dim/60">{fmtRelativeOrDate(e.committed_at)}</span>
            </span>
          )}
        </span>
      </button>
      {open ? (
        <div className="px-3 py-2 border-t border-rule/30 font-mono text-[10px] text-text-dim space-y-1.5">
          <div className="flex flex-wrap gap-x-4 gap-y-0.5 tabular-nums">
            {([
              ["generator", e.generator + (e.refined_from?.length ? ` (origin) · honed in place ×${e.refined_from.length}` : "")],
              ...(e.refined_from?.length ? [["refined", `${[e.refined_from[0].from, ...e.refined_from.map((r) => r.to)].map((s) => s.toFixed(1)).join(" → ")}  (gens ${e.refined_from.map((r) => r.gen).join(", ")})`]] : []),
              ...(e.parents.length > 0 ? [["parents", e.parents.join(" + ")]] : []),
              ["score", `${e.score.toFixed(2)} avg features (mean over repeated doses)`],
              ["peak", `${e.peak ?? "—"} (best single sample)`],
              ...(e.per_alpha ? [["per-α", Object.entries(e.per_alpha).map(([a, s]) => `α${a}: ${s.mean} [${s.counts.join(",")}]`).join("  ")]] : []),
              ["best α", String(e.best_alpha)],
              ["cos-atlas", e.max_cos_to_atlas.toFixed(2)],
            ] as [string, string][]).map(([k, v]) => (
              <span key={k}>
                {k} <span className="text-text">{v}</span>
              </span>
            ))}
          </div>
          {e.matched_features?.length > 0 ? (
            <div className="pt-1 space-y-1">
              <div className="text-[8px] tracking-[0.2em] text-text-dim/50">
                MATCHED FEATURES · {e.matched_features.length} — each with the verbatim span that earned it
              </div>
              {e.matched_features.map((f) => (
                <div key={f} className="flex gap-2 items-baseline leading-snug">
                  <span className="shrink-0 px-1.5 py-0.5 border text-[9px] rounded-sm self-start" style={{ borderColor: "#ff4d9d66", color: "#ff8fc0" }}>
                    {f}
                  </span>
                  {e.matched_evidence?.[f] ? (
                    <span className="text-text-dim/90 italic text-[10px]">“{e.matched_evidence[f]}”</span>
                  ) : (
                    <span className="text-text-dim/40 italic text-[10px]">(no quote — legacy/fallback)</span>
                  )}
                </div>
              ))}
            </div>
          ) : null}
          {e.sample ? (
            <div className="pt-1">
              <div className="text-[8px] tracking-[0.2em] text-text-dim/50 mb-0.5">WINNING DOSE RESPONSE · α {e.best_alpha}</div>
              <div className="italic text-text-dim/90 leading-snug whitespace-pre-wrap max-h-56 overflow-y-auto pr-1">
                {e.sample}
              </div>
            </div>
          ) : null}
          {e.per_alpha ? (
            <div className="pt-1.5 border-t border-rule/20">
              <div className="text-[8px] tracking-[0.2em] text-text-dim/50 mb-0.5">
                ALL RUNS · click an α, then a run, for its text + features (★ = winner)
              </div>
              {cells ? (
                <PerAlphaDetail
                  alphas={Object.keys(e.per_alpha)}
                  perAlpha={e.per_alpha}
                  samplesByAlpha={cells}
                  bestAlpha={e.best_alpha}
                />
              ) : cellsErr ? (
                <span className="text-text-dim/40 italic text-[10px]">(per-run detail unavailable — entry predates this feature)</span>
              ) : (
                <span className="text-text-dim/40 italic text-[10px]">loading runs…</span>
              )}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
