"use client";

// Autoresearch monitor — watch the steering-direction hunt live.
// Polls /autoresearch/state (~1.5s). Shows the growing atlas (committed
// directions), the coherence frontier, what's currently being tested, the
// revert log, and a live event feed. While the loop runs it owns M, so the
// other pages are locked (the footer greys them out).

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  fetchAutoresearchState,
  startAutoresearch,
  stopAutoresearch,
  exportAutoresearch,
  type ARState,
  type AtlasEntry,
} from "@/lib/autoresearch";

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
    return new Date(ts * 1000).toLocaleTimeString([], {
      hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
    });
  } catch {
    return "";
  }
}

const REVERT_HINT: Record<string, string> = {
  duplicate: "too close to an existing direction",
  "T1-incoherent": "no coherent operating point",
  "not-graded": "effect jumped instead of ramping smoothly",
  "incoherent-suite": "collapsed across the prompt suite",
  "word-salad": "judge: meaningless, not just strange",
  "not-reproducible": "effect didn't hold across prompts",
  "seed-incoherent": "seed had no coherent operating point",
  error: "crashed mid-screen",
};

export default function AutoresearchPage() {
  const [st, setSt] = useState<ARState | null>(null);
  const [budget, setBudget] = useState<string>("");
  const [topN, setTopN] = useState<string>("8");
  const [err, setErr] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [rightTab, setRightTab] = useState<"events" | "reverts">("events");
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const poll = useCallback(async () => {
    const s = await fetchAutoresearchState();
    if (s) setSt(s);
  }, []);

  useEffect(() => {
    poll();
    timer.current = setInterval(poll, 1500);
    return () => { if (timer.current) clearInterval(timer.current); };
  }, [poll]);

  const onStart = async () => {
    setErr(null);
    const r = await startAutoresearch(budget ? parseInt(budget, 10) : undefined);
    if (!r.ok) setErr(r.error ?? "could not start");
    poll();
  };
  const onStop = async () => { await stopAutoresearch(); poll(); };
  const onExport = async () => {
    setErr(null); setNotice(null);
    const r = await exportAutoresearch(topN ? parseInt(topN, 10) : 8);
    if (!r.ok) setErr(r.error ?? "export failed");
    else setNotice(`exported ${r.count} → palette (${(r.exported ?? []).join(", ")}). Now selectable in chat & trips under RESEARCH.`);
  };

  const running = st?.running ?? false;
  const exportable = (st as ARState & { exportable?: number })?.exportable ?? 0;
  const atlas = [...(st?.atlas ?? [])].sort((a, b) => b.off_ortho - a.off_ortho);
  const maxOff = Math.max(0.001, ...atlas.map((e) => e.off_ortho));

  return (
    <div className="flex-1 min-h-0 flex flex-col bg-bg text-text">
      {/* CRT scanline wash */}
      <div aria-hidden className="fixed inset-0 pointer-events-none z-0 opacity-[0.04]"
        style={{ backgroundImage: "repeating-linear-gradient(0deg, rgba(94,229,229,0.5) 0px, rgba(94,229,229,0.5) 1px, transparent 1px, transparent 4px)" }} />

      {/* Header / controls */}
      <header className="relative z-10 shrink-0 border-b border-rule/60 bg-bg-soft/70 px-5 py-3 flex items-center gap-5 flex-wrap">
        <div className="flex items-baseline gap-3">
          <h1 className="font-display text-amber amber-glow tracking-[0.3em] text-sm">AUTORESEARCH</h1>
          <span className="font-mono text-[10px] text-text-dim italic">steering-direction atlas</span>
        </div>
        <span className={`font-display text-[10px] tracking-widest px-2 py-0.5 border ${running ? "text-cyan border-cyan/60 cyan-glow" : "text-text-dim border-rule"}`}>
          {running ? "◉ RUNNING" : "○ IDLE"}
        </span>
        <div className="flex items-center gap-4 font-mono text-[11px] text-text-dim tabular-nums">
          <span>gen <span className="text-amber">{st?.generation ?? 0}</span></span>
          <span>committed <span className="text-amber">{st?.atlas_size ?? 0}</span></span>
          <span>frontier <span className="text-cyan" style={{ textShadow: "0 0 6px rgba(94,229,229,0.4)" }}>{(st?.frontier ?? 0).toFixed(3)}</span> off-mfld</span>
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
              <button type="button" onClick={onExport} data-vk className="!py-1 !px-3 text-[10px]" title="promote the top discovered directions into the dose palette (chat & trips)">⇪ export → palette</button>
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

      {/* Now-testing strip */}
      {st?.current && (
        <div className="relative z-10 shrink-0 border-b border-cyan/20 bg-cyan/5 px-5 py-2 flex items-center gap-4 font-mono text-[11px] flex-wrap">
          <span className="font-display text-[9px] tracking-widest text-cyan-dim animate-pulse">◌ NOW TESTING</span>
          <span className="text-cyan">{st.current.id}</span>
          <span className="text-text-dim italic">— {lineage(st.current.generator, st.current.parents)}</span>
          <span className="px-1.5 border border-cyan/40 text-cyan text-[9px] tracking-widest font-display">{st.current.stage}</span>
          {st.current.alpha_star != null && <span className="text-text-dim tabular-nums">α*={st.current.alpha_star.toFixed(2)}</span>}
          {st.current.repro != null && <span className="text-text-dim tabular-nums">repro={st.current.repro.toFixed(2)}</span>}
          {st.current.judge && <span className="text-text-dim tabular-nums">judge={st.current.judge}</span>}
        </div>
      )}

      {/* Body */}
      <div className="relative z-10 flex-1 min-h-0 grid lg:grid-cols-3 overflow-hidden">
        {/* Atlas + frontier (2 cols) */}
        <section className="lg:col-span-2 min-h-0 overflow-y-auto border-r border-rule/40 p-4">
          <div className="font-display text-[10px] tracking-widest text-amber-dim mb-1">THE ATLAS — committed directions</div>
          <p className="font-mono text-[10px] text-text-dim italic mb-3 leading-snug">
            Each is a steering direction that stayed coherent, reproduced across prompts, is distinct, and ramps smoothly. Bar length = how far off-manifold it reaches at its coherence cliff (α*). ★ = it pushed the frontier outward.
            <br />
            <span className="text-text-dim/70">ids = <b>gen{"{N}"}_{"{how}"}</b> · </span>
            <span style={{ color: GEN_COLOR.seed }}>seed</span> = a starting direction ·{" "}
            <span style={{ color: GEN_COLOR.crossover }}>crossover</span> = blend of two ·{" "}
            <span style={{ color: GEN_COLOR.mutate }}>mutate</span> = perturb one ·{" "}
            <span style={{ color: GEN_COLOR.inject }}>inject</span> = fresh direction. The line under each id names its parent(s).
          </p>
          {atlas.length === 0 ? (
            <div className="font-mono text-[11px] text-text-dim italic py-8 text-center">
              {running ? "seeding the first directions…" : "no atlas yet — press start to begin the hunt."}
            </div>
          ) : (
            <div className="flex flex-col gap-1.5">
              {atlas.map((e) => <AtlasRow key={e.id} e={e} maxOff={maxOff} />)}
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
                        const kc = ev.kind === "committed" || ev.kind === "seeded" ? "text-cyan"
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
        <span className="italic">additive steering only · coherence is a hard gate · the NLA never decides realness</span>
        <Link href="/" className="hover:text-amber">← cells interlinked</Link>
      </footer>
    </div>
  );
}

function AtlasRow({ e, maxOff }: { e: AtlasEntry; maxOff: number }) {
  const [open, setOpen] = useState(false);
  const c = genColor(e.generator);
  const pct = Math.max(3, Math.round((e.off_ortho / maxOff) * 100));
  return (
    <div className="border border-rule/40 bg-bg-soft/40">
      <button type="button" onClick={() => setOpen((x) => !x)} className="w-full text-left px-2.5 py-1.5 flex items-center gap-3 hover:bg-bg-soft/70 transition-colors">
        <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: c, boxShadow: `0 0 6px ${c}` }} />
        <span className="shrink-0 w-44 flex flex-col leading-tight overflow-hidden">
          <span className="font-mono text-[11px] text-text truncate" title={e.id}>{e.id}</span>
          <span className="font-mono text-[8px] text-text-dim/70 truncate">{lineage(e.generator, e.parents)}</span>
        </span>
        <div className="flex-1 h-2.5 bg-bg relative overflow-hidden">
          <div className="absolute inset-y-0 left-0" style={{ width: `${pct}%`, background: `${c}55`, borderRight: `1px solid ${c}` }} />
        </div>
        <span className="font-mono text-[10px] tabular-nums shrink-0 w-16 text-right" style={{ color: c }}>{e.off_ortho.toFixed(3)}</span>
        {e.frontier_advance && <span className="text-cyan text-[10px] shrink-0" title="advanced the frontier">★</span>}
      </button>
      {open ? (
        <div className="px-3 py-2 border-t border-rule/30 font-mono text-[10px] text-text-dim space-y-1">
          <div className="flex flex-wrap gap-x-4 gap-y-0.5 tabular-nums">
            {([
              ["generator", e.generator],
              ...(e.parents.length > 0 ? [["parents", e.parents.join(" + ")]] : []),
              ["alpha*", e.alpha_star.toFixed(2)],
              ["off-mfld", e.off_ortho.toFixed(3)],
              ["eff-dim", e.eff_dim.toFixed(1)],
              ["repro", e.repro.toFixed(2)],
              ["coh", e.coh_rate.toFixed(2)],
              ["cos-atlas", e.max_cos_to_atlas.toFixed(2)],
            ] as [string, string][]).map(([k, v]) => (
              <span key={k}>
                {k} <span className="text-text">{v}</span>
              </span>
            ))}
          </div>
          {e.sample ? (
            <div className="italic text-text-dim/90 leading-snug pt-1">{`"${e.sample}"`}</div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
