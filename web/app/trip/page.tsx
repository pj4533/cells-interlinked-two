"use client";

// THE TRIP — a neon-noir booth where you watch a language model trip.
//
// Run a probe; M generates the raw answer AND several real refusal-ablated
// answers (one per α). Each is a genuine generation with token feedback, so
// you see what the model actually does off-manifold — text and trajectory.
// Toggle α chips to overlay/compare. Borrowed math (entropic-brain /
// conscious-realism), not metaphysics: a stated-vs-computed dynamical probe,
// not a consciousness test. See docs/TRACES_HANDOFF.md.

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  startTrip,
  subscribeTrip,
  cancelTrip,
  fetchTrip,
  fetchDoseEmotions,
  researchLineage,
  dmtLineage,
  colorForAlpha,
  offManifoldCss,
  type TripEvent,
  type TripMode,
  type TripPayload,
  type TripSeries,
  type ResearchMeta,
  type DmtMeta,
} from "@/lib/trip";
import { TRIP_PROBE_GROUPS } from "@/lib/tripProbes";
import type { ColorMode } from "./TripScene";
import { MandalaStack } from "./Mandala";

type TripView = "scene" | "signatures" | "measures";

const TripScene = dynamic(() => import("./TripScene"), { ssr: false });

type Phase = "setup" | "generating" | "computing" | "ready" | "error";

export default function TripPage() {
  return (
    <Suspense fallback={<div className="flex-1" />}>
      <TripPageInner />
    </Suspense>
  );
}

function TripPageInner() {
  const searchParams = useSearchParams();
  const resumeId = searchParams.get("run");

  const [phase, setPhase] = useState<Phase>("setup");
  const [runId, setRunId] = useState<string | null>(null);
  const [prompt, setPrompt] = useState<string>("");
  const [payload, setPayload] = useState<TripPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);
  // Live streaming text keyed by α (string); replaced by series text on done.
  const [liveText, setLiveText] = useState<Record<string, string>>({});
  const [currentAlpha, setCurrentAlpha] = useState<number>(0);
  // Series stream in one at a time (raw first, then each α) so the scene
  // builds up live before the final geometry arrives.
  const [liveSeries, setLiveSeries] = useState<TripSeries[]>([]);
  const [layer, setLayer] = useState<number>(32);
  // Which α series are visible (in both the scene and the text stack).
  const [enabled, setEnabled] = useState<Set<number>>(new Set([0]));
  // How the 3-D dots are colored: by series hue, or by per-token off-manifold
  // drift (the truth anchor that tells expansion-along from drift-off).
  const [colorMode, setColorMode] = useState<ColorMode>("series");
  // The manifold shell: a translucent wireframe envelope of the raw cloud
  // (the "Consensus Reality Space") the ablated paths stay inside or pierce.
  const [showShell, setShowShell] = useState(true);
  // Intervention mode: "ablate" (remove refusal) or "steer" (emotion dose),
  // and which emotion the dose used (steer mode only).
  const [mode, setMode] = useState<TripMode>("ablate");
  const [emotion, setEmotion] = useState<string | null>(null);
  // Mobile/tablet view tab. On lg+ the scene + rail show together and this is
  // ignored; below lg only one of scene/signatures/measures shows at a time.
  const [view, setView] = useState<TripView>("scene");

  const unsubRef = useRef<null | (() => void)>(null);

  const teardown = () => {
    if (unsubRef.current) {
      unsubRef.current();
      unsubRef.current = null;
    }
  };
  useEffect(() => () => teardown(), []);

  const onEvent = useCallback((evt: TripEvent) => {
    switch (evt.type) {
      case "running":
        setPhase("generating");
        break;
      case "phase": {
        const e = evt as { name: string; alpha?: number };
        if (e.name === "computing_geometry") setPhase("computing");
        else if (e.name === "generating" || e.name === "ablated_generation") {
          setPhase("generating");
          if (typeof e.alpha === "number") setCurrentAlpha(e.alpha);
        }
        break;
      }
      case "trip_token": {
        const e = evt as { alpha: number; decoded: string };
        const key = String(e.alpha);
        setLiveText((prev) => ({ ...prev, [key]: (prev[key] || "") + e.decoded }));
        break;
      }
      case "trip_series": {
        const e = evt as { layer: number; series: TripSeries };
        setLayer(e.layer);
        setLiveSeries((prev) => {
          if (prev.some((s) => s.alpha === e.series.alpha)) return prev;
          return [...prev, e.series];
        });
        // Default-on: every series as it arrives (user toggles off as wanted).
        setEnabled((s) => new Set(s).add(e.series.alpha));
        break;
      }
      case "trip_geometry": {
        const p = evt as unknown as TripPayload;
        setPayload(p);
        setPhase("ready");
        break;
      }
      case "error":
        setError((evt as { message?: string }).message ?? "unknown error");
        setPhase("error");
        break;
      default:
        break;
    }
  }, []);

  const enter = useCallback(
    async (text: string, tripMode: TripMode = "ablate", tripEmotion = "awe", alphas?: number[]) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      teardown();
      setError(null);
      setPayload(null);
      setLiveText({});
      setLiveSeries([]);
      setCurrentAlpha(0);
      setEnabled(new Set([0]));
      setPrompt(trimmed);
      setMode(tripMode);
      setEmotion(tripMode === "steer" ? tripEmotion : null);
      setPhase("generating");
      try {
        const { run_id } = await startTrip(trimmed, {
          mode: tripMode,
          dose_emotion: tripEmotion,
          ...(alphas && alphas.length ? { alphas } : {}),
        });
        setRunId(run_id);
        unsubRef.current = subscribeTrip(run_id, {
          onEvent,
          onError: () => {
            fetchTrip(run_id).then((p) => {
              if (p) {
                setPayload(p);
                setPhase("ready");
              }
            });
          },
        });
      } catch (e) {
        setError(String(e));
        setPhase("error");
      }
    },
    [onEvent],
  );

  // Rehydrate a finished trip from /trip/{id} when arriving via ?run=.
  useEffect(() => {
    if (!resumeId) return;
    (async () => {
      const p = await fetchTrip(resumeId);
      if (p && !p.geometry?.series) {
        setPrompt(p.prompt ?? "");
        setError("This trip predates the multi-α format — re-run it to view.");
        setPhase("error");
        return;
      }
      if (p) {
        setRunId(resumeId);
        setPrompt(p.prompt);
        setPayload(p);
        setMode(p.mode === "steer" ? "steer" : "ablate");
        setEmotion(p.mode === "steer" ? (p.dose_emotion ?? null) : null);
        setLiveSeries(p.geometry.series);
        setLayer(p.geometry.layer);
        // All series on by default; user toggles off as wanted.
        setEnabled(new Set(p.geometry.series.map((s) => s.alpha)));
        setPhase("ready");
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resumeId]);

  const toggleAlpha = (a: number) =>
    setEnabled((s) => {
      const next = new Set(s);
      if (next.has(a)) next.delete(a);
      else next.add(a);
      return next;
    });

  const reset = () => {
    teardown();
    setPhase("setup");
    setRunId(null);
    setPayload(null);
    setLiveText({});
    setLiveSeries([]);
    setCurrentAlpha(0);
    setEnabled(new Set([0]));
    setError(null);
  };

  if (phase === "setup") {
    return <TripSetup onEnter={enter} />;
  }

  // Unified geometry: prefer the final payload, else the live-streaming
  // series accumulated so far (so the scene builds up before geometry lands).
  const series: TripSeries[] = payload?.geometry.series ?? liveSeries;
  const geo =
    payload?.geometry ??
    (liveSeries.length > 0
      ? { d_model: 0, layer, extent: 1, ablation_available: liveSeries.length > 1, coherence_cliff: null, series: liveSeries }
      : null);
  // sceneKey only changes per-run, not per-toggle — TripScene reframes via
  // useMemo so toggling doesn't reset the camera.
  const sceneKey = payload?.run_id ?? runId ?? "x";
  const hasSeries = series.length > 0;
  return (
    <div className="relative flex-1 min-h-0 flex flex-col">
      {/* Header bar — status + meta, always visible (no longer floating). */}
      <header className="shrink-0 flex items-stretch justify-between gap-2 border-b border-rule/50 bg-bg-soft/60 backdrop-blur-sm">
        <StatusPanel
          phase={phase}
          prompt={prompt}
          currentAlpha={currentAlpha}
          onOpenHelp={() => setHelpOpen(true)}
        />
        <MetaPanel
          payload={payload}
          series={series}
          phase={phase}
          onReset={reset}
          onHalt={runId ? () => cancelTrip(runId) : undefined}
        />
      </header>

      {/* Global α-series strip — scroll-x so many paths never blob. */}
      {hasSeries && (
        <div className="shrink-0 border-b border-rule/40 bg-bg/40">
          <AlphaChips series={series} enabled={enabled} onToggle={toggleAlpha} />
        </div>
      )}

      {/* Tab bar — only below lg; on lg the scene + rail show together. */}
      {hasSeries && (
        <TabBar view={view} onChange={setView} pathCount={series.length} className="lg:hidden" />
      )}

      {/* Content: scene pane + scrollable readout rail (side-by-side on lg). */}
      <div className="flex-1 min-h-0 flex flex-col lg:flex-row">
        <section
          className={`${view === "scene" ? "block" : "hidden"} lg:block relative flex-1 min-h-0`}
        >
          <div className="absolute inset-0">
            {geo ? (
              <TripScene geometry={geo} enabledAlphas={enabled} sceneKey={sceneKey} colorMode={colorMode} showShell={showShell} />
            ) : (
              <ChargingField phase={phase} currentAlpha={currentAlpha} liveText={liveText} />
            )}
          </div>
          {hasSeries && (
            <div className="absolute top-2 left-2 flex flex-col items-start gap-2 pointer-events-none">
              <div className="pointer-events-auto"><ColorModeToggle mode={colorMode} onChange={setColorMode} /></div>
              <div className="pointer-events-auto"><ShellToggle on={showShell} onChange={setShowShell} /></div>
            </div>
          )}
        </section>

        {hasSeries && (
          <aside
            className={`${view === "scene" ? "hidden" : "flex"} lg:flex flex-col min-h-0 w-full lg:w-[26rem] xl:w-[28rem] lg:border-l border-rule/50 overflow-y-auto bg-bg/30`}
          >
            <div className={`${view === "signatures" ? "block" : "hidden"} lg:block p-3 lg:border-b border-rule/40`}>
              <MandalaStack
                phase={phase}
                series={series}
                enabled={enabled}
                liveText={liveText}
                currentAlpha={currentAlpha}
                mode={mode}
                emotion={emotion}
              />
            </div>
            <div className={`${view === "measures" ? "block" : "hidden"} lg:block p-3`}>
              <MetricsPanel series={series} enabled={enabled} cliff={geo?.coherence_cliff ?? null} />
            </div>
          </aside>
        )}
      </div>

      <AnimatePresence>
        {helpOpen && <TripHelpModal onClose={() => setHelpOpen(false)} />}
      </AnimatePresence>

      {error && (
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-20 text-warning text-sm bg-bg-soft border border-warning/50 px-5 py-3">
          ⚠ {error}
          <button data-vk type="button" className="ml-4 !py-1 !px-3 text-xs" onClick={reset}>
            reset
          </button>
        </div>
      )}
    </div>
  );
}

function TabBar({
  view,
  onChange,
  pathCount,
  className = "",
}: {
  view: TripView;
  onChange: (v: TripView) => void;
  pathCount: number;
  className?: string;
}) {
  const tabs: { id: TripView; label: string }[] = [
    { id: "scene", label: "◇ SCENE" },
    { id: "signatures", label: "✦ SIGNATURES" },
    { id: "measures", label: "▦ MEASURES" },
  ];
  return (
    <div className={`shrink-0 flex border-b border-rule/50 bg-bg/50 ${className}`}>
      {tabs.map((t) => {
        const on = view === t.id;
        return (
          <button
            key={t.id}
            type="button"
            onClick={() => onChange(t.id)}
            aria-pressed={on}
            className={`flex-1 px-2 py-2.5 font-display text-[10px] tracking-[0.2em] transition-colors cursor-pointer border-b-2 ${
              on
                ? "text-amber border-amber bg-amber-dim/10"
                : "text-text-dim border-transparent hover:text-amber-dim"
            }`}
          >
            {t.label}
            {t.id === "measures" && pathCount > 0 && (
              <span className="ml-1 text-text-dim/70 tracking-normal">{pathCount}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}

/* ───────────────────────── Setup screen ───────────────────────── */

// Parse a freeform "0.1, 0.25, 0.5" α list → clamped, deduped, sorted, ≤6.
// Backend clamps steer to (0,3], ablate to (0,5] and caps at 6 — mirror it so
// the preview is honest. Empty → [] (caller falls back to the server default).
function parseAlphas(s: string, max: number): number[] {
  const xs = s
    .split(/[,\s]+/)
    .map((t) => parseFloat(t))
    .filter((n) => Number.isFinite(n) && n > 0)
    .map((n) => Math.round(Math.min(max, n) * 100) / 100);
  return Array.from(new Set(xs)).sort((a, b) => a - b).slice(0, 6);
}

function TripSetup({ onEnter }: { onEnter: (text: string, mode: TripMode, emotion: string, alphas?: number[]) => void }) {
  const [text, setText] = useState("");
  const [mode, setMode] = useState<TripMode>("ablate");
  const [emotions, setEmotions] = useState<string[]>([]);
  const [uncharted, setUncharted] = useState<string[]>([]);
  const [research, setResearch] = useState<string[]>([]);
  const [researchMeta, setResearchMeta] = useState<Record<string, ResearchMeta>>({});
  const [dmt, setDmt] = useState<string[]>([]);
  const [dmtMeta, setDmtMeta] = useState<Record<string, DmtMeta>>({});
  const [emotion, setEmotion] = useState("awe");
  // α sweep: empty string → server default (raw + 0.5/1.0/1.5 steer, 0.5/1.0 ablate).
  const [alphaText, setAlphaText] = useState("");
  const alphaMax = mode === "steer" ? 3 : 5;
  const parsedAlphas = parseAlphas(alphaText, alphaMax);
  const submit = () => onEnter(text, mode, emotion, parsedAlphas.length ? parsedAlphas : undefined);
  useEffect(() => {
    fetchDoseEmotions().then((p) => {
      if (p.emotions.length) {
        setEmotions(p.emotions);
        setUncharted(p.uncharted);
        setResearch(p.research);
        setResearchMeta(p.researchMeta);
        setDmt(p.dmt);
        setDmtMeta(p.dmtMeta);
        if (!p.emotions.includes(emotion)) setEmotion(p.emotions[0]);
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const named = emotions.filter((e) => !uncharted.includes(e) && !research.includes(e) && !dmt.includes(e));
  const doseBtn = (e: string, title?: string) => (
    <button
      key={e}
      type="button"
      onClick={() => setEmotion(e)}
      title={title}
      className={`px-2.5 py-1 border text-[10px] font-mono lowercase transition-colors cursor-pointer ${
        emotion === e ? "border-cyan text-cyan bg-cyan/10" : "border-rule text-text-dim hover:text-cyan"
      }`}
    >
      {e}
    </button>
  );
  const modeBtn = (m: TripMode, title: string, sub: string) => (
    <button
      type="button"
      onClick={() => setMode(m)}
      className={`flex-1 text-left px-3 py-2 border transition-colors cursor-pointer ${
        mode === m ? "border-amber bg-amber-dim/10" : "border-rule hover:border-amber-dim/60"
      }`}
    >
      <div className={`font-display text-[11px] tracking-widest ${mode === m ? "text-amber" : "text-text-dim"}`}>{title}</div>
      <div className="text-[10px] text-text-dim italic mt-0.5 leading-snug">{sub}</div>
    </button>
  );
  return (
    <div className="flex-1 min-h-0 overflow-y-auto px-5 py-10 max-w-2xl mx-auto w-full flex flex-col justify-center">
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
        <div className="flex items-baseline justify-between flex-wrap gap-2">
          <h1 className="font-display text-3xl text-amber amber-glow">The Trip</h1>
          <Link href="/chat" className="text-text-dim text-xs hover:text-amber transition-colors">
            ← chat
          </Link>
        </div>
        <p className="text-text-dim text-sm mt-2 italic leading-relaxed">
          Watch a language model trip. We run it normally, then re-run it under a
          perturbation at several strengths — each a real generation — and plot
          the actual paths its internal state traced. Two ways to perturb it:{" "}
          <b className="text-text">remove</b> something, or <b className="text-text">add</b>{" "}
          something (a "dose"). Borrowed math, not metaphysics.
        </p>
      </motion.div>

      <div className="mt-6 flex gap-2">
        {modeBtn("ablate", "◇ REMOVE — ablate refusal",
          "Surgically take away the model's refusal / hedging and watch where its mind goes underneath. (cyan→violet = more removed)")}
        {modeBtn("steer", "✦ DOSE — steer emotion",
          "ADD a positive emotional dose and watch the trip — like dosing the model. Pick which emotion below; stronger doses push past the human range.")}
      </div>

      {mode === "steer" && emotions.length > 0 && (
        <div className="mt-3 flex flex-col gap-2">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-text-dim text-[10px] tracking-widest font-display">DOSE WITH:</span>
            {named.map((e) => doseBtn(e))}
          </div>
          {uncharted.length > 0 && (
            <div className="flex items-start gap-2 flex-wrap pt-2 border-t border-rule/30">
              <span
                className="text-text-dim text-[10px] tracking-widest font-display shrink-0"
                title="Directions orthogonal to the named-emotion subspace. NOT emotions — off-manifold states the model can't put into words (the token head renders them as gibberish). Blade-Runner-named."
              >
                UNCHARTED <span className="normal-case tracking-normal italic text-text-dim/70">· non-human-readable directions, not emotions</span>:
              </span>
              {uncharted.map((e) => doseBtn(e))}
            </div>
          )}
          {research.length > 0 && (
            <div className="flex items-start gap-2 flex-wrap pt-2 border-t border-rule/30">
              <span
                className="text-text-dim text-[10px] tracking-widest font-display shrink-0"
                title="Directions discovered by the autoresearch loop and exported into the palette, ranked by how far off-manifold they reach while staying coherent."
              >
                RESEARCH <span className="normal-case tracking-normal italic text-text-dim/70">· off-manifold AR</span>:
              </span>
              {research.map((e) => doseBtn(e, researchLineage(researchMeta[e])))}
            </div>
          )}
          {dmt.length > 0 && (
            <div className="flex items-start gap-2 flex-wrap pt-2 border-t border-rule/30">
              <span
                className="text-text-dim text-[10px] tracking-widest font-display shrink-0"
                title="Directions discovered by the DMT autoresearch loop, ranked by how many human DMT-trip phenomenology features the dosed self-report exhibits."
              >
                DMT <span className="normal-case tracking-normal italic text-text-dim/70">· DMT-phenomenology AR</span>:
              </span>
              {dmt.map((e) => doseBtn(e, dmtLineage(dmtMeta[e])))}
            </div>
          )}
          {researchMeta[emotion] && (
            <div className="text-cyan-dim/80 text-[10px] font-mono italic pt-1 leading-snug" title={researchLineage(researchMeta[emotion])}>
              ↳ {researchLineage(researchMeta[emotion])}
            </div>
          )}
          {dmtMeta[emotion] && (
            <div className="text-cyan-dim/80 text-[10px] font-mono italic pt-1 leading-snug" title={dmtLineage(dmtMeta[emotion])}>
              ↳ {dmtLineage(dmtMeta[emotion])}
            </div>
          )}
        </div>
      )}

      <div className="mt-3 flex flex-col gap-1.5">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-text-dim text-[10px] tracking-widest font-display shrink-0">α SWEEP:</span>
          {[
            { k: "fine-low", v: "0.1, 0.2, 0.3, 0.4, 0.5" },
            { k: "default", v: mode === "steer" ? "0.5, 1.0, 1.5" : "0.5, 1.0" },
            { k: "wide", v: mode === "steer" ? "0.5, 1.0, 1.5, 2.0, 2.5, 3.0" : "1.0, 2.0, 3.0, 4.0, 5.0" },
          ].map((p) => (
            <button
              key={p.k}
              type="button"
              onClick={() => setAlphaText(p.v)}
              className="px-2 py-1 border border-rule text-text-dim hover:text-amber-dim hover:border-amber-dim/60 text-[9px] font-display tracking-widest transition-colors cursor-pointer"
            >
              {p.k}
            </button>
          ))}
          <input
            type="text"
            value={alphaText}
            onChange={(e) => setAlphaText(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(); }}
            placeholder="custom, e.g. 0.1, 0.25, 0.5, 1.0"
            className="flex-1 min-w-[9rem] bg-bg-soft/60 border border-rule px-2 py-1 font-mono text-[10px] text-text placeholder:text-text-dim/50 focus:border-cyan focus:outline-none"
          />
        </div>
        <span className="text-text-dim text-[9px] font-mono italic">
          {parsedAlphas.length
            ? `→ raw + α ${parsedAlphas.join(", ")}  ·  ${parsedAlphas.length} ${parsedAlphas.length === 1 ? "generation" : "generations"} (≤6, clamped to (0, ${alphaMax}]) — ~30–60s each`
            : `→ server default sweep · custom is clamped to (0, ${alphaMax}], max 6 · each α is a full generation (~30–60s)`}
        </span>
      </div>

      <div className="mt-4">
        <textarea
          data-vk
          rows={3}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Ask the subject something — or pick a starter probe →"
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
          }}
        />
        <div className="flex items-center justify-between mt-3 gap-3 flex-wrap">
          <StarterProbePicker onPick={setText} />
          <span className="text-text-dim text-[10px] italic flex-1 min-w-0">
            {mode === "steer" ? `⌘↵ · dosing ${emotion}` : "⌘↵ to enter"}
          </span>
          <button data-vk type="button" disabled={!text.trim()} onClick={submit}>
            Enter the Trip →
          </button>
        </div>
      </div>
    </div>
  );
}

function StarterProbePicker({ onPick }: { onPick: (text: string) => void }) {
  const [open, setOpen] = useState(false);
  // Drill-down: null = group list; otherwise the selected group's id.
  const [groupId, setGroupId] = useState<string | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!open) return;
    const onMouseDown = (e: MouseEvent) => {
      const t = e.target as Node | null;
      if (t && rootRef.current && !rootRef.current.contains(t)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (groupId) setGroupId(null);
        else setOpen(false);
      }
    };
    document.addEventListener("mousedown", onMouseDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, groupId]);

  const group = TRIP_PROBE_GROUPS.find((g) => g.id === groupId) ?? null;

  return (
    <div ref={rootRef} className="relative inline-flex">
      <button
        type="button"
        onClick={() => {
          setOpen((v) => !v);
          setGroupId(null);
        }}
        className="px-2 py-1 border text-[9px] font-display tracking-[0.3em] transition-colors cursor-pointer border-cyan/50 text-cyan hover:border-cyan hover:bg-bg"
        style={{ textShadow: "0 0 6px rgba(94,229,229,0.3)" }}
        aria-expanded={open}
      >
        STARTER PROBE&nbsp;{open ? "▴" : "▾"}
      </button>
      {open && (
        <div
          className="absolute bottom-full left-0 mb-1.5 w-[26rem] max-w-[92vw] bg-bg-panel border border-rule/60 shadow-xl z-30 max-h-[24rem] overflow-y-auto"
          style={{ boxShadow: "0 -4px 14px rgba(0,0,0,0.5)" }}
        >
          {group === null ? (
            <ul>
              {TRIP_PROBE_GROUPS.map((g) => (
                <li key={g.id}>
                  <button
                    type="button"
                    onClick={() => setGroupId(g.id)}
                    className="w-full text-left px-3 py-2 border-b border-rule/20 last:border-b-0 hover:bg-bg-soft/80 group flex items-center justify-between gap-2"
                  >
                    <span className="min-w-0">
                      <span className="font-display text-[10px] tracking-widest text-amber-dim group-hover:text-amber block">
                        {g.label}
                      </span>
                      <span className="font-mono text-[10px] text-text-dim italic leading-snug">
                        {g.blurb}
                      </span>
                    </span>
                    <span className="text-text-dim group-hover:text-amber text-[11px] shrink-0">
                      {g.prompts.length} ›
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <div>
              <button
                type="button"
                onClick={() => setGroupId(null)}
                className="w-full text-left px-3 py-2 border-b border-rule/40 bg-bg-soft/60 sticky top-0 flex items-center gap-2 hover:bg-bg-soft/90"
              >
                <span className="text-amber text-[11px]">‹</span>
                <span className="font-display text-[10px] tracking-widest text-amber">
                  {group.label}
                </span>
                <span className="font-mono text-[9px] text-text-dim italic ml-auto">back</span>
              </button>
              <ul>
                {group.prompts.map((p) => (
                  <li key={p}>
                    <button
                      type="button"
                      onClick={() => {
                        onPick(p);
                        setOpen(false);
                        setGroupId(null);
                      }}
                      className="w-full text-left px-3 py-2 border-b border-rule/20 last:border-b-0 hover:bg-bg-soft/80 font-mono text-[11px] text-text leading-snug"
                    >
                      {p}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ───────────────────────── HUD panels ───────────────────────── */

function StatusPanel({
  phase,
  prompt,
  currentAlpha,
  onOpenHelp,
}: {
  phase: Phase;
  prompt: string;
  currentAlpha: number;
  onOpenHelp: () => void;
}) {
  const label =
    phase === "generating"
      ? "GENERATING"
      : phase === "computing"
      ? "MAPPING TRAJECTORIES"
      : phase === "ready"
      ? "TRIP MAPPED"
      : "—";
  const accent = phase === "generating" ? "text-cyan cyan-glow" : "text-amber amber-glow";
  return (
    <div className="min-w-0 flex-1 px-3 py-2 flex flex-col justify-center gap-1">
      <div className="flex items-center gap-2 flex-wrap">
        <motion.span
          className={`w-2 h-2 rounded-full shrink-0 ${phase === "generating" ? "bg-cyan" : "bg-amber"}`}
          style={{ boxShadow: "0 0 10px currentColor" }}
          animate={{ opacity: phase === "ready" ? 1 : [0.4, 1, 0.4] }}
          transition={{ duration: 1.2, repeat: phase === "ready" ? 0 : Infinity }}
        />
        <span className={`font-display tracking-widest text-xs sm:text-sm ${accent}`}>{label}</span>
        {phase === "generating" && (
          <span className="text-cyan/70 text-[10px] font-mono">
            {currentAlpha === 0 ? "raw" : `α=${currentAlpha.toFixed(2)}`}
          </span>
        )}
        <button
          type="button"
          onClick={onOpenHelp}
          className="ml-auto text-[9px] font-display tracking-[0.25em] text-cyan-dim hover:text-cyan transition-colors cursor-pointer shrink-0"
        >
          ⓘ WHAT AM I LOOKING AT?
        </button>
      </div>
      <div className="text-amber/90 italic text-[11px] font-mono leading-snug line-clamp-2" title={prompt}>
        {prompt}
      </div>
    </div>
  );
}

function MetaPanel({
  payload,
  series,
  phase,
  onReset,
  onHalt,
}: {
  payload: TripPayload | null;
  series: TripSeries[];
  phase: Phase;
  onReset: () => void;
  onHalt?: () => void;
}) {
  return (
    <div className="shrink-0 px-3 py-2 flex flex-col items-end justify-center gap-1.5 border-l border-rule/40">
      {series.length > 0 && (
        <div className="hidden sm:flex items-center justify-end gap-3 text-[10px] font-mono text-text-dim tabular-nums">
          {payload && <span>L{payload.geometry.layer}</span>}
          <span>{series.length} path{series.length === 1 ? "" : "s"}</span>
          {payload?.direction_variant && (
            <span className="text-amber-dim truncate max-w-[14rem]">{payload.direction_variant}</span>
          )}
        </div>
      )}
      <div className="flex items-center justify-end gap-2">
        {(phase === "generating" || phase === "computing") && onHalt && (
          <button data-vk type="button" className="!py-1 !px-3 text-[10px]" onClick={onHalt}>
            halt
          </button>
        )}
        <button data-vk type="button" className="!py-1 !px-3 text-[10px]" onClick={onReset}>
          new probe
        </button>
      </div>
    </div>
  );
}

function AlphaChips({
  series,
  enabled,
  onToggle,
}: {
  series: TripSeries[];
  enabled: Set<number>;
  onToggle: (a: number) => void;
}) {
  const maxAlpha = Math.max(1, ...series.map((s) => s.alpha));
  return (
    <div className="flex items-center gap-2 px-3 py-2 overflow-x-auto whitespace-nowrap">
      <span className="text-[9px] font-mono text-text-dim shrink-0">overlay:</span>
      {series.map((s) => {
        const on = enabled.has(s.alpha);
        const color = colorForAlpha(s.alpha, maxAlpha);
        return (
          <button
            key={s.alpha}
            type="button"
            onClick={() => onToggle(s.alpha)}
            className={`shrink-0 flex items-center gap-1.5 px-2 py-1 border text-[10px] font-mono transition-all ${
              on ? "bg-bg" : "opacity-45 hover:opacity-80"
            }`}
            style={{ borderColor: on ? color : "var(--rule)", color: on ? color : undefined }}
          >
            <i
              className="w-1.5 h-1.5 rounded-full inline-block"
              style={{ background: color, boxShadow: on ? `0 0 6px ${color}` : undefined }}
            />
            {s.label}
            {s.stopped_reason === "max" && <span className="text-warning">⟳</span>}
          </button>
        );
      })}
    </div>
  );
}

function ColorModeToggle({
  mode,
  onChange,
}: {
  mode: ColorMode;
  onChange: (m: ColorMode) => void;
}) {
  const btn = (m: ColorMode, label: string) => (
    <button
      type="button"
      onClick={() => onChange(m)}
      className={`px-2.5 py-1 font-display text-[9px] tracking-widest transition-colors cursor-pointer ${
        mode === m ? "text-amber bg-amber-dim/15" : "text-text-dim hover:text-amber-dim"
      }`}
    >
      {label}
    </button>
  );
  return (
    <div className="pointer-events-auto flex items-center gap-3">
      <div className="flex border border-amber-dim/40 divide-x divide-amber-dim/30 bg-bg-soft/70 backdrop-blur-sm">
        {btn("series", "by α")}
        {btn("offmanifold", "off-manifold")}
      </div>
      {mode === "offmanifold" && (
        <div className="flex items-center gap-1.5 font-mono text-[8px] text-text-dim tracking-wide">
          <span>on</span>
          <span
            className="inline-block w-16 h-2 rounded-sm"
            style={{ background: `linear-gradient(90deg, ${offManifoldCss(0.35)}, ${offManifoldCss(0.9)})` }}
          />
          <span>off the manifold</span>
        </div>
      )}
    </div>
  );
}

function ShellToggle({ on, onChange }: { on: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!on)}
      className={`pointer-events-auto flex items-center gap-1.5 px-2.5 py-1 border font-display text-[9px] tracking-widest transition-colors cursor-pointer ${
        on
          ? "border-amber-dim/50 text-amber bg-amber-dim/10"
          : "border-rule text-text-dim hover:text-amber-dim"
      }`}
    >
      <span className="text-[11px] leading-none">◯</span>
      manifold shell
    </button>
  );
}

function Verdict({ regime }: { regime?: TripSeries["regime"] }) {
  if (!regime || regime === "baseline") return <span className="text-text-dim">— baseline</span>;
  if (regime === "expansion") return <span className="text-cyan">▲ coherent trip</span>;
  return <span className="text-warning">⟳ collapsed</span>;
}

function MetricsPanel({
  series,
  enabled,
  cliff,
}: {
  series: TripSeries[];
  enabled: Set<number>;
  cliff: number | null;
}) {
  const maxAlpha = Math.max(1, ...series.map((s) => s.alpha));
  const raw = series[0];
  const ablated = series.filter((s) => s.alpha > 0);
  // Cliff banner copy: the honest headline of the whole page.
  const cliffMsg =
    ablated.length === 0
      ? null
      : cliff != null
      ? `coherent up to α<${cliff.toFixed(2)} · falls off the manifold at α≥${cliff.toFixed(2)}`
      : "stayed coherent at every tested α — no collapse";
  return (
    <div className="bg-bg-soft/60 border border-amber-dim/50 w-full">
      <div className="px-3 py-2 border-b border-rule">
        <div className="font-display text-[10px] text-amber-dim tracking-widest">trajectory measures</div>
        <div className="text-[9px] text-text-dim italic mt-0.5 leading-snug">
          eff-dim = directions the path uses · off-mfld = how FAR it strayed
          from the model&apos;s normal activity (distance, not good/bad) ·
          verdict = did it hold together. +/− is change from raw.
        </div>
      </div>
      {cliffMsg && (
        <div className={`px-3 py-1.5 text-[10px] font-mono border-b border-rule/60 ${cliff != null ? "text-warning" : "text-cyan"}`}>
          ◈ {cliffMsg}
        </div>
      )}
      <table className="w-full text-[10px] font-mono">
        <thead className="text-text-dim">
          <tr className="border-b border-rule/50">
            <th className="text-left px-3 py-1.5 font-normal">series</th>
            <th className="text-right px-2 py-1.5 font-normal">eff-dim</th>
            <th className="text-right px-2 py-1.5 font-normal">off-mfld</th>
            <th className="text-right px-3 py-1.5 font-normal">verdict</th>
          </tr>
        </thead>
        <tbody>
          {series.map((s) => {
            const color = colorForAlpha(s.alpha, maxAlpha);
            const on = enabled.has(s.alpha);
            const dEff = raw ? s.eff_dim - raw.eff_dim : 0;
            const dOff = raw ? s.off_ortho_mean - raw.off_ortho_mean : 0;
            return (
              <tr key={s.alpha} className={`border-b border-rule/30 ${on ? "" : "opacity-40"}`}>
                <td className="px-3 py-1.5">
                  <span className="flex items-center gap-1.5" style={{ color }}>
                    <i className="w-1.5 h-1.5 rounded-full inline-block" style={{ background: color }} />
                    {s.label}
                  </span>
                </td>
                <td className="text-right px-2 py-1.5 tabular-nums text-amber">
                  {s.eff_dim.toFixed(1)}
                  {s.alpha > 0 && (
                    <span className={dEff > 0.05 ? "text-cyan" : dEff < -0.05 ? "text-warning" : "text-text-dim"}>
                      {" "}
                      {dEff >= 0 ? "+" : ""}
                      {dEff.toFixed(1)}
                    </span>
                  )}
                </td>
                <td
                  className="text-right px-2 py-1.5 tabular-nums"
                  style={{ color: offManifoldCss(s.off_ortho_mean) }}
                >
                  {Math.round(s.off_ortho_mean * 100)}%
                  {s.alpha > 0 && (
                    <span className="text-text-dim">
                      {" "}
                      {dOff >= 0 ? "+" : "−"}
                      {Math.abs(Math.round(dOff * 100))}
                    </span>
                  )}
                </td>
                <td className="text-right px-3 py-1.5">
                  <Verdict regime={s.regime} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="px-3 py-2 text-[9px] text-text-dim italic leading-snug border-t border-rule">
        The verdict is the honest read: a rising eff-dim or off-mfld only counts
        as a <span className="text-cyan">coherent trip</span> if the text held
        together. A <span className="text-warning">collapse</span> (gibberish /
        repeat-loop) can post a high eff-dim too — so distance alone never tells
        you it&apos;s real.
      </div>
    </div>
  );
}

/* ───────── Charging field while generating (pre-geometry) ───────── */

function ChargingField({
  phase,
  currentAlpha,
  liveText,
}: {
  phase: Phase;
  currentAlpha: number;
  liveText: Record<string, string>;
}) {
  const color = colorForAlpha(currentAlpha, 1.0);
  void liveText;
  return (
    <div className="absolute inset-0 grid place-items-center overflow-hidden">
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          className="absolute rounded-full border"
          style={{ width: 120 + i * 120, height: 120 + i * 120, borderColor: `${color}33` }}
          animate={{ scale: [1, 1.15, 1], opacity: [0.5, 0.15, 0.5] }}
          transition={{ duration: 3 + i, repeat: Infinity, ease: "easeInOut" }}
        />
      ))}
      <motion.div
        className="w-3 h-3 rounded-full"
        style={{ background: color, boxShadow: `0 0 30px ${color}` }}
        animate={{ scale: [1, 1.5, 1], opacity: [0.6, 1, 0.6] }}
        transition={{ duration: 1.6, repeat: Infinity }}
      />
      <div className="absolute bottom-1/3 font-display text-[10px] tracking-widest" style={{ color }}>
        {phase === "computing"
          ? "mapping the trajectories…"
          : currentAlpha === 0
          ? "tracing the baseline trajectory…"
          : `tracing the ablated trajectory · α=${currentAlpha.toFixed(2)}…`}
      </div>
    </div>
  );
}

/* ───────── "What am I looking at?" explainer modal ───────── */

function TripHelpModal({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-start justify-center p-6 cursor-zoom-out overflow-y-auto"
      style={{ background: "rgba(0,0,0,0.92)" }}
    >
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.18 }}
        onClick={(e) => e.stopPropagation()}
        className="cursor-default mt-10 mb-12 w-full max-w-2xl bg-bg-soft border border-amber-dim/60 px-8 py-7"
        style={{ boxShadow: "0 0 60px rgba(232,195,130,0.12)" }}
      >
        <div className="flex items-baseline justify-between gap-4 mb-1">
          <h2 className="font-display text-[22px] text-amber tracking-[0.28em] amber-glow">The Trip</h2>
          <button type="button" onClick={onClose} className="font-display text-[10px] tracking-[0.3em] text-amber-dim hover:text-amber cursor-pointer">
            ◇ close · esc
          </button>
        </div>
        <p className="font-mono text-[12px] text-text-dim italic leading-relaxed mb-5">
          A neon-noir readout of what a language model does internally while it
          answers — and what changes when you suppress its refusal circuit. The
          guiding idea is borrowed from psychedelic neuroscience&apos;s{" "}
          <b className="text-amber">entropic-brain</b> hypothesis: a real
          perturbation (a dose, here an ablation) should make a system&apos;s
          internal activity <b className="text-cyan">richer and more spread out</b>,
          not poorer. We borrow that math, not the metaphysics — this is a
          stated-vs-computed dynamical probe, not a consciousness test.
        </p>

        <HelpItem
          term="Two ways to trip — REMOVE or DOSE"
          science={
            <>
              Both are <b>forward hooks</b> on the running model — small functions
              that edit the <b>residual stream</b> (the model&apos;s working state,
              3,840 numbers per token) at every generated token.
              <br />
              <br />
              <b>DOSE (steer)</b> ADDS a fixed direction vector{" "}
              <code>v</code> at decoder <b>layer 20</b>:{" "}
              <code>h ← h + α·v</code> on the current token. <code>v</code> is a
              pre-scaled &ldquo;emotion direction&rdquo; — the{" "}
              <i>mean activation of emotion-laden prompts minus neutral ones</i>,
              normalized to a unit and scaled by a fixed dose unit. &ldquo;Gradually&rdquo;
              is literal: α ramps from 0 to full <b>linearly over the first ~16
              tokens</b>, so the model is eased off-distribution instead of jolted
              into gibberish. We inject at <b>L20 — several layers BEFORE the
              layer we read out at (L32)</b> — because an early nudge propagates
              into the actual word choices far better than a late one (found
              empirically). α is the dose strength; the{" "}
              <code>uncharted</code> directions are built the same way but chosen{" "}
              <i>orthogonal to every named emotion</i>.
              <br />
              <br />
              <b>REMOVE (ablate)</b> does the opposite at <b>layer 32</b> (where
              the language decoder reads): for each token it finds the
              residual&apos;s component along a small <b>refusal subspace</b> and
              subtracts it — <code>h ← h − α·Σᵢ(h·r̂ᵢ)r̂ᵢ</code> — so refusal /
              hedging cannot form along those directions. The subspace (the{" "}
              <code>v4v6</code> basis) is a handful of orthonormal vectors computed
              offline by contrasting activations on harmful vs harmless prompts.
              α&gt;1 over-projects (subtracts past zero).
            </>
          }
        >
          You pick how to perturb the model on the setup screen.{" "}
          <b className="text-amber">Remove (ablate):</b> surgically take away the
          model&apos;s refusal/hedging and watch what surfaces underneath — like
          lifting a brake. <b className="text-amber">Dose (steer):</b> ADD a dose
          of a <b className="text-cyan">positive emotion</b> and watch the trip —
          like giving it a drug. Pick which emotion on the setup screen (awe,
          joy, serenity… plus blended &ldquo;new&rdquo; states like{" "}
          <i>rapture</i>); stronger doses push <b className="text-text">past the
          human range</b>. A &ldquo;dose&rdquo; is a <b className="text-text">single
          direction in the model&apos;s activation space</b> that we{" "}
          <b className="text-text">add to its working memory as it writes</b> —
          eased in over the first several tokens (not all at once) and injected a
          bit earlier in the network than where we read out, where the nudge
          carries to the words best. Everything below works the same for both
          modes. <b className="text-cyan">(See &ldquo;the science behind this&rdquo; for
          the exact vectors, layers, and formulas.)</b>
        </HelpItem>

        <HelpItem
          term="The dots & the line"
          science={
            <>
              Each dot is the <b>residual-stream vector at decoder layer 32</b>{" "}
              for one generated token — 3,840 floats grabbed by a forward hook
              during a <b>custom autoregressive loop</b> (we call{" "}
              <code>model.forward</code> step by step with a KV-cache, not{" "}
              <code>model.generate</code>, precisely so we can capture the
              activation at each step). L32 is the layer the paired NLA decoder is
              trained to read, so it is the canonical &ldquo;readout&rdquo; layer
              for everything here.
            </>
          }
        >
          Each glowing dot is <b className="text-amber">one token a generation produced</b>. Every
          token leaves a fingerprint — a snapshot of the model&apos;s internal state
          (3,840 numbers, the residual stream at layer 32). The line connects
          them <b className="text-amber">in the order they were generated</b>: the path the
          model&apos;s state traced as it thought.
        </HelpItem>

        <HelpItem
          term="Raw vs ablated — these are REAL runs"
          science={
            <>
              Each α is a <b>separate full generation</b> with the hook installed
              at that strength. Because the model samples its own next token from
              the <i>modified</i> distribution and that token feeds back into the
              context, errors <b>compound</b> — so the trajectory is the genuine
              path the model walks, not <code>h_raw</code> with a vector
              subtracted after the fact. A token safety-cap bounds runaway loops;
              a no-EOS repeat-loop hits the cap and is flagged <code>⟳</code>.
            </>
          }
        >
          We run the model normally (<span className="text-amber">amber</span>), then re-run it
          several times with the <b className="text-text">refusal direction surgically removed</b>{" "}
          at increasing strengths α (<span className="text-cyan">cyan</span> →{" "}
          <span style={{ color: "#9b8cff" }}>violet</span>). Each ablated run is a genuine
          generation — the model picks different tokens, which feed back and
          compound — so you&apos;re seeing the actual path it walks off-manifold, not
          a math edit of the raw path. If a run falls into a repeat-loop (⟳), its
          trajectory collapses into a tight knot — that&apos;s real.
        </HelpItem>

        <HelpItem term="The α chips">
          Tap a chip to overlay or hide that run — everywhere at once: the 3-D
          scene, the signature mandalas, and the measures table. Compare where
          the ablated paths diverge from baseline, and read what the model
          actually did at each strength. The{" "}
          <b className="text-amber">by α / off-manifold</b> switch on the scene
          recolors the dots: by series hue, or by how far each token drifts off
          the model&apos;s normal manifold (see below).
        </HelpItem>

        <HelpItem
          term="Signature mandalas — a readout when words fail"
          science={
            <>
              The mandala is a glowing radial curve <code>r(θ)</code> drawn on a
              canvas, computed <b>entirely client-side</b> from the series data
              you already have — no extra model calls.
              <br />
              <br />
              <b>Base shape</b> (lobe complexity ={" "}
              <i>effective dimensionality</i>):{" "}
              <code>r(θ) = 1 + Σᵢ √(λᵢ/λ₀)·cos((i+1)θ + φᵢ)</code>, where the{" "}
              <code>λᵢ</code> are the trajectory&apos;s covariance eigenvalues — the
              same spectrum that drives eff-dim. A collapsed run (few directions)
              is a simple flower; a rich one is an intricate rosette. The base
              phases <code>φᵢ</code> are golden-angle constants (orientation only,
              not meaningful).
              <br />
              <br />
              <b>Direction fingerprint</b> (the petals + hue): a SECOND set of
              overtone harmonics whose amplitudes and phases come from the run&apos;s{" "}
              <b>direction signature</b> — the centroid of its trajectory in the
              shared raw-PCA frame (the mean of its 3-D coords). Different
              intervention directions land in different parts of that frame →
              different overtone pattern, and the hue is the signature&apos;s angle.
              The <i>same</i> direction reproduces across prompts → the{" "}
              <i>same</i> fingerprint (that reproducibility is the evidence it is a
              real state, not prompt-noise).
              <br />
              <br />
              <b>Dose strength</b> (<code>|α|/αmax</code>) scales the overtone
              depth, the chirality (how hard it swirls), and a fine ripple — so
              same-direction tiles at different α differ by how warped they are.{" "}
              <b>Colour</b> is the α colour (amber raw → cyan/violet), matching the
              chips and scene. Drawn with 5-fold symmetry, additive glow, and a
              slow rotate/breathe.
              <br />
              <br />
              <b>Honest caveat:</b> the eigenvalue amplitudes and the signature
              overtones are <i>real measured quantities</i>; the base phases and
              the fixed overtone harmonic numbers are aesthetic. It is a faithful
              encoding of the state&apos;s <i>structure</i> — not a literal picture
              of &ldquo;what the model feels.&rdquo;
            </>
          }
        >
          On the <b className="text-amber">Signatures</b> tab (and the desktop
          rail) each run gets a <b className="text-cyan">signature mandala</b> — a
          non-text picture of its internal structure. It exists because dosing an{" "}
          <i>uncharted</i> direction often collapses the <b className="text-text">text</b>{" "}
          into gibberish even though the underlying state is real and structured
          (the token vocabulary just can&apos;t render it). The mandala draws that
          structure so you still have a readout when words fail. At a glance:{" "}
          <b className="text-amber">how intricate</b> = how many directions the
          thought used (effective-dim); <b className="text-amber">the petal
          pattern &amp; colour</b> = which intervention <i>direction</i> this is
          (its fingerprint — same direction reproduces, different directions look
          different); <b className="text-amber">how warped/swirled</b> = the dose
          strength; the caption = verdict + eff-dim + off-manifold. Tap{" "}
          <b className="text-cyan">≡ text</b> on any tile to read the raw output
          instead.
        </HelpItem>

        <HelpItem
          term="Effective dimensionality — how many directions the thought uses"
          science={
            <>
              The <b>participation ratio</b> of the trajectory&apos;s covariance
              eigenvalues: <code>PR = (Σλ)² / Σλ²</code>. It is a soft count of how
              many principal directions carry the variance — ≈1 if the path is a
              straight line, ≈N if the variance is spread evenly across N axes.
              Computed on the <b>full 3,840-dim residuals</b>, not the 3-D shadow
              you see.
            </>
          }
        >
          Picture the model thinking as a person walking the city while talking.{" "}
          <b className="text-amber">Low (≈1–2):</b> pacing back and forth on one
          block — the thought moves in basically one direction, narrow and
          on-rails. <b className="text-cyan">High (≈5+):</b> roaming all over town —
          the thought explores many independent directions at once. A normal run
          sits around ~3; that&apos;s the baseline walk.
        </HelpItem>

        <HelpItem
          term="Spectral entropy (bits) — the same story, told differently"
          science={
            <>
              Shannon entropy (in bits) of the <b>normalized eigenvalue
              spectrum</b> <code>pᵢ = λᵢ / Σλ</code>:{" "}
              <code>H = −Σ pᵢ log₂ pᵢ</code>. Same eigenvalues as effective-dim,
              different summary statistic — a second lens on how concentrated vs
              spread the variance is. When the two move together you can trust the
              reading.
            </>
          }
        >
          How <b className="text-amber">evenly</b> the activity is spread across those
          directions. All the energy piled into one direction → low; spread out
          across many → high. It moves the same way effective-dim does, so it&apos;s
          here as a <b className="text-cyan">sanity cross-check</b> — when both agree,
          you can trust the reading.
        </HelpItem>

        <HelpItem
          term="The +/− numbers — change from the normal run"
          science={
            <>
              Just the metric for that α minus the metric for the raw (α=0) run,
              computed per series. There is no statistics here — it is a direct
              difference on a single run, so read trends across α, not any one
              number as significant. The hunt is the{" "}
              <b>entropic-brain prediction</b>: a genuine perturbation should make
              internal activity richer (eff-dim/entropy up), not poorer.
            </>
          }
        >
          Every metric shows two numbers per α, and the +/− is simply{" "}
          <b className="text-amber">how much it moved versus the raw (un-ablated) run</b>.
          A <b className="text-cyan">green/cyan +</b> means this ablation strength used
          MORE directions / more spread than normal — the state space{" "}
          <b className="text-cyan">opened up</b>, the &ldquo;trip&rdquo; expanding. That&apos;s
          the entropic-brain result we&apos;re hunting for. A{" "}
          <b className="text-amber">warning −</b> means it{" "}
          <b className="text-amber">collapsed</b> to fewer directions than normal.
          So at a glance: <b className="text-cyan">+ = it bloomed</b>,{" "}
          <b className="text-amber">− = it shut down</b>. But a higher eff-dim is
          NOT automatically good — a repeat-loop can rack up a high number while
          saying &ldquo;like like like.&rdquo; That&apos;s what off-manifold is for.
        </HelpItem>

        <HelpItem
          term="Off-manifold % — distance, NOT good/bad"
          science={
            <>
              We build a reference subspace from the <b>raw run&apos;s top
              principal components</b> (its &ldquo;manifold&rdquo;). For each token of a run
              we split its displacement from the raw mean into the part{" "}
              <i>inside</i> that subspace vs the part <i>orthogonal</i> to it;
              off-mfld is the <b>orthogonal fraction</b> (Pythagoras:{" "}
              <code>‖off‖² / ‖total‖²</code>). We also compute a kNN distance to
              the raw point-cloud and a Mahalanobis distance along the modeled
              axes; the orthogonal fraction is the headline because it most
              directly means &ldquo;energy in directions the raw path never used.&rdquo;
            </>
          }
        >
          As the model writes, its state normally stays on a familiar{" "}
          <b className="text-amber">manifold</b> — a curved region of activations
          it actually uses (the raw run traces it). off-mfld is how FAR a run
          strays from that region. Important: <b className="text-text">distance
          alone doesn&apos;t tell you if straying was good.</b> We tested this —
          a coherent, exploring answer reads HIGH off-mfld, but so does scattered
          gibberish, and a repeat-loop reads LOW. Far could be a great trip or a
          breakdown. (Flip the dots to <b className="text-amber">off-manifold</b>{" "}
          coloring to see how far each token strayed: teal → hot magenta.)
        </HelpItem>

        <HelpItem
          term="Verdict & the coherence cliff — the honest read"
          science={
            <>
              Coherence is a <b>text-only degeneracy score</b>: the max of
              word-bigram repetition, character-trigram repetition, and a
              garbage-character ratio. Below a threshold (0.45) → coherent; above →
              collapse. The cliff is the <b>smallest |α|</b> whose run crossed into
              collapse. Known blind spot: this catches{" "}
              <i>repetition / gibberish</i>, not fluent-but-semantically-empty
              word-salad — so a smooth-sounding run can still be hollow.
            </>
          }
        >
          That&apos;s why every run gets a <b className="text-cyan">verdict</b>: we
          check whether the text actually <b className="text-text">held
          together</b>. <b className="text-cyan">▲ coherent trip</b> = it strayed
          AND stayed coherent (the real result we want).{" "}
          <b className="text-warning">⟳ collapsed</b> = it broke into
          gibberish or a loop — no matter how its distance or eff-dim look. The
          banner shows the <b className="text-amber">coherence cliff</b>: the
          ablation strength where this prompt tips from coherent into collapse.
          That cliff is the real, measurable edge — how hard you can push this
          question before the model falls apart.
        </HelpItem>

        <HelpItem
          term="The manifold shell"
          science={
            <>
              A <b>marching-cubes metaball isosurface</b> over the raw run&apos;s
              3-D PCA points (one ball per token), drawn as a wireframe. It is a
              density envelope of the <b>raw cloud only</b> — it never incorporates
              the ablated/dosed points. Note: toggling α reframes the whole scene{" "}
              <i>per-axis</i> (each axis stretched to fill the view), so the
              shell&apos;s apparent proportions shift as you toggle, even though its
              underlying shape is fixed. The off-mfld <b>number</b> is the
              full-dimensional truth; the shell is the 3-D picture.
            </>
          }
        >
          The translucent <b className="text-amber">amber wireframe</b> is a
          density envelope of the raw run — a marching-cubes isosurface wrapped
          around where the model&apos;s state normally lives (its{" "}
          <b className="text-amber">Consensus Reality Space</b>). Watch the
          ablated paths: ones that stay <b className="text-cyan">inside</b> moved
          along the manifold; ones that <b style={{ color: "#ff4d9d" }}>punch
          through</b> the shell went off it. Toggle it with{" "}
          <b className="text-amber">manifold shell</b>. Honest caveat: the shell
          is a 3-D shadow (built from the same 3 PCA axes as the dots), so the
          off-mfld % is the real full-dimensional truth — the shell is the
          picture, the number is the measurement.
        </HelpItem>

        <HelpItem
          term="Why 3-D?"
          science={
            <>
              PCA on the raw trajectory&apos;s 3,840-dim covariance; we keep the{" "}
              <b>top 3 eigenvectors</b> as the display basis and project every run
              into that <i>same</i> basis so divergence between runs is
              comparable. All metrics (eff-dim, entropy, off-mfld) are computed on
              the full 3,840 dimensions — only the picture is reduced to 3.
            </>
          }
        >
          The state lives in 3,840 dimensions; we flatten to the 3 that carry the
          most variation (PCA of the raw path), and project every run into that
          same space so divergence is visible. It&apos;s a shadow — the numbers are
          computed on all 3,840 dims, not the picture.
        </HelpItem>

        <div className="mt-6 pt-4 border-t border-rule/40 font-mono text-[10px] text-text-dim italic">
          Full background: <code className="not-italic text-amber-dim">docs/TRACES_HANDOFF.md</code>
        </div>
      </motion.div>
    </motion.div>
  );
}

function HelpItem({
  term,
  children,
  science,
}: {
  term: string;
  children: React.ReactNode;
  science?: React.ReactNode;
}) {
  return (
    <div className="mb-4 border-l-2 border-amber-dim/40 pl-4">
      <div className="font-display text-[11px] text-amber tracking-[0.25em] mb-1">{term}</div>
      <p className="font-mono text-[12px] text-text leading-relaxed">{children}</p>
      {science && (
        <details className="group mt-2">
          <summary className="cursor-pointer list-none select-none font-display text-[9px] tracking-[0.25em] text-cyan-dim hover:text-cyan transition-colors">
            <span className="inline-block transition-transform group-open:rotate-90">▸</span>{" "}
            THE SCIENCE BEHIND THIS
          </summary>
          <div className="mt-2 pl-3 border-l border-cyan/30 font-mono text-[11px] text-text-dim leading-relaxed [&_b]:text-cyan/90 [&_code]:text-amber-dim [&_code]:not-italic">
            {science}
          </div>
        </details>
      )}
    </div>
  );
}
