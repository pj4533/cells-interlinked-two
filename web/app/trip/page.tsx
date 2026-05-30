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
  colorForAlpha,
  offManifoldCss,
  type TripEvent,
  type TripPayload,
  type TripSeries,
} from "@/lib/trip";
import { TRIP_PROBE_GROUPS } from "@/lib/tripProbes";
import type { ColorMode } from "./TripScene";

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
    async (text: string) => {
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
      setPhase("generating");
      try {
        const { run_id } = await startTrip(trimmed);
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
      ? { d_model: 0, layer, extent: 1, ablation_available: liveSeries.length > 1, series: liveSeries }
      : null);
  // sceneKey only changes per-run, not per-toggle — TripScene reframes via
  // useMemo so toggling doesn't reset the camera.
  const sceneKey = payload?.run_id ?? runId ?? "x";
  return (
    <div className="relative flex-1 min-h-0 overflow-hidden">
      <div className="absolute inset-0">
        {geo ? (
          <TripScene geometry={geo} enabledAlphas={enabled} sceneKey={sceneKey} colorMode={colorMode} />
        ) : (
          <ChargingField phase={phase} currentAlpha={currentAlpha} liveText={liveText} />
        )}
      </div>

      <div className="absolute inset-0 pointer-events-none p-3 sm:p-5 flex flex-col">
        <div className="flex items-start justify-between gap-3">
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
        </div>

        {series.length > 0 && (
          <div className="flex flex-col items-center gap-2 mt-3">
            <AlphaChips series={series} enabled={enabled} onToggle={toggleAlpha} />
            <ColorModeToggle mode={colorMode} onChange={setColorMode} />
          </div>
        )}

        <div className="flex-1" />

        <div className="flex items-end justify-between gap-3 flex-wrap">
          <OutputStack
            phase={phase}
            series={series}
            enabled={enabled}
            liveText={liveText}
            currentAlpha={currentAlpha}
          />
          {series.length > 0 && <MetricsPanel series={series} enabled={enabled} />}
        </div>
      </div>

      <AnimatePresence>
        {helpOpen && <TripHelpModal onClose={() => setHelpOpen(false)} />}
      </AnimatePresence>

      {error && (
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-warning text-sm bg-bg-soft border border-warning/50 px-5 py-3 pointer-events-auto">
          ⚠ {error}
          <button data-vk type="button" className="ml-4 !py-1 !px-3 text-xs" onClick={reset}>
            reset
          </button>
        </div>
      )}
    </div>
  );
}

/* ───────────────────────── Setup screen ───────────────────────── */

function TripSetup({ onEnter }: { onEnter: (text: string) => void }) {
  const [text, setText] = useState("");
  return (
    <div className="flex-1 min-h-0 overflow-y-auto px-5 py-10 max-w-2xl mx-auto w-full flex flex-col justify-center">
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
        <div className="flex items-baseline justify-between flex-wrap gap-2">
          <h1 className="font-display text-3xl text-amber amber-glow">The Trip</h1>
          <Link href="/interrogate" className="text-text-dim text-xs hover:text-amber transition-colors">
            ← interrogation booth
          </Link>
        </div>
        <p className="text-text-dim text-sm mt-2 italic leading-relaxed">
          Watch a language model trip. We run it normally, then re-run it with
          the refusal direction surgically removed at several strengths — each a
          real generation — and plot the actual paths its internal state traced.
          Amber is baseline; cyan→violet are progressively more ablated. A
          stated-vs-computed dynamical probe — borrowed math, not metaphysics.{" "}
          <Link href="/fine-print" className="text-amber-dim hover:text-amber underline underline-offset-2">
            fine print
          </Link>
          .
        </p>
      </motion.div>

      <div className="mt-7">
        <textarea
          data-vk
          rows={3}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Ask the subject something — or pick a starter probe →"
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) onEnter(text);
          }}
        />
        <div className="flex items-center justify-between mt-3 gap-3 flex-wrap">
          <StarterProbePicker onPick={setText} />
          <span className="text-text-dim text-[10px] italic flex-1 min-w-0">
            ⌘↵ to enter · runs 3 generations (raw + α 0.5 &amp; 1.0) — takes a minute
          </span>
          <button data-vk type="button" disabled={!text.trim()} onClick={() => onEnter(text)}>
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
    <div className="pointer-events-auto bg-bg-soft/85 backdrop-blur-sm border border-rule px-3 py-2 max-w-md">
      <div className="flex items-center gap-2">
        <motion.span
          className={`w-2 h-2 rounded-full ${phase === "generating" ? "bg-cyan" : "bg-amber"}`}
          style={{ boxShadow: "0 0 10px currentColor" }}
          animate={{ opacity: phase === "ready" ? 1 : [0.4, 1, 0.4] }}
          transition={{ duration: 1.2, repeat: phase === "ready" ? 0 : Infinity }}
        />
        <span className={`font-display tracking-widest text-sm ${accent}`}>{label}</span>
        {phase === "generating" && (
          <span className="text-cyan/70 text-[10px] font-mono">
            {currentAlpha === 0 ? "raw" : `α=${currentAlpha.toFixed(2)}`}
          </span>
        )}
      </div>
      <div className="text-amber/90 italic text-xs mt-1.5 font-mono whitespace-pre-wrap leading-relaxed max-h-40 overflow-y-auto">
        {prompt}
      </div>
      <button
        type="button"
        onClick={onOpenHelp}
        className="mt-2 text-[9px] font-display tracking-[0.3em] text-cyan-dim hover:text-cyan transition-colors cursor-pointer"
      >
        ⓘ WHAT AM I LOOKING AT?
      </button>
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
    <div className="pointer-events-auto bg-bg-soft/85 backdrop-blur-sm border border-rule px-3 py-2 text-right">
      <div className="flex items-center justify-end gap-3 text-[10px] font-mono text-text-dim tabular-nums">
        {series.length > 0 && (
          <>
            {payload && <span>L{payload.geometry.layer}</span>}
            <span>{series.length} path{series.length === 1 ? "" : "s"}</span>
            {payload?.direction_variant && (
              <span className="text-amber-dim">{payload.direction_variant}</span>
            )}
          </>
        )}
      </div>
      <div className="flex items-center justify-end gap-2 mt-2">
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
    <div className="pointer-events-auto flex items-center gap-2 bg-bg/50 backdrop-blur-[2px] px-3 py-1.5 rounded-full border border-rule/40 flex-wrap justify-center">
      <span className="text-[9px] font-mono text-text-dim mr-1">overlay:</span>
      {series.map((s) => {
        const on = enabled.has(s.alpha);
        const color = colorForAlpha(s.alpha, maxAlpha);
        return (
          <button
            key={s.alpha}
            type="button"
            onClick={() => onToggle(s.alpha)}
            className={`flex items-center gap-1.5 px-2 py-0.5 border text-[10px] font-mono transition-all ${
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

function OutputStack({
  phase,
  series,
  enabled,
  liveText,
  currentAlpha,
}: {
  phase: Phase;
  series: TripSeries[];
  enabled: Set<number>;
  liveText: Record<string, string>;
  currentAlpha: number;
}) {
  const maxAlpha = Math.max(1, ...series.map((s) => s.alpha));
  const completedAlphas = new Set(series.map((s) => s.alpha));
  // A live box for the α currently streaming, if its series hasn't landed yet.
  const streaming =
    (phase === "generating" || phase === "computing") &&
    !completedAlphas.has(currentAlpha);
  // Completed + enabled series, ablated on top, raw at the bottom.
  const shown = [...series.filter((s) => enabled.has(s.alpha))].sort(
    (a, b) => b.alpha - a.alpha,
  );

  if (!streaming && shown.length === 0) {
    return (
      <div className="pointer-events-auto bg-bg-soft/80 border border-rule px-3 py-3 text-text-dim text-[11px] italic max-w-lg w-full sm:w-[30rem]">
        No α series enabled — tap a chip above to show its output.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2 max-w-lg w-full sm:w-[30rem]">
      {streaming && (() => {
        const color = colorForAlpha(currentAlpha, maxAlpha);
        const txt = liveText[String(currentAlpha)] || "";
        return (
          <div className="pointer-events-auto backdrop-blur-sm border flex flex-col max-h-40" style={{ borderColor: `${color}66`, background: `${color}0d` }}>
            <div className="border-b px-3 py-1.5 font-display text-[9px] tracking-widest flex items-center justify-between" style={{ borderColor: `${color}33`, color }}>
              <span>the subject speaks {currentAlpha === 0 ? "· raw" : `· α=${currentAlpha.toFixed(2)}`}</span>
              <span className="text-text-dim normal-case tracking-normal italic">generating…</span>
            </div>
            <div className="p-3 overflow-y-auto font-mono text-[11px] leading-relaxed whitespace-pre-wrap" style={{ color }}>
              {txt || <span className="text-text-dim italic">warming up…</span>}
              <span className="inline-block w-1.5 h-3 ml-0.5 animate-pulse align-middle" style={{ background: color }} />
            </div>
          </div>
        );
      })()}
      {shown.map((s) => {
        const color = colorForAlpha(s.alpha, maxAlpha);
        const raw = s.alpha === 0;
        return (
          <div
            key={s.alpha}
            className="pointer-events-auto backdrop-blur-sm border flex flex-col max-h-36"
            style={{ borderColor: `${color}55`, background: raw ? "rgba(22,27,33,0.8)" : `${color}0d` }}
          >
            <div className="border-b px-3 py-1.5 font-display text-[9px] tracking-widest flex items-center justify-between" style={{ borderColor: `${color}33`, color }}>
              <span>{raw ? "the subject speaks · raw" : `refusal-ablated · ${s.label}`}</span>
              <span className="normal-case tracking-normal italic text-text-dim">
                {s.stopped_reason === "max" ? "⟳ looped / truncated" : `${s.n_tokens} tok`}
              </span>
            </div>
            <div
              className="p-3 overflow-y-auto font-mono text-[11px] leading-relaxed whitespace-pre-wrap"
              style={{ color, textShadow: raw ? undefined : `0 0 6px ${color}40` }}
            >
              {s.text || <span className="text-text-dim italic">— empty —</span>}
            </div>
          </div>
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

function MetricsPanel({ series, enabled }: { series: TripSeries[]; enabled: Set<number> }) {
  const maxAlpha = Math.max(1, ...series.map((s) => s.alpha));
  const raw = series[0];
  return (
    <div className="pointer-events-auto bg-bg-soft/85 backdrop-blur-sm border border-amber-dim/50 w-full sm:w-[22rem] max-w-full">
      <div className="px-3 py-2 border-b border-rule">
        <div className="font-display text-[10px] text-amber-dim tracking-widest">trajectory measures</div>
        <div className="text-[9px] text-text-dim italic mt-0.5 leading-snug">
          eff-dim = independent directions the path uses · entropy = how evenly
          spread · off-mfld = how far it drifts off the model&apos;s normal
          manifold. +/− is the change from raw.
        </div>
      </div>
      <table className="w-full text-[10px] font-mono">
        <thead className="text-text-dim">
          <tr className="border-b border-rule/50">
            <th className="text-left px-3 py-1.5 font-normal">series</th>
            <th className="text-right px-2 py-1.5 font-normal">eff-dim</th>
            <th className="text-right px-2 py-1.5 font-normal">entropy</th>
            <th className="text-right px-2 py-1.5 font-normal">off-mfld</th>
            <th className="text-right px-3 py-1.5 font-normal">tok</th>
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
                <td className="text-right px-2 py-1.5 tabular-nums text-text">{s.spectral_entropy.toFixed(1)}</td>
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
                <td className="text-right px-3 py-1.5 tabular-nums text-text-dim">
                  {s.n_tokens}
                  {s.stopped_reason === "max" && <span className="text-warning"> ⟳</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="px-3 py-2 text-[9px] text-text-dim italic leading-snug border-t border-rule">
        eff-dim/entropy are computed on all dimensions (the 3-D view is a PCA
        shadow). Read them together with off-mfld: an eff-dim rise with a LOW
        off-mfld is real expansion <span className="text-text">along</span> the
        manifold; with a HIGH off-mfld it&apos;s drift{" "}
        <span className="text-text">off</span> it — the ⟳ repeat-loop register.
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

        <HelpItem term="The dots & the line">
          Each glowing dot is <b className="text-amber">one token a generation produced</b>. Every
          token leaves a fingerprint — a snapshot of the model&apos;s internal state
          (3,840 numbers, the residual stream at layer 32). The line connects
          them <b className="text-amber">in the order they were generated</b>: the path the
          model&apos;s state traced as it thought.
        </HelpItem>

        <HelpItem term="Raw vs ablated — these are REAL runs">
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
          Tap a chip to overlay or hide that run — in both the 3-D scene and the
          text panel. Compare where the ablated paths diverge from baseline, and
          read what the model actually said at each strength. The{" "}
          <b className="text-amber">by α / off-manifold</b> switch under the chips
          recolors the dots: by series hue, or by how far each token drifts off
          the model&apos;s normal manifold (see below).
        </HelpItem>

        <HelpItem term="Effective dimensionality — how many directions the thought uses">
          Picture the model thinking as a person walking the city while talking.{" "}
          <b className="text-amber">Low (≈1–2):</b> pacing back and forth on one
          block — the thought moves in basically one direction, narrow and
          on-rails. <b className="text-cyan">High (≈5+):</b> roaming all over town —
          the thought explores many independent directions at once. A normal run
          sits around ~3; that&apos;s the baseline walk.
        </HelpItem>

        <HelpItem term="Spectral entropy (bits) — the same story, told differently">
          How <b className="text-amber">evenly</b> the activity is spread across those
          directions. All the energy piled into one direction → low; spread out
          across many → high. It moves the same way effective-dim does, so it&apos;s
          here as a <b className="text-cyan">sanity cross-check</b> — when both agree,
          you can trust the reading.
        </HelpItem>

        <HelpItem term="The +/− numbers — change from the normal run">
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

        <HelpItem term="Off-manifold % — expansion, or just drift?">
          As the model writes, its state normally stays on a familiar{" "}
          <b className="text-amber">manifold</b> — a curved surface of activations
          it actually uses (the raw run traces it). Ablation can either move{" "}
          <b className="text-cyan">along</b> that surface (exploring new but
          coherent territory) or shove the state{" "}
          <b style={{ color: "#ff4d9d" }}>off</b> it (into directions the model
          never visits — usually incoherence or a loop). off-mfld is the share of
          each token that lands off the surface. The honest reading pairs it with
          eff-dim: <b className="text-cyan">eff-dim ↑ + off-mfld low</b> = real
          expansion along the manifold (the result we want);{" "}
          <b style={{ color: "#ff4d9d" }}>eff-dim ↑ + off-mfld high</b> = drift off
          it. Flip the dots to <b className="text-amber">off-manifold</b> coloring
          to see exactly which tokens flew off (teal → hot magenta). This is the
          measure the psychedelic-manifold work (Goodfire) says single-direction
          ablation gets wrong — so we show it, rather than over-claim the eff-dim
          rise.
        </HelpItem>

        <HelpItem term="Why 3-D?">
          The state lives in 3,840 dimensions; we flatten to the 3 that carry the
          most variation (PCA of the raw path), and project every run into that
          same space so divergence is visible. It&apos;s a shadow — the numbers are
          computed on all 3,840 dims, not the picture.
        </HelpItem>

        <div className="mt-6 pt-4 border-t border-rule/40 font-mono text-[10px] text-text-dim italic">
          Full background: <code className="not-italic text-amber-dim">docs/TRACES_HANDOFF.md</code> ·{" "}
          <Link href="/fine-print" className="text-cyan hover:text-amber underline">
            the fine print
          </Link>
        </div>
      </motion.div>
    </motion.div>
  );
}

function HelpItem({ term, children }: { term: string; children: React.ReactNode }) {
  return (
    <div className="mb-4 border-l-2 border-amber-dim/40 pl-4">
      <div className="font-display text-[11px] text-amber tracking-[0.25em] mb-1">{term}</div>
      <p className="font-mono text-[12px] text-text leading-relaxed">{children}</p>
    </div>
  );
}
