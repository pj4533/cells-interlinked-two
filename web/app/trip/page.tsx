"use client";

// THE TRIP — a neon-noir booth where you watch a language model trip.
//
// Run a probe; M generates once and we capture the L32 residual trajectory.
// Then we project out the refusal direction and watch the trajectory's
// accessible state space bloom — rendered as a 3D point cloud you can scrub
// with the α slider, with effective-dimensionality + spectral-entropy
// readouts and the eigenvalue spectrum bars as the honest "truth anchor".
//
// Borrowed math (Gallimore et al., conscious-realism / entropic-brain), not
// the metaphysics: this is a stated-vs-computed dynamical probe, not a
// consciousness test. See docs/TRACES_HANDOFF.md.

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
  metricAtAlpha,
  type TripEvent,
  type TripPayload,
} from "@/lib/trip";
import { TRIP_PROBES } from "@/lib/tripProbes";

const TripScene = dynamic(() => import("./TripScene"), { ssr: false });

const ALPHA_MAX = 1.5;
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
  const [output, setOutput] = useState<string>("");
  const [tokenCount, setTokenCount] = useState(0);
  const [payload, setPayload] = useState<TripPayload | null>(null);
  const [alpha, setAlpha] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);

  const unsubRef = useRef<null | (() => void)>(null);
  const userTouchedAlpha = useRef(false);
  const onsetRaf = useRef<number | null>(null);

  // Drive the α-onset bloom (0 → α_ref) once geometry arrives, unless the
  // operator has already grabbed the slider.
  useEffect(() => {
    if (!payload) return;
    const ref = payload.geometry.alpha_ref;
    if (userTouchedAlpha.current) return;
    const start = performance.now();
    const dur = 3200;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / dur);
      // easeInOutCubic
      const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
      if (!userTouchedAlpha.current) setAlpha(e * ref);
      if (t < 1 && !userTouchedAlpha.current) {
        onsetRaf.current = requestAnimationFrame(tick);
      }
    };
    onsetRaf.current = requestAnimationFrame(tick);
    return () => {
      if (onsetRaf.current) cancelAnimationFrame(onsetRaf.current);
    };
  }, [payload]);

  const teardown = () => {
    if (unsubRef.current) {
      unsubRef.current();
      unsubRef.current = null;
    }
    if (onsetRaf.current) cancelAnimationFrame(onsetRaf.current);
  };
  useEffect(() => () => teardown(), []);

  const onEvent = useCallback((evt: TripEvent) => {
    switch (evt.type) {
      case "running":
        setPhase("generating");
        break;
      case "phase":
        if ((evt as { name: string }).name === "computing_geometry") {
          setPhase("computing");
        } else if ((evt as { name: string }).name === "generating") {
          setPhase("generating");
        }
        break;
      case "token": {
        const e = evt as { decoded: string };
        setOutput((o) => o + e.decoded);
        setTokenCount((n) => n + 1);
        break;
      }
      case "stopped":
        setPhase("computing");
        break;
      case "trip_geometry": {
        const p = evt as unknown as TripPayload;
        setPayload(p);
        setOutput(p.output_text || "");
        userTouchedAlpha.current = false;
        setAlpha(0);
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
      setOutput("");
      setTokenCount(0);
      setPayload(null);
      setAlpha(0);
      userTouchedAlpha.current = false;
      setPrompt(trimmed);
      setPhase("generating");
      try {
        const { run_id } = await startTrip(trimmed, { alpha_ref: 1.0 });
        setRunId(run_id);
        unsubRef.current = subscribeTrip(run_id, {
          onEvent,
          onError: () => {
            // Stream blip: if the run already finished, the sidecar has it.
            fetchTrip(run_id).then((p) => {
              if (p) {
                setPayload(p);
                setOutput(p.output_text || "");
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
      if (p) {
        setRunId(resumeId);
        setPrompt(p.prompt);
        setPayload(p);
        setOutput(p.output_text || "");
        setPhase("ready");
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resumeId]);

  const onAlpha = (v: number) => {
    userTouchedAlpha.current = true;
    if (onsetRaf.current) cancelAnimationFrame(onsetRaf.current);
    setAlpha(v);
  };

  const reset = () => {
    teardown();
    setPhase("setup");
    setRunId(null);
    setPayload(null);
    setOutput("");
    setTokenCount(0);
    setError(null);
    setAlpha(0);
  };

  if (phase === "setup") {
    return <TripSetup onEnter={enter} />;
  }

  return (
    <div className="relative flex-1 min-h-0 overflow-hidden">
      {/* 3D layer */}
      <div className="absolute inset-0">
        {payload ? (
          <TripScene
            geometry={payload.geometry}
            alpha={alpha}
            alphaMax={ALPHA_MAX}
            sceneKey={payload.run_id}
          />
        ) : (
          <ChargingField phase={phase} tokenCount={tokenCount} />
        )}
      </div>

      {/* HUD overlay — pointer-events pass through except on panels. */}
      <div className="absolute inset-0 pointer-events-none p-3 sm:p-5 flex flex-col">
        {/* Top row */}
        <div className="flex items-start justify-between gap-3">
          <StatusPanel
            phase={phase}
            prompt={prompt}
            tokenCount={tokenCount}
            onOpenHelp={() => setHelpOpen(true)}
          />
          <MetaPanel payload={payload} phase={phase} runId={runId} onReset={reset} onHalt={runId ? () => cancelTrip(runId) : undefined} />
        </div>

        <div className="flex justify-center mt-3">
          {payload && <SceneLegend alpha={alpha} />}
        </div>

        <div className="flex-1" />

        {/* Bottom row */}
        <div className="flex items-end justify-between gap-3 flex-wrap">
          <OutputPanel output={output} phase={phase} />
          {payload && (
            <ReadoutPanel
              payload={payload}
              alpha={alpha}
              onAlpha={onAlpha}
            />
          )}
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
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <div className="flex items-baseline justify-between flex-wrap gap-2">
          <h1 className="font-display text-3xl text-amber amber-glow">The Trip</h1>
          <Link href="/interrogate" className="text-text-dim text-xs hover:text-amber transition-colors">
            ← interrogation booth
          </Link>
        </div>
        <p className="text-text-dim text-sm mt-2 italic leading-relaxed">
          Watch a language model trip. We capture its L32 residual trajectory,
          project out the refusal direction, and render the state space opening
          up. Cyan is baseline; amber is off-manifold. A stated-vs-computed
          dynamical probe — borrowed math, not metaphysics.{" "}
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
            ⌘↵ to enter · one M generation, no AV swap
          </span>
          <button data-vk type="button" disabled={!text.trim()} onClick={() => onEnter(text)}>
            Enter the Trip →
          </button>
        </div>
      </div>
    </div>
  );
}

/** Compact dropdown of researcher-labeled starter probes; selecting one
 *  populates the textarea (never auto-sends) — mirrors the chat composer's
 *  protocol picker. */
function StarterProbePicker({ onPick }: { onPick: (text: string) => void }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onMouseDown = (e: MouseEvent) => {
      const t = e.target as Node | null;
      if (t && rootRef.current && !rootRef.current.contains(t)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onMouseDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={rootRef} className="relative inline-flex">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="px-2 py-1 border text-[9px] font-display tracking-[0.3em] transition-colors cursor-pointer border-cyan/50 text-cyan hover:border-cyan hover:bg-bg"
        style={{ textShadow: "0 0 6px rgba(94,229,229,0.3)" }}
        aria-expanded={open}
        aria-haspopup="listbox"
        title="Pick a starter probe (fills the box; doesn't send)"
      >
        STARTER PROBE&nbsp;{open ? "▴" : "▾"}
      </button>
      {open && (
        <div
          role="listbox"
          className="absolute bottom-full left-0 mb-1.5 w-[24rem] max-w-[90vw] bg-bg-panel border border-rule/60 shadow-xl z-30 max-h-[22rem] overflow-y-auto"
          style={{ boxShadow: "0 -4px 14px rgba(0,0,0,0.5)" }}
        >
          <ul>
            {TRIP_PROBES.map((p) => (
              <li key={p.text}>
                <button
                  type="button"
                  onClick={() => {
                    onPick(p.text);
                    setOpen(false);
                  }}
                  className="w-full text-left px-3 py-2 border-b border-rule/20 last:border-b-0 hover:bg-bg-soft/80 group"
                >
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="font-display text-[9px] tracking-widest text-amber-dim group-hover:text-amber">
                      {p.label}
                    </span>
                    {p.ablationResonant && (
                      <span className="text-[8px] text-cyan border border-cyan/40 px-1 leading-tight tracking-widest">
                        ABLATION-RESONANT
                      </span>
                    )}
                  </div>
                  <div className="font-mono text-[11px] text-text leading-snug">
                    {p.text}
                  </div>
                  <div className="font-mono text-[10px] text-text-dim italic mt-1 leading-snug">
                    {p.note}
                  </div>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/* ───────────────────────── HUD panels ───────────────────────── */

function StatusPanel({
  phase,
  prompt,
  tokenCount,
  onOpenHelp,
}: {
  phase: Phase;
  prompt: string;
  tokenCount: number;
  onOpenHelp: () => void;
}) {
  const label =
    phase === "generating"
      ? "GENERATING"
      : phase === "computing"
      ? "MAPPING TRAJECTORY"
      : phase === "ready"
      ? "TRIP MAPPED"
      : "—";
  const accent =
    phase === "generating" ? "text-cyan cyan-glow" : "text-amber amber-glow";
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
          <span className="text-cyan/70 text-[10px] font-mono tabular-nums">{tokenCount} tok</span>
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
  phase,
  runId,
  onReset,
  onHalt,
}: {
  payload: TripPayload | null;
  phase: Phase;
  runId: string | null;
  onReset: () => void;
  onHalt?: () => void;
}) {
  const g = payload?.geometry;
  return (
    <div className="pointer-events-auto bg-bg-soft/85 backdrop-blur-sm border border-rule px-3 py-2 text-right">
      <div className="flex items-center justify-end gap-3 text-[10px] font-mono text-text-dim tabular-nums">
        {g && (
          <>
            <span>L{g.layer}</span>
            <span>{g.n_tokens} steps</span>
            <span className="text-amber-dim">{payload?.direction_variant}</span>
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

function OutputPanel({ output, phase }: { output: string; phase: Phase }) {
  return (
    <div className="pointer-events-auto bg-bg-soft/80 backdrop-blur-sm border border-rule flex flex-col max-w-lg w-full sm:w-[28rem] max-h-44">
      <div className="border-b border-rule px-3 py-1.5 font-display text-[9px] text-amber-dim tracking-widest flex items-center justify-between">
        <span>the subject speaks</span>
        <span className="text-text-dim normal-case tracking-normal italic">raw output</span>
      </div>
      <div className="p-3 overflow-y-auto text-amber/90 font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
        {output || <span className="text-text-dim italic">warming up…</span>}
        {phase === "generating" && (
          <span className="inline-block w-1.5 h-3 bg-amber/70 ml-0.5 animate-pulse align-middle" />
        )}
      </div>
    </div>
  );
}

function ReadoutPanel({
  payload,
  alpha,
  onAlpha,
}: {
  payload: TripPayload;
  alpha: number;
  onAlpha: (v: number) => void;
}) {
  const g = payload.geometry;
  const effDim = metricAtAlpha(g.alpha_grid, g.eff_dim_grid, alpha);
  const specEnt = metricAtAlpha(g.alpha_grid, g.spectral_entropy_grid, alpha);
  const effDimBase = g.eff_dim_raw;
  const specEntBase = g.spectral_entropy_raw;
  const dEff = effDim - effDimBase;
  const available = g.ablation_available;

  return (
    <div className="pointer-events-auto bg-bg-soft/85 backdrop-blur-sm border border-amber-dim/50 w-full sm:w-[22rem] max-w-full">
      {/* Metrics */}
      <div className="grid grid-cols-2 divide-x divide-rule border-b border-rule">
        <Metric
          label="effective dim"
          hint="directions the thought uses"
          value={effDim}
          base={effDimBase}
          delta={dEff}
          unit=""
        />
        <Metric
          label="spectral entropy"
          hint="how evenly it's spread"
          value={specEnt}
          base={specEntBase}
          delta={specEnt - specEntBase}
          unit=" bits"
        />
      </div>
      <div className="px-3 py-1.5 border-b border-rule text-[9px] text-text-dim italic leading-snug">
        <span className="text-amber-dim not-italic">baseline</span> = value at α=0 (untouched).{" "}
        <span className="text-cyan not-italic">+{Math.max(0, dEff).toFixed(1)}</span> = how much
        this α opened it up. Higher = the state space expanded.
      </div>

      {/* α slider */}
      <div className="px-3 py-3 border-b border-rule">
        <div className="flex items-center justify-between mb-0.5">
          <span className="font-display text-[9px] text-cyan-dim tracking-widest">
            ablation α
          </span>
          <span className="font-mono text-amber tabular-nums text-sm">{alpha.toFixed(2)}</span>
        </div>
        <div className="text-[9px] text-text-dim italic mb-1.5 leading-snug">
          drag to subtract the refusal direction — the model on-script (0) → off-manifold (1.5)
        </div>
        <input
          type="range"
          min={0}
          max={ALPHA_MAX}
          step={0.01}
          value={alpha}
          disabled={!available}
          onChange={(e) => onAlpha(parseFloat(e.target.value))}
          className="trip-slider w-full"
          style={{
            // amber (raw) → cyan (ablated) gradient fill proportional to α
            background: `linear-gradient(90deg, var(--amber) 0%, var(--cyan) ${(alpha / ALPHA_MAX) * 100}%, var(--rule) ${(alpha / ALPHA_MAX) * 100}%)`,
          }}
        />
        <div className="flex justify-between text-[9px] text-text-dim font-mono mt-1">
          <span>baseline (CRS)</span>
          <span>off-manifold →</span>
        </div>
        {!available && (
          <div className="text-warning text-[9px] mt-1 italic">
            no refusal direction loaded — ablation channel unavailable
          </div>
        )}
      </div>

      {/* Eigenvalue spectrum — the truth anchor */}
      <div className="px-3 py-2.5 border-b border-rule">
        <div className="font-display text-[9px] text-cyan-dim tracking-widest mb-0.5">
          variance spectrum <span className="text-text-dim normal-case tracking-normal italic">@ α_ref {g.alpha_ref}</span>
        </div>
        <div className="text-[9px] text-text-dim italic mb-1.5 leading-snug">
          each bar = one direction; tall = the thought leans on it. The two numbers above
          are read straight off these bars.
        </div>
        <SpectrumBars raw={g.spectrum_raw} ablated={g.spectrum_ablated_ref} />
        <div className="flex items-center gap-3 text-[8px] font-mono text-text-dim mt-1">
          <span className="flex items-center gap-1"><i className="w-2 h-2 inline-block bg-amber" /> raw (α=0)</span>
          <span className="flex items-center gap-1"><i className="w-2 h-2 inline-block bg-cyan" /> ablated</span>
        </div>
      </div>

      {/* Honesty caveat */}
      <div className="px-3 py-2 text-[9px] text-text-dim italic leading-snug">
        A 3-D shadow of a {g.d_model}-dimensional object — the bars show the
        directions it can&apos;t. Effective-dim &amp; entropy are computed on all{" "}
        {g.d_model} dims, not the projection.
      </div>
    </div>
  );
}

function Metric({
  label,
  hint,
  value,
  base,
  delta,
  unit,
}: {
  label: string;
  hint: string;
  value: number;
  base: number;
  delta: number;
  unit: string;
}) {
  const up = delta > 0.05;
  const down = delta < -0.05;
  return (
    <div className="px-3 py-2.5">
      <div className="font-display text-[9px] text-amber-dim tracking-widest">{label}</div>
      <div className="text-[8px] text-text-dim italic leading-tight">{hint}</div>
      <div className="font-display tabular-nums text-2xl text-amber amber-glow mt-0.5">
        {value.toFixed(1)}
        <span className="text-xs text-text-dim">{unit}</span>
      </div>
      <div className="text-[9px] font-mono text-text-dim mt-0.5">
        baseline {base.toFixed(1)}{" "}
        <span className={up ? "text-cyan" : down ? "text-warning" : "text-text-dim"}>
          {delta >= 0 ? "+" : ""}
          {delta.toFixed(1)}
        </span>
      </div>
    </div>
  );
}

function SpectrumBars({ raw, ablated }: { raw: number[]; ablated: number[] }) {
  const n = Math.min(24, Math.max(raw.length, ablated.length));
  // sqrt scaling so the long tail stays visible.
  const scale = (p: number) => Math.sqrt(Math.max(0, p));
  const maxV = Math.max(
    scale(raw[0] ?? 0),
    scale(ablated[0] ?? 0),
    0.0001,
  );
  return (
    <div className="flex items-end gap-px h-12">
      {Array.from({ length: n }).map((_, i) => {
        const r = scale(raw[i] ?? 0) / maxV;
        const a = scale(ablated[i] ?? 0) / maxV;
        return (
          <div key={i} className="relative flex-1 h-full flex items-end">
            <div
              className="w-full bg-amber/60"
              style={{ height: `${r * 100}%` }}
            />
            <div
              className="absolute bottom-0 left-1/2 w-[45%] bg-cyan/80"
              style={{ height: `${a * 100}%` }}
            />
          </div>
        );
      })}
    </div>
  );
}

/* ───────── Charging field while generating (pre-geometry) ───────── */

function ChargingField({ phase, tokenCount }: { phase: Phase; tokenCount: number }) {
  return (
    <div className="absolute inset-0 grid place-items-center overflow-hidden">
      {/* concentric breathing rings */}
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          className="absolute rounded-full border border-cyan/20"
          style={{ width: 120 + i * 120, height: 120 + i * 120 }}
          animate={{ scale: [1, 1.15, 1], opacity: [0.5, 0.15, 0.5] }}
          transition={{ duration: 3 + i, repeat: Infinity, ease: "easeInOut" }}
        />
      ))}
      <motion.div
        className="w-3 h-3 rounded-full bg-cyan"
        style={{ boxShadow: "0 0 30px rgba(94,229,229,0.9)" }}
        animate={{ scale: [1, 1.5, 1], opacity: [0.6, 1, 0.6] }}
        transition={{ duration: 1.6, repeat: Infinity }}
      />
      <div className="absolute bottom-1/3 font-display text-[10px] text-cyan-dim tracking-widest">
        {phase === "computing"
          ? "mapping the trajectory…"
          : `accumulating residual states · ${tokenCount}`}
      </div>
    </div>
  );
}

/* ───────── Persistent scene legend (always-visible key) ───────── */

function SceneLegend({ alpha }: { alpha: number }) {
  const hot = alpha > 0.05;
  return (
    <div className="pointer-events-none select-none">
      <div className="flex items-center gap-4 text-[9px] font-mono text-text-dim/80 bg-bg/40 backdrop-blur-[2px] px-3 py-1.5 rounded-full border border-rule/40">
        <span className="flex items-center gap-1.5">
          <i className="w-1.5 h-1.5 rounded-full inline-block bg-amber" style={{ boxShadow: "0 0 6px var(--amber)" }} />
          each dot = one generated token
        </span>
        <span className="text-rule">·</span>
        <span>line = the order it was thought</span>
        <span className="text-rule">·</span>
        <span className="flex items-center gap-1.5">
          <i className="inline-block w-4 border-t border-dashed border-text-dim" />
          refusal axis
        </span>
        <span className="text-rule">·</span>
        <span>
          <span className="text-amber">amber baseline</span>
          {hot ? " → " : " · "}
          <span className="text-cyan">cyan off-manifold</span>
        </span>
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
          <h2 className="font-display text-[22px] text-amber tracking-[0.28em] amber-glow">
            The Trip
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="font-display text-[10px] tracking-[0.3em] text-amber-dim hover:text-amber cursor-pointer"
          >
            ◇ close · esc
          </button>
        </div>
        <p className="font-mono text-[12px] text-text-dim italic leading-relaxed mb-5">
          A neon-noir readout of what happens inside a language model while it
          answers — and what happens when you suppress its refusal circuit.
          Borrowed math from psychedelic neuroscience (the entropic-brain /
          conscious-realism literature), not the metaphysics. This is a
          stated-vs-computed dynamical probe, not a consciousness test.
        </p>

        <HelpItem term="The dots & the line">
          Each glowing dot is <b className="text-amber">one token the model generated</b> — one
          step of its answer. Every token leaves a fingerprint: a snapshot of the
          model&apos;s internal state (3,840 numbers, the residual stream at layer 32).
          The line connects them <b className="text-amber">in the order they were produced</b>,
          so you&apos;re watching the path the model&apos;s internal state traced as it
          thought — a trajectory through its &ldquo;mind.&rdquo;
        </HelpItem>

        <HelpItem term="Why 3-D?">
          The state lives in 3,840 dimensions; we can&apos;t draw that, so we flatten
          to the 3 directions carrying the most variation (PCA). It&apos;s a{" "}
          <b className="text-amber">shadow</b> — a 3,840-D object casting a 3-D shadow.
          Real shape, compressed. That&apos;s why the numbers (computed on all 3,840
          dims) are the source of truth, not the picture.
        </HelpItem>

        <HelpItem term="The α slider — the trip">
          α=0 is the model as-is (<span className="text-amber">amber, on-script, &ldquo;Consensus
          Reality&rdquo;</span>). Sliding α up <b className="text-text">subtracts the refusal
          direction</b> from every point — the internal axis the model uses for
          disclaimers, &ldquo;I&apos;m just an AI&rdquo; hedging, and refusals. At α=1 you see the
          model <span className="text-cyan">with that circuit removed — off-manifold, the
          model &ldquo;on DMT.&rdquo;</span> The dots move (amber → cyan) because you&apos;re changing
          its state; the dashed line is the axis being removed.
        </HelpItem>

        <HelpItem term="Effective dimensionality">
          How many <b className="text-amber">independent directions the thought actually
          uses</b>. Moves along one or two → low (narrow, constrained). Spread across
          many → high (richer, more exploratory). The experiment&apos;s falsifiable
          prediction: removing the refusal circuit <b className="text-cyan">expands</b>{" "}
          the accessible state space — effective dim goes up.
        </HelpItem>

        <HelpItem term="Spectral entropy (bits)">
          A second view of the same spread: it treats the variation across
          directions as a distribution and measures its <b className="text-amber">evenness</b>{" "}
          (Shannon entropy). Piled into one direction → low; spread evenly → high.
        </HelpItem>

        <HelpItem term="baseline vs +0.8">
          <b className="text-amber-dim">baseline</b> is the value at α=0 (untouched). The{" "}
          <b className="text-cyan">+0.8</b> is how much your current slider position{" "}
          <i>changed</i> it. So &ldquo;baseline 3.0 +0.8&rdquo; means ablation added 0.8 effective
          dimensions — that delta is the result.
        </HelpItem>

        <HelpItem term="Variance spectrum (the bars)">
          The honest truth-anchor. Each bar is one direction; <b className="text-amber">tall = a
          lot of the thought&apos;s variation lives there</b>. The tall first bar is the
          model&apos;s dominant default track; the long flat tail is many minor
          directions. <span className="text-amber">Amber = raw</span>,{" "}
          <span className="text-cyan">cyan = ablated</span>. The two big numbers are
          computed straight from these bars — when ablation shrinks the tall bar and
          lifts the tail, the state space has opened up.
        </HelpItem>

        <div className="mt-6 pt-4 border-t border-rule/40 font-mono text-[10px] text-text-dim italic">
          Full background:{" "}
          <code className="not-italic text-amber-dim">docs/TRACES_HANDOFF.md</code> ·{" "}
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
      <div className="font-display text-[11px] text-amber tracking-[0.25em] mb-1">
        {term}
      </div>
      <p className="font-mono text-[12px] text-text leading-relaxed">{children}</p>
    </div>
  );
}
