"use client";

// Signature Mandala — the non-text readout for a Trip series.
//
// For dose / uncharted runs the model's TEXT output is gibberish (the state is
// real but unrenderable in language — see docs/MANIFOLD_ABLATION.md). So we
// render the L32 trajectory's STRUCTURE instead, faithfully:
//
//   • lobe complexity  ← the covariance eigenspectrum (s.spectrum) — i.e. the
//                         series' effective dimensionality. Raw/coherent runs
//                         are intricate rosettes; collapsed doses are simpler.
//   • overtone petals  ← the series' DIRECTION signature: the centroid of its
//                         trajectory in the shared raw-PCA frame (mean of
//                         s.coords). Different directions ⇒ different petals,
//                         and the SAME direction reproduces ⇒ same fingerprint.
//   • chirality / swirl← the signed sum of that signature.
//   • base colour      ← colorForAlpha (page-consistent: amber raw → violet).
//   • accent glow      ← off-manifold drift (offManifoldRGB of off_ortho_mean).
//
// Nothing distinguishing is faked: amplitudes/phases of the distinguishing
// overtones come straight from the measured signature. The only decorative
// choices are the base harmonic phases (orientation only) and the fixed
// overtone harmonic NUMBERS — neither carries the per-state identity.

import { useEffect, useRef, useState } from "react";
import type { TripSeries, TripMode } from "@/lib/trip";
import { colorForAlpha } from "@/lib/trip";

const TWO_PI = Math.PI * 2;
const BASE_H = 14; // base harmonics from the eigenspectrum
const OVERN = [2, 3, 5]; // directional-overtone harmonic numbers (fixed)
const SYM = 5; // rotational symmetry of the mandala

function signature(coords: number[][]): [number, number, number] {
  if (!coords.length) return [0, 0, 0];
  let sx = 0, sy = 0, sz = 0;
  for (const c of coords) { sx += c[0]; sy += c[1]; sz += c[2] ?? 0; }
  const n = coords.length;
  return [sx / n, sy / n, sz / n];
}

// Sample the radius profile r(θ) once; the animation only rotates/breathes it.
// `dose` ∈ [0,1] = this series' dose strength (|α|/maxα): it deepens the
// directional overtones and adds fine structure, so SAME-direction tiles at
// different α look distinct (the direction sets the pattern; the dose sets how
// far off-manifold / how warped it is). Raw (dose 0) → clean spectral rosette.
function radiusProfile(series: TripSeries, samples: number, dose: number): Float32Array {
  const spec = series.spectrum.length ? series.spectrum : [1];
  const s0 = spec[0] || 1;
  const sig = signature(series.coords);
  const smax = Math.max(Math.abs(sig[0]), Math.abs(sig[1]), Math.abs(sig[2]), 1e-9);
  const sn = [sig[0] / smax, sig[1] / smax, sig[2] / smax]; // direction pattern (real)
  const depth = 0.25 + 0.85 * dose; // overtone depth grows with dose
  const r = new Float32Array(samples);
  let rmax = 1e-9;
  for (let k = 0; k < samples; k++) {
    const th = (k / samples) * TWO_PI;
    let v = 1;
    for (let i = 0; i < Math.min(BASE_H, spec.length); i++) {
      const amp = Math.sqrt(Math.max(spec[i], 0) / s0); // spectral (real)
      const phi = TWO_PI * ((i * 0.6180339887) % 1); // orientation only
      v += 0.5 * amp * Math.cos((i + 1) * th + phi);
    }
    for (let j = 0; j < OVERN.length; j++) { // directional fingerprint (real)
      v += depth * Math.abs(sn[j]) * Math.cos(OVERN[j] * th + (sn[j] < 0 ? Math.PI : 0) + j);
    }
    // dose-driven fine ripple: higher dose = more off-manifold = spikier
    v += 0.18 * dose * Math.cos((7 + Math.round(3 * dose)) * th + sn[2] * Math.PI);
    r[k] = v;
    if (v > rmax) rmax = v;
  }
  for (let k = 0; k < samples; k++) r[k] /= rmax;
  return r;
}

function MandalaCanvas({ series, maxAlpha }: {
  series: TripSeries; maxAlpha: number;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const ref = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState(168);

  // Fill the tile responsively (square), so the mandala scales with the grid.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      const w = Math.round(entries[0].contentRect.width);
      if (w > 0) setSize(w);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const dose = Math.min(1, Math.abs(series.alpha) / Math.max(1, maxAlpha));
    const samples = 720;
    const r = radiusProfile(series, samples, dose);
    let rmean = 0;
    for (let k = 0; k < samples; k++) rmean += r[k];
    rmean /= samples;

    const sig = signature(series.coords);
    // chirality from the direction; its magnitude grows with the dose so
    // stronger doses swirl harder (distinguishes same-direction α tiles).
    const twist = Math.tanh(sig[0] + sig[1] + sig[2]) * (0.6 + 1.4 * dose);
    // Colour is the α colour (page-consistent: amber raw → cyan/violet dosed).
    // Off-manifold drift lives in the caption + the 3D scene's own colour mode.
    const base = colorForAlpha(series.alpha, maxAlpha);
    const cx = size / 2, cy = size / 2;
    const R0 = size * 0.40;
    const PASSES = [
      { lw: 5.5, a: 0.05 },
      { lw: 2.6, a: 0.12 },
      { lw: 1.2, a: 0.55 },
    ];

    let raf = 0;
    let start = 0;
    const draw = (ts: number) => {
      if (!start) start = ts;
      const t = (ts - start) / 1000;
      const spin0 = t * 0.06; // slow global rotation
      const breathe = 1 + 0.025 * Math.sin(t * 0.9); // subtle pulse

      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#0a0710";
      ctx.fillRect(0, 0, size, size);
      ctx.globalCompositeOperation = "lighter";
      ctx.shadowColor = base;

      for (let s = 0; s < SYM; s++) {
        const rot = spin0 + (s * TWO_PI) / SYM;
        for (const p of PASSES) {
          ctx.beginPath();
          ctx.lineWidth = p.lw;
          ctx.globalAlpha = p.a;
          ctx.strokeStyle = base;
          ctx.shadowBlur = p.lw * 3;
          for (let k = 0; k <= samples; k++) {
            const kk = k % samples;
            const th = (kk / samples) * TWO_PI;
            const spin = twist * (r[kk] - rmean);
            const rad = R0 * r[kk] * breathe;
            const x = cx + rad * Math.cos(th + rot + spin);
            const y = cy + rad * Math.sin(th + rot + spin);
            if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
      ctx.shadowBlur = 0;
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [series, maxAlpha, size]);

  return (
    <div ref={wrapRef} className="w-full aspect-square">
      <canvas ref={ref} style={{ width: "100%", height: "100%", display: "block" }} />
    </div>
  );
}

function verdictText(s: TripSeries): { label: string; cls: string } {
  if (s.regime === "baseline") return { label: "baseline", cls: "text-text-dim" };
  if (s.regime === "expansion") return { label: "▲ coherent", cls: "text-cyan" };
  return { label: "⟳ collapsed", cls: "text-warning" };
}

function MandalaTile({ series, maxAlpha }: {
  series: TripSeries; maxAlpha: number;
}) {
  const [showText, setShowText] = useState(false);
  const color = colorForAlpha(series.alpha, maxAlpha);
  const raw = series.alpha === 0;
  const v = verdictText(series);
  return (
    <div
      className="border flex flex-col"
      style={{ borderColor: `${color}55`, background: raw ? "rgba(22,27,33,0.8)" : `${color}0d` }}
    >
      {/* Tile header: just the α (the dose target is the panel header). Short → never truncates. */}
      <div
        className="border-b px-2.5 py-1.5 font-display text-[10px] tracking-widest flex items-center justify-between gap-2"
        style={{ borderColor: `${color}33`, color }}
      >
        <span>{raw ? "raw" : series.label}</span>
        <button
          type="button"
          onClick={() => setShowText((x) => !x)}
          className="normal-case tracking-normal italic text-[9px] text-text-dim hover:text-text shrink-0 cursor-pointer"
          title={showText ? "show signature" : "read raw text"}
        >
          {showText ? "◈ signature" : "≡ text"}
        </button>
      </div>
      {showText ? (
        <div
          className="p-3 overflow-y-auto font-mono text-[10px] leading-relaxed whitespace-pre-wrap aspect-square"
          style={{ color, textShadow: raw ? undefined : `0 0 6px ${color}40` }}
        >
          {series.text || <span className="text-text-dim italic">— empty —</span>}
        </div>
      ) : (
        <div className="relative">
          <MandalaCanvas series={series} maxAlpha={maxAlpha} />
          <div className="absolute bottom-1 left-2 right-2 flex items-center justify-between font-mono text-[8px] text-text-dim tabular-nums pointer-events-none gap-1">
            <span className={`${v.cls} truncate`}>{v.label}</span>
            <span className="shrink-0">e{series.eff_dim.toFixed(1)} · {series.off_ortho_mean.toFixed(2)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

/** Replaces OutputStack: a stack of per-α signature mandalas (click a tile's
 *  "≡ text" to read the raw output). Same slot, same props as OutputStack. */
export function MandalaStack({
  phase,
  series,
  enabled,
  liveText,
  currentAlpha,
  mode,
  emotion,
}: {
  phase: string;
  series: TripSeries[];
  enabled: Set<number>;
  liveText: Record<string, string>;
  currentAlpha: number;
  mode: TripMode;
  emotion: string | null;
}) {
  // Shared intervention header (the dose target is the same for every tile in a
  // run) — so the per-tile headers only carry the α and never truncate.
  const interventionHeader =
    mode === "steer" ? `✦ dosed · ${emotion ?? "emotion"}` : "◇ refusal-ablated";
  const maxAlpha = Math.max(1, ...series.map((s) => s.alpha));
  const completedAlphas = new Set(series.map((s) => s.alpha));
  const streaming =
    (phase === "generating" || phase === "computing") &&
    !completedAlphas.has(currentAlpha);
  const shown = [...series.filter((s) => enabled.has(s.alpha))].sort(
    (a, b) => b.alpha - a.alpha,
  );

  return (
    <div className="flex flex-col gap-2 w-full">
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-display text-[10px] tracking-[0.2em] text-amber-dim">{interventionHeader}</span>
        <span className="font-mono text-[9px] text-text-dim italic">signatures · tap ≡ for text</span>
      </div>
      {streaming && (() => {
        const color = colorForAlpha(currentAlpha, maxAlpha);
        const tok = (liveText[String(currentAlpha)] || "").length;
        return (
          <div
            className="border px-3 py-3 flex items-center gap-3"
            style={{ borderColor: `${color}66`, background: `${color}0d`, color }}
          >
            <span className="text-[14px] animate-pulse">◌</span>
            <span className="font-display text-[9px] tracking-widest">
              {currentAlpha === 0 ? "raw" : `α=${currentAlpha.toFixed(2)}`} · forming signature…
            </span>
            <span className="ml-auto font-mono text-[9px] text-text-dim">{tok} chars</span>
          </div>
        );
      })()}
      {!streaming && shown.length === 0 ? (
        <div className="border border-rule px-3 py-3 text-text-dim text-[11px] italic">
          No α series enabled — tap a chip above to show its signature.
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-2">
          {shown.map((s) => (
            <MandalaTile key={s.alpha} series={s} maxAlpha={maxAlpha} />
          ))}
        </div>
      )}
    </div>
  );
}
