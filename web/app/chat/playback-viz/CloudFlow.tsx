"use client";

import { useEffect, useRef } from "react";

import { getAnalyser } from "./audio-graph";
import type { PlaybackVizProps } from "./types";

/** Audio-reactive cloud / nebula visualization.
 *
 *  Multiple translucent layers stacked on a canvas, each driven by a
 *  different frequency band of the playing audio. Each layer is a
 *  filled curve (top half above center, mirrored below) whose shape
 *  is computed from live `getByteFrequencyData()` samples; layers
 *  use different per-frame smoothing factors and slight horizontal
 *  offsets so they read as overlapping clouds rather than a single
 *  oscilloscope trace.
 *
 *  Rendering is imperative via requestAnimationFrame — no React
 *  re-renders during playback. Canvas blur + shadow gives the soft,
 *  edges-blending-into-each-other quality.
 *
 *  Knobs to iterate on:
 *    LAYERS array     – per-layer alpha, blur, scale, band range,
 *                       smoothing, offset
 *    CONTROL_POINTS   – curve resolution (higher = smoother)
 *    RISE_CONSTANT    – how fast the envelope can grow per frame
 *    FALL_CONSTANT    – how fast it decays
 */
export function CloudFlow({ accent, accentRgb }: PlaybackVizProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const c2d = canvas.getContext("2d", { alpha: true });
    if (!c2d) return;

    // ── HiDPI sizing. The backing store gets bumped to dpr × CSS
    // pixels, but the drawing transform is reset to (dpr,dpr) so
    // all draw calls use CSS pixels. Critical for Retina/iPad:
    // ctx.filter blur values are in CSS pixels, and without the
    // transform a `blur(18px)` looks half as wide on a dpr=2
    // screen vs a dpr=1 one. cssW/cssH are what we draw to.
    const fitToParent = (): { cssW: number; cssH: number } | null => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return null;
      const targetW = Math.max(1, Math.round(rect.width * dpr));
      const targetH = Math.max(1, Math.round(rect.height * dpr));
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }
      c2d.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { cssW: rect.width, cssH: rect.height };
    };

    const analyser = getAnalyser();
    const binCount = analyser?.frequencyBinCount ?? 256;
    const freqBuf = new Uint8Array(binCount);

    // ── Layer config.
    //
    // Speech audio at 48 kHz has most energy in the bottom ~15% of
    // the frequency bins (200 Hz–4 kHz formants). If we spread the
    // layers evenly across 0..1 of the spectrum, the higher layers
    // see almost no signal and look dead. Compensation:
    //   - Pack the bands into the speech-dense low end (≤ 0.35).
    //   - Give higher-band layers a `gain` multiplier so their
    //     smaller raw amplitude still translates into visible motion.
    //   - Bump alphas and amplitudes overall so the cloud reads as
    //     a *bright* nebula instead of a dim wash.
    const LAYERS = [
      // 0) Bass anchor — the slow-breathing body of the cloud.
      {
        alpha: 0.38,
        blurPx: 18,
        scaleY: 1.35,
        offsetX: 0.0,
        bandStart: 0.00,
        bandEnd: 0.05,
        gain: 1.0,
        rise: 0.45,
        fall: 0.04,
      },
      // 1) Low-mid — first formants, carries the vowel core.
      {
        alpha: 0.30,
        blurPx: 12,
        scaleY: 1.20,
        offsetX: 0.07,
        bandStart: 0.03,
        bandEnd: 0.12,
        gain: 1.25,
        rise: 0.55,
        fall: 0.06,
      },
      // 2) Mid — upper formants, consonant onsets.
      {
        alpha: 0.26,
        blurPx: 9,
        scaleY: 1.45,
        offsetX: -0.06,
        bandStart: 0.08,
        bandEnd: 0.22,
        gain: 1.55,
        rise: 0.65,
        fall: 0.09,
      },
      // 3) Sibilants — fast-moving brightness, the "shimmer."
      {
        alpha: 0.22,
        blurPx: 6,
        scaleY: 1.15,
        offsetX: 0.04,
        bandStart: 0.18,
        bandEnd: 0.45,
        gain: 1.95,
        rise: 0.80,
        fall: 0.14,
      },
      // 4) High sparkle — sub-second texture inside the cloud.
      {
        alpha: 0.16,
        blurPx: 3,
        scaleY: 0.95,
        offsetX: -0.03,
        bandStart: 0.30,
        bandEnd: 0.75,
        gain: 2.4,
        rise: 0.92,
        fall: 0.22,
      },
    ];

    const CONTROL_POINTS = 48;
    // Per-layer time-smoothed envelopes (one float per control point).
    // Asymmetric rise/fall constants give the "swell on attack, decay
    // gently" feel that reads as breath.
    const envs = LAYERS.map(() => new Float32Array(CONTROL_POINTS));

    // Feature detection: ctx.filter (used for per-layer Gaussian
    // blur) only landed in Safari/iPadOS 17. On older versions the
    // assignment is silently ignored, leaving hard-edged shapes.
    // When unsupported we compensate with bigger shadowBlur + an
    // extra fill pass so layers still bleed into each other via
    // their halos rather than stacking like sharp foils.
    const supportsFilter = typeof (c2d as { filter?: string }).filter !== "undefined";

    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const dims = fitToParent();
      if (!dims) return;

      const W = dims.cssW;
      const H = dims.cssH;
      // clearRect uses transformed coords (CSS pixels) since we
      // setTransform'd to (dpr, dpr). H/W are CSS, so this clears
      // the entire visible area.
      c2d.clearRect(0, 0, W, H);

      if (analyser) {
        analyser.getByteFrequencyData(freqBuf);
      }

      for (let li = 0; li < LAYERS.length; li++) {
        const layer = LAYERS[li];
        const env = envs[li];

        // Map this layer's band of the spectrum onto its control
        // points. We sample with a linear walk so adjacent control
        // points see similar bins (avoids jagged shape).
        const startIdx = Math.floor(layer.bandStart * binCount);
        const endIdx = Math.floor(layer.bandEnd * binCount);
        const bandSize = Math.max(1, endIdx - startIdx);
        for (let i = 0; i < CONTROL_POINTS; i++) {
          const t = i / (CONTROL_POINTS - 1);
          const bi = Math.min(
            binCount - 1,
            startIdx + Math.floor(t * bandSize),
          );
          // Light spatial smoothing across the band so single hot
          // bins don't poke through.
          const raw =
            (freqBuf[Math.max(0, bi - 1)] +
              freqBuf[bi] +
              freqBuf[Math.min(binCount - 1, bi + 1)]) /
            3 /
            255;
          // Per-layer gain so higher-band layers (which have less
          // raw energy) still drive visible motion. Clamp to 1 so
          // we don't blow past the canvas.
          const target = Math.min(1, raw * layer.gain);
          const prev = env[i];
          const k = target > prev ? layer.rise : layer.fall;
          env[i] = prev + (target - prev) * k;
        }

        // Render this layer as a filled symmetric blob: top half
        // sweeps left-to-right above center, then mirror back below.
        c2d.beginPath();
        const centerY = H * 0.5;
        // 0.58 lets a fully-saturated env reach (and slightly past)
        // the canvas edge; with the per-layer scaleY > 1 multipliers
        // on top, peaks bleed off-canvas, which the blur turns into
        // soft halos rather than hard clips.
        const maxAmp = H * 0.58 * layer.scaleY;
        const offsetPx = W * layer.offsetX;
        const halfPad = 32; // bleed off-canvas so cloud edges don't
        //                     hard-clip against the column borders.

        // Top edge: 0..N
        for (let i = 0; i < CONTROL_POINTS; i++) {
          const x =
            ((i / (CONTROL_POINTS - 1)) * (W + halfPad * 2) -
              halfPad) +
            offsetPx;
          const amp = env[i] * maxAmp;
          if (i === 0) {
            c2d.moveTo(x, centerY - amp);
          } else {
            const prevT = (i - 1) / (CONTROL_POINTS - 1);
            const prevX =
              (prevT * (W + halfPad * 2) - halfPad) + offsetPx;
            const prevAmp = env[i - 1] * maxAmp;
            const midX = (prevX + x) / 2;
            const midY = ((centerY - prevAmp) + (centerY - amp)) / 2;
            c2d.quadraticCurveTo(prevX, centerY - prevAmp, midX, midY);
            if (i === CONTROL_POINTS - 1) {
              c2d.lineTo(x, centerY - amp);
            }
          }
        }
        // Bottom edge: N..0 (mirrored)
        for (let i = CONTROL_POINTS - 1; i >= 0; i--) {
          const x =
            ((i / (CONTROL_POINTS - 1)) * (W + halfPad * 2) -
              halfPad) +
            offsetPx;
          const amp = env[i] * maxAmp;
          if (i === CONTROL_POINTS - 1) {
            c2d.lineTo(x, centerY + amp);
          } else {
            const nextT = (i + 1) / (CONTROL_POINTS - 1);
            const nextX =
              (nextT * (W + halfPad * 2) - halfPad) + offsetPx;
            const nextAmp = env[i + 1] * maxAmp;
            const midX = (nextX + x) / 2;
            const midY = ((centerY + nextAmp) + (centerY + amp)) / 2;
            c2d.quadraticCurveTo(nextX, centerY + nextAmp, midX, midY);
            if (i === 0) c2d.lineTo(x, centerY + amp);
          }
        }
        c2d.closePath();

        // Soft fill + glow. Canvas filter blur is the secret sauce
        // for the cloud edge — it diffuses the shape so layers
        // bleed into each other instead of stacking like sharp foils.
        // shadowBlur stacks on top of filter blur for extra halo.
        //
        // On Safari < 17 / iPadOS < 17, ctx.filter is undefined; we
        // skip the assignment and lean entirely on shadowBlur (with
        // a larger radius) for the cloud effect. The result is less
        // soft but still nebulous.
        if (supportsFilter) {
          c2d.filter = `blur(${layer.blurPx}px)`;
        }
        c2d.fillStyle = `rgba(${accentRgb},${layer.alpha})`;
        const shadowFactor = supportsFilter ? 2.4 : 4.2;
        c2d.shadowColor = `rgba(${accentRgb},${Math.min(0.8, layer.alpha * 2.4)})`;
        c2d.shadowBlur = layer.blurPx * shadowFactor;
        c2d.fill();
        // Double-fill the bass / mid layers (li=0,1) to push them
        // brighter without affecting the higher detail layers.
        if (li <= 1) {
          c2d.shadowBlur = layer.blurPx * (supportsFilter ? 1.2 : 2.4);
          c2d.fill();
        }
      }

      // Reset state we touched so it doesn't carry into the next
      // frame's clear. Guard `filter` for older Safari where the
      // property doesn't exist.
      if (supportsFilter) c2d.filter = "none";
      c2d.shadowBlur = 0;
    };
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
    };
  }, [accent, accentRgb]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full"
      // No CSS background — the canvas is fully transparent and the
      // surrounding column's tint shows through.
      aria-hidden
    />
  );
}
