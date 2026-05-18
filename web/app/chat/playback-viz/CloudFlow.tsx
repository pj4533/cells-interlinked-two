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

    // ── HiDPI sizing. We re-measure on every frame because the
    // canvas can be remounted with a different size between turns
    // and a ResizeObserver round-trip would add lag.
    const dpr = window.devicePixelRatio || 1;
    const fitToParent = () => {
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return false;
      const targetW = Math.max(1, Math.round(rect.width * dpr));
      const targetH = Math.max(1, Math.round(rect.height * dpr));
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }
      return true;
    };
    fitToParent();

    const analyser = getAnalyser();
    const binCount = analyser?.frequencyBinCount ?? 256;
    const freqBuf = new Uint8Array(binCount);

    // ── Layer config. Lower-frequency layers carry the main mass;
    // upper-frequency layers add texture and high-rate motion.
    const LAYERS = [
      // Big slow bass cloud — fills most of the canvas, anchors the
      // shape, very smooth in time so it breathes rather than jitters.
      {
        alpha: 0.20,
        blurPx: 14,
        scaleY: 1.05,
        offsetX: 0.0,
        bandStart: 0.00,
        bandEnd: 0.35,
        rise: 0.30,
        fall: 0.05,
      },
      // Mid cloud — slightly smaller scale, opposite horizontal offset
      // so it doesn't perfectly track the bass layer.
      {
        alpha: 0.16,
        blurPx: 10,
        scaleY: 0.90,
        offsetX: 0.06,
        bandStart: 0.18,
        bandEnd: 0.55,
        rise: 0.40,
        fall: 0.07,
      },
      // Upper-mid wisp — taller and thinner, gives the "highlight"
      // edge of the cloud.
      {
        alpha: 0.12,
        blurPx: 7,
        scaleY: 1.20,
        offsetX: -0.05,
        bandStart: 0.30,
        bandEnd: 0.75,
        rise: 0.55,
        fall: 0.10,
      },
      // High-frequency speckle — fast-rising/falling, low alpha so it
      // reads as texture inside the cloud.
      {
        alpha: 0.08,
        blurPx: 4,
        scaleY: 0.80,
        offsetX: 0.03,
        bandStart: 0.45,
        bandEnd: 0.95,
        rise: 0.85,
        fall: 0.18,
      },
    ];

    const CONTROL_POINTS = 48;
    // Per-layer time-smoothed envelopes (one float per control point).
    // Asymmetric rise/fall constants give the "swell on attack, decay
    // gently" feel that reads as breath.
    const envs = LAYERS.map(() => new Float32Array(CONTROL_POINTS));

    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      if (!fitToParent()) return;

      const W = canvas.width;
      const H = canvas.height;
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
          const v = (
            (freqBuf[Math.max(0, bi - 1)] +
              freqBuf[bi] +
              freqBuf[Math.min(binCount - 1, bi + 1)]) /
            3
          ) / 255;
          // Asymmetric envelope follower: snap up fast, decay slow.
          const target = v;
          const prev = env[i];
          const k = target > prev ? layer.rise : layer.fall;
          env[i] = prev + (target - prev) * k;
        }

        // Render this layer as a filled symmetric blob: top half
        // sweeps left-to-right above center, then mirror back below.
        c2d.beginPath();
        const centerY = H * 0.5;
        const maxAmp = H * 0.46 * layer.scaleY;
        const offsetPx = W * layer.offsetX;
        const halfPad = 24; // bleed off-canvas so cloud edges don't
        //                    hard-clip against the column borders.

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
        c2d.filter = `blur(${layer.blurPx}px)`;
        c2d.fillStyle = `rgba(${accentRgb},${layer.alpha})`;
        c2d.shadowColor = `rgba(${accentRgb},${Math.min(0.4, layer.alpha * 1.8)})`;
        c2d.shadowBlur = layer.blurPx * 1.5;
        c2d.fill();
      }

      // Reset state we touched so it doesn't carry into the next
      // frame's clear.
      c2d.filter = "none";
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
