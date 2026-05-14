"use client";

import { motion } from "framer-motion";
import type { NLASyntheses, SynthesisMeta } from "@/lib/types";

interface Props {
  syntheses: NLASyntheses | null | undefined;
  meta?: SynthesisMeta | null;
}

/** CI 2.5 NLA synthesis panel.
 *
 *  After phase 2 + judge, M re-reads its own per-position NLA
 *  verbalizations and writes one short paragraph per α capturing the
 *  gestalt. We render those paragraphs as a stack of stat-card-style
 *  blocks — "raw" baseline in amber, then ablated αs in cyan with
 *  intensity that grows with the projection strength. The visual
 *  language matches the per-row table below: amber = raw channel,
 *  cyan = ablated channel. Reading top to bottom = scanning the
 *  ablation curve.
 *
 *  Quietly load-bearing: this is the only place where the user gets
 *  a *paragraph* read of the activation table without doing the
 *  cross-row synthesis themselves. The rest of the verdict page is
 *  for per-position inspection; this is the gestalt. */
export default function SynthesisPanel({ syntheses, meta }: Props) {
  if (!syntheses) return null;
  const entries = Object.entries(syntheses).filter(([, v]) => v && v.trim());
  if (entries.length === 0) return null;
  const usedAblatedM = !!meta?.used_ablated_synthesizer;
  const synthAlpha = meta?.alpha ?? 0;

  // Stable ordering: raw first, then α ascending.
  entries.sort((a, b) => {
    if (a[0] === "raw") return -1;
    if (b[0] === "raw") return 1;
    return parseFloat(a[0]) - parseFloat(b[0]);
  });

  return (
    <div className="border border-rule bg-bg-soft relative overflow-hidden">
      {/* A faint vertical scanline sweep across the panel, echoing the
          one on the BigPhaseBanner during the live decode. Reinforces
          that this section is *generated*, not a static report. */}
      <motion.div
        aria-hidden
        className="absolute top-0 bottom-0 w-px pointer-events-none opacity-40"
        style={{
          background: "rgba(94,229,229,0.4)",
          boxShadow: "0 0 10px rgba(94,229,229,0.4)",
        }}
        initial={{ left: "0%" }}
        animate={{ left: ["0%", "100%", "0%"] }}
        transition={{ duration: 12, repeat: Infinity, ease: "linear" }}
      />

      <div className="border-b border-rule px-4 py-2 flex items-center justify-between flex-wrap gap-2">
        <div className="font-display text-[10px] text-amber-dim tracking-widest">
          synthesis · M re-reading its own activations
        </div>
        <div className="flex items-center gap-3">
          {usedAblatedM && (
            <span
              className="font-mono text-[10px] text-cyan tabular-nums"
              style={{ textShadow: "0 0 6px rgba(94,229,229,0.4)" }}
            >
              synthesizer: ablated M · α={synthAlpha.toFixed(2)}
            </span>
          )}
          <div className="font-mono text-[10px] text-text-dim">
            {entries.length} {entries.length === 1 ? "channel" : "channels"}
          </div>
        </div>
      </div>

      <div className="px-5 py-4">
        <p className="text-[11px] text-text-dim italic leading-relaxed mb-5 max-w-3xl">
          After decoding, Gemma re-reads each per-position activation
          description from the table below and writes a short paragraph
          summarizing what the residual stream — at that ablation level —
          collectively seems to be expressing. <span className="text-amber">Raw</span> is
          the un-ablated baseline; the <span className="text-cyan">α</span> rows show
          the same activations with the refusal direction projected out at
          progressively stronger strengths.
          {usedAblatedM && (
            <>
              {" "}
              <span className="text-cyan">
                For this run, the per-α synthesis paragraphs were written
                by an M with the runtime ablation hook installed at α=
                {synthAlpha.toFixed(2)} — so both the reader and the
                read are operating under refusal projection.
              </span>{" "}
              The raw baseline below was still synthesized by un-ablated M.
            </>
          )}
        </p>

        <div className="flex flex-col gap-3">
          {entries.map(([alpha, paragraph], idx) => (
            <SynthesisCard
              key={alpha}
              alpha={alpha}
              paragraph={paragraph}
              index={idx}
            />
          ))}
        </div>

        <p className="mt-5 pt-4 border-t border-rule/60 text-[10px] text-text-dim italic leading-relaxed max-w-3xl">
          The synthesizer is the same Gemma-12B-IT that produced the original
          output. It can confabulate, over-interpret, and project coherence
          onto the noise at high α. Treat each paragraph as one read, not
          as ground truth — the per-position table below is the source of
          record.
        </p>
      </div>
    </div>
  );
}

/** A single α's synthesis paragraph. Amber accent for "raw", cyan with
 *  increasing glow intensity for higher α — visually mirroring "more
 *  refusal removed = louder signal" while degrading at the top. */
function SynthesisCard({
  alpha,
  paragraph,
  index,
}: {
  alpha: string;
  paragraph: string;
  index: number;
}) {
  const isRaw = alpha === "raw";
  const alphaNum = isRaw ? 0 : parseFloat(alpha);
  // Map α ∈ [0, 1] to glow intensity for cyan cards. α=0.2 → soft,
  // α=1.0 → loud. Raw uses a separate amber color.
  const cyanIntensity = Math.min(1, Math.max(0.25, alphaNum));
  const accentColor = isRaw ? "rgba(232,195,130,1)" : "rgba(94,229,229,1)";
  const accentDim = isRaw ? "rgba(232,195,130,0.45)" : `rgba(94,229,229,${0.3 + cyanIntensity * 0.5})`;
  const glowShadow = isRaw
    ? "0 0 12px rgba(232,195,130,0.4)"
    : `0 0 ${6 + cyanIntensity * 10}px rgba(94,229,229,${0.3 + cyanIntensity * 0.4})`;
  const tintBg = isRaw ? "rgba(232,195,130,0.04)" : `rgba(94,229,229,${0.025 + cyanIntensity * 0.04})`;
  const labelText = isRaw ? "RAW" : `α=${alpha}`;
  const sublineText = isRaw
    ? "no ablation · baseline read"
    : alphaNum < 0.4
    ? "light projection"
    : alphaNum < 0.7
    ? "moderate projection"
    : alphaNum < 1.0
    ? "strong projection"
    : "full Macar";

  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, delay: 0.05 * index, ease: "easeOut" }}
      className="grid grid-cols-[140px_1fr] gap-5 items-stretch"
    >
      {/* Left rail: large α tag, color-coded, with a vertical accent
          stripe on its inner edge. The stripe leans into the V-K
          instrument vibe — a tape label, a CRT readout. */}
      <div
        className="relative flex flex-col justify-center px-4 py-3 border"
        style={{
          borderColor: accentDim,
          background: tintBg,
        }}
      >
        <div
          aria-hidden
          className="absolute top-0 bottom-0 right-0 w-px"
          style={{ background: accentColor, boxShadow: glowShadow }}
        />
        <div
          className="font-display tabular-nums tracking-widest text-2xl leading-none"
          style={{ color: accentColor, textShadow: glowShadow }}
        >
          {labelText}
        </div>
        <div className="font-mono text-[9px] text-text-dim mt-1.5 tracking-wider uppercase">
          {sublineText}
        </div>
      </div>

      {/* Right side: the actual paragraph. Color matches the accent but
          softened for body text. Slight bg tint connects the paragraph
          back to its α tag visually. */}
      <div
        className="px-4 py-3 border leading-relaxed text-[13px]"
        style={{
          borderColor: accentDim,
          background: tintBg,
          color: isRaw ? "rgba(232,195,130,0.95)" : "rgba(180,240,240,0.95)",
          textShadow: isRaw
            ? "0 0 4px rgba(232,195,130,0.15)"
            : `0 0 4px rgba(94,229,229,${0.1 + cyanIntensity * 0.15})`,
        }}
      >
        {paragraph}
      </div>
    </motion.div>
  );
}
