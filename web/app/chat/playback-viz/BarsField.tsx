"use client";

import type { PlaybackVizProps } from "./types";

/** Spectrum-style bar field. Each bar runs its own CSS keyframe
 *  scaleY pulse with a unique duration + delay so the field reads
 *  like a real-time analyser without an actual analyser. Doesn't
 *  consume the audio graph — kept around as the original /
 *  reference visualization.
 */
export function BarsField({ accent, accentRgb }: PlaybackVizProps) {
  return (
    <div
      className="absolute inset-0 flex items-end justify-between gap-[2px]"
      style={{ transformOrigin: "bottom" }}
    >
      <style>{`
        @keyframes ci-bar-a {
          0%   { transform: scaleY(0.12); }
          18%  { transform: scaleY(0.85); }
          37%  { transform: scaleY(0.32); }
          54%  { transform: scaleY(0.72); }
          71%  { transform: scaleY(0.20); }
          88%  { transform: scaleY(0.58); }
          100% { transform: scaleY(0.15); }
        }
        @keyframes ci-bar-b {
          0%   { transform: scaleY(0.45); }
          22%  { transform: scaleY(0.18); }
          41%  { transform: scaleY(0.92); }
          63%  { transform: scaleY(0.30); }
          82%  { transform: scaleY(0.68); }
          100% { transform: scaleY(0.50); }
        }
        @keyframes ci-bar-c {
          0%   { transform: scaleY(0.28); }
          15%  { transform: scaleY(0.62); }
          33%  { transform: scaleY(0.14); }
          50%  { transform: scaleY(0.80); }
          67%  { transform: scaleY(0.35); }
          85%  { transform: scaleY(0.55); }
          100% { transform: scaleY(0.22); }
        }
      `}</style>
      {Array.from({ length: 32 }).map((_, i) => {
        const variant = ["ci-bar-a", "ci-bar-b", "ci-bar-c"][i % 3];
        const dur = 0.65 + ((i * 13) % 11) * 0.06;
        const delay = ((i * 17) % 23) * 0.045;
        return (
          <div
            key={i}
            className="flex-1 rounded-[1px] h-full origin-bottom"
            style={{
              background: accent,
              boxShadow: `0 0 6px ${accent}, 0 0 12px rgba(${accentRgb},0.65)`,
              opacity: 0.95,
              animation: `${variant} ${dur}s ${delay}s ease-in-out infinite alternate`,
              willChange: "transform",
            }}
          />
        );
      })}
    </div>
  );
}
