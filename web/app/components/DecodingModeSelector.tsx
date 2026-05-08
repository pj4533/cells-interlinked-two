"use client";

import {
  DECODING_MODES,
  DECODING_MODE_LABELS,
  modeDescription,
  type DecodingMode,
} from "@/lib/decodingModes";

interface Props {
  active: DecodingMode;
  pooled: boolean;
  onChange: (m: DecodingMode, pooled: boolean) => void;
  busy?: boolean;
  /** Used by /autorun where the toggle is non-default → glow. The /interrogate
   *  picker leaves this off so the row reads as a normal config control. */
  glowWhenNonDefault?: boolean;
}

/** Mirrors ProbeSetSelector's chrome — same chip-row visual treatment.
 *  Adds a "POOLED" pill on the right that toggles whether each sample
 *  decodes one position or a mean-pooled window of adjacent positions. */
export default function DecodingModeSelector({
  active,
  pooled,
  onChange,
  busy,
  glowWhenNonDefault,
}: Props) {
  const isActive = active !== "per-token" || pooled;
  const pooledDisabled = active === "per-token";
  return (
    <div
      className="border border-rule bg-bg-soft px-5 py-4 flex items-center gap-5 flex-wrap"
      style={
        glowWhenNonDefault && isActive
          ? {
              borderColor: "var(--amber-dim)",
              boxShadow: "0 0 18px rgba(232, 195, 130, 0.18)",
            }
          : undefined
      }
    >
      <div
        role="radiogroup"
        aria-label="decoding mode"
        className="inline-flex shrink-0 border border-rule"
      >
        {DECODING_MODES.map((m, i) => {
          const selected = m === active;
          const isLast = i === DECODING_MODES.length - 1;
          return (
            <button
              key={m}
              type="button"
              role="radio"
              aria-checked={selected}
              disabled={busy || selected}
              onClick={() => onChange(m, pooled)}
              className="px-3 py-1.5 font-display text-[10px] tracking-widest transition-colors disabled:cursor-default"
              style={{
                background: selected ? "var(--amber-dim)" : "var(--bg)",
                color: selected ? "var(--bg)" : "var(--text-dim)",
                borderRight: isLast ? "none" : "1px solid var(--rule)",
              }}
            >
              {DECODING_MODE_LABELS[m]}
            </button>
          );
        })}
      </div>

      <button
        type="button"
        role="switch"
        aria-checked={pooled}
        disabled={busy || pooledDisabled}
        onClick={() => onChange(active, !pooled)}
        className="inline-flex items-center gap-2 px-3 py-1.5 font-display text-[10px] tracking-widest border transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        style={{
          background: pooled && !pooledDisabled ? "var(--cyan-dim)" : "var(--bg)",
          color: pooled && !pooledDisabled ? "var(--bg)" : "var(--text-dim)",
          borderColor: pooled && !pooledDisabled
            ? "var(--cyan-dim)"
            : "var(--rule)",
        }}
        title={
          pooledDisabled
            ? "Pooled is a no-op at per-token (windows of 1)"
            : pooled
            ? "Pooled ON — each pick is a window of activations mean-pooled into one decode"
            : "Pooled OFF — each pick is a single position"
        }
      >
        {pooled && !pooledDisabled ? "POOLED ON" : "POOLED OFF"}
      </button>

      <div className="flex-1 min-w-[18rem]">
        <div
          className={`font-display text-sm tracking-widest ${
            isActive ? "text-amber amber-glow" : "text-text-dim"
          }`}
        >
          NLA DECODING
        </div>
        <div className="text-[10px] text-text-dim italic mt-0.5 leading-snug">
          {modeDescription(active, pooled)}
        </div>
      </div>
    </div>
  );
}
