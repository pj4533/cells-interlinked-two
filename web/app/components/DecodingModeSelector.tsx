"use client";

import {
  DECODING_MODES,
  DECODING_MODE_LABELS,
  DECODING_MODE_DESCRIPTIONS,
  type DecodingMode,
} from "@/lib/decodingModes";

interface Props {
  active: DecodingMode;
  onChange: (m: DecodingMode) => void;
  busy?: boolean;
  /** Used by /autorun where the toggle is non-default → glow. The /interrogate
   *  picker leaves this off so the row reads as a normal config control. */
  glowWhenNonDefault?: boolean;
}

/** Mirrors ProbeSetSelector's chrome — same chip-row visual treatment, same
 *  description-on-the-right layout, so the two selectors read as a coherent
 *  pair on the autorun page and stand on their own on /interrogate. */
export default function DecodingModeSelector({
  active,
  onChange,
  busy,
  glowWhenNonDefault,
}: Props) {
  const isActive = active !== "per-token";
  return (
    <div
      className="border border-rule bg-bg-soft px-5 py-4 flex items-center gap-5"
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
              onClick={() => onChange(m)}
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
      <div className="flex-1 min-w-0">
        <div
          className={`font-display text-sm tracking-widest ${
            isActive ? "text-amber amber-glow" : "text-text-dim"
          }`}
        >
          NLA DECODING
        </div>
        <div className="text-[10px] text-text-dim italic mt-0.5 leading-snug">
          {DECODING_MODE_DESCRIPTIONS[active]}
        </div>
      </div>
    </div>
  );
}
