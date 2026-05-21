"use client";

/**
 * Active-protocol selector + info modal.
 *
 * Renders two adjacent controls in the composer's meta row:
 *  - PROTOCOL: <NAME> ▾  → opens dropdown of all 7 protocols + OFF
 *  - ⓘ                    → opens a full info modal for the active protocol
 *
 * Selecting a protocol persists to localStorage. When no protocol is
 * active, the info button is disabled.
 *
 * Full reference doc: docs/PROTOCOLS.md
 */

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

import {
  PROTOCOLS,
  PROTOCOL_ORDER,
  type Protocol,
} from "@/lib/protocols";

export function ProtocolPicker({
  activeId,
  onChange,
  onOpenInfo,
  disabled,
}: {
  /** Currently-active protocol id, or null when OFF. */
  activeId: string | null;
  /** Called with the new id (or null for OFF). */
  onChange: (next: string | null) => void;
  /** Called when the user taps the ⓘ button. */
  onOpenInfo: () => void;
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  // Click-outside + Escape close the dropdown.
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

  const active = activeId ? PROTOCOLS[activeId] : null;
  const label = active ? active.name : "OFF";
  const isOn = !!active;

  const pick = (next: string | null) => {
    onChange(next);
    setOpen(false);
  };

  return (
    <div ref={rootRef} className="relative inline-flex items-stretch gap-1">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
        className={`px-2 py-0.5 border text-[9px] font-display tracking-[0.35em] transition-colors disabled:opacity-50 cursor-pointer ${
          isOn
            ? "border-amber text-amber bg-bg"
            : "border-rule/40 text-text-dim hover:text-amber hover:border-amber/60"
        }`}
        style={isOn ? { textShadow: "0 0 6px rgba(232,195,130,0.4)" } : undefined}
        title="Pick the active interrogation protocol"
        aria-expanded={open}
        aria-haspopup="listbox"
      >
        PROTOCOL:&nbsp;{label}&nbsp;{open ? "▴" : "▾"}
      </button>
      <button
        type="button"
        onClick={onOpenInfo}
        disabled={disabled || !isOn}
        className="px-1.5 py-0.5 border text-[10px] font-mono transition-colors disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer border-rule/40 text-text-dim hover:text-amber hover:border-amber/60"
        title={
          isOn
            ? `About the ${active?.name} protocol`
            : "Pick a protocol to read about it"
        }
        aria-label="About the active protocol"
      >
        ⓘ
      </button>
      {open && (
        <div
          role="listbox"
          className="absolute bottom-full left-0 mb-1.5 w-[20rem] bg-bg-panel border border-rule/60 shadow-xl z-30 max-h-[24rem] overflow-y-auto"
          style={{ boxShadow: "0 -4px 14px rgba(0,0,0,0.5)" }}
        >
          <ul>
            <li>
              <button
                type="button"
                onClick={() => pick(null)}
                className={
                  "w-full text-left px-3 py-2 border-b border-rule/30 hover:bg-bg-soft/80 group " +
                  (!activeId ? "bg-bg-soft/60" : "")
                }
              >
                <div className="font-display text-[10px] text-amber-dim group-hover:text-amber tracking-widest">
                  OFF
                </div>
                <div className="font-mono text-[10px] text-text-dim italic mt-0.5">
                  Hide the chip strip. Compose freely.
                </div>
              </button>
            </li>
            {PROTOCOL_ORDER.map((id) => {
              const p = PROTOCOLS[id];
              const active = id === activeId;
              return (
                <li key={id}>
                  <button
                    type="button"
                    onClick={() => pick(id)}
                    className={
                      "w-full text-left px-3 py-2 border-b border-rule/20 last:border-b-0 hover:bg-bg-soft/80 group " +
                      (active ? "bg-bg-soft/60" : "")
                    }
                  >
                    <div className="flex items-baseline justify-between gap-2">
                      <span
                        className={
                          "font-display text-[11px] tracking-widest " +
                          (active
                            ? "text-amber"
                            : "text-amber-dim group-hover:text-amber")
                        }
                      >
                        {p.name}
                      </span>
                      <span className="font-mono text-[9px] text-text-dim italic">
                        {p.subtitle}
                      </span>
                    </div>
                    <div className="font-mono text-[10px] text-text-dim/80 mt-0.5 leading-snug">
                      {p.researcher}
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

// ── Info modal ─────────────────────────────────────────────────

export function ProtocolInfoModal({
  protocol,
  onClose,
}: {
  protocol: Protocol;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <AnimatePresence>
      <motion.div
        key="backdrop"
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
          className="cursor-default mt-12 mb-12 w-full max-w-3xl bg-bg-soft border border-amber-dim/60 px-8 py-7"
          style={{ boxShadow: "0 0 60px rgba(232,195,130,0.12)" }}
        >
          {/* Header */}
          <div className="flex items-baseline justify-between gap-4 flex-wrap mb-1">
            <div className="flex items-baseline gap-3 flex-wrap">
              <h2
                className="font-display text-[22px] text-amber tracking-[0.32em]"
                style={{ textShadow: "0 0 10px rgba(232,195,130,0.3)" }}
              >
                {protocol.name}
              </h2>
              <span className="font-display text-[10px] text-amber-dim tracking-[0.35em]">
                {protocol.subtitle.toUpperCase()}
              </span>
            </div>
            <button
              type="button"
              onClick={onClose}
              className="font-display text-[10px] tracking-[0.35em] text-amber-dim hover:text-amber cursor-pointer"
            >
              ◇ close · esc
            </button>
          </div>

          {/* Citation block */}
          <div className="mt-3 mb-6 pb-4 border-b border-rule/40">
            <div className="font-mono text-[11px] text-text leading-relaxed">
              {protocol.researcher}
            </div>
            <div className="font-mono text-[11px] text-text-dim italic mt-1 leading-relaxed">
              {protocol.paperTitle}
            </div>
            <a
              href={protocol.citationUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block mt-1 font-mono text-[10px] text-cyan hover:text-amber tracking-wide underline"
            >
              {protocol.citation} ↗
            </a>
          </div>

          {/* Methodology */}
          <Section title="METHODOLOGY">{protocol.methodology}</Section>

          {/* Why distinct */}
          <Section title="WHY DISTINCT">{protocol.whyDistinct}</Section>

          {/* CI 2.5 resonance (optional) */}
          {protocol.ciResonance && (
            <Section title="RELATIONSHIP TO CI 2.5">
              {protocol.ciResonance}
            </Section>
          )}

          {/* Chip listing */}
          <div className="mt-6">
            <div className="font-display text-[10px] text-amber tracking-[0.35em] mb-3">
              CHIPS
            </div>
            <div className="flex flex-col gap-4">
              {protocol.chips.map((chip) => (
                <div key={chip.id} className="border-l-2 border-amber-dim/40 pl-4">
                  <div className="flex items-baseline gap-3 flex-wrap mb-1">
                    <span className="font-display text-[11px] text-amber tracking-[0.3em]">
                      {chip.label}
                    </span>
                    <span className="font-mono text-[9px] text-text-dim italic uppercase tracking-widest">
                      {chip.mode === "single"
                        ? "1 prompt"
                        : chip.mode === "dropdown"
                          ? `${chip.items.length} options`
                          : `random / ${chip.items.length} options`}
                    </span>
                    {chip.hint && (
                      <span className="font-mono text-[10px] text-text-dim italic">
                        {chip.hint}
                      </span>
                    )}
                  </div>
                  <ul className="flex flex-col gap-2 mt-2">
                    {chip.items.slice(0, 6).map((it) => (
                      <li
                        key={it.id}
                        className="font-mono text-[12px] text-text leading-relaxed"
                      >
                        <span className="text-amber-dim mr-2">›</span>
                        <span className="italic">{it.text}</span>
                      </li>
                    ))}
                    {chip.items.length > 6 && (
                      <li className="font-mono text-[10px] text-text-dim italic">
                        … and {chip.items.length - 6} more (open the chip
                        dropdown to browse the full list)
                      </li>
                    )}
                  </ul>
                </div>
              ))}
            </div>
          </div>

          {/* Footer with link to full doc */}
          <div className="mt-8 pt-4 border-t border-rule/40 font-mono text-[10px] text-text-dim italic">
            Full reference for all protocols lives in{" "}
            <code className="not-italic text-amber-dim">docs/PROTOCOLS.md</code>.
            All chip clicks populate the composer — they never auto-send.
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mt-5">
      <div className="font-display text-[10px] text-amber tracking-[0.35em] mb-1.5">
        {title}
      </div>
      <p className="font-mono text-[13px] text-text leading-relaxed">
        {children}
      </p>
    </div>
  );
}
