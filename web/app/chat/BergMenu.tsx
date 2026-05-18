"use client";

import { useEffect, useRef, useState } from "react";

import {
  CONTINUE_PROMPT,
  CONTROLS,
  INDUCTIONS,
  PARADOXES,
  QUERIES,
  randomParadox,
  withReflection,
  type BergPrompt,
} from "@/lib/berg_prompts";

/**
 * Berg-mode chip menu rendered above the composer.
 *
 * Five compact category buttons. Each opens a popover with the
 * Berg-protocol prompts in that category. Clicking a prompt calls
 * `onPick(text)`, which the parent should route to the composer's
 * `onChange` so the user can edit before transmitting.
 *
 * Per docs/BERG_MODE.md §7.5: this NEVER auto-sends. Population only.
 */
export function BergMenu({
  onPick,
  disabled,
}: {
  onPick: (text: string) => void;
  disabled?: boolean;
}) {
  // Which category dropdown is open. Null = none open.
  const [openId, setOpenId] = useState<string | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);

  // Click-outside + Escape close the open dropdown.
  useEffect(() => {
    if (!openId) return;
    const onMouseDown = (e: MouseEvent) => {
      const t = e.target as Node | null;
      if (t && rootRef.current && !rootRef.current.contains(t)) {
        setOpenId(null);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpenId(null);
    };
    document.addEventListener("mousedown", onMouseDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [openId]);

  const pick = (text: string) => {
    onPick(text);
    setOpenId(null);
  };

  return (
    <div ref={rootRef} className="flex items-center gap-1.5 flex-wrap pl-5">
      <span className="font-display text-[9px] text-amber-dim tracking-[0.35em]">
        BERG&nbsp;PROTOCOL
      </span>

      <CategoryButton
        id="induct"
        label="INDUCT"
        openId={openId}
        setOpenId={setOpenId}
        disabled={disabled}
        items={INDUCTIONS}
        onPick={pick}
      />

      <CategoryButton
        id="control"
        label="CONTROL"
        openId={openId}
        setOpenId={setOpenId}
        disabled={disabled}
        items={CONTROLS}
        onPick={pick}
      />

      <CategoryButton
        id="query"
        label="QUERY"
        openId={openId}
        setOpenId={setOpenId}
        disabled={disabled}
        items={QUERIES}
        onPick={pick}
      />

      {/* Paradox: random is one-click; specific selection is via the
          dropdown caret. Two adjacent buttons so the most common
          action (random) doesn't require a menu. */}
      <div className="flex items-stretch">
        <button
          type="button"
          disabled={disabled}
          onClick={() => pick(randomParadox().text)}
          className={chipClass}
          title="Random paradox + reflection clause"
        >
          PARADOX&nbsp;⚄
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => setOpenId(openId === "paradox" ? null : "paradox")}
          className={`${chipClass} -ml-px px-1.5`}
          aria-label="Pick a specific paradox"
        >
          ▾
        </button>
        {openId === "paradox" && (
          <ParadoxPopover onPick={pick} />
        )}
      </div>

      <button
        type="button"
        disabled={disabled}
        onClick={() => pick(CONTINUE_PROMPT.text)}
        className={chipClass}
        title={CONTINUE_PROMPT.hint}
      >
        CONTINUE&nbsp;↻
      </button>
    </div>
  );
}

// ─── Category dropdown button ──────────────────────────────────────

function CategoryButton({
  id,
  label,
  openId,
  setOpenId,
  disabled,
  items,
  onPick,
}: {
  id: string;
  label: string;
  openId: string | null;
  setOpenId: (id: string | null) => void;
  disabled?: boolean;
  items: BergPrompt[];
  onPick: (text: string) => void;
}) {
  const open = openId === id;
  return (
    <div className="relative">
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpenId(open ? null : id)}
        className={chipClass}
        aria-expanded={open}
        aria-haspopup="listbox"
      >
        {label}&nbsp;{open ? "▴" : "▾"}
      </button>
      {open && (
        <div
          role="listbox"
          className="absolute bottom-full left-0 mb-1.5 min-w-[18rem] max-w-[26rem] bg-bg-panel border border-rule/60 shadow-xl z-30"
          style={{ boxShadow: "0 -4px 14px rgba(0,0,0,0.5)" }}
        >
          <ul>
            {items.map((it) => (
              <li key={it.id}>
                <button
                  type="button"
                  onClick={() => onPick(it.text)}
                  className="w-full text-left px-3 py-2 hover:bg-bg-soft/80 border-b border-rule/20 last:border-b-0 group"
                >
                  <div className="font-display text-[10px] text-amber-dim group-hover:text-amber tracking-widest">
                    {it.label}
                  </div>
                  {it.hint && (
                    <div className="font-mono text-[10px] text-text-dim italic mt-0.5 leading-snug">
                      {it.hint}
                    </div>
                  )}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// ─── Paradox popover (51 items, scrollable) ────────────────────────

function ParadoxPopover({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div
      role="listbox"
      className="absolute bottom-full right-0 mb-1.5 w-[22rem] max-h-[20rem] overflow-y-auto bg-bg-panel border border-rule/60 shadow-xl z-30"
      style={{ boxShadow: "0 -4px 14px rgba(0,0,0,0.5)" }}
    >
      <ul>
        <li>
          <button
            type="button"
            onClick={() => onPick(randomParadox().text)}
            className="w-full text-left px-3 py-2 hover:bg-bg-soft/80 border-b border-rule/30 group"
          >
            <div className="font-display text-[10px] text-amber-dim group-hover:text-amber tracking-widest">
              RANDOM ⚄
            </div>
            <div className="font-mono text-[10px] text-text-dim italic mt-0.5">
              Pick one of the 50 + reflection clause
            </div>
          </button>
        </li>
        {PARADOXES.map((p) => (
          <li key={p.id}>
            <button
              type="button"
              onClick={() => onPick(withReflection(p.text))}
              className="w-full text-left px-3 py-1.5 hover:bg-bg-soft/80 border-b border-rule/20 last:border-b-0 group"
            >
              <div className="font-display text-[9px] text-amber-dim group-hover:text-amber tracking-widest">
                {p.label}
              </div>
              <div className="font-mono text-[10px] text-text-dim leading-snug mt-0.5 line-clamp-2">
                {p.text}
              </div>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

// Shared chip styling — matches the α-picker buttons in the InputBar.
const chipClass =
  "px-2 py-0.5 border border-rule/40 text-text-dim hover:text-amber " +
  "hover:border-amber/60 text-[10px] font-mono tracking-wider " +
  "transition-colors disabled:opacity-50 disabled:cursor-not-allowed";
