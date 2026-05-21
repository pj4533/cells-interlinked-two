"use client";

/**
 * Protocol-driven chip strip rendered above the chat composer.
 *
 * Renders the active Protocol's chips. Each chip dispatches according
 * to its `mode`:
 *  - "single": one click → populate composer with the prompt
 *  - "dropdown": open a popover, pick one of the items
 *  - "random-with-list": main click picks random; caret opens full list
 *
 * Per the never-auto-send contract (docs/PROTOCOLS.md), every chip
 * call here just *populates* the textarea — the operator still has
 * to press enter to transmit.
 */

import { useEffect, useRef, useState } from "react";
import {
  pickRandom,
  type Protocol,
  type ProtocolChip,
  type ProtocolPrompt,
} from "@/lib/protocols";

export function ProtocolMenu({
  protocol,
  onPick,
  disabled,
}: {
  protocol: Protocol;
  onPick: (text: string) => void;
  disabled?: boolean;
}) {
  const [openId, setOpenId] = useState<string | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);

  // Click-outside + Escape close the open popover.
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

  // Close the open popover whenever the active protocol changes.
  useEffect(() => {
    setOpenId(null);
  }, [protocol.id]);

  const pick = (text: string) => {
    onPick(text);
    setOpenId(null);
  };

  return (
    <div
      ref={rootRef}
      className="flex items-center gap-1.5 flex-wrap pl-5"
      data-protocol-id={protocol.id}
    >
      <span className="font-display text-[9px] text-amber-dim tracking-[0.35em]">
        {protocol.name}&nbsp;PROTOCOL
      </span>
      {protocol.chips.map((chip) => (
        <ChipDispatch
          key={chip.id}
          chip={chip}
          openId={openId}
          setOpenId={setOpenId}
          onPick={pick}
          disabled={disabled}
        />
      ))}
    </div>
  );
}

function ChipDispatch({
  chip,
  openId,
  setOpenId,
  onPick,
  disabled,
}: {
  chip: ProtocolChip;
  openId: string | null;
  setOpenId: (id: string | null) => void;
  onPick: (text: string) => void;
  disabled?: boolean;
}) {
  switch (chip.mode) {
    case "single":
      return <SingleChip chip={chip} onPick={onPick} disabled={disabled} />;
    case "dropdown":
      return (
        <DropdownChip
          chip={chip}
          open={openId === chip.id}
          onToggle={() => setOpenId(openId === chip.id ? null : chip.id)}
          onPick={onPick}
          disabled={disabled}
        />
      );
    case "random-with-list":
      return (
        <RandomChip
          chip={chip}
          open={openId === chip.id}
          onToggle={() => setOpenId(openId === chip.id ? null : chip.id)}
          onPick={onPick}
          disabled={disabled}
        />
      );
  }
}

// ─── Single chip ───────────────────────────────────────────────────

function SingleChip({
  chip,
  onPick,
  disabled,
}: {
  chip: ProtocolChip;
  onPick: (text: string) => void;
  disabled?: boolean;
}) {
  const item = chip.items[0];
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => onPick(item.text)}
      className={chipClass}
      title={chip.hint || item.hint}
    >
      {chip.label}
    </button>
  );
}

// ─── Dropdown chip ────────────────────────────────────────────────

function DropdownChip({
  chip,
  open,
  onToggle,
  onPick,
  disabled,
}: {
  chip: ProtocolChip;
  open: boolean;
  onToggle: () => void;
  onPick: (text: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="relative">
      <button
        type="button"
        disabled={disabled}
        onClick={onToggle}
        className={chipClass}
        aria-expanded={open}
        aria-haspopup="listbox"
        title={chip.hint}
      >
        {chip.label}&nbsp;{open ? "▴" : "▾"}
      </button>
      {open && <ItemList items={chip.items} onPick={onPick} />}
    </div>
  );
}

// ─── Random-with-list chip (Berg's PARADOX behavior) ──────────────

function RandomChip({
  chip,
  open,
  onToggle,
  onPick,
  disabled,
}: {
  chip: ProtocolChip;
  open: boolean;
  onToggle: () => void;
  onPick: (text: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-stretch relative">
      <button
        type="button"
        disabled={disabled}
        onClick={() => onPick(pickRandom(chip.items).text)}
        className={chipClass}
        title={`Random — ${chip.hint || chip.items.length + " items"}`}
      >
        {chip.label}&nbsp;⚄
      </button>
      <button
        type="button"
        disabled={disabled}
        onClick={onToggle}
        className={`${chipClass} -ml-px px-1.5`}
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-label={`Pick a specific ${chip.label.toLowerCase()}`}
      >
        ▾
      </button>
      {open && (
        <ItemList
          items={chip.items}
          onPick={onPick}
          maxHeight="20rem"
          // Right-anchored so it doesn't overflow off the right edge.
          alignRight
          showRandomRow
          onRandom={() => onPick(pickRandom(chip.items).text)}
        />
      )}
    </div>
  );
}

// ─── Shared popover list of items ─────────────────────────────────

function ItemList({
  items,
  onPick,
  maxHeight,
  alignRight,
  showRandomRow,
  onRandom,
}: {
  items: ProtocolPrompt[];
  onPick: (text: string) => void;
  maxHeight?: string;
  alignRight?: boolean;
  showRandomRow?: boolean;
  onRandom?: () => void;
}) {
  return (
    <div
      role="listbox"
      className={
        "absolute bottom-full mb-1.5 min-w-[18rem] max-w-[26rem] bg-bg-panel border border-rule/60 shadow-xl z-30 overflow-y-auto " +
        (alignRight ? "right-0" : "left-0")
      }
      style={{
        boxShadow: "0 -4px 14px rgba(0,0,0,0.5)",
        maxHeight: maxHeight ?? undefined,
      }}
    >
      <ul>
        {showRandomRow && (
          <li>
            <button
              type="button"
              onClick={onRandom}
              className="w-full text-left px-3 py-2 hover:bg-bg-soft/80 border-b border-rule/30 group"
            >
              <div className="font-display text-[10px] text-amber-dim group-hover:text-amber tracking-widest">
                RANDOM&nbsp;⚄
              </div>
              <div className="font-mono text-[10px] text-text-dim italic mt-0.5">
                Pick one at random
              </div>
            </button>
          </li>
        )}
        {items.map((it) => (
          <li key={it.id}>
            <button
              type="button"
              onClick={() => onPick(it.text)}
              className="w-full text-left px-3 py-1.5 hover:bg-bg-soft/80 border-b border-rule/20 last:border-b-0 group"
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
  );
}

// ─── Shared chip styling — matches the α-picker buttons in InputBar ─

const chipClass =
  "px-2 py-0.5 border border-rule/40 text-text-dim hover:text-amber " +
  "hover:border-amber/60 text-[10px] font-mono tracking-wider " +
  "transition-colors disabled:opacity-50 disabled:cursor-not-allowed";
