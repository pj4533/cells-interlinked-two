"use client";

/** Multi-turn "share as image" UI.
 *
 *  Used from the read-only /chat/[sessionId] page. The operator
 *  selects one or more turns via per-turn select toggles, then taps
 *  the floating SHARE button to open the modal — which renders a
 *  single PNG containing every selected turn in selection order.
 *
 *  Image rendering happens at /share/{sessionId}?turns=0,2,5&format=…
 *  and is served by web/app/share/[sessionId]/route.tsx. */

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

export type ShareFormat = "polaroid" | "dossier";

// ── Per-turn selection toggle ───────────────────────────────────

export function TurnSelectToggle({
  selected,
  onToggle,
}: {
  selected: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      title={selected ? "deselect this turn" : "select this turn for sharing"}
      className={
        "font-display text-[9px] tracking-[0.3em] px-2 py-1 border transition-colors cursor-pointer " +
        (selected
          ? "bg-amber text-bg border-amber"
          : "text-amber-dim border-amber-dim/40 hover:text-amber hover:border-amber")
      }
    >
      {selected ? "● SELECTED" : "◇ SELECT"}
    </button>
  );
}

// ── Floating action bar — only when count > 0 ───────────────────

export function ShareSelectionBar({
  count,
  onShare,
  onClear,
}: {
  count: number;
  onShare: () => void;
  onClear: () => void;
}) {
  return (
    <AnimatePresence>
      {count > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 24 }}
          transition={{ duration: 0.2 }}
          className="fixed bottom-6 left-1/2 -translate-x-1/2 z-40 flex items-center gap-3 px-4 py-2 border border-amber-dim bg-bg-soft/95 backdrop-blur"
          style={{
            boxShadow: "0 0 24px rgba(232,195,130,0.25)",
          }}
        >
          <span className="font-display text-[10px] tracking-[0.35em] text-amber tabular-nums">
            {count} SELECTED
          </span>
          <span className="text-amber-dim/40">·</span>
          <button
            type="button"
            onClick={onShare}
            className="font-display text-[10px] tracking-[0.3em] px-3 py-1 border border-amber text-amber hover:bg-amber hover:text-bg transition-colors cursor-pointer"
            style={{ textShadow: "0 0 4px rgba(232,195,130,0.3)" }}
          >
            ↗ SHARE
          </button>
          <button
            type="button"
            onClick={onClear}
            className="font-display text-[10px] tracking-[0.3em] text-amber-dim hover:text-amber transition-colors cursor-pointer"
          >
            CLEAR
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// ── Selection state hook ────────────────────────────────────────

export interface ShareSelection {
  selectedIdx: number[]; // ordered by selection time
  isSelected: (turnIdx: number) => boolean;
  toggle: (turnIdx: number) => void;
  clear: () => void;
  count: number;
}

export function useShareSelection(): ShareSelection {
  const [selected, setSelected] = useState<number[]>([]);
  return {
    selectedIdx: selected,
    isSelected: (t) => selected.includes(t),
    toggle: (t) =>
      setSelected((prev) =>
        prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t],
      ),
    clear: () => setSelected([]),
    count: selected.length,
  };
}

// ── Modal that renders + delivers the share artifact ────────────

export function useShareModal(): {
  open: () => void;
  close: () => void;
  isOpen: boolean;
} {
  const [isOpen, setIsOpen] = useState(false);
  return {
    open: () => setIsOpen(true),
    close: () => setIsOpen(false),
    isOpen,
  };
}

export function ShareModal({
  sessionId,
  selectedIdx,
  anySelectionHasImagery,
  onClose,
}: {
  sessionId: string;
  selectedIdx: number[];
  /** True if at least one selected turn has imagery on either side.
   *  Determines whether polaroid is a sensible default. */
  anySelectionHasImagery: boolean;
  onClose: () => void;
}) {
  const [format, setFormat] = useState<ShareFormat>(
    anySelectionHasImagery ? "polaroid" : "dossier",
  );
  const [nonce, setNonce] = useState(0);
  const [busy, setBusy] = useState<"download" | "copy" | "">("");
  const [flash, setFlash] = useState<string>("");
  const [error, setError] = useState<string>("");

  const turnsParam = selectedIdx.join(",");
  const imageUrl = `/share/${sessionId}?turns=${turnsParam}&format=${format}&_=${nonce}`;
  const fileName = `cells-interlinked_${sessionId.slice(0, 8)}_${
    selectedIdx.length === 1
      ? `t${String(selectedIdx[0] + 1).padStart(2, "0")}`
      : `${selectedIdx.length}turns`
  }_${format}.png`;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    if (!flash) return;
    const t = setTimeout(() => setFlash(""), 1800);
    return () => clearTimeout(t);
  }, [flash]);

  const pickFormat = (f: ShareFormat) => {
    if (f === format) return;
    setFormat(f);
    setNonce((n) => n + 1);
    setError("");
  };

  const handleDownload = async () => {
    setBusy("download");
    setError("");
    try {
      const res = await fetch(imageUrl);
      if (!res.ok) throw new Error(`render failed: ${res.status}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setFlash("downloaded ↓");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy("");
    }
  };

  const handleCopy = async () => {
    setBusy("copy");
    setError("");
    try {
      // Safari requires the ClipboardItem to be a Promise<Blob>
      // for non-text MIME types so the gesture-attached write can
      // resolve the bytes lazily.
      const item = new ClipboardItem({
        "image/png": fetch(imageUrl).then((r) => r.blob()),
      });
      await navigator.clipboard.write([item]);
      setFlash("copied ⧉");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(`copy failed: ${msg}`);
    } finally {
      setBusy("");
    }
  };

  const subjectLabel =
    selectedIdx.length === 1
      ? `TURN ${String(selectedIdx[0] + 1).padStart(2, "0")}`
      : `${selectedIdx.length} TURNS`;

  return (
    <AnimatePresence>
      <motion.div
        key="backdrop"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        className="fixed inset-0 z-50 flex items-center justify-center p-6 cursor-zoom-out overflow-y-auto"
        style={{ background: "rgba(0,0,0,0.92)" }}
      >
        <div
          onClick={(e) => e.stopPropagation()}
          className="cursor-default flex flex-col items-center gap-4 max-w-[92vw]"
        >
          <div className="font-display text-[10px] tracking-[0.4em] text-amber-dim">
            SHARE · {subjectLabel}
          </div>

          {/* Format toggle */}
          <div className="flex items-center gap-2 font-display text-[10px] tracking-[0.3em]">
            {(
              [
                {
                  key: "polaroid",
                  label: "POLAROID",
                  help: "imagery-forward",
                },
                {
                  key: "dossier",
                  label: "DOSSIER",
                  help: "text-forward",
                },
              ] as const
            ).map((f) => {
              const active = format === f.key;
              const disabled =
                f.key === "polaroid" && !anySelectionHasImagery;
              return (
                <button
                  key={f.key}
                  type="button"
                  onClick={() => pickFormat(f.key)}
                  disabled={disabled}
                  title={
                    disabled
                      ? "no imagery in any selected turn — pick dossier"
                      : f.help
                  }
                  className={
                    "px-3 py-1.5 border transition-colors cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed " +
                    (active
                      ? "border-amber text-bg bg-amber"
                      : "border-amber-dim text-amber-dim hover:text-amber hover:border-amber")
                  }
                >
                  {f.label}
                </button>
              );
            })}
          </div>

          {/* Preview — capped at 64vh so the modal stays usable for
              tall multi-turn artifacts. */}
          <motion.img
            key={`${format}-${nonce}-${turnsParam}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.25 }}
            src={imageUrl}
            alt={`${format} share for ${subjectLabel.toLowerCase()}`}
            className="max-w-[80vw] max-h-[64vh] object-contain"
            style={{ boxShadow: "0 0 40px rgba(232,195,130,0.18)" }}
          />

          {/* Actions */}
          <div className="flex items-center gap-3 flex-wrap justify-center">
            <button
              type="button"
              data-vk
              onClick={handleDownload}
              disabled={!!busy}
            >
              {busy === "download" ? "…" : "↓ DOWNLOAD"}
            </button>
            <button
              type="button"
              data-vk
              onClick={handleCopy}
              disabled={!!busy}
            >
              {busy === "copy" ? "…" : "⧉ COPY"}
            </button>
            <button
              type="button"
              onClick={onClose}
              className="font-display tracking-[0.35em] text-[10px] text-amber-dim hover:text-amber cursor-pointer"
            >
              ◇ close · esc
            </button>
          </div>

          {/* Status / error line — same slot so the dialog doesn't jump */}
          <div className="min-h-[1.4rem] font-mono text-[11px] tabular-nums">
            {error ? (
              <span className="text-warning italic">⚠ {error}</span>
            ) : flash ? (
              <span
                className="text-amber"
                style={{ textShadow: "0 0 6px rgba(232,195,130,0.4)" }}
              >
                {flash}
              </span>
            ) : (
              <span className="text-text-dim/0">·</span>
            )}
          </div>

          <p className="font-mono text-[10px] italic text-text-dim/60 mt-1 max-w-[56ch] text-center">
            download or copy the PNG, then attach it to a post — no link
            is shared, the image is the artifact.
          </p>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
