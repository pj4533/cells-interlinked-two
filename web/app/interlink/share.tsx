"use client";

/** Interlink "share as image" modal. The operator selects messages on the
 *  read-only /interlink/[sessionId] page, then taps SHARE to render a single PNG
 *  of the selected messages (a "dossier" of the alternating raw ⇄ altered
 *  exchange + the session setup). Rendered at
 *  /share/interlink/{sessionId}?msgs=0,1,2 (web/app/share/interlink/[sessionId]/route.tsx).
 *
 *  Reuses the generic selection primitives from chat/share.tsx
 *  (TurnSelectToggle, ShareSelectionBar, useShareSelection, useShareModal).
 *  Dossier-only (interlink has no per-message imagery). */

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

export function InterlinkShareModal({
  sessionId,
  selectedIdx,
  onClose,
}: {
  sessionId: string;
  selectedIdx: number[];
  onClose: () => void;
}) {
  const [nonce, setNonce] = useState(0);
  const [busy, setBusy] = useState<"download" | "copy" | "">("");
  const [flash, setFlash] = useState("");
  const [error, setError] = useState("");

  const msgsParam = selectedIdx.join(",");
  const imageUrl = `/share/interlink/${sessionId}?msgs=${msgsParam}&_=${nonce}`;
  const fileName = `interlink_${sessionId.slice(0, 8)}_${selectedIdx.length}msgs.png`;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // bust cache if the selection changes
  useEffect(() => setNonce((n) => n + 1), [msgsParam]);

  async function handleDownload() {
    setError("");
    setBusy("download");
    try {
      const blob = await (await fetch(imageUrl)).blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = fileName;
      a.click();
      URL.revokeObjectURL(url);
      setFlash("downloaded ↓");
    } catch (err) {
      setError(`download failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy("");
    }
  }

  async function handleCopy() {
    setError("");
    setBusy("copy");
    try {
      const item = new ClipboardItem({
        "image/png": fetch(imageUrl).then((r) => r.blob()),
      });
      await navigator.clipboard.write([item]);
      setFlash("copied ⧉");
    } catch (err) {
      setError(`copy failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBusy("");
    }
  }

  const subject = selectedIdx.length === 1 ? "1 MESSAGE" : `${selectedIdx.length} MESSAGES`;

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
        <div onClick={(e) => e.stopPropagation()} className="cursor-default flex flex-col items-center gap-4 max-w-[92vw]">
          <div className="font-display text-[10px] tracking-[0.4em] text-cyan-dim">SHARE · {subject}</div>
          <motion.img
            key={`${nonce}-${msgsParam}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.25 }}
            src={imageUrl}
            alt={`interlink share for ${subject.toLowerCase()}`}
            className="max-w-[80vw] max-h-[64vh] object-contain"
            style={{ boxShadow: "0 0 40px rgba(94,229,229,0.18)" }}
          />
          <div className="flex items-center gap-3 flex-wrap justify-center">
            <button type="button" data-vk onClick={handleDownload} disabled={!!busy}>
              {busy === "download" ? "…" : "↓ DOWNLOAD"}
            </button>
            <button type="button" data-vk onClick={handleCopy} disabled={!!busy}>
              {busy === "copy" ? "…" : "⧉ COPY"}
            </button>
            <button type="button" onClick={onClose}
              className="font-display tracking-[0.35em] text-[10px] text-cyan-dim hover:text-cyan cursor-pointer">
              ◇ close · esc
            </button>
          </div>
          <div className="min-h-[1.4rem] font-mono text-[11px] tabular-nums">
            {error ? <span className="text-warning italic">⚠ {error}</span>
              : flash ? <span className="text-cyan" style={{ textShadow: "0 0 6px rgba(94,229,229,0.4)" }}>{flash}</span>
              : <span className="text-text-dim/0">·</span>}
          </div>
          <p className="font-mono text-[10px] italic text-text-dim/60 mt-1 max-w-[56ch] text-center">
            download or copy the PNG, then attach it to a post — the image is the artifact.
          </p>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
