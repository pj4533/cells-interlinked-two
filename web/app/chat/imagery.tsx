"use client";

/** Shared chat-mode imagery UI — used by both the live /chat page
 *  (where image phase walks through prompt → generating → done) and
 *  the read-only archive detail page (where every image is already
 *  "done" or "error" by the time it's rendered). Keeping the same
 *  component for both means the lightbox modal, thumbnail styling,
 *  and prompt-detail toggle stay identical regardless of which page
 *  the user is on.
 */

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { imageUrl as resolveImageUrl } from "@/lib/chat";

/** Phase of a single channel's image pipeline. The archive page only
 *  ever passes "done" or "error" since persisted turns are terminal;
 *  the live /chat page passes the full sequence as SSE events
 *  arrive. */
export type ImagePhase =
  | "idle"
  | "prompt"
  | "generating"
  | "done"
  | "error";

/** Per-channel imagery slot: tight square at the right edge of the
 *  response text. Walks through prompt → generating → (thumbnail |
 *  error). Thumbnail is tappable and opens the fullscreen lightbox.
 *  Prompt text is NOT rendered here — only inside the modal — so the
 *  column's main content (the channel reply) stays uncluttered. */
export function ChannelImageBlock({
  accent,
  phase,
  imageUrl,
  prompt,
  imageError,
  onOpen,
}: {
  /** rgba(...,1) full-opacity color for borders + glow. */
  accent: string;
  phase: ImagePhase;
  /** Relative URL into the /chat-images static mount. */
  imageUrl: string;
  prompt: string;
  imageError: string;
  onOpen: (url: string, caption: string) => void;
}) {
  return (
    <div className="shrink-0">
      {phase === "done" && imageUrl ? (
        <button
          type="button"
          onClick={() => onOpen(imageUrl, prompt)}
          className="group block relative"
          title="tap to enlarge"
        >
          <motion.img
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4 }}
            src={resolveImageUrl(imageUrl)}
            alt="channel image"
            className="h-24 w-24 object-cover cursor-zoom-in transition-transform group-hover:scale-[1.03]"
            style={{
              boxShadow: `0 0 12px ${accent.replace("1)", "0.25)")}`,
              border: `1px solid ${accent.replace("1)", "0.4)")}`,
            }}
          />
          <span
            aria-hidden
            className="absolute bottom-1 right-1 font-display text-[8px] tracking-widest text-text/70 px-1 bg-bg/70 opacity-0 group-hover:opacity-100 transition-opacity"
          >
            ↗ TAP
          </span>
        </button>
      ) : phase === "error" ? (
        <div
          className="h-24 w-24 flex items-center justify-center text-center text-[9px] font-mono italic px-1"
          style={{
            color: "rgba(220,140,80,0.85)",
            border: "1px dashed rgba(220,140,80,0.4)",
          }}
          title={imageError}
        >
          image
          <br />
          failed
        </div>
      ) : (
        <div
          className="h-24 w-24 relative overflow-hidden"
          style={{
            border: `1px dashed ${accent.replace("1)", "0.35)")}`,
            background: accent.replace("1)", "0.04)"),
          }}
          title="generating image…"
        >
          <motion.span
            aria-hidden
            initial={{ y: "-100%" }}
            animate={{ y: "200%" }}
            transition={{
              duration: 1.6,
              repeat: Infinity,
              ease: "linear",
            }}
            className="absolute inset-x-0 h-1/3"
            style={{
              background: `linear-gradient(180deg, transparent 0%, ${accent.replace(
                "1)",
                "0.35)",
              )} 50%, transparent 100%)`,
            }}
          />
          <div
            className="absolute inset-0 flex items-center justify-center font-display text-[9px] tracking-widest animate-pulse"
            style={{ color: accent }}
          >
            {phase === "generating" ? "◇ RENDERING" : "◇ COMPOSING"}
          </div>
        </div>
      )}
    </div>
  );
}

/** Fullscreen image preview. Renders the image, the description as
 *  caption, and a collapsible "prompt sent to nano banana" detail
 *  panel below. Closes on backdrop click, the close button, or Esc. */
export function ImageLightbox({
  url,
  caption,
  onClose,
}: {
  url: string;
  caption: string;
  onClose: () => void;
}) {
  const [showPrompt, setShowPrompt] = useState(false);
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
      className="fixed inset-0 z-50 flex flex-col items-center justify-center p-6 cursor-zoom-out overflow-y-auto"
      style={{ background: "rgba(0,0,0,0.88)" }}
    >
      <motion.img
        initial={{ scale: 0.96, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        src={resolveImageUrl(url)}
        onClick={(e) => e.stopPropagation()}
        alt={caption || "generated image"}
        className="max-w-[92vw] max-h-[80vh] object-contain cursor-default"
        style={{ boxShadow: "0 0 40px rgba(232,195,130,0.18)" }}
      />
      {caption && (
        <p
          className="mt-4 max-w-[80vw] text-center font-mono text-[12px] italic text-text-dim leading-relaxed"
          onClick={(e) => e.stopPropagation()}
        >
          {caption}
        </p>
      )}
      {caption && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            setShowPrompt((v) => !v);
          }}
          className="mt-3 font-display tracking-[0.3em] text-[9px] text-amber-dim hover:text-amber"
        >
          {showPrompt ? "▾" : "▸"} prompt sent to nano banana
        </button>
      )}
      {showPrompt && caption && (
        <div
          onClick={(e) => e.stopPropagation()}
          className="mt-2 max-w-[80vw] cursor-text"
        >
          <pre
            className="font-mono text-[11px] leading-relaxed text-text-dim/90 whitespace-pre-wrap break-words px-4 py-3 border border-rule/40"
            style={{ background: "rgba(0,0,0,0.55)" }}
          >
{caption}
          </pre>
          <p className="mt-2 text-center font-mono text-[9px] italic text-text-dim/60">
            exact bytes sent to gemini-2.5-flash-image
          </p>
        </div>
      )}
      <button
        type="button"
        onClick={onClose}
        className="mt-3 font-display tracking-[0.35em] text-[10px] text-amber-dim hover:text-amber"
      >
        ◇ close · esc
      </button>
    </motion.div>
  );
}

/** Convenience hook: wires the lightbox state + Esc keybind. Returns
 *  an opener function and the lightbox-state pair so the parent can
 *  conditionally render the modal. */
export function useImageLightbox(): {
  open: (url: string, caption: string) => void;
  url: string | null;
  caption: string;
  close: () => void;
} {
  const [url, setUrl] = useState<string | null>(null);
  const [caption, setCaption] = useState<string>("");

  useEffect(() => {
    if (!url) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setUrl(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [url]);

  return {
    open: (u: string, c: string) => {
      setUrl(u);
      setCaption(c);
    },
    url,
    caption,
    close: () => setUrl(null),
  };
}
