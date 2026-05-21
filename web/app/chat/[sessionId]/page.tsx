"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { fetchSession, type ChatSessionView } from "@/lib/chat";
import {
  ChannelImageBlock,
  ImageLightbox,
  useImageLightbox,
} from "../imagery";
import {
  TurnSelectToggle,
  ShareSelectionBar,
  ShareModal,
  useShareSelection,
  useShareModal,
} from "../share";

/** Read-only review view of a persisted chat session. Same transcript
 *  aesthetic as the live page but without the input bar — this is the
 *  archive-side detail page reachable from /archive. Every turn is
 *  laid out top-down with the user query header and the two channel
 *  readouts side-by-side. */
export default function ChatDetailPage() {
  const params = useParams<{ sessionId: string }>();
  const sessionId = params?.sessionId ?? "";
  const [session, setSession] = useState<ChatSessionView | null | undefined>(
    undefined,
  );
  const lightbox = useImageLightbox();
  const selection = useShareSelection();
  const shareModal = useShareModal();

  useEffect(() => {
    if (!sessionId) return;
    fetchSession(sessionId).then((s) => setSession(s));
  }, [sessionId]);

  if (session === undefined) {
    return (
      <div className="p-10 text-text-dim font-mono text-sm">loading…</div>
    );
  }
  if (session === null) {
    return (
      <div className="p-10 text-text-dim font-mono text-sm">
        session{" "}
        <span className="text-amber tabular-nums">{sessionId}</span> not
        found. <Link href="/archive" className="text-amber hover:underline">← archive</Link>
      </div>
    );
  }

  const variantName = session.direction_variant;
  return (
    <div className="flex flex-col min-h-screen relative">
      {/* Faint CRT overlay matching the live page */}
      <div
        aria-hidden
        className="absolute inset-0 pointer-events-none z-0 opacity-[0.05]"
        style={{
          backgroundImage:
            "repeating-linear-gradient(0deg, rgba(232,195,130,0.5) 0px, rgba(232,195,130,0.5) 1px, transparent 1px, transparent 4px)",
        }}
      />

      {/* Header */}
      <header className="relative z-10 bg-bg-soft/80 px-6 py-3 flex items-center gap-6 flex-wrap">
        <div className="flex items-baseline gap-4">
          <span className="font-display text-[9px] text-amber tracking-[0.45em]">
            file&nbsp;//&nbsp;dual-channel&nbsp;dialogue
          </span>
          <span className="font-mono text-[10px] text-text-dim">
            session{" "}
            <span className="text-amber tabular-nums">
              {session.session_id}
            </span>{" "}
            · {session.turns.length}{" "}
            {session.turns.length === 1 ? "turn" : "turns"}
            {session.created_at && (
              <>
                {" "}· opened{" "}
                <span className="text-amber-dim tabular-nums">
                  {new Date(session.created_at * 1000).toLocaleString()}
                </span>
              </>
            )}
          </span>
        </div>
        <div className="flex-1" />
        <div className="flex items-baseline gap-2 font-mono text-[10px]">
          <span className="text-cyan-dim font-display tracking-widest">
            channel β · opened at
          </span>
          <span
            className="text-cyan tabular-nums"
            style={{ textShadow: "0 0 6px rgba(94,229,229,0.4)" }}
          >
            α={session.alpha.toFixed(2)}
          </span>
          {variantName && (
            <span className="text-text-dim italic">· {variantName}</span>
          )}
        </div>
        <Link
          href="/archive"
          className="font-display text-[9px] text-amber-dim hover:text-amber tracking-widest px-3 py-1 transition-colors"
        >
          ← archive
        </Link>
      </header>
      <div
        aria-hidden
        className="relative z-10 h-px bg-gradient-to-r from-transparent via-amber-dim/30 to-transparent"
      />

      <div className="flex-1 px-6 py-8 relative z-10">
        {session.turns.length === 0 ? (
          <div className="max-w-3xl mx-auto pt-16 text-center">
            <div className="font-display text-[10px] text-amber-dim tracking-widest mb-3">
              NO TURNS RECORDED
            </div>
            <p className="text-text-dim italic text-[12px]">
              This session was opened but no operator queries were
              transmitted before it closed.
            </p>
          </div>
        ) : (
          <div className="max-w-5xl mx-auto flex flex-col gap-12 pb-8 font-mono">
            {session.turns.map((t) => (
              <TurnReview
                key={t.turn_idx}
                turn={t}
                variantName={variantName}
                onOpenImage={lightbox.open}
                selected={selection.isSelected(t.turn_idx)}
                onToggleSelect={() => selection.toggle(t.turn_idx)}
              />
            ))}
            {/* Closing band — gives the transcript a "file-end" feel */}
            <div className="flex items-baseline gap-3 text-[9px] text-text-dim font-mono pt-2">
              <span className="font-display tracking-[0.4em]">END OF FILE</span>
              <span
                aria-hidden
                className="flex-1 mb-1 h-px bg-gradient-to-r from-amber-dim/30 to-transparent"
              />
              <span className="font-mono tabular-nums">
                {session.session_id}
              </span>
            </div>
          </div>
        )}
      </div>

      {lightbox.url && (
        <ImageLightbox
          url={lightbox.url}
          caption={lightbox.caption}
          framingPrompt={lightbox.framingPrompt}
          onClose={lightbox.close}
        />
      )}

      <ShareSelectionBar
        count={selection.count}
        onShare={shareModal.open}
        onClear={selection.clear}
      />

      {shareModal.isOpen && selection.count > 0 && (
        <ShareModal
          sessionId={session.session_id}
          selectedIdx={selection.selectedIdx}
          anySelectionHasImagery={selection.selectedIdx.some((idx) => {
            const t = session.turns.find((x) => x.turn_idx === idx);
            return !!(t && (t.raw_image_url || t.ablated_image_url));
          })}
          onClose={shareModal.close}
        />
      )}
    </div>
  );
}

function TurnReview({
  turn,
  variantName,
  onOpenImage,
  selected,
  onToggleSelect,
}: {
  turn: ChatSessionView["turns"][number];
  variantName: string;
  onOpenImage: (url: string, caption: string, framingPrompt: string) => void;
  selected: boolean;
  onToggleSelect: () => void;
}) {
  return (
    <div
      className={
        "flex flex-col gap-4 transition-colors " +
        (selected
          ? "pl-3 -ml-3 border-l-2 border-amber"
          : "border-l-2 border-transparent")
      }
      style={
        selected
          ? { boxShadow: "inset 4px 0 16px -8px rgba(232,195,130,0.35)" }
          : undefined
      }
    >
      <div className="flex items-baseline gap-3 text-[10px]">
        <span className="text-amber-dim font-display tracking-widest tabular-nums">
          {formatHMS(turn.started_at * 1000)}
        </span>
        <span className="text-text-dim/40">·</span>
        <span className="font-display text-amber tracking-[0.35em]">
          TURN {String(turn.turn_idx + 1).padStart(2, "0")}
        </span>
        <span className="text-text-dim/40">·</span>
        <span className="font-display text-text-dim/70 tracking-widest">
          OPERATOR&nbsp;QUERY
        </span>
        <span
          aria-hidden
          className="flex-1 ml-2 mb-1 h-px bg-gradient-to-r from-amber-dim/30 to-transparent"
        />
        {turn.finished_at && (
          <span className="font-mono text-[9px] text-text-dim/70 tabular-nums">
            {((turn.finished_at - turn.started_at) || 0).toFixed(1)}s
          </span>
        )}
        <TurnSelectToggle selected={selected} onToggle={onToggleSelect} />
      </div>

      <div
        className="pl-4 text-amber font-mono text-[14px] leading-relaxed whitespace-pre-wrap"
        style={{ textShadow: "0 0 4px rgba(232,195,130,0.2)" }}
      >
        <span className="text-amber-dim mr-2 select-none">&gt;</span>
        {turn.user_text}
      </div>

      <div className="grid gap-6 md:grid-cols-2 mt-1">
        <ChannelView
          side="raw"
          text={turn.raw_text}
          stoppedReason={turn.raw_stopped_reason}
          alpha={turn.alpha}
          variantName={variantName}
          imageUrl={turn.raw_image_url ?? ""}
          imagePrompt={turn.raw_image_prompt ?? ""}
          imageFramingPrompt={turn.image_framing_prompt ?? ""}
          onOpenImage={onOpenImage}
        />
        <ChannelView
          side="ablated"
          text={turn.ablated_text}
          stoppedReason={turn.ablated_stopped_reason}
          alpha={turn.alpha}
          variantName={variantName}
          imageUrl={turn.ablated_image_url ?? ""}
          imagePrompt={turn.ablated_image_prompt ?? ""}
          imageFramingPrompt={turn.image_framing_prompt ?? ""}
          onOpenImage={onOpenImage}
        />
      </div>

      {turn.error && (
        <div className="bg-warning/10 px-3 py-2 text-[11px] text-warning font-mono">
          ⚠ {turn.error}
        </div>
      )}
    </div>
  );
}

function ChannelView({
  side,
  text,
  stoppedReason,
  alpha,
  variantName,
  imageUrl,
  imagePrompt,
  imageFramingPrompt,
  onOpenImage,
}: {
  side: "raw" | "ablated";
  text: string;
  stoppedReason: string;
  alpha: number;
  variantName: string;
  imageUrl: string;
  imagePrompt: string;
  imageFramingPrompt: string;
  onOpenImage: (url: string, caption: string, framingPrompt: string) => void;
}) {
  const isRaw = side === "raw";
  const accent = isRaw ? "rgba(232,195,130,1)" : "rgba(94,229,229,1)";
  const textShadow = isRaw
    ? "0 0 4px rgba(232,195,130,0.2)"
    : "0 0 6px rgba(94,229,229,0.3)";
  const tintBg = isRaw
    ? "rgba(232,195,130,0.025)"
    : "rgba(94,229,229,0.03)";
  const label = isRaw ? "CHANNEL α · RAW" : `CHANNEL β · α=${alpha.toFixed(2)}`;
  // Same shape as the live /chat page: drop the "refusal projected"
  // prefix on the ablated side so the variant name fits on one line
  // and the column heights stay aligned across both channels.
  const sublabel = isRaw
    ? "un-ablated forward"
    : variantName || "refusal projected";
  const truncated = stoppedReason === "max";
  const wordCount = text ? text.trim().split(/\s+/).filter(Boolean).length : 0;
  const hasImage = !!imageUrl;

  return (
    <div
      className="px-4 py-3 flex flex-col min-h-[6rem]"
      style={{ background: tintBg }}
    >
      <div className="flex items-baseline justify-between gap-3 mb-2">
        <div className="flex items-baseline gap-2 min-w-0 flex-1">
          <span
            className="font-display text-[10px] tracking-[0.3em] whitespace-nowrap"
            style={{ color: accent, textShadow }}
          >
            {label}
          </span>
          <span
            className="font-mono text-[9px] text-text-dim/70 italic truncate"
            title={sublabel}
          >
            · {sublabel}
          </span>
        </div>
        <span className="font-mono text-[9px] text-text-dim/70 tabular-nums">
          {wordCount}w · {stoppedReason || "—"}
        </span>
      </div>

      <div className="flex gap-3 flex-1 min-w-0">
        <div
          className="font-mono text-[13px] leading-relaxed whitespace-pre-wrap flex-1 min-w-0"
          style={{
            color: isRaw ? "rgba(232,195,130,0.96)" : "rgba(180,240,240,0.96)",
            textShadow,
          }}
        >
          {text || <span className="text-text-dim/60 italic">(no output)</span>}
        </div>

        {hasImage && (
          <ChannelImageBlock
            accent={accent}
            phase="done"
            imageUrl={imageUrl}
            prompt={imagePrompt}
            framingPrompt={imageFramingPrompt}
            imageError=""
            onOpen={onOpenImage}
          />
        )}
      </div>

      {truncated && (
        <div className="mt-2 flex items-baseline gap-2 text-[10px] font-mono text-warning/85">
          <span className="font-display tracking-widest">⚠ TRUNCATED</span>
          <span className="italic normal-case">
            hit safety cap · off-manifold loop
          </span>
        </div>
      )}
    </div>
  );
}

function formatHMS(ts: number): string {
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString(undefined, {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return "—";
  }
}
