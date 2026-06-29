"use client";

// Read-only review of a past Interlink conversation: exactly how it was set up
// (opener, goal, β intervention, α, ramp, who opened, thinking) + the full
// transcript, with per-message selection → share-as-image.

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { fetchInterlinkSession, type InterlinkMessage, type InterlinkSide } from "@/lib/interlink";
import { TurnSelectToggle, ShareSelectionBar, useShareSelection, useShareModal } from "@/app/chat/share";
import { InterlinkShareModal } from "@/app/interlink/share";

const RAW_COLOR = "#e8c382";
const BETA_COLOR = "#5ee5e5";

export default function InterlinkReview({ params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = use(params);
  const [data, setData] = useState<Awaited<ReturnType<typeof fetchInterlinkSession>>>(null);
  const [loading, setLoading] = useState(true);
  const selection = useShareSelection();
  const shareModal = useShareModal();

  useEffect(() => {
    fetchInterlinkSession(sessionId).then((d) => {
      setData(d);
      setLoading(false);
    });
  }, [sessionId]);

  if (loading) return <main className="min-h-screen bg-bg text-text-dim p-6 font-mono text-sm">loading…</main>;
  if (!data) return <main className="min-h-screen bg-bg text-text-dim p-6 font-mono text-sm">session not found.</main>;

  const cfg = data.config as Record<string, unknown>;
  const mode = String(cfg.mode ?? "steer");
  const alpha = Number(cfg.alpha ?? 0);

  return (
    <main className="min-h-screen bg-bg text-text px-4 py-5 max-w-3xl mx-auto font-mono">
      <div className="flex items-center gap-3 mb-3 flex-wrap">
        <h1 className="font-display text-cyan tracking-[0.3em] text-sm">INTERLINK</h1>
        <span className="text-[10px] text-text-dim tracking-widest uppercase border border-rule px-2 py-0.5">{data.status}</span>
        <span className="text-[10px] text-text-dim">{data.messages.length} messages</span>
        <Link href="/archive" className="ml-auto text-[10px] text-text-dim hover:text-cyan">← archive</Link>
      </div>

      {/* SETUP — exactly how the session was configured */}
      <div className="mb-5 border border-rule/60 bg-bg-soft/30 px-3 py-3 text-[11px]">
        <div className="font-display text-[8px] tracking-[0.3em] text-cyan-dim mb-2">SESSION SETUP</div>
        <Row label="β CHANNEL">
          {mode === "steer"
            ? <>dose <span className="text-cyan">{String(cfg.dose_emotion)}</span> @L20</>
            : <>refusal ablation @L32</>}
          {" · "}<span className="tabular-nums">α={alpha.toFixed(2)}</span>
          {mode === "steer" && <> · ramp {String(cfg.dose_ramp)}</>}
        </Row>
        <Row label="OPENS">{cfg.first_speaker === "raw" ? "raw (baseline) speaks first" : "altered speaks first"}</Row>
        <Row label="THINKING">{cfg.thinking ? "on" : "off"}</Row>
        <Row label="OPENER">
          <span className="text-text/90 whitespace-pre-wrap">{data.opener}</span>
        </Row>
        {data.opener && cfg.goal ? (
          <Row label="SHARED GOAL">
            <span className="text-text-dim/90 italic whitespace-pre-wrap">{String(cfg.goal)}</span>
          </Row>
        ) : null}
      </div>

      {/* TRANSCRIPT */}
      <div className="flex flex-col gap-3">
        <Bubble side={data.opener_side} label="OPENER" text={data.opener} thinking="" isOpener />
        {[...data.messages].sort((a, b) => a.idx - b.idx).map((m: InterlinkMessage) => (
          <Bubble
            key={m.idx}
            side={m.side}
            label={m.side === "raw" ? "RAW" : "ALTERED"}
            text={m.text}
            thinking={m.thinking}
            stopped={m.stopped_reason}
            selected={selection.isSelected(m.idx)}
            onToggleSelect={() => selection.toggle(m.idx)}
          />
        ))}
      </div>

      <ShareSelectionBar count={selection.count} onShare={shareModal.open} onClear={selection.clear} />
      {shareModal.isOpen && selection.count > 0 && (
        <InterlinkShareModal
          sessionId={sessionId}
          selectedIdx={[...selection.selectedIdx].sort((a, b) => a - b)}
          onClose={shareModal.close}
        />
      )}
    </main>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-2 mb-1.5">
      <span className="text-text-dim/45 shrink-0 w-[5.5rem] text-[9px] tracking-wide uppercase pt-0.5">{label}</span>
      <span className="min-w-0 flex-1">{children}</span>
    </div>
  );
}

function Bubble({
  side, label, text, thinking, isOpener, stopped, selected, onToggleSelect,
}: {
  side: InterlinkSide; label: string; text: string; thinking: string;
  isOpener?: boolean; stopped?: string;
  selected?: boolean; onToggleSelect?: () => void;
}) {
  const color = side === "raw" ? RAW_COLOR : BETA_COLOR;
  const alignRight = side === "beta";
  return (
    <div className={`flex ${alignRight ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[88%] border px-3 py-2 ${isOpener ? "border-dashed" : ""}`}
        style={{ borderColor: color + "66", background: color + "0d" }}>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[8px] tracking-[0.25em]" style={{ color }}>{isOpener ? "OPENER" : label}</span>
          {stopped && stopped !== "eos" && <span className="text-[8px] text-text-dim/60">[{stopped}]</span>}
          {!isOpener && onToggleSelect && (
            <span className="ml-auto"><TurnSelectToggle selected={!!selected} onToggle={onToggleSelect} /></span>
          )}
        </div>
        {thinking && (
          <details className="mb-1">
            <summary className="text-[9px] text-text-dim/60 cursor-pointer">thinking</summary>
            <p className="text-[10px] text-text-dim/70 italic whitespace-pre-wrap leading-snug mt-1">{thinking}</p>
          </details>
        )}
        <p className="text-[12px] whitespace-pre-wrap leading-snug text-text">{text}</p>
      </div>
    </div>
  );
}
