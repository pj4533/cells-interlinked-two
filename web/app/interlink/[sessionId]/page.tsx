"use client";

// Read-only review of a past Interlink conversation.

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { fetchInterlinkSession, type InterlinkMessage, type InterlinkSide } from "@/lib/interlink";

const RAW_COLOR = "#e8c382";
const BETA_COLOR = "#5ee5e5";

export default function InterlinkReview({ params }: { params: Promise<{ sessionId: string }> }) {
  const { sessionId } = use(params);
  const [data, setData] = useState<Awaited<ReturnType<typeof fetchInterlinkSession>>>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchInterlinkSession(sessionId).then((d) => {
      setData(d);
      setLoading(false);
    });
  }, [sessionId]);

  if (loading) return <main className="min-h-screen bg-bg text-text-dim p-6 font-mono text-sm">loading…</main>;
  if (!data) return <main className="min-h-screen bg-bg text-text-dim p-6 font-mono text-sm">session not found.</main>;

  const cfg = data.config as Record<string, unknown>;
  return (
    <main className="min-h-screen bg-bg text-text px-4 py-5 max-w-3xl mx-auto font-mono">
      <div className="flex items-center gap-3 mb-3 flex-wrap">
        <h1 className="font-display text-cyan tracking-[0.3em] text-sm">INTERLINK</h1>
        <span className="text-[10px] text-text-dim">{data.status} · {data.messages.length} messages</span>
        <span className="text-[10px] text-text-dim">
          {cfg.mode === "steer" ? `dose ${cfg.dose_emotion} @α${cfg.alpha}` : `ablate @α${cfg.alpha}`}
        </span>
        <Link href="/interlink" className="ml-auto text-[10px] text-text-dim hover:text-cyan">← interlink</Link>
      </div>
      <div className="flex flex-col gap-3">
        <Bubble side={data.opener_side} label="OPENER" text={data.opener} thinking="" isOpener />
        {[...data.messages].sort((a, b) => a.idx - b.idx).map((m: InterlinkMessage) => (
          <Bubble key={m.idx} side={m.side} label={m.side === "raw" ? "RAW" : "ALTERED"}
            text={m.text} thinking={m.thinking} stopped={m.stopped_reason} />
        ))}
      </div>
    </main>
  );
}

function Bubble({
  side, label, text, thinking, isOpener, stopped,
}: {
  side: InterlinkSide; label: string; text: string; thinking: string;
  isOpener?: boolean; stopped?: string;
}) {
  const color = side === "raw" ? RAW_COLOR : BETA_COLOR;
  const alignRight = side === "beta";
  return (
    <div className={`flex ${alignRight ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[88%] border px-3 py-2 ${isOpener ? "border-dashed" : ""}`}
        style={{ borderColor: color + "66", background: color + "0d" }}>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[8px] tracking-[0.25em]" style={{ color }}>
            {isOpener ? "OPENER" : label}
          </span>
          {stopped && stopped !== "eos" && <span className="text-[8px] text-text-dim/60">[{stopped}]</span>}
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
