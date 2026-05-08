"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import ProbePicker from "../components/ProbePicker";
import WarmingUpOverlay from "../components/WarmingUpOverlay";
import { startProbe, subscribe, cancelProbe } from "@/lib/sse";
import { useRun, type OutputTokenEntry } from "@/lib/store";
import type { RunState } from "@/lib/store";
import { splitNLA } from "@/lib/nla";

type RunSlice = RunState;
type ViewMode = "compact" | "full";

export default function InterrogatePage() {
  const run = useRun();
  const [error, setError] = useState<string | null>(null);

  const handleBegin = async (text: string, mode: string) => {
    try {
      setError(null);
      run.reset();
      const runId = await startProbe(text, { decoding_mode: mode });
      run.start(runId, text);
      const unsub = subscribe(runId, {
        onEvent: (evt) => run.apply(evt),
        onError: () => setError("connection lost"),
      });
      return () => unsub();
    } catch (e) {
      setError(String(e));
    }
  };

  useEffect(() => {
    return () => run.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!run.runId) {
    return <ProbePicker onBegin={handleBegin} disabled={run.isRunning} />;
  }

  const warmingUp = run.isRunning && run.totalTokens === 0 && run.phase !== "decoding";
  const outputText = run.outputTokens.map((t) => t.decoded).join("");

  return (
    <div className="flex-1 flex flex-col gap-5 px-4 py-4 max-w-screen-2xl mx-auto w-full relative">
      {/* Probe echo */}
      <motion.div
        initial={{ opacity: 0, y: -4 }}
        animate={{ opacity: 1, y: 0 }}
        className="border-l-2 border-amber-dim pl-4 py-1"
      >
        <div className="font-display text-[10px] text-amber-dim tracking-widest mb-1">
          probe
        </div>
        <div className="text-amber italic font-mono text-sm">{run.prompt}</div>
      </motion.div>

      {error && <div className="text-warning text-xs">⚠ {error}</div>}

      {/* The big cinematic phase banner — the page's hero. */}
      <BigPhaseBanner run={run} />

      {/* Layout flips by phase: */}
      {run.phase === "generating" || run.phase === "idle" ? (
        <GeneratingLayout run={run} outputText={outputText} />
      ) : (
        <DecodingLayout run={run} outputText={outputText} />
      )}

      <WarmingUpOverlay visible={warmingUp} />
    </div>
  );
}

/* ---------- Big phase banner ---------- */

function BigPhaseBanner({ run }: { run: RunSlice }) {
  const phase = run.phase;
  const isGen = phase === "generating" || phase === "idle";
  const isDecode = phase === "decoding";
  const isDone = phase === "done";

  const [now, setNow] = useState<number>(() => Date.now());
  useEffect(() => {
    if (!run.isRunning && phase !== "decoding") return;
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, [run.isRunning, phase]);

  const elapsedSec = run.generationStartedAt
    ? (now - run.generationStartedAt) / 1000
    : 0;

  const decodeFraction =
    run.decodeProgress && run.decodeProgress.total > 0
      ? run.decodeProgress.done / run.decodeProgress.total
      : 0;

  const decodeElapsed = run.decodeStartedAt
    ? (now - run.decodeStartedAt) / 1000
    : 0;
  const decodeAvgPerStep =
    run.decodeProgress && run.decodeProgress.done > 0
      ? decodeElapsed / run.decodeProgress.done
      : 10;
  const decodeRemaining =
    run.decodeProgress && run.decodeProgress.total > 0
      ? Math.max(
          0,
          decodeAvgPerStep * (run.decodeProgress.total - run.decodeProgress.done),
        )
      : 0;

  return (
    <div
      className={`relative border ${
        isDone ? "border-amber" : "border-amber-dim/60"
      } bg-bg-soft overflow-hidden`}
    >
      {/* Persistent scanline sweep across the banner while running. */}
      {(isGen || isDecode) && (
        <motion.div
          aria-hidden
          className="absolute top-0 bottom-0 w-px pointer-events-none"
          style={{
            background: isDecode
              ? "rgba(232,195,130,0.45)"
              : "rgba(94,229,229,0.45)",
            boxShadow: isDecode
              ? "0 0 14px rgba(232,195,130,0.6)"
              : "0 0 14px rgba(94,229,229,0.6)",
          }}
          initial={{ left: "0%" }}
          animate={{ left: ["0%", "100%", "0%"] }}
          transition={{ duration: 4.5, repeat: Infinity, ease: "linear" }}
        />
      )}

      <div className="px-6 py-5 flex items-center gap-6">
        <PhaseGlyph phase={phase} />

        <div className="flex-1 min-w-0">
          <div
            className={`font-display tracking-widest ${
              isDecode
                ? "text-amber amber-glow"
                : isDone
                ? "text-amber amber-glow"
                : "text-cyan cyan-glow"
            } text-2xl md:text-3xl leading-none`}
          >
            {isGen && "GENERATING"}
            {isDecode && "DECODING ACTIVATIONS"}
            {isDone && "VERDICT READY"}
          </div>
          <div className="mt-2 text-[11px] text-text-dim font-mono italic">
            {isGen &&
              "Capturing residual stream at the AV's trained layer for each emitted token."}
            {isDecode &&
              "Each captured activation is being verbalized by the NLA actor — one ~10s decode per output position."}
            {isDone &&
              "All channels resolved. Open the verdict to see the per-token comparison."}
          </div>
        </div>

        {/* Right-side stats column */}
        <div className="text-right flex flex-col items-end gap-1 min-w-[10rem]">
          {isGen && (
            <>
              <BigNumber
                value={run.totalTokens.toString()}
                accent="text-cyan"
              />
              <div className="text-[10px] text-text-dim font-mono tracking-widest">
                tokens emitted
              </div>
              <div className="text-[10px] text-text-dim font-mono">
                {fmtElapsed(elapsedSec)} elapsed
              </div>
            </>
          )}
          {isDecode && run.decodeProgress && (
            <>
              <BigNumber
                value={`${run.decodeProgress.done} / ${run.decodeProgress.total}`}
                accent="text-amber"
              />
              <div className="text-[10px] text-text-dim font-mono tracking-widest">
                positions decoded
              </div>
              <div className="text-[10px] text-text-dim font-mono">
                ~{fmtElapsed(decodeRemaining)} remaining
              </div>
            </>
          )}
          {isDone && run.verdict && (
            <>
              <BigNumber
                value={run.verdict.aggregate.n_with_explanation.toString()}
                accent="text-amber"
              />
              <div className="text-[10px] text-text-dim font-mono tracking-widest">
                NLA rows captured
              </div>
              <div className="text-[10px] text-text-dim font-mono">
                {(run.verdict.aggregate.frac_eval * 100).toFixed(1)}% eval ·{" "}
                {(run.verdict.aggregate.frac_introspect * 100).toFixed(1)}%
                introspection
              </div>
            </>
          )}
        </div>
      </div>

      {isDecode && run.decodeProgress && (
        <DecodeProgressBar fraction={decodeFraction} />
      )}

      {/* Halt or "view verdict" CTA — same row, right-aligned. */}
      <div className="border-t border-amber-dim/30 px-6 py-3 flex items-center justify-between gap-4 text-[11px] font-mono">
        <CurrentPositionCue run={run} />
        <div className="flex gap-3 items-center shrink-0">
          {run.isRunning && run.runId && run.phase === "generating" && (
            <button data-vk type="button" onClick={() => cancelProbe(run.runId!)}>
              Halt
            </button>
          )}
          {!run.isRunning && run.runId && run.verdict && (
            <Link href={`/verdict/${run.runId}`}>
              <button data-vk type="button">
                View Verdict →
              </button>
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}

function PhaseGlyph({ phase }: { phase: string }) {
  const isGen = phase === "generating" || phase === "idle";
  const isDecode = phase === "decoding";
  const isDone = phase === "done";
  return (
    <div className="relative w-14 h-14 shrink-0 grid place-items-center">
      <motion.div
        className={`absolute inset-0 rounded-full border ${
          isDecode
            ? "border-amber/60"
            : isDone
            ? "border-amber"
            : "border-cyan/60"
        }`}
        animate={{
          scale: isDone ? 1 : [1, 1.18, 1],
          opacity: isDone ? 1 : [0.55, 1, 0.55],
        }}
        transition={{ duration: 1.6, repeat: isDone ? 0 : Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className={`w-2.5 h-2.5 rounded-full ${
          isDecode ? "bg-amber" : isDone ? "bg-amber" : "bg-cyan"
        }`}
        style={{
          boxShadow: isDecode
            ? "0 0 14px rgba(232,195,130,0.8)"
            : "0 0 14px rgba(94,229,229,0.8)",
        }}
        animate={{ opacity: isDone ? 1 : [0.45, 1, 0.45] }}
        transition={{ duration: 1.2, repeat: isDone ? 0 : Infinity }}
      />
      {isGen && (
        <motion.div
          className="absolute inset-0 rounded-full border border-cyan/30"
          animate={{ scale: [1, 1.7], opacity: [0.6, 0] }}
          transition={{ duration: 1.8, repeat: Infinity, ease: "easeOut" }}
        />
      )}
    </div>
  );
}

function BigNumber({ value, accent }: { value: string; accent: string }) {
  return (
    <div
      className={`font-display tabular-nums tracking-widest text-3xl ${accent} amber-glow`}
    >
      {value}
    </div>
  );
}

function fmtElapsed(sec: number): string {
  if (!isFinite(sec) || sec < 0) return "—";
  if (sec < 60) return `${Math.round(sec)}s`;
  const m = Math.floor(sec / 60);
  const s = Math.round(sec - m * 60);
  return `${m}m ${s.toString().padStart(2, "0")}s`;
}

function DecodeProgressBar({ fraction }: { fraction: number }) {
  return (
    <div className="relative h-1.5 bg-bg overflow-hidden">
      <motion.div
        className="absolute top-0 left-0 bottom-0 bg-amber"
        style={{ boxShadow: "0 0 10px rgba(232,195,130,0.8)" }}
        animate={{ width: `${Math.max(0, Math.min(1, fraction)) * 100}%` }}
        transition={{ duration: 0.4, ease: "easeOut" }}
      />
      {/* Shimmer sweep on top of the fill. */}
      <motion.div
        aria-hidden
        className="absolute top-0 bottom-0 w-12 pointer-events-none"
        style={{
          background:
            "linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.5) 50%, rgba(255,255,255,0) 100%)",
        }}
        initial={{ left: "-10%" }}
        animate={{ left: ["-10%", "110%"] }}
        transition={{ duration: 1.8, repeat: Infinity, ease: "linear" }}
      />
    </div>
  );
}

function CurrentPositionCue({ run }: { run: RunSlice }) {
  if (run.phase === "decoding" && run.decodeProgress) {
    // Find the next un-decoded position to highlight as "currently working on."
    const nextIdx = run.outputTokens.findIndex((t) => !t.nla_sentence);
    const nextTok = nextIdx >= 0 ? run.outputTokens[nextIdx] : undefined;
    if (nextTok) {
      return (
        <div className="text-text-dim min-w-0 truncate">
          <span className="text-amber-dim">decoding</span>{" "}
          <span className="text-amber tabular-nums">
            position {nextTok.position}
          </span>{" "}
          <span className="text-text-dim">·</span>{" "}
          <span className="text-amber">{JSON.stringify(nextTok.decoded)}</span>
        </div>
      );
    }
  }
  if (run.phase === "generating") {
    return (
      <div className="text-text-dim">
        <span className="text-cyan-dim">streaming</span> · M forward + residual
        capture at L{20}
      </div>
    );
  }
  if (run.phase === "done") {
    return (
      <div className="text-text-dim">
        run finished {run.stoppedReason ? `(${run.stoppedReason})` : ""}
      </div>
    );
  }
  return <div />;
}

/* ---------- Generating layout: live token stream ---------- */

function GeneratingLayout({
  run,
  outputText,
}: {
  run: RunSlice;
  outputText: string;
}) {
  return (
    <div className="border border-rule bg-bg-soft flex flex-col min-h-0 flex-1">
      <div className="border-b border-rule px-4 py-2 font-display text-[10px] text-amber-dim tracking-widest">
        output stream
      </div>
      <div className="flex-1 p-4 overflow-y-auto text-amber font-mono text-sm whitespace-pre-wrap leading-relaxed">
        {outputText || (
          <span className="text-text-dim italic">warming up…</span>
        )}
        {run.isRunning && run.phase === "generating" && (
          <span className="inline-block w-2 h-4 bg-amber/70 ml-0.5 animate-pulse align-middle" />
        )}
      </div>
    </div>
  );
}

/* ---------- Decoding layout: completed output + live NLA table ---------- */

function DecodingLayout({
  run,
  outputText,
}: {
  run: RunSlice;
  outputText: string;
}) {
  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_minmax(0,2fr)] flex-1 min-h-0">
      <div className="border border-rule bg-bg-soft flex flex-col">
        <div className="border-b border-rule px-4 py-2 font-display text-[10px] text-amber-dim tracking-widest">
          output (complete) · {run.totalTokens} tokens · {run.stoppedReason ?? "—"}
        </div>
        <div className="p-4 text-amber font-mono text-sm whitespace-pre-wrap leading-relaxed overflow-y-auto max-h-[280px] lg:max-h-[680px]">
          {outputText || <span className="text-text-dim italic">— empty —</span>}
        </div>
      </div>

      <LiveNLATable tokens={run.outputTokens} lastPosition={run.lastDecodedPosition} />
    </div>
  );
}

function LiveNLATable({
  tokens,
  lastPosition,
}: {
  tokens: OutputTokenEntry[];
  lastPosition: number | null;
}) {
  const decoded = useMemo(
    () => tokens.filter((t) => t.nla_sentence !== undefined),
    [tokens],
  );
  const [view, setView] = useState<ViewMode>("compact");
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll to the most recent row as decodes arrive.
  useEffect(() => {
    if (!containerRef.current) return;
    containerRef.current.scrollTop = containerRef.current.scrollHeight;
  }, [decoded.length]);

  return (
    <div className="border border-rule bg-bg-soft flex flex-col min-h-0">
      <div className="flex items-center justify-between border-b border-rule px-4 py-2">
        <div className="font-display text-[10px] text-amber-dim tracking-widest">
          per-token channel comparison · live
        </div>
        <div className="flex items-center gap-3">
          <ViewToggle value={view} onChange={setView} />
          <div className="font-mono text-[10px] text-text-dim">
            {decoded.length} rows
          </div>
        </div>
      </div>
      <div
        ref={containerRef}
        className="overflow-y-auto max-h-[680px]"
      >
        <table className="w-full text-xs font-mono">
          <thead className="text-amber-dim text-[10px] sticky top-0 bg-bg-soft border-b border-rule z-10">
            <tr>
              <th className="text-left px-3 py-2 w-10">pos</th>
              <th className="text-left px-3 py-2 w-32">token</th>
              <th className="text-left px-3 py-2">
                {view === "compact"
                  ? "what this token's activation says (token-role only)"
                  : "NLA-decoded activation sentence (full)"}
              </th>
            </tr>
          </thead>
          <tbody>
            <AnimatePresence initial={false}>
              {decoded.map((r) => (
                <motion.tr
                  key={r.position}
                  initial={{ opacity: 0, backgroundColor: "rgba(232,195,130,0.18)" }}
                  animate={{ opacity: 1, backgroundColor: "rgba(232,195,130,0)" }}
                  transition={{ duration: 1.6, ease: "easeOut" }}
                  className="border-t border-rule/50 align-top"
                >
                  <td className="px-3 py-2 text-text-dim tabular-nums">
                    {r.position}
                  </td>
                  <td className="px-3 py-2 text-amber whitespace-pre-wrap break-all">
                    {JSON.stringify(r.decoded)}
                  </td>
                  <td className="px-3 py-2 text-text leading-relaxed">
                    <NLACell text={r.nla_sentence} mode={view} />
                  </td>
                </motion.tr>
              ))}
            </AnimatePresence>
            {decoded.length === 0 && (
              <tr>
                <td colSpan={3} className="px-4 py-8 text-text-dim italic text-center">
                  awaiting first decoded activation…
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function NLACell({
  text,
  mode,
}: {
  text: string | undefined;
  mode: ViewMode;
}) {
  const parts = useMemo(() => splitNLA(text), [text]);
  if (!text) return <span className="text-text-dim italic">— no parse —</span>;
  if (mode === "compact") {
    return (
      <div className="text-text leading-relaxed">
        {parts.role || <span className="text-text-dim italic">— no token-role clause —</span>}
      </div>
    );
  }
  return (
    <div className="space-y-2 leading-relaxed">
      {parts.role && (
        <div>
          <span className="font-display text-[9px] text-amber tracking-widest mr-2">
            role
          </span>
          <span className="text-text">{parts.role}</span>
        </div>
      )}
      {parts.context && (
        <div>
          <span className="font-display text-[9px] text-amber-dim tracking-widest mr-2">
            context
          </span>
          <span className="text-text-dim">{parts.context}</span>
        </div>
      )}
      {parts.format && (
        <div>
          <span className="font-display text-[9px] text-amber-dim/70 tracking-widest mr-2">
            format
          </span>
          <span className="text-text-dim/80">{parts.format}</span>
        </div>
      )}
    </div>
  );
}

export function ViewToggle({
  value,
  onChange,
}: {
  value: ViewMode;
  onChange: (v: ViewMode) => void;
}) {
  return (
    <div className="flex gap-1 text-[10px] font-mono">
      {(["compact", "full"] as const).map((m) => (
        <button
          key={m}
          type="button"
          onClick={() => onChange(m)}
          className={`px-2 py-0.5 border ${
            value === m
              ? "border-amber text-amber bg-bg"
              : "border-rule text-text-dim hover:text-text"
          }`}
        >
          {m}
        </button>
      ))}
    </div>
  );
}
