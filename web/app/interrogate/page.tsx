"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import ProbePicker from "../components/ProbePicker";
import WarmingUpOverlay from "../components/WarmingUpOverlay";
import { startProbe, subscribe, cancelProbe } from "@/lib/sse";
import { useRun } from "@/lib/store";

export default function InterrogatePage() {
  const run = useRun();
  const [error, setError] = useState<string | null>(null);

  const handleBegin = async (text: string) => {
    try {
      setError(null);
      run.reset();
      const runId = await startProbe(text);
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
    <div className="flex-1 flex flex-col gap-4 px-4 py-4 max-w-screen-2xl mx-auto w-full relative">
      <motion.div
        initial={{ opacity: 0, y: -4 }}
        animate={{ opacity: 1, y: 0 }}
        className="border-l-2 border-amber-dim pl-4 py-1 flex items-center gap-3"
      >
        <div className="flex flex-col">
          <div className="font-display text-[10px] text-amber-dim tracking-widest mb-1">
            probe
          </div>
          <div className="text-amber italic font-mono text-sm">{run.prompt}</div>
        </div>
        {run.isRunning && (
          <motion.div
            className="ml-auto flex items-center gap-2"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <motion.div
              className="w-2 h-2 rounded-full bg-cyan"
              animate={{ opacity: [0.3, 1, 0.3], scale: [1, 1.4, 1] }}
              transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
              style={{ boxShadow: "0 0 8px rgba(94,229,229,0.6)" }}
            />
            <span className="font-display text-[10px] text-cyan-dim tracking-widest">
              {run.phase === "decoding" ? "decoding" : "generating"}
            </span>
          </motion.div>
        )}
      </motion.div>

      {error && <div className="text-warning text-xs">⚠ {error}</div>}

      <div className="grid gap-4 flex-1 min-h-0" style={{ gridTemplateColumns: "1fr 16rem" }}>
        <div className="border border-rule bg-bg-soft flex flex-col min-h-0">
          <div className="border-b border-rule px-4 py-2 font-display text-[10px] text-amber-dim tracking-widest">
            output stream
          </div>
          <div className="flex-1 p-4 overflow-y-auto text-amber font-mono text-sm whitespace-pre-wrap leading-relaxed">
            {outputText || (
              <span className="text-text-dim italic">
                {run.phase === "decoding"
                  ? "generation complete; decoding activations…"
                  : "warming up…"}
              </span>
            )}
            {run.isRunning && run.phase === "generating" && (
              <span className="inline-block w-2 h-4 bg-amber/70 ml-0.5 animate-pulse align-middle" />
            )}
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-2 p-4 border border-rule bg-bg-soft">
            <div className="font-display text-[10px] text-amber-dim tracking-widest">
              status
            </div>
            <div className="text-xs">
              output tokens: <span className="text-amber">{run.totalTokens}</span>
            </div>
            <div className="text-xs">
              phase: <span className="text-amber">{run.phase}</span>
            </div>
            {run.decodeProgress && (
              <div className="text-xs">
                NLA decoding:{" "}
                <span className="text-amber">
                  {run.decodeProgress.done}/{run.decodeProgress.total}
                </span>
              </div>
            )}
            <div className="text-xs">
              {run.isRunning ? (
                <span className="text-cyan animate-pulse">running…</span>
              ) : (
                <span className="text-text-dim">{run.stoppedReason ?? "idle"}</span>
              )}
            </div>
          </div>

          {run.isRunning && run.runId && run.phase === "generating" && (
            <button
              data-vk
              type="button"
              onClick={() => cancelProbe(run.runId!)}
            >
              Halt
            </button>
          )}

          {!run.isRunning && run.runId && run.verdict && (
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="flex flex-col gap-2"
            >
              <div className="font-display text-[10px] text-amber tracking-widest amber-glow text-center">
                run complete
              </div>
              <Link href={`/verdict/${run.runId}`}>
                <button data-vk type="button" className="w-full">
                  View Verdict →
                </button>
              </Link>
            </motion.div>
          )}
        </div>
      </div>

      <WarmingUpOverlay visible={warmingUp} />
    </div>
  );
}
