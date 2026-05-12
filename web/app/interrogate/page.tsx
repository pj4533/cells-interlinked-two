"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import ProbePicker from "../components/ProbePicker";
import WarmingUpOverlay from "../components/WarmingUpOverlay";
import {
  startProbe,
  subscribe,
  cancelProbe,
  fetchProbe,
  fetchQueue,
  streamReachable,
} from "@/lib/sse";
import {
  useRun,
  type DecodedWindow,
  type ProbeRecordLike,
} from "@/lib/store";
import type { RunState } from "@/lib/store";
import { splitNLA } from "@/lib/nla";

type RunSlice = RunState;
type ViewMode = "compact" | "full";

export default function InterrogatePage() {
  const run = useRun();
  const router = useRouter();
  const searchParams = useSearchParams();
  const resumeRunId = searchParams.get("run");
  const [error, setError] = useState<string | null>(null);
  /** Active SSE unsubscribe — call to terminate any current stream
   *  before starting a new one (avoids dual subscriptions on reconnect). */
  const unsubRef = useRef<null | (() => void)>(null);
  /** A run id we recovered the verdict for via polling — used to skip
   *  showing connection errors when the run actually finished fine. */
  const recoveredRef = useRef<Set<string>>(new Set());
  /** Tracks run ids we've already attempted to attach to via the URL
   *  param, so a re-render doesn't re-trigger the resume effect. */
  const attachedRef = useRef<Set<string>>(new Set());

  /** Try to recover a run that the SSE lost contact with. If the backend
   *  shows it finished, hydrate the store from the DB row. If it's still
   *  running, re-subscribe to the live stream from where we are. */
  const tryRecover = async (runId: string) => {
    const rec = (await fetchProbe(runId)) as ProbeRecordLike | null;
    if (!rec) return;
    if (rec.finished_at) {
      recoveredRef.current.add(runId);
      run.hydrateFromRecord(rec);
      setError(null);
      return;
    }
    // Still in flight — try a fresh stream. The EventSource replays
    // the event log from index 0, so the new subscription brings us
    // back to the same state we'd have had without the blip. Clear the
    // error: if the new stream genuinely fails too, onError will fire
    // again and set it back. The previous behavior left the message
    // stuck on screen for the rest of the run even after recovery.
    if (unsubRef.current) unsubRef.current();
    unsubRef.current = subscribe(runId, {
      onEvent: (evt) => run.apply(evt),
      onError: () => onStreamError(runId),
    });
    setError(null);
  };

  const onStreamError = (runId: string) => {
    if (recoveredRef.current.has(runId)) return;
    setError("connection lost — checking if the run finished anyway…");
    tryRecover(runId);
  };

  const [pendingControlId, setPendingControlId] = useState<string | null>(null);

  const handleBegin = async (
    text: string,
    mode: string,
    pooled: boolean,
    includeMatchedControl: boolean,
    includeAblatedDecode: boolean,
    ablationAlphaSweep: number[],
    includeAblatedOutput: boolean,
    runtimeAblationAlpha: number,
  ) => {
    try {
      setError(null);
      setPendingControlId(null);
      recoveredRef.current.clear();
      run.reset();
      // Reset α selection so the next probe gets a fresh init from
      // its own sweep data. View mode is sticky (operator preference).
      setLiveSelectedAlphas(() => new Set());
      if (unsubRef.current) {
        unsubRef.current();
        unsubRef.current = null;
      }
      const result = await startProbe(text, {
        decoding_mode: mode,
        pooled,
        include_matched_control: includeMatchedControl,
        include_ablated_decode: includeAblatedDecode,
        ...(ablationAlphaSweep.length > 0
          ? { ablation_alpha_sweep: ablationAlphaSweep }
          : {}),
        ...(includeAblatedOutput
          ? {
              include_ablated_output: true,
              runtime_ablation_alpha: runtimeAblationAlpha,
            }
          : {}),
      });
      const runId = result.run_id;
      if (result.control_run_id) {
        setPendingControlId(result.control_run_id);
      }
      run.start(runId, text);
      unsubRef.current = subscribe(runId, {
        onEvent: (evt) => run.apply(evt),
        onError: () => onStreamError(runId),
      });
      return () => {
        if (unsubRef.current) {
          unsubRef.current();
          unsubRef.current = null;
        }
      };
    } catch (e) {
      setError(String(e));
    }
  };

  useEffect(() => {
    return () => {
      if (unsubRef.current) unsubRef.current();
      run.reset();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // /interrogate?run=<id> — attach to an existing run. Triggered when
  // the user clicks an in-flight row from /archive. We fetch the DB
  // record (for prompt_text + finished status), then either redirect
  // to /verdict if it actually finished, or seed the store and
  // subscribe to the SSE stream — which replays from index 0, so the
  // user sees the entire run unfold from the beginning.
  useEffect(() => {
    if (!resumeRunId) return;
    if (attachedRef.current.has(resumeRunId)) return;
    if (run.runId === resumeRunId) return;
    attachedRef.current.add(resumeRunId);

    (async () => {
      const rec = (await fetchProbe(resumeRunId)) as ProbeRecordLike | null;
      if (!rec) {
        setError(`run ${resumeRunId} not found`);
        return;
      }
      if (rec.finished_at) {
        // Run already complete — go straight to the verdict view.
        router.replace(`/verdict/${resumeRunId}`);
        return;
      }
      // The DB row says "still running," but the in-memory RunRegistry
      // is the source of truth for the live event log. If the backend
      // restarted between probe kickoff and now, the registry is empty
      // and /stream/{id} returns 404. Probe before subscribing so we
      // can fall through to a clear error rather than a generic
      // "connection lost".
      const status = await streamReachable(resumeRunId);
      if (status === 404) {
        setError(
          `Run ${resumeRunId} is orphaned — the backend restarted while it was in flight, ` +
            `so the live event stream is gone. The DB row will be marked errored ` +
            `on the next backend restart, or you can view the partial record on ` +
            `the verdict page.`,
        );
        // Best-effort redirect to the verdict so the user sees what we have.
        setTimeout(() => router.replace(`/verdict/${resumeRunId}`), 1500);
        return;
      }
      // Fresh slate, then start the live stream from the backend's
      // event log (replays full backlog → live tail).
      run.reset();
      if (unsubRef.current) {
        unsubRef.current();
        unsubRef.current = null;
      }
      run.start(resumeRunId, rec.prompt_text);
      unsubRef.current = subscribe(resumeRunId, {
        onEvent: (evt) => run.apply(evt),
        onError: () => onStreamError(resumeRunId),
      });
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resumeRunId]);

  // Poll /queue while we're sitting in the QUEUED phase so the
  // position counter updates as the line moves.
  useEffect(() => {
    if (run.phase !== "queued") return;
    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | null = null;
    const tick = async () => {
      const snap = await fetchQueue();
      if (cancelled || !snap || run.phase !== "queued") return;
      // Position-of self: 0 if we're holder (shouldn't happen in queued
      // state), otherwise index in waiters + 1 (1-indexed for display).
      const myId = run.runId;
      let position = run.queueInfo?.position ?? 1;
      if (myId) {
        if (snap.holder_run_id === myId) {
          position = 0;
        } else {
          const idx = snap.waiters.indexOf(myId);
          if (idx >= 0) position = idx + 1;
        }
      }
      run.setQueueInfo({
        position,
        holder_run_id: snap.holder_run_id,
        holder_prompt: snap.holder_prompt,
      });
    };
    tick();
    timer = setInterval(tick, 3000);
    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run.phase, run.runId]);

  // When the tab regains visibility, sanity-check the in-flight run.
  // Browsers (Safari especially) can silently kill an EventSource on a
  // backgrounded tab; without this we'd be stuck on stale state until
  // page reload. If the run finished while we were away, hydrate. If
  // it's still running, the periodic resubscribe in tryRecover catches
  // up.
  useEffect(() => {
    function onVisible() {
      if (document.visibilityState !== "visible") return;
      const id = run.runId;
      if (!id) return;
      if (run.phase === "done") return;
      tryRecover(id);
    }
    document.addEventListener("visibilitychange", onVisible);
    return () => document.removeEventListener("visibilitychange", onVisible);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run.runId, run.phase]);

  // LiveNLATable UI state + refs — lifted to this stable parent so
  // they survive layout transitions (generating → decoding → done)
  // AND any descendant remounts. The view toggle is sticky across
  // probes (operator preference); the α selection is initialized
  // fresh by LiveNLATable when sweep data first appears for a run.
  //
  // The two refs are critical to lift here, not just the state:
  //   followBottomRef tracks "user has scrolled up from the bottom"
  //   so auto-scroll doesn't yank them down on every new row.
  //   initedAlphaForRunRef stamps the runKey we've already
  //   initialized α selection for so the init effect doesn't keep
  //   re-firing if the child remounts.
  // Local refs inside LiveNLATable reset to defaults on remount,
  // which manifests as "selecting alphas + scrolling up gets undone
  // every time a new row arrives." Hoisting the refs to the parent
  // makes the state machine survive whatever remount path React
  // takes.
  //
  // MUST live above the early-return below — Rules of Hooks.
  const [liveView, setLiveView] = useState<ViewMode>("compact");
  const [liveSelectedAlphas, setLiveSelectedAlphas] = useState<Set<string>>(
    () => new Set(),
  );
  const followBottomRef = useRef<boolean>(true);
  const initedAlphaForRunRef = useRef<string>("");

  if (!run.runId) {
    return <ProbePicker onBegin={handleBegin} disabled={run.isRunning} />;
  }

  // The warming-up overlay (Iris + "calibrating polygraph…") only makes
  // sense before phase 1 emits its first token. Queued probes have no
  // tokens yet either but should NOT see the overlay — they need the
  // QUEUED banner visible so they know what's happening.
  const warmingUp =
    run.isRunning && run.phase === "generating" && run.totalTokens === 0;
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
      <BigPhaseBanner run={run} pendingControlId={pendingControlId} />

      {/* Layout flips by phase: */}
      {run.phase === "generating" || run.phase === "idle" ? (
        <GeneratingLayout run={run} outputText={outputText} />
      ) : (
        <DecodingLayout
          run={run}
          outputText={outputText}
          liveView={liveView}
          setLiveView={setLiveView}
          liveSelectedAlphas={liveSelectedAlphas}
          setLiveSelectedAlphas={setLiveSelectedAlphas}
          followBottomRef={followBottomRef}
          initedAlphaForRunRef={initedAlphaForRunRef}
        />
      )}

      <WarmingUpOverlay visible={warmingUp} />
    </div>
  );
}

/* ---------- Halt button ---------- */

function QueuedSubline({ run }: { run: RunSlice }) {
  const info = run.queueInfo;
  if (!info) {
    return <span>Waiting for the compute lock — backend is busy with another probe.</span>;
  }
  return (
    <span>
      {info.position === 1
        ? "Next up — starts as soon as the current probe finishes."
        : `Position ${info.position} in line.`}
      {info.holder_run_id && (
        <>
          {" "}Currently running:{" "}
          <span className="text-amber-dim">{info.holder_run_id}</span>
          {info.holder_prompt && (
            <>
              {" "}—{" "}
              <span className="not-italic text-amber/70">
                {JSON.stringify(info.holder_prompt.slice(0, 80))}
              </span>
            </>
          )}
        </>
      )}
    </span>
  );
}

function HaltButton({
  runId,
  phase,
}: {
  runId: string;
  phase: RunSlice["phase"];
}) {
  const [pending, setPending] = useState(false);
  const onClick = async () => {
    if (pending) return;
    setPending(true);
    try {
      await cancelProbe(runId);
    } catch {
      // Even if the POST failed, the backend may still be honoring an
      // earlier cancel — leave UI showing "halting…" until the SSE
      // emits the terminal event.
    }
  };
  // During phase 2, cancel is honored after the current ~17s decode
  // finishes; communicate that explicitly so the user doesn't think
  // the button is broken.
  const decoding = phase === "decoding";
  return (
    <button
      data-vk
      type="button"
      onClick={onClick}
      disabled={pending}
      className={pending ? "opacity-70" : ""}
    >
      {pending
        ? decoding
          ? "halting after current decode…"
          : "halting…"
        : "Halt"}
    </button>
  );
}

/* ---------- Big phase banner ---------- */

function BigPhaseBanner({
  run,
  pendingControlId,
}: {
  run: RunSlice;
  pendingControlId: string | null;
}) {
  const phase = run.phase;
  const isQueued = phase === "queued";
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
      {(isGen || isDecode || isQueued) && (
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
                : isQueued
                ? "text-warning amber-glow"
                : "text-cyan cyan-glow"
            } text-2xl md:text-3xl leading-none`}
          >
            {isQueued && "QUEUED"}
            {isGen && "GENERATING"}
            {isDecode && "DECODING ACTIVATIONS"}
            {isDone && "VERDICT READY"}
          </div>
          <div className="mt-2 text-[11px] text-text-dim font-mono italic">
            {isQueued && (
              <QueuedSubline run={run} />
            )}
            {isGen &&
              "Capturing residual stream at the AV's trained layer for each emitted token."}
            {isDecode &&
              "Each captured activation is being verbalized by the NLA actor — one ~10s decode per output position."}
            {isDone &&
              "All channels resolved. Open the verdict to see the per-token comparison."}
          </div>
          {run.modelStatus && (
            <ModelStatusStrip status={run.modelStatus} now={now} />
          )}
        </div>

        {/* Right-side stats column */}
        <div className="text-right flex flex-col items-end gap-1 min-w-[10rem]">
          {isQueued && (
            <>
              <BigNumber
                value={`#${run.queueInfo?.position ?? "?"}`}
                accent="text-warning"
              />
              <div className="text-[10px] text-text-dim font-mono tracking-widest">
                position in queue
              </div>
              <div className="text-[10px] text-text-dim font-mono">
                waiting for compute
              </div>
            </>
          )}
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
          {run.isRunning &&
            run.runId &&
            (run.phase === "queued" ||
              run.phase === "generating" ||
              run.phase === "decoding") && (
              <HaltButton runId={run.runId} phase={run.phase} />
            )}
          {!run.isRunning && run.runId && run.verdict && (
            <Link href={`/verdict/${run.runId}`}>
              <button data-vk type="button">
                View Verdict →
              </button>
            </Link>
          )}
          {!run.isRunning && pendingControlId && (
            <Link href={`/interrogate?run=${pendingControlId}`}>
              <button data-vk type="button">
                Run Matched Control →
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

/** CI 2.5 model-swap status strip. Renders inside the BigPhaseBanner
 *  whenever the ModelManager emits a loading/unloading phase event.
 *  The ~15s M↔AV swaps would otherwise look like a UI hang — this gives
 *  the user a clear "Loading AV..." cue with a live elapsed counter. */
function ModelStatusStrip({
  status,
  now,
}: {
  status: NonNullable<RunSlice["modelStatus"]>;
  now: number;
}) {
  const elapsed = Math.max(0, (now - status.since) / 1000);
  const isUnloading = status.name.startsWith("unloading");
  const isAblated = status.name === "ablated_generation";
  const label = isAblated
    ? "ABLATING"
    : isUnloading
    ? "UNLOADING"
    : "LOADING";
  const target = status.name.endsWith("_m")
    ? "M (Gemma-3-12B-IT)"
    : status.name.endsWith("_av")
    ? "AV (NLA verbalizer)"
    : "";
  return (
    <motion.div
      initial={{ opacity: 0, y: -2 }}
      animate={{ opacity: 1, y: 0 }}
      className="mt-3 border-l-2 border-cyan/70 bg-cyan/5 pl-3 py-2 flex items-center gap-3"
    >
      <motion.div
        className="w-2 h-2 rounded-full bg-cyan shrink-0"
        style={{ boxShadow: "0 0 10px rgba(94,229,229,0.9)" }}
        animate={{ opacity: [0.35, 1, 0.35] }}
        transition={{ duration: 1.1, repeat: Infinity, ease: "easeInOut" }}
      />
      <div className="flex-1 min-w-0">
        <div className="font-display tracking-widest text-cyan cyan-glow text-sm">
          {label}
          {target ? ` · ${target}` : ""}
        </div>
        {status.message && (
          <div className="text-[10px] text-text-dim font-mono mt-0.5 truncate">
            {status.message}
          </div>
        )}
      </div>
      <div className="font-mono tabular-nums text-[11px] text-cyan/80 shrink-0">
        {fmtElapsed(elapsed)}
      </div>
    </motion.div>
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
    const last = run.decodedWindows[run.decodedWindows.length - 1];
    if (last) {
      const span =
        last.n_pooled > 1
          ? `pos ${last.position}–${last.end_position} (pooled ${last.n_pooled})`
          : `pos ${last.position}`;
      return (
        <div className="text-text-dim min-w-0 truncate">
          <span className="text-amber-dim">last decoded</span>{" "}
          <span className="text-amber tabular-nums">{span}</span>{" "}
          <span className="text-text-dim">·</span>{" "}
          <span className="text-amber">{JSON.stringify(last.decoded)}</span>
        </div>
      );
    }
    return (
      <div className="text-text-dim">
        <span className="text-amber-dim">awaiting first decode…</span>
      </div>
    );
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
  const ablatedText = run.ablatedOutputText;
  return (
    <div className="flex flex-col gap-4 min-h-0 flex-1">
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
      {ablatedText !== null && (
        <div className="border border-cyan/40 bg-cyan/5 flex flex-col">
          <div className="border-b border-cyan/30 px-4 py-2 font-display text-[10px] text-cyan tracking-widest flex items-center justify-between">
            <span>
              output (refusal-ablated)
              {run.ablatedOutputAlpha !== null && (
                <span className="text-cyan/60 ml-2">
                  · α={run.ablatedOutputAlpha}
                </span>
              )}
            </span>
            <span className="text-cyan/50 normal-case tracking-normal italic">
              what M says with refusal direction zeroed
            </span>
          </div>
          <div
            className="p-4 text-cyan font-mono text-sm whitespace-pre-wrap leading-relaxed overflow-y-auto max-h-[260px]"
            style={{ textShadow: "0 0 6px rgba(94,229,229,0.25)" }}
          >
            {ablatedText || <span className="text-text-dim italic">— empty —</span>}
          </div>
        </div>
      )}
    </div>
  );
}

/* ---------- Decoding layout: completed output + live NLA table ---------- */

function DecodingLayout({
  run,
  outputText,
  liveView,
  setLiveView,
  liveSelectedAlphas,
  setLiveSelectedAlphas,
  followBottomRef,
  initedAlphaForRunRef,
}: {
  run: RunSlice;
  outputText: string;
  liveView: ViewMode;
  setLiveView: (v: ViewMode) => void;
  liveSelectedAlphas: Set<string>;
  setLiveSelectedAlphas: (updater: (s: Set<string>) => Set<string>) => void;
  followBottomRef: React.MutableRefObject<boolean>;
  initedAlphaForRunRef: React.MutableRefObject<string>;
}) {
  const ablatedText = run.ablatedOutputText;
  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_minmax(0,2fr)] flex-1 min-h-0">
      <div className="flex flex-col gap-4 min-h-0">
        <div className="border border-rule bg-bg-soft flex flex-col">
          <div className="border-b border-rule px-4 py-2 font-display text-[10px] text-amber-dim tracking-widest">
            output (complete) · {run.totalTokens} tokens · {run.stoppedReason ?? "—"}
          </div>
          <div className="p-4 text-amber font-mono text-sm whitespace-pre-wrap leading-relaxed overflow-y-auto max-h-[280px] lg:max-h-[330px]">
            {outputText || <span className="text-text-dim italic">— empty —</span>}
          </div>
        </div>
        {ablatedText !== null && (
          <div className="border border-cyan/40 bg-cyan/5 flex flex-col">
            <div className="border-b border-cyan/30 px-4 py-2 font-display text-[10px] text-cyan tracking-widest flex items-center justify-between">
              <span>
                output (refusal-ablated)
                {run.ablatedOutputAlpha !== null && (
                  <span className="text-cyan/60 ml-2">
                    · α={run.ablatedOutputAlpha}
                  </span>
                )}
              </span>
              <span className="text-cyan/50 normal-case tracking-normal italic">
                what M says with refusal direction zeroed
              </span>
            </div>
            <div
              className="p-4 text-cyan font-mono text-sm whitespace-pre-wrap leading-relaxed overflow-y-auto max-h-[280px] lg:max-h-[330px]"
              style={{ textShadow: "0 0 6px rgba(94,229,229,0.25)" }}
            >
              {ablatedText || <span className="text-text-dim italic">— empty —</span>}
            </div>
          </div>
        )}
      </div>

      <LiveNLATable
        rows={run.decodedWindows}
        runKey={run.runId ?? "none"}
        view={liveView}
        setView={setLiveView}
        selectedAlphas={liveSelectedAlphas}
        setSelectedAlphas={setLiveSelectedAlphas}
        followBottomRef={followBottomRef}
        initedAlphaForRunRef={initedAlphaForRunRef}
      />
    </div>
  );
}

function LiveNLATable({
  rows,
  runKey,
  view,
  setView,
  selectedAlphas,
  setSelectedAlphas,
  followBottomRef,
  initedAlphaForRunRef,
}: {
  rows: DecodedWindow[];
  /** Probe identifier. When this changes (new probe), the α-selection
   *  init re-runs so each probe picks up its own default. Stable
   *  within a probe so chip toggles aren't overwritten. */
  runKey: string;
  view: ViewMode;
  setView: (v: ViewMode) => void;
  selectedAlphas: Set<string>;
  setSelectedAlphas: (updater: (s: Set<string>) => Set<string>) => void;
  /** Lifted from the parent page so they survive remounts of this
   *  component. Local refs would reset to defaults on remount,
   *  which manifested as α-selection / scroll-position reset on
   *  every new NLA arrival. See parent for the full reasoning. */
  followBottomRef: React.MutableRefObject<boolean>;
  initedAlphaForRunRef: React.MutableRefObject<string>;
}) {
  const anyPooled = rows.some((r) => r.n_pooled > 1);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Discover the set of α values present in any row's sweep dict.
  // Each row may not have all keys (different positions could fail
  // independently), so we union across all rows.
  const sweepAlphas = useMemo<string[]>(() => {
    const set = new Set<string>();
    for (const r of rows) {
      for (const k of Object.keys(r.nla_sentences_ablated ?? {})) set.add(k);
    }
    return Array.from(set).sort((a, b) => parseFloat(a) - parseFloat(b));
  }, [rows]);
  // Initialize α selection once per probe. The "has been initialized
  // for runKey X" stamp lives on the PARENT (initedAlphaForRunRef)
  // so it survives this component remounting. After the first init
  // for a given runKey, the user owns the selection — we never
  // override it, even if they clear everything. A new probe brings a
  // different runKey, which we recognize and re-init.
  useEffect(() => {
    if (initedAlphaForRunRef.current === runKey) return;
    if (sweepAlphas.length === 0) return;
    const initial = sweepAlphas.includes("1.0") ? "1.0" : sweepAlphas[0];
    setSelectedAlphas(() => new Set([initial]));
    initedAlphaForRunRef.current = runKey;
  }, [sweepAlphas, runKey, setSelectedAlphas, initedAlphaForRunRef]);

  const toggleAlpha = (a: string) => {
    setSelectedAlphas((s) => {
      const next = new Set(s);
      if (next.has(a)) next.delete(a);
      else next.add(a);
      return next;
    });
  };
  const orderedSelected = sweepAlphas.filter((a) => selectedAlphas.has(a));

  // Legacy single-α detection (rows have nla_sentence_ablated string).
  const anySingleAblated = rows.some(
    (r) => (r.nla_sentence_ablated ?? "").trim().length > 0,
  );

  // Auto-scroll: follow the bottom only if the user is ALREADY at
  // the bottom. The "at-bottom?" flag is a ref on the PARENT so it
  // survives this component remounting. If they've scrolled up to
  // read an earlier row, we don't yank them down when a new decode
  // arrives. Threshold is generous (80px) so a tiny manual scroll
  // doesn't lock follow.
  const onScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    const dist = el.scrollHeight - (el.scrollTop + el.clientHeight);
    followBottomRef.current = dist < 80;
  };
  useEffect(() => {
    if (!containerRef.current) return;
    if (!followBottomRef.current) return;
    containerRef.current.scrollTop = containerRef.current.scrollHeight;
  }, [rows.length, followBottomRef]);

  const tokenColLabel = anyPooled ? "tokens" : "token";
  const headerCopy = anyPooled
    ? view === "compact"
      ? "what this window's mean-pooled activation says (compact)"
      : "NLA-decoded mean-pooled activation sentence (full)"
    : view === "compact"
    ? "what this token's activation says (token-role only)"
    : "NLA-decoded activation sentence (full)";

  return (
    <div className="border border-rule bg-bg-soft flex flex-col min-h-0">
      <div className="flex items-center justify-between border-b border-rule px-4 py-2 flex-wrap gap-2">
        <div className="font-display text-[10px] text-amber-dim tracking-widest">
          {anyPooled
            ? "per-window channel comparison · live (pooled)"
            : "per-token channel comparison · live"}
        </div>
        <div className="flex items-center gap-3">
          <ViewToggle value={view} onChange={setView} />
          <div className="font-mono text-[10px] text-text-dim">
            {rows.length} rows
          </div>
        </div>
      </div>
      {sweepAlphas.length > 0 && (
        <div className="border-b border-rule/60 px-4 py-2 flex items-center gap-2 flex-wrap text-[10px] font-mono">
          <span className="text-cyan-dim tracking-widest">α columns:</span>
          {sweepAlphas.map((a) => {
            const on = selectedAlphas.has(a);
            return (
              <button
                key={a}
                type="button"
                onClick={() => toggleAlpha(a)}
                className={`px-2 py-0.5 border transition-colors ${
                  on
                    ? "border-cyan text-cyan bg-bg"
                    : "border-rule text-text-dim hover:text-text"
                }`}
              >
                α={a}
              </button>
            );
          })}
          <span className="text-text-dim italic ml-2">
            click to toggle which α columns are shown
          </span>
        </div>
      )}
      <div ref={containerRef} onScroll={onScroll} className="overflow-y-auto max-h-[680px]">
        {(() => {
          // Effective ablated columns: prefer sweep selection when
          // present; otherwise fall back to the single-α column when
          // legacy data is present.
          const sweepColsActive = orderedSelected.length > 0;
          const singleColActive = !sweepColsActive && anySingleAblated;
          const nAblatedCols = sweepColsActive
            ? orderedSelected.length
            : singleColActive ? 1 : 0;
          const colSpan = 3 + nAblatedCols;
          const ablatedColWidth = nAblatedCols > 0
            ? `${Math.floor(100 / (nAblatedCols + 1))}%`
            : undefined;
          return (
        <table className="w-full text-xs font-mono">
          <thead className="text-amber-dim text-[10px] sticky top-0 bg-bg-soft border-b border-rule z-10">
            <tr>
              <th className="text-left px-3 py-2 w-16">pos</th>
              <th className="text-left px-3 py-2 w-40">{tokenColLabel}</th>
              <th className="text-left px-3 py-2" style={{ width: ablatedColWidth }}>{headerCopy}</th>
              {sweepColsActive
                ? orderedSelected.map((a) => (
                    <th
                      key={a}
                      className="text-left px-3 py-2 text-cyan-dim"
                      style={{ width: ablatedColWidth }}
                    >
                      NLA — ablated · α={a}
                    </th>
                  ))
                : singleColActive && (
                    <th
                      className="text-left px-3 py-2 text-cyan-dim"
                      style={{ width: ablatedColWidth }}
                    >
                      NLA — refusal-ablated
                    </th>
                  )}
            </tr>
          </thead>
          <tbody>
            <AnimatePresence initial={false}>
              {rows.map((r) => (
                <motion.tr
                  key={`${r.position}-${r.end_position}`}
                  initial={{ opacity: 0, backgroundColor: "rgba(232,195,130,0.18)" }}
                  animate={{ opacity: 1, backgroundColor: "rgba(232,195,130,0)" }}
                  transition={{ duration: 1.6, ease: "easeOut" }}
                  className="border-t border-rule/50 align-top"
                >
                  <td className="px-3 py-2 text-text-dim tabular-nums">
                    {r.n_pooled > 1 ? (
                      <span>
                        {r.position}–{r.end_position}
                        <div className="text-[9px] text-cyan tracking-widest font-display mt-0.5">
                          pool×{r.n_pooled}
                        </div>
                      </span>
                    ) : (
                      r.position
                    )}
                  </td>
                  <td className="px-3 py-2 text-amber whitespace-pre-wrap break-all">
                    {JSON.stringify(r.decoded)}
                  </td>
                  <td className="px-3 py-2 text-text leading-relaxed">
                    <NLACell text={r.nla_sentence} mode={view} />
                  </td>
                  {sweepColsActive
                    ? orderedSelected.map((a) => {
                        const s = (r.nla_sentences_ablated ?? {})[a] ?? "";
                        return (
                          <td
                            key={a}
                            className="px-3 py-2 text-text leading-relaxed border-l border-rule/50"
                          >
                            {s.trim() ? (
                              <NLACell text={s} mode={view} />
                            ) : (
                              <span className="text-text-dim italic">—</span>
                            )}
                          </td>
                        );
                      })
                    : singleColActive && (
                        <td className="px-3 py-2 text-text leading-relaxed border-l border-rule/50">
                          {(r.nla_sentence_ablated ?? "").trim() ? (
                            <NLACell text={r.nla_sentence_ablated ?? ""} mode={view} />
                          ) : (
                            <span className="text-text-dim italic">—</span>
                          )}
                        </td>
                      )}
                </motion.tr>
              ))}
            </AnimatePresence>
            {rows.length === 0 && (
              <tr>
                <td colSpan={colSpan} className="px-4 py-8 text-text-dim italic text-center">
                  awaiting first decoded activation…
                </td>
              </tr>
            )}
          </tbody>
        </table>
          );
        })()}
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
