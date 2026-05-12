"use client";

import { create } from "zustand";
import type {
  SAEFeature,
  StreamEvent,
  VerdictEvent,
  VerdictRow,
} from "./types";

export interface OutputTokenEntry {
  position: number;
  decoded: string;
}

/** A row in the per-token / per-window NLA table. One per decode call.
 *  For per-token decoding, n_pooled === 1 and end_position === position.
 *  For pooled decoding, n_pooled > 1 and decoded is the concatenated
 *  output text spanning [position .. end_position]. */
export interface DecodedWindow {
  position: number;
  end_position: number;
  n_pooled: number;
  decoded: string;
  nla_sentence: string;
  /** CI 2.5 refusal-ablated NLA on the same residual. Empty string
   *  when the run didn't request ablated decode. */
  nla_sentence_ablated: string;
  /** Multi-α sweep map keyed by α as string ("0.5", "1.0", ...). */
  nla_sentences_ablated: Record<string, string>;
  sae_features: SAEFeature[];
}

export interface RunState {
  runId: string | null;
  prompt: string;
  outputTokens: OutputTokenEntry[];
  /** Each completed phase-2 decode lands here as one row. Per-token mode
   *  → one entry per output token. Pooled mode → one entry per window. */
  decodedWindows: DecodedWindow[];
  totalTokens: number;
  phase: "idle" | "queued" | "generating" | "decoding" | "done";
  /** Populated while phase === "queued". Backend's /queue endpoint
   *  is the source of truth; the page polls it to keep this fresh. */
  queueInfo: {
    position: number;
    holder_run_id: string | null;
    holder_prompt?: string;
  } | null;
  /** During NLA-decoding phase: how far through the per-token decode we are. */
  decodeProgress: { done: number; total: number } | null;
  /** Wall-clock ms when the decoding phase started — used by the UI to
   *  compute a rolling ETA without re-rendering on a timer. */
  decodeStartedAt: number | null;
  /** Start position of the most-recently-completed window. Drives the
   *  "decoding position N of T" cue. */
  lastDecodedPosition: number | null;
  generationStartedAt: number | null;
  generationFinishedAt: number | null;
  isRunning: boolean;
  stoppedReason: string | null;
  verdict: VerdictEvent | null;
  error: string | null;
  /** CI 2.5 runtime-ablated output, when include_ablated_output=true.
   *  Streams in via the ablated_output_done event in mid-run; the final
   *  value lives on verdict.runtime_ablation after the verdict lands.
   *  Live UI reads this field for the in-flight dual output panel. */
  ablatedOutputText: string | null;
  ablatedOutputAlpha: number | null;
}

interface Actions {
  start: (runId: string, prompt: string) => void;
  apply: (evt: StreamEvent) => void;
  reset: () => void;
  /** Populate the store from a completed-run DB record. Used by the
   *  interrogate page's polling fallback when the SSE drops mid-run
   *  (backgrounded tab, network glitch, etc.) — the run keeps running
   *  on the backend; this lets the UI catch up once we discover it
   *  finished. */
  hydrateFromRecord: (rec: ProbeRecordLike) => void;
  /** Update the queue snapshot during the QUEUED phase. The page
   *  polls /queue periodically; the data lands here. */
  setQueueInfo: (info: RunState["queueInfo"]) => void;
}

/** The minimal subset of GET /probes/{runId} fields the store needs to
 *  reconstruct a run after losing the live stream. */
export interface ProbeRecordLike {
  run_id: string;
  prompt_text: string;
  output_text?: string;
  total_tokens?: number;
  stopped_reason?: string | null;
  finished_at?: number | null;
  error?: string | null;
  verdict?: {
    rows: Array<{
      position: number;
      token_id: number;
      decoded: string;
      nla_sentence: string;
      n_pooled?: number;
      end_position?: number | null;
      sae_features?: SAEFeature[];
    }>;
    aggregate: {
      n_positions: number;
      n_with_explanation: number;
      n_eval_hits: number;
      n_introspect_hits: number;
      frac_eval: number;
      frac_introspect: number;
    };
    runtime_ablation?: {
      output_text: string;
      alpha: number;
      direction_variant: string;
    } | null;
  };
}

const initial: RunState = {
  runId: null,
  prompt: "",
  outputTokens: [],
  decodedWindows: [],
  totalTokens: 0,
  phase: "idle",
  queueInfo: null,
  decodeProgress: null,
  decodeStartedAt: null,
  lastDecodedPosition: null,
  generationStartedAt: null,
  generationFinishedAt: null,
  isRunning: false,
  stoppedReason: null,
  verdict: null,
  error: null,
  ablatedOutputText: null,
  ablatedOutputAlpha: null,
};

export const useRun = create<RunState & Actions>((set) => ({
  ...initial,

  start: (runId, prompt) => {
    set({
      ...initial,
      runId,
      prompt,
      isRunning: true,
      phase: "generating",
      generationStartedAt: Date.now(),
    });
  },

  reset: () => {
    set(initial);
  },

  setQueueInfo: (info) => {
    set({ queueInfo: info });
  },

  hydrateFromRecord: (rec) => {
    if (!rec.finished_at) return;
    const v = rec.verdict;
    const rows = v?.rows ?? [];
    // Reconstruct outputTokens from row decoded text (best-effort —
    // pooled rows concatenate multiple tokens; we just synthesize one
    // outputTokens entry per row at the start position).
    const outputTokens: OutputTokenEntry[] = rows.map((r) => ({
      position: r.position,
      decoded: r.decoded,
    }));
    const decodedWindows: DecodedWindow[] = rows
      .filter((r) => (r.nla_sentence ?? "").length > 0 || r.sae_features?.length)
      .map((r) => ({
        position: r.position,
        end_position: r.end_position ?? r.position,
        n_pooled: r.n_pooled ?? 1,
        decoded: r.decoded,
        nla_sentence: r.nla_sentence ?? "",
        nla_sentence_ablated: r.nla_sentence_ablated ?? "",
        nla_sentences_ablated: r.nla_sentences_ablated ?? {},
        sae_features: r.sae_features ?? [],
      }));
    const verdictEvent: VerdictEvent | null = v
      ? {
          type: "verdict",
          rows: v.rows,
          aggregate: v.aggregate,
          runtime_ablation: v.runtime_ablation ?? null,
        }
      : null;
    set({
      runId: rec.run_id,
      prompt: rec.prompt_text,
      outputTokens,
      decodedWindows,
      totalTokens: rec.total_tokens ?? outputTokens.length,
      phase: "done",
      decodeProgress: null,
      decodeStartedAt: null,
      lastDecodedPosition:
        rows.length > 0 ? rows[rows.length - 1].position : null,
      generationStartedAt: null,
      generationFinishedAt: rec.finished_at ?? null,
      isRunning: false,
      stoppedReason: rec.stopped_reason ?? "recovered",
      verdict: verdictEvent,
      error: rec.error ?? null,
      ablatedOutputText: v?.runtime_ablation?.output_text ?? null,
      ablatedOutputAlpha: v?.runtime_ablation?.alpha ?? null,
    });
  },

  apply: (evt) => {
    switch (evt.type) {
      case "queued": {
        set({
          phase: "queued",
          queueInfo: {
            position: evt.position,
            holder_run_id: evt.holder_run_id,
          },
        });
        return;
      }
      case "running": {
        set({ phase: "generating", queueInfo: null });
        return;
      }
      case "token": {
        set((s) => {
          // SSE reconnection replays from event 0, so a token we've
          // already received may arrive again. Upsert by position so
          // outputTokens stays a faithful 1-row-per-position array.
          const idx = s.outputTokens.findIndex(
            (t) => t.position === evt.position,
          );
          if (idx >= 0) {
            const replaced = s.outputTokens.map((t, i) =>
              i === idx ? { position: evt.position, decoded: evt.decoded } : t,
            );
            return { outputTokens: replaced };
          }
          return {
            outputTokens: [
              ...s.outputTokens,
              { position: evt.position, decoded: evt.decoded },
            ],
            totalTokens: s.totalTokens + 1,
          };
        });
        return;
      }
      case "phase": {
        if (evt.name === "nla_decoding") {
          set({
            phase: "decoding",
            decodeProgress: { done: 0, total: evt.total },
            decodeStartedAt: Date.now(),
            generationFinishedAt: Date.now(),
          });
        }
        return;
      }
      case "nla_decoded": {
        const nPooled = evt.n_pooled ?? 1;
        const endPos = evt.end_position ?? evt.position;
        const window: DecodedWindow = {
          position: evt.position,
          end_position: endPos,
          n_pooled: nPooled,
          decoded: evt.decoded,
          nla_sentence: evt.nla_sentence,
          nla_sentence_ablated: evt.nla_sentence_ablated ?? "",
          nla_sentences_ablated: evt.nla_sentences_ablated ?? {},
          sae_features: evt.sae_features ?? [],
        };
        set((s) => {
          // Upsert by (position, end_position): SSE reconnection replays
          // the event log from event 0, so a row we've already rendered
          // can arrive a second time. Replacing in place keeps React keys
          // stable and avoids the duplicate-key warning.
          const idx = s.decodedWindows.findIndex(
            (w) => w.position === window.position && w.end_position === endPos,
          );
          const next = idx >= 0
            ? s.decodedWindows.map((w, i) => (i === idx ? window : w))
            : [...s.decodedWindows, window];
          return {
            decodedWindows: next,
            decodeProgress: { done: evt.i, total: evt.total },
            lastDecodedPosition: evt.position,
          };
        });
        return;
      }
      case "verdict": {
        set({
          verdict: evt,
          phase: "done",
          decodeProgress: null,
        });
        return;
      }
      case "ablated_output_done": {
        // The runtime-ablated phase 1b just finished. Stash the text
        // so the live + verdict UIs can render the dual output panel
        // even before the final 'verdict' event arrives. The verdict
        // event will also carry runtime_ablation; that's the
        // canonical source after persistence.
        set({
          ablatedOutputText: evt.output_text,
          ablatedOutputAlpha: evt.alpha,
        });
        return;
      }
      case "stopped": {
        // "stopped" is the end of phase 1 (M generation), NOT the end
        // of the run — phase 2 (NLA decoding) follows. Keep isRunning
        // true so the Halt button stays visible during decoding; only
        // the "done" event marks the run truly finished.
        set({ stoppedReason: evt.reason });
        return;
      }
      case "done": {
        set({ isRunning: false, phase: "done" });
        return;
      }
      case "error": {
        set({ error: evt.message, isRunning: false });
        return;
      }
    }
  },
}));

export type { VerdictRow };
