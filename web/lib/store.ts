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
  phase: "idle" | "generating" | "decoding" | "done";
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
}

interface Actions {
  start: (runId: string, prompt: string) => void;
  apply: (evt: StreamEvent) => void;
  reset: () => void;
}

const initial: RunState = {
  runId: null,
  prompt: "",
  outputTokens: [],
  decodedWindows: [],
  totalTokens: 0,
  phase: "idle",
  decodeProgress: null,
  decodeStartedAt: null,
  lastDecodedPosition: null,
  generationStartedAt: null,
  generationFinishedAt: null,
  isRunning: false,
  stoppedReason: null,
  verdict: null,
  error: null,
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

  apply: (evt) => {
    switch (evt.type) {
      case "token": {
        set((s) => ({
          outputTokens: [
            ...s.outputTokens,
            { position: evt.position, decoded: evt.decoded },
          ],
          totalTokens: s.totalTokens + 1,
        }));
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
          sae_features: evt.sae_features ?? [],
        };
        set((s) => ({
          decodedWindows: [...s.decodedWindows, window],
          decodeProgress: { done: evt.i, total: evt.total },
          lastDecodedPosition: evt.position,
        }));
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
      case "stopped": {
        set({ stoppedReason: evt.reason, isRunning: false });
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
