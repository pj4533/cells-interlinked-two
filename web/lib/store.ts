"use client";

import { create } from "zustand";
import type { StreamEvent, VerdictEvent, VerdictRow } from "./types";

export interface OutputTokenEntry {
  position: number;
  decoded: string;
  /** Filled in by the "nla_decoded" event during phase 2. */
  nla_sentence?: string;
}

export interface RunState {
  runId: string | null;
  prompt: string;
  outputTokens: OutputTokenEntry[];
  totalTokens: number;
  phase: "idle" | "generating" | "decoding" | "done";
  /** During NLA-decoding phase: how far through the per-token decode we are. */
  decodeProgress: { done: number; total: number } | null;
  /** Wall-clock ms when the decoding phase started — used by the UI to
   *  compute a rolling ETA without re-rendering on a timer. */
  decodeStartedAt: number | null;
  /** Position of the most-recently-completed NLA decode. Drives the
   *  "decoding position N of T" cue without scanning the array. */
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
        set((s) => {
          const out = s.outputTokens.slice();
          const idx = out.findIndex((t) => t.position === evt.position);
          if (idx >= 0) {
            out[idx] = { ...out[idx], nla_sentence: evt.nla_sentence };
          }
          return {
            outputTokens: out,
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
