// DMT autoresearch client — start/stop the DMT-phenomenology hunt and poll its
// live state for the monitor page. Mirrors lib/autoresearch.ts but targets the
// /autoresearch-dmt/* endpoints and the DMT atlas shape (score + matched
// features instead of off-manifold geometry).

import type { RevertEntry } from "./autoresearch";

function apiBase(): string {
  if (typeof window !== "undefined") {
    const override = process.env.NEXT_PUBLIC_API_BASE;
    if (override) return override;
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
}
const API = apiBase();

export interface DmtAtlasEntry {
  id: string;
  parents: string[];
  generator: string; // seed | crossover | mutate | inject
  score: number; // count of DMT phenomenology features present
  best_alpha: number;
  best_prompt: string;
  matched_features: string[];
  max_cos_to_atlas: number;
  frontier_advance: boolean;
  frontier_at_commit: number;
  sample: string; // best dose self-report
  committed_at: number;
}

export interface DmtEvent {
  ts: number;
  kind: string;
  msg: string;
  entry?: DmtAtlasEntry;
  revert?: RevertEntry;
}

export interface DmtCurrentCandidate {
  id: string;
  generator: string;
  parents?: string[];
  stage: string; // distinct | score
  score?: number;
  best_alpha?: number;
  max_cos?: number;
}

export interface DmtARState {
  running: boolean;
  stop_requested: boolean;
  generation: number;
  frontier: number; // best feature-count reached
  started_at: number | null;
  atlas_size: number;
  atlas: DmtAtlasEntry[];
  reverts: RevertEntry[];
  recent_events: DmtEvent[];
  current: DmtCurrentCandidate | null;
}

export async function fetchDmtState(): Promise<DmtARState | null> {
  try {
    const res = await fetch(`${API}/autoresearch-dmt/state`, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as DmtARState;
  } catch {
    return null;
  }
}

export async function startDmt(budget?: number): Promise<{ ok: boolean; error?: string; already_running?: boolean }> {
  const res = await fetch(`${API}/autoresearch-dmt/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(budget ? { budget } : {}),
  });
  return res.json();
}

export async function stopDmt(): Promise<{ ok: boolean; was_running?: boolean }> {
  const res = await fetch(`${API}/autoresearch-dmt/stop`, { method: "POST" });
  return res.json();
}

export async function exportDmt(topN: number): Promise<{ ok: boolean; error?: string; exported?: string[]; count?: number }> {
  const res = await fetch(`${API}/autoresearch-dmt/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ top_n: topN }),
  });
  return res.json();
}
