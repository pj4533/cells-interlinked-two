// DMT autoresearch client — start/stop the DMT-phenomenology hunt and poll its
// live state for the monitor page. Mirrors lib/autoresearch.ts but targets the
// /autoresearch-dmt/* endpoints and the DMT atlas shape (score + matched
// features instead of off-manifold geometry).

export interface RevertEntry {
  id: string;
  generator: string;
  parents: string[];
  reason: string; // duplicate | T1-incoherent | not-graded | incoherent-suite | word-salad | not-reproducible | error
  detail: string;
  ts: number;
}

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
  score: number; // MEAN DMT-feature count over repeated doses (averaged/reliable, ~0–6, a float)
  peak?: number; // best single sample's feature count (the highest one observed)
  per_alpha?: Record<string, { mean: number; counts: number[] }>; // per-α mean + raw counts
  refined_from?: { gen: number; from: number; to: number }[]; // in-place hone history (origin generator unchanged)
  best_alpha: number;
  best_prompt: string;
  matched_features: string[];
  matched_evidence: Record<string, string>; // feature id -> verbatim quote that earned it
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

// One individual dose (run): its feature count, the features it matched, the
// verbatim evidence span per feature, and the full dose-response text.
export interface DmtSample {
  count: number;
  features: string[];
  evidence: Record<string, string>;
  text: string;
}

// Live per-α / per-sample progress of the candidate currently being scored,
// published by the backend during _score_candidate (absent in the "distinct"
// stage and on older backends). counts[]/samples[] grow per completed sample.
export interface DmtProgress {
  alphas: string[];
  samples_per_cell: number;
  samples_total: number;
  samples_done: number;
  per_alpha: Record<string, { mean: number; counts: number[]; samples?: DmtSample[] }>;
  best: {
    score: number;
    best_alpha: number | null;
    matched_features: string[];
    matched_evidence?: Record<string, string>;
    sample: string;
  };
}

// Lazy-loaded per-α / per-dose detail for a committed atlas entry.
export interface DmtCellsDetail {
  id: string;
  best_alpha: number | null;
  cells: Record<string, DmtSample[]>;
}

export async function fetchDmtCells(id: string): Promise<DmtCellsDetail | null> {
  try {
    const res = await fetch(`${API}/autoresearch-dmt/cells/${encodeURIComponent(id)}`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return (await res.json()) as DmtCellsDetail;
  } catch {
    return null;
  }
}

export interface DmtCurrentCandidate {
  id: string;
  generator: string;
  parents?: string[];
  stage: string; // distinct | score
  score?: number;
  best_alpha?: number;
  max_cos?: number;
  progress?: DmtProgress;
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
