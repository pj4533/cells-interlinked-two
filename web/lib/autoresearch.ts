// Autoresearch client — start/stop the steering-direction hunt and poll its
// live state for the monitor page. Events are low-frequency, so we poll
// /autoresearch/state (~2s) rather than open an SSE stream.

function apiBase(): string {
  if (typeof window !== "undefined") {
    const override = process.env.NEXT_PUBLIC_API_BASE;
    if (override) return override;
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
}
const API = apiBase();

export interface AtlasEntry {
  id: string;
  parents: string[];
  generator: string; // seed | crossover | mutate | inject
  alpha_star: number;
  off_ortho: number;
  eff_dim: number;
  coh_rate: number;
  repro: number;
  max_cos_to_atlas: number;
  frontier_advance: boolean;
  frontier_at_commit: number;
  sample: string;
  committed_at: number;
}

export interface RevertEntry {
  id: string;
  generator: string;
  parents: string[];
  reason: string; // duplicate | T1-incoherent | not-graded | incoherent-suite | word-salad | not-reproducible | error
  detail: string;
  ts: number;
}

export interface AREvent {
  ts: number;
  kind: string;
  msg: string;
}

export interface CurrentCandidate {
  id: string;
  generator: string;
  parents?: string[];
  stage: string; // distinct | T1 | smoothness | T2
  alpha_star?: number;
  t1_off_ortho?: number;
  coh_rate?: number;
  repro?: number;
  judge?: string;
  max_cos?: number;
}

export interface ARState {
  running: boolean;
  stop_requested: boolean;
  generation: number;
  frontier: number;
  started_at: number | null;
  atlas_size: number;
  atlas: AtlasEntry[];
  reverts: RevertEntry[];
  recent_events: AREvent[];
  current: CurrentCandidate | null;
}

export async function fetchAutoresearchState(): Promise<ARState | null> {
  try {
    const res = await fetch(`${API}/autoresearch/state`, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as ARState;
  } catch {
    return null;
  }
}

export async function startAutoresearch(budget?: number): Promise<{ ok: boolean; error?: string; already_running?: boolean }> {
  const res = await fetch(`${API}/autoresearch/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(budget ? { budget } : {}),
  });
  return res.json();
}

export async function stopAutoresearch(): Promise<{ ok: boolean; was_running?: boolean }> {
  const res = await fetch(`${API}/autoresearch/stop`, { method: "POST" });
  return res.json();
}
