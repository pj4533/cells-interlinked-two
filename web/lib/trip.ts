// Trip View client: start a trip run, subscribe to its SSE stream.
//
// Multi-α design: the backend runs the model at several discrete ablation
// strengths — raw (α=0) plus each requested α — each a REAL generation
// (text + actual L32 trajectory). No continuous slider; the UI toggles which
// α series to overlay. All series share one PCA basis (raw's), so the ablated
// paths visibly diverge from baseline.

function apiBase(): string {
  if (typeof window !== "undefined") {
    const override = process.env.NEXT_PUBLIC_API_BASE;
    if (override) return override;
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
}
const API = apiBase();

export interface TripSeries {
  alpha: number; // 0 = raw
  label: string; // "raw" | "α=1.00"
  coords: number[][]; // [N][3] in the shared raw-PCA basis
  tokens: string[];
  text: string;
  n_tokens: number;
  eff_dim: number;
  spectral_entropy: number;
  spectrum: number[];
  stopped_reason: string;
}

export interface TripGeometry {
  d_model: number;
  layer: number;
  extent: number;
  ablation_available: boolean;
  series: TripSeries[]; // series[0] = raw
}

export interface TripPayload {
  run_id: string;
  prompt: string;
  seed: number | null;
  direction_variant: string;
  alphas: number[];
  geometry: TripGeometry;
  created_at: number;
}

export type TripEvent =
  | { type: "queued"; holder_run_id: string | null; position: number }
  | { type: "running" }
  | { type: "phase"; name: string; alpha?: number }
  | { type: "trip_token"; alpha: number; position: number; decoded: string }
  | { type: "trip_series"; layer: number; series: TripSeries }
  | ({ type: "trip_geometry" } & TripPayload)
  | { type: "done" }
  | { type: "error"; message?: string }
  | { type: string; [k: string]: unknown };

export async function startTrip(
  prompt: string,
  opts?: { alphas?: number[]; temperature?: number; seed?: number },
): Promise<{ run_id: string }> {
  const res = await fetch(`${API}/trip`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, ...(opts ?? {}) }),
  });
  if (!res.ok) throw new Error(`trip start failed: ${res.status}`);
  return res.json();
}

export async function cancelTrip(runId: string): Promise<void> {
  await fetch(`${API}/cancel/${runId}`, { method: "POST" }).catch(() => {});
}

export async function fetchTrip(runId: string): Promise<TripPayload | null> {
  try {
    const res = await fetch(`${API}/trip/${runId}`);
    if (!res.ok) return null;
    return (await res.json()) as TripPayload;
  } catch {
    return null;
  }
}

export interface TripSubscribeHandlers {
  onEvent: (evt: TripEvent) => void;
  onError?: (err: Event) => void;
  onClose?: () => void;
}

export function subscribeTrip(
  runId: string,
  h: TripSubscribeHandlers,
): () => void {
  const es = new EventSource(`${API}/stream/${runId}`);
  const eventTypes = [
    "queued",
    "running",
    "phase",
    "trip_token",
    "trip_series",
    "trip_geometry",
    "done",
    "error",
    "ping",
  ];
  let cleanlyClosed = false;
  for (const t of eventTypes) {
    es.addEventListener(t, (e: MessageEvent) => {
      if (t === "ping") return;
      if (typeof e.data !== "string") return;
      try {
        h.onEvent(JSON.parse(e.data) as TripEvent);
      } catch (err) {
        console.error("trip sse parse error", err, e.data);
        return;
      }
      if (t === "done" || t === "error") {
        cleanlyClosed = true;
        es.close();
        h.onClose?.();
      }
    });
  }
  es.onerror = (err) => {
    if (cleanlyClosed) return;
    h.onError?.(err);
  };
  return () => {
    cleanlyClosed = true;
    es.close();
  };
}

// ── Color per α (shared by scene + chips + text) ───────────────────────────
// amber = raw; ablated ramps cyan → violet as α rises (all "cool = ablated").
export function colorForAlpha(alpha: number, maxAlpha: number): string {
  if (alpha <= 0) return "#e8c382"; // amber (raw)
  const t = maxAlpha > 0 ? Math.min(1, alpha / maxAlpha) : 0;
  // cyan #5ee5e5 → violet #9b8cff
  const c0 = [0x5e, 0xe5, 0xe5];
  const c1 = [0x9b, 0x8c, 0xff];
  const r = Math.round(c0[0] + (c1[0] - c0[0]) * t);
  const g = Math.round(c0[1] + (c1[1] - c0[1]) * t);
  const b = Math.round(c0[2] + (c1[2] - c0[2]) * t);
  return `rgb(${r}, ${g}, ${b})`;
}
