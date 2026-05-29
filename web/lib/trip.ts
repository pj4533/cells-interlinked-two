// Trip View client: start a trip run, subscribe to its SSE stream, and the
// pure-math helpers that drive the realtime α-morph in the browser.
//
// The backend ships the RAW projected trajectory plus two rank-1 morph
// helpers (refusal_axis, refusal_component). Because the ablation is rank-1,
// the ablated cloud at any α is an EXACT linear function we can evaluate at
// 60fps with no backend round-trip:
//
//   coords_ablated(α)[i] = coords_raw[i] − α · refusal_component[i] · refusal_axis
//
// That's the whole reason the slider feels instant.

function apiBase(): string {
  if (typeof window !== "undefined") {
    const override = process.env.NEXT_PUBLIC_API_BASE;
    if (override) return override;
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
}
const API = apiBase();

export interface TripGeometry {
  n_tokens: number;
  d_model: number;
  layer: number;
  tokens: string[];
  coords_raw: number[][]; // [N][3]
  refusal_axis: number[]; // [3]
  refusal_component: number[]; // [N]
  spectrum_raw: number[];
  eff_dim_raw: number;
  eff_dim_ablated: number;
  spectral_entropy_raw: number;
  spectral_entropy_ablated: number;
  alpha_grid: number[];
  eff_dim_grid: number[];
  spectral_entropy_grid: number[];
  spectrum_ablated_ref: number[];
  alpha_ref: number;
  extent: number;
  ablation_available: boolean;
}

export interface TripPayload {
  run_id: string;
  prompt: string;
  output_text: string;
  stopped_reason: string;
  total_tokens: number;
  seed: number | null;
  direction_variant: string;
  geometry: TripGeometry;
  created_at: number;
  // The actual runtime-ablated generation at alpha_ref (what M *says*
  // off-manifold). Null if no refusal direction was loaded.
  output_text_ablated?: string | null;
  ablated_alpha?: number;
  ablated_stopped_reason?: string;
}

// Discriminated-ish SSE event shape for the trip stream.
export type TripEvent =
  | { type: "queued"; holder_run_id: string | null; position: number }
  | { type: "running" }
  | { type: "phase"; name: string; total: number }
  | { type: "token"; position: number; token_id: number; decoded: string }
  | { type: "stopped"; reason: string; total_tokens: number }
  | ({ type: "trip_geometry" } & TripPayload)
  | { type: "ablated_token"; position: number; decoded: string }
  | { type: "ablated_output_done"; output_text: string; alpha: number; stopped_reason: string }
  | { type: "done" }
  | { type: "error"; message?: string }
  | { type: string; [k: string]: unknown };

export async function startTrip(
  prompt: string,
  opts?: { alpha_ref?: number; temperature?: number; seed?: number },
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
  // Reuses the shared /cancel/{id} endpoint (RunRegistry is common).
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
    "token",
    "stopped",
    "trip_geometry",
    "ablated_token",
    "ablated_output_done",
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

// ── α-morph math (pure, runs every frame) ──────────────────────────────────

/** Fill a Float32Array [N*3] with the ablated trajectory at strength α.
 *  Reuses the provided buffer so we don't allocate per frame. */
export function ablatedCoordsInto(
  out: Float32Array,
  g: TripGeometry,
  alpha: number,
): Float32Array {
  const N = g.coords_raw.length;
  const ax = g.refusal_axis;
  const comp = g.refusal_component;
  for (let i = 0; i < N; i++) {
    const c = g.coords_raw[i];
    const k = alpha * (comp[i] ?? 0);
    out[i * 3 + 0] = c[0] - k * ax[0];
    out[i * 3 + 1] = c[1] - k * ax[1];
    out[i * 3 + 2] = c[2] - k * ax[2];
  }
  return out;
}

export function rawCoordsInto(out: Float32Array, g: TripGeometry): Float32Array {
  const N = g.coords_raw.length;
  for (let i = 0; i < N; i++) {
    const c = g.coords_raw[i];
    out[i * 3 + 0] = c[0];
    out[i * 3 + 1] = c[1];
    out[i * 3 + 2] = c[2];
  }
  return out;
}

/** Linear-interpolate a per-α grid metric (eff_dim / spectral_entropy) at an
 *  arbitrary α. The grid is uniform from 0..max. */
export function metricAtAlpha(
  grid: number[],
  values: number[],
  alpha: number,
): number {
  if (grid.length === 0) return 0;
  if (grid.length === 1) return values[0];
  const lo = grid[0];
  const hi = grid[grid.length - 1];
  if (alpha <= lo) return values[0];
  if (alpha >= hi) return values[values.length - 1];
  const span = (hi - lo) / (grid.length - 1);
  const f = (alpha - lo) / span;
  const i = Math.floor(f);
  const t = f - i;
  return values[i] * (1 - t) + values[i + 1] * t;
}
