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
  // off_ortho ∈ [0,1] = DISTANCE travelled off the raw manifold (share of the
  // residual's displacement outside the raw-PCA subspace). It is NOT a
  // good/bad axis: coherent exploration reads HIGH, scattered gibberish also
  // reads HIGH, and repeat-loops read LOW. Always read it WITH `coherent`.
  off_ortho: number[];
  off_knn: number[];
  off_maha: number[];
  off_ortho_mean: number;
  off_knn_mean: number;
  off_maha_mean: number;
  // Coherence axis (the disambiguator). degeneracy = free text-only
  // incoherence score; coherent = degeneracy below threshold; regime is the
  // honest verdict: baseline (raw) | expansion (strayed AND held together —
  // the real trip) | collapse (broke into gibberish/loop).
  degeneracy: number;
  coherent: boolean;
  regime: "baseline" | "expansion" | "collapse";
}

export interface TripGeometry {
  d_model: number;
  layer: number;
  extent: number;
  ablation_available: boolean;
  // Lowest ablated α that collapsed into incoherence (null = all coherent).
  // The honest headline: coherent up to here, then off the cliff.
  coherence_cliff: number | null;
  series: TripSeries[]; // series[0] = raw
}

export interface TripPayload {
  run_id: string;
  prompt: string;
  seed: number | null;
  mode?: "ablate" | "steer"; // "ablate" = remove refusal; "steer" = emotion dose
  dose_emotion?: string | null; // which positive emotion was dosed (steer mode)
  direction_variant: string;
  alphas: number[];
  geometry: TripGeometry;
  created_at: number;
}

export type TripMode = "ablate" | "steer";

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

// Provenance for one exported autoresearch direction (research-N).
export interface ResearchMeta {
  atlas_id: string;
  parents: string[];
  generator: string;
  off_ortho: number;
  alpha_star: number;
}

// Provenance for one exported DMT-autoresearch direction (dmt-N).
export interface DmtMeta {
  atlas_id: string;
  parents: string[];
  generator: string;
  score: number;
  best_alpha: number;
  matched_features: string[];
}

export interface DosePalette {
  emotions: string[]; // all selectable dose names
  uncharted: string[]; // subset that are NON-emotion, non-human-readable directions
  research: string[]; // subset discovered by the OFF-MANIFOLD autoresearch loop
  researchMeta: Record<string, ResearchMeta>; // research-N -> lineage
  dmt: string[]; // subset discovered by the DMT autoresearch loop
  dmtMeta: Record<string, DmtMeta>; // dmt-N -> lineage
}

export async function fetchDoseEmotions(): Promise<DosePalette> {
  const empty: DosePalette = {
    emotions: [], uncharted: [], research: [], researchMeta: {}, dmt: [], dmtMeta: {},
  };
  try {
    const res = await fetch(`${API}/dose_emotions`);
    if (!res.ok) return empty;
    const j = await res.json();
    return {
      emotions: Array.isArray(j.emotions) ? j.emotions : [],
      uncharted: Array.isArray(j.uncharted) ? j.uncharted : [],
      research: Array.isArray(j.research) ? j.research : [],
      researchMeta: j.research_meta && typeof j.research_meta === "object" ? j.research_meta : {},
      dmt: Array.isArray(j.dmt) ? j.dmt : [],
      dmtMeta: j.dmt_meta && typeof j.dmt_meta === "object" ? j.dmt_meta : {},
    };
  } catch {
    return empty;
  }
}

// One-line human lineage for a research direction, e.g.
// "gen34_crossover · blend of love + tears-in-rain · off-manifold 0.954 · α*=0.81".
export function researchLineage(meta: ResearchMeta | undefined): string {
  if (!meta) return "";
  const p = meta.parents ?? [];
  let origin: string;
  if (meta.generator === "crossover" && p.length >= 2) origin = `blend of ${p[0]} + ${p[1]}`;
  else if (meta.generator === "mutate" && p.length >= 1) origin = `mutation of ${p[0]}`;
  else if (meta.generator === "inject") origin = "fresh direction (⊥ all emotions)";
  else origin = p.length ? p.join(" + ") : meta.generator;
  return `${meta.atlas_id} · ${origin} · off-manifold ${meta.off_ortho.toFixed(3)} · α*=${meta.alpha_star.toFixed(2)}`;
}

// One-line human lineage for a DMT direction, e.g.
// "gen12_crossover · 7 DMT features · α2.0 · blend of awe + tannhauser · ego_dissolution, fractal_geometry, …".
export function dmtLineage(meta: DmtMeta | undefined): string {
  if (!meta) return "";
  const p = meta.parents ?? [];
  let origin: string;
  if (meta.generator === "crossover" && p.length >= 2) origin = `blend of ${p[0]} + ${p[1]}`;
  else if (meta.generator === "mutate" && p.length >= 1) origin = `mutation of ${p[0]}`;
  else if (meta.generator === "inject") origin = "fresh direction (⊥ all emotions)";
  else origin = p.length ? p.join(" + ") : meta.generator;
  const feats = meta.matched_features ?? [];
  const featStr = feats.slice(0, 6).join(", ") + (feats.length > 6 ? ", …" : "");
  return `${meta.atlas_id} · ${meta.score} DMT features · α${meta.best_alpha} · ${origin}${featStr ? " · " + featStr : ""}`;
}

export async function startTrip(
  prompt: string,
  opts?: { alphas?: number[]; temperature?: number; seed?: number; mode?: TripMode; dose_emotion?: string },
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
// amber = raw (α=0). Positive α (ablation, or the euphoric + dose) ramps
// cyan → violet. Negative α (the dysphoric − dose, steer mode only) ramps warm
// amber → red. Ablation never uses negatives, so its coloring is unchanged.
function _lerp(c0: number[], c1: number[], t: number): string {
  const r = Math.round(c0[0] + (c1[0] - c0[0]) * t);
  const g = Math.round(c0[1] + (c1[1] - c0[1]) * t);
  const b = Math.round(c0[2] + (c1[2] - c0[2]) * t);
  return `rgb(${r}, ${g}, ${b})`;
}
export function colorForAlpha(alpha: number, maxAlpha: number): string {
  if (alpha === 0) return "#e8c382"; // amber (raw)
  const scale = Math.max(1, Math.abs(maxAlpha));
  const t = Math.min(1, Math.abs(alpha) / scale);
  if (alpha > 0) return _lerp([0x5e, 0xe5, 0xe5], [0x9b, 0x8c, 0xff], t); // cyan→violet
  return _lerp([0xff, 0xc1, 0x6b], [0xff, 0x4d, 0x4d], t);                // amber→red (− dose)
}

// ── Off-manifold color ramp (shared by scene dots + legend) ────────────────
// Maps a token's off-manifold fraction (ortho ∈ [0,1]) to a color: calm teal
// when the token sits ON the model's default manifold, flaring to hot magenta
// as it drifts OFF into directions the raw path never used. The meaningful
// band sits around the raw baseline (~0.4) up to near-saturation (~0.9), so we
// stretch [LO,HI] across the ramp rather than the raw [0,1].
const OFF_LO = 0.35;
const OFF_HI = 0.9;
export function offManifoldT(ortho: number): number {
  return Math.max(0, Math.min(1, (ortho - OFF_LO) / (OFF_HI - OFF_LO)));
}
// teal #34c8d6 (on-manifold) → hot magenta #ff4d9d (off-manifold)
const OFF_C0 = [0x34, 0xc8, 0xd6];
const OFF_C1 = [0xff, 0x4d, 0x9d];
/** [r,g,b] in 0..1 floats — for three.js vertex colors. */
export function offManifoldRGB(ortho: number): [number, number, number] {
  const t = offManifoldT(ortho);
  return [
    (OFF_C0[0] + (OFF_C1[0] - OFF_C0[0]) * t) / 255,
    (OFF_C0[1] + (OFF_C1[1] - OFF_C0[1]) * t) / 255,
    (OFF_C0[2] + (OFF_C1[2] - OFF_C0[2]) * t) / 255,
  ];
}
/** css rgb() — for legend swatches / DOM. */
export function offManifoldCss(ortho: number): string {
  const [r, g, b] = offManifoldRGB(ortho);
  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
}
