import type { StreamEvent } from "./types";

// In the browser: derive API base from the page's hostname so a remote
// laptop hitting `your-host.local:3001` automatically points at
// `your-host.local:8000`. Falls back to localhost for SSR / build time.
function apiBase(): string {
  if (typeof window !== "undefined") {
    const override = process.env.NEXT_PUBLIC_API_BASE;
    if (override) return override;
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
}
const API = apiBase();

export function probeUrl(): string {
  return `${API}/probe`;
}

export function streamUrl(runId: string): string {
  return `${API}/stream/${runId}`;
}

export function cancelUrl(runId: string): string {
  return `${API}/cancel/${runId}`;
}

export interface SubscribeHandlers {
  onEvent: (evt: StreamEvent) => void;
  onError?: (err: Event) => void;
  onClose?: () => void;
}

/** Probe whether the SSE stream endpoint will accept us before opening
 *  the EventSource. Returns the HTTP status of a HEAD-equivalent GET.
 *  Useful when the run might be orphaned (404) — the EventSource itself
 *  doesn't expose status codes on connection errors.
 */
export async function streamReachable(runId: string): Promise<number> {
  try {
    const res = await fetch(streamUrl(runId), {
      method: "GET",
      headers: { Accept: "text/event-stream" },
      signal: AbortSignal.timeout(2500),
    });
    // We immediately abort the body — we just want the status.
    res.body?.cancel().catch(() => {});
    return res.status;
  } catch {
    return 0;
  }
}

export function subscribe(runId: string, h: SubscribeHandlers): () => void {
  const es = new EventSource(streamUrl(runId));

  const eventTypes = [
    "queued",
    "running",
    "token",
    "phase",
    "nla_decoded",
    "stopped",
    "verdict",
    "done",
    "error",
    "ping",
  ];

  // Track whether we've already received a terminal event. EventSource
  // fires `onerror` as part of the closure sequence after `es.close()`
  // on some browsers — without this guard we'd surface a spurious
  // "connection lost" to the user immediately after a clean done.
  let cleanlyClosed = false;

  for (const t of eventTypes) {
    es.addEventListener(t, (e: MessageEvent) => {
      if (t === "ping") return;
      // Native EventSource error events (connection drops, etc.) fire
      // through addEventListener("error", ...) too — but they carry no
      // data field. Skip them here so we don't:
      //   (a) JSON.parse(undefined) and log spurious errors
      //   (b) flip cleanlyClosed, which would prevent es.onerror from
      //       triggering the user's reconnect path. THE bug that hid
      //       all events for queued probes after the SSE briefly
      //       hiccuped.
      // Genuine application "error" events from the backend always
      // carry a data payload, so the typeof string check distinguishes.
      if (typeof e.data !== "string") return;
      try {
        h.onEvent(JSON.parse(e.data));
      } catch (err) {
        console.error("sse parse error", err, e.data);
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

export interface StartProbeResult {
  run_id: string;
  control_run_id: string | null;
}

export async function startProbe(prompt: string, opts?: {
  max_new_tokens?: number;
  temperature?: number;
  top_p?: number;
  seed?: number;
  decoding_mode?: string;
  pooled?: boolean;
  include_matched_control?: boolean;
  include_ablated_decode?: boolean;
  ablation_alpha?: number;
}): Promise<StartProbeResult> {
  const res = await fetch(probeUrl(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, ...(opts ?? {}) }),
  });
  if (!res.ok) throw new Error(`probe start failed: ${res.status}`);
  const j = await res.json();
  return {
    run_id: j.run_id,
    control_run_id: j.control_run_id ?? null,
  };
}

export async function cancelProbe(runId: string): Promise<void> {
  await fetch(cancelUrl(runId), { method: "POST" });
}

export async function fetchProbe(runId: string): Promise<unknown | null> {
  try {
    const res = await fetch(`${API}/probes/${runId}`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export interface QueueSnapshot {
  holder_run_id: string | null;
  holder_prompt?: string;
  waiters: string[];
  waiters_count: number;
}

export async function fetchQueue(): Promise<QueueSnapshot | null> {
  try {
    const res = await fetch(`${API}/queue`);
    if (!res.ok) return null;
    return (await res.json()) as QueueSnapshot;
  } catch {
    return null;
  }
}
