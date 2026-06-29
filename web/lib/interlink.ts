// Interlink — model-to-model auto-conversation client. Start/stop the loop and
// subscribe to its session-level SSE stream. Mirrors lib/chat.ts conventions.

function apiBase(): string {
  if (typeof window !== "undefined") {
    const override = process.env.NEXT_PUBLIC_API_BASE;
    if (override) return override;
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
}
const API = apiBase();

export type InterlinkSide = "raw" | "beta";
export type InterlinkMode = "steer" | "ablate";

export interface InterlinkConfig {
  mode: InterlinkMode;
  dose_emotion: string | null;
  alpha: number;
  dose_ramp: number;
  opener: string;
  goal: string;
  first_speaker: InterlinkSide;
  opener_side: InterlinkSide;
  thinking: boolean;
}

export interface InterlinkMessage {
  idx: number;
  side: InterlinkSide;
  text: string;
  thinking: string;
  stopped_reason: string;
  started_at?: number;
  finished_at?: number | null;
}

export interface InterlinkState {
  running: boolean;
  status: string; // idle | running | stopped | done | error
  session_id: string | null;
  config: Partial<InterlinkConfig>;
  opener: string;
  opener_side: InterlinkSide;
  current: { idx: number; side: InterlinkSide } | null;
  messages: InterlinkMessage[];
}

export interface StartParams {
  mode: InterlinkMode;
  doseEmotion: string | null;
  alpha: number;
  doseRamp: number;
  opener: string;
  goal: string;
  firstSpeaker: InterlinkSide;
  thinking: boolean;
}

export async function startInterlink(
  p: StartParams,
): Promise<{ ok: boolean; session_id?: string; error?: string }> {
  const res = await fetch(`${API}/interlink/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mode: p.mode,
      dose_emotion: p.doseEmotion,
      alpha: p.alpha,
      dose_ramp: p.doseRamp,
      opener: p.opener,
      goal: p.goal,
      first_speaker: p.firstSpeaker,
      thinking: p.thinking,
    }),
  });
  return res.json();
}

export async function stopInterlink(): Promise<{ ok: boolean; was_running?: boolean }> {
  const res = await fetch(`${API}/interlink/stop`, { method: "POST" });
  return res.json();
}

export async function fetchInterlinkState(): Promise<InterlinkState | null> {
  try {
    const res = await fetch(`${API}/interlink/state`, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as InterlinkState;
  } catch {
    return null;
  }
}

export async function fetchInterlinkSession(sessionId: string): Promise<{
  session_id: string;
  config: Partial<InterlinkConfig>;
  opener: string;
  opener_side: InterlinkSide;
  messages: InterlinkMessage[];
  status: string;
} | null> {
  try {
    const res = await fetch(`${API}/interlink/sessions/${encodeURIComponent(sessionId)}`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    const j = await res.json();
    return {
      session_id: j.session_id,
      config: j,
      opener: j.opener,
      opener_side: j.first_speaker === "raw" ? "beta" : "raw",
      messages: j.messages ?? [],
      status: j.status,
    };
  } catch {
    return null;
  }
}

export type InterlinkEvent =
  | { type: "message_start"; idx: number; side: InterlinkSide }
  | { type: "interlink_token"; side: InterlinkSide; channel: "thought" | "answer"; decoded: string }
  | ({ type: "message_done" } & InterlinkMessage)
  | { type: "conversation_done"; status: string }
  | { type: "error"; message: string };

export function subscribeInterlink(
  sessionId: string,
  handlers: {
    onEvent: (evt: InterlinkEvent) => void;
    onError?: () => void;
    onClose?: () => void;
  },
): () => void {
  const url = `${API}/interlink/stream/${encodeURIComponent(sessionId)}`;
  const es = new EventSource(url);
  // Custom-typed SSE events are only delivered if there's an addEventListener
  // for the exact name — keep this list in sync with routes_interlink.py.
  const eventTypes = [
    "message_start",
    "interlink_token",
    "message_done",
    "conversation_done",
    "error",
    "ping",
  ];
  for (const t of eventTypes) {
    es.addEventListener(t, (e: MessageEvent) => {
      if (t === "ping") return;
      try {
        const data = JSON.parse(e.data);
        handlers.onEvent({ type: t, ...data } as InterlinkEvent);
        if (t === "conversation_done") {
          es.close();
          handlers.onClose?.();
        }
      } catch {
        /* ignore malformed */
      }
    });
  }
  es.onerror = () => {
    handlers.onError?.();
  };
  return () => es.close();
}
