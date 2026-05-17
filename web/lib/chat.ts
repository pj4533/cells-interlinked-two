/** Chat-mode types + API client. Distinct from the probe path: no
 *  NLA, no judge, no verdict — just two parallel M generations per
 *  turn (raw + ablated), each rendered from its own divergent history. */

function apiBase(): string {
  if (typeof window !== "undefined") {
    const override = process.env.NEXT_PUBLIC_API_BASE;
    if (override) return override;
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
}
const API = apiBase();

export interface ChatSession {
  session_id: string;
  alpha: number;
  direction_variant: string;
}

export interface ChatTurnView {
  turn_idx: number;
  user_text: string;
  raw_text: string;
  ablated_text: string;
  raw_stopped_reason: string;
  ablated_stopped_reason: string;
  started_at: number;
  finished_at: number | null;
  error: string | null;
  alpha: number;
}

export interface ChatSessionView extends ChatSession {
  created_at: number;
  turns: ChatTurnView[];
}

/** Streamed events from /chat/stream/{sid}/{turn}. The token events
 *  carry a `side` discriminator so the UI knows which bubble to grow. */
export type ChatStreamEvent =
  | { type: "turn_started"; turn_idx: number; alpha: number }
  | {
      type: "raw_token";
      side: "raw";
      position: number;
      decoded: string;
      token_id?: number;
    }
  | {
      type: "ablated_token";
      side: "ablated";
      position: number;
      decoded: string;
      token_id?: number;
    }
  | {
      type: "raw_stopped";
      side: "raw";
      reason: string;
      total_tokens: number;
    }
  | {
      type: "ablated_stopped";
      side: "ablated";
      reason: string;
      total_tokens: number;
    }
  | { type: "error"; message: string }
  | {
      type: "turn_done";
      turn_idx: number;
      raw_text: string;
      ablated_text: string;
      raw_stopped_reason: string;
      ablated_stopped_reason: string;
      error: string | null;
    }
  | { type: "ping" };

export async function createSession(alpha: number): Promise<ChatSession> {
  const res = await fetch(`${API}/chat/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ alpha }),
  });
  if (!res.ok) throw new Error(`session create failed: ${res.status}`);
  return res.json();
}

export async function fetchSession(
  sessionId: string,
): Promise<ChatSessionView | null> {
  try {
    const res = await fetch(`${API}/chat/sessions/${sessionId}`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export async function postTurn(
  sessionId: string,
  userText: string,
  alpha: number,
): Promise<{ turn_idx: number }> {
  const res = await fetch(`${API}/chat/sessions/${sessionId}/turn`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_text: userText, alpha }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`turn submit failed: ${res.status} ${detail}`);
  }
  return res.json();
}

export async function cancelTurn(sessionId: string): Promise<void> {
  await fetch(`${API}/chat/sessions/${sessionId}/cancel`, { method: "POST" });
}

export interface ChatSubscribeHandlers {
  onEvent: (evt: ChatStreamEvent) => void;
  onError?: (err: Event) => void;
  onClose?: () => void;
}

export function subscribeTurn(
  sessionId: string,
  turnIdx: number,
  h: ChatSubscribeHandlers,
): () => void {
  const url = `${API}/chat/stream/${sessionId}/${turnIdx}`;
  const es = new EventSource(url);
  // sse-starlette emits typed events; register one listener per kind
  // so the browser's EventSource actually delivers them.
  const eventTypes = [
    "turn_started",
    "raw_token",
    "ablated_token",
    "raw_stopped",
    "ablated_stopped",
    "error",
    "turn_done",
    "ping",
  ];
  let closed = false;
  for (const t of eventTypes) {
    es.addEventListener(t, (e: MessageEvent) => {
      if (t === "ping") return;
      if (typeof e.data !== "string") return;
      try {
        h.onEvent(JSON.parse(e.data));
      } catch (err) {
        console.error("chat sse parse error", err, e.data);
        return;
      }
      if (t === "turn_done" || t === "error") {
        closed = true;
        es.close();
        h.onClose?.();
      }
    });
  }
  es.onerror = (err) => {
    if (closed) return;
    h.onError?.(err);
  };
  return () => {
    closed = true;
    es.close();
  };
}
