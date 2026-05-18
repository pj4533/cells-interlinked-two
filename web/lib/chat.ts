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
      voice_mode?: VoiceModeWire | boolean;
      raw_speech?: string;
      raw_style?: string;
      ablated_speech?: string;
      ablated_style?: string;
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

export type VoiceModeWire = "off" | "both" | "raw" | "ablated";

export async function postTurn(
  sessionId: string,
  userText: string,
  alpha: number,
  voiceMode: VoiceModeWire = "off",
): Promise<{ turn_idx: number }> {
  const res = await fetch(`${API}/chat/sessions/${sessionId}/turn`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_text: userText,
      alpha,
      voice_mode: voiceMode,
    }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`turn submit failed: ${res.status} ${detail}`);
  }
  return res.json();
}

/** Cap TTS input so a runaway generation can't tie up the playback
 *  pipeline for minutes. We let the model emit whatever it wants in
 *  `<speech>` and reveal the full text after audio finishes — but
 *  only the first N words actually get spoken.
 *
 *  Tries to cut at the last sentence boundary inside the budget so
 *  the audio doesn't end mid-clause. Falls back to a hard word-cap
 *  with a trailing ellipsis if no sentence-end lands in range. */
export function truncateForSpeech(
  text: string,
  maxWords = 80,
): { spoken: string; truncated: boolean; wordsKept: number; wordsTotal: number } {
  const trimmed = (text || "").trim();
  if (!trimmed) {
    return { spoken: "", truncated: false, wordsKept: 0, wordsTotal: 0 };
  }
  const words = trimmed.split(/\s+/);
  if (words.length <= maxWords) {
    return {
      spoken: trimmed,
      truncated: false,
      wordsKept: words.length,
      wordsTotal: words.length,
    };
  }
  // Within the budget, pick the latest sentence-ending punctuation.
  const head = words.slice(0, maxWords).join(" ");
  // Scan all `.!?` characters in the head and keep the index of the
  // latest one followed by whitespace or end-of-string. Linear-time,
  // no ES2018 regex flags required.
  let sentenceEnd = -1;
  for (let i = 0; i < head.length; i++) {
    const c = head[i];
    if (c !== "." && c !== "!" && c !== "?") continue;
    const next = head[i + 1];
    if (next === undefined || /\s/.test(next)) {
      sentenceEnd = i;
    }
  }
  if (sentenceEnd > 0 && sentenceEnd > head.length * 0.4) {
    // Use the sentence-end only if it captures more than ~40% of
    // the budget — otherwise the cut feels too short.
    const spoken = head.slice(0, sentenceEnd + 1);
    return {
      spoken,
      truncated: true,
      wordsKept: spoken.split(/\s+/).length,
      wordsTotal: words.length,
    };
  }
  // No usable sentence boundary — hard-cap with ellipsis so the TTS
  // delivery sounds intentional rather than chopped.
  return {
    spoken: head + "…",
    truncated: true,
    wordsKept: maxWords,
    wordsTotal: words.length,
  };
}

/** Server-side TTS proxy: returns an MP3 blob URL the browser can
 * feed straight into an <audio> element. The OpenAI key never
 * leaves the server. */
export async function fetchSpeechClip(
  text: string,
  style: string,
  side: "raw" | "ablated",
): Promise<string> {
  const res = await fetch(`${API}/tts/speak`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, style, side }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`tts failed: ${res.status} ${detail}`);
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
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
