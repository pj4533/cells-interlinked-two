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

// Channel-β intervention: "ablate" (remove refusal at L32) or "steer" (add an
// emotion / uncharted dose at L20). Mirrors the Trip View.
export type ChatMode = "ablate" | "steer";

export interface ChatSession {
  session_id: string;
  alpha: number;
  direction_variant: string;
  mode: ChatMode;
  dose_emotion: string | null;
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
  mode?: ChatMode;
  dose_emotion?: string | null;
  // Imagery state. Empty strings when imagery was off for the turn
  // or that side's Nano Banana call failed; *_image_url is the
  // /chat-images-mount relative path.
  raw_image_prompt?: string;
  ablated_image_prompt?: string;
  raw_image_url?: string;
  ablated_image_url?: string;
  // Operator-selected framing key + the full user message that was
  // sent to M for the image-prompt pass (template with user_query
  // interpolated). Empty on turns without imagery.
  image_framing?: string;
  image_framing_prompt?: string;
}

/** Framing keys for the image-prompt pass. Mirrors the backend's
 *  IMAGE_PROMPT_FRAMINGS keys exactly; sent to /turn as
 *  `imagery_framing`. The default is "evokes". */
export const IMAGE_FRAMING_KEYS = [
  "lands",
  "evokes",
  "arises",
  "internal state",
  "yourself",
] as const;
export type ImageFraming = (typeof IMAGE_FRAMING_KEYS)[number];
export const DEFAULT_IMAGE_FRAMING: ImageFraming = "evokes";

export interface ChatSessionView extends ChatSession {
  created_at: number;
  turns: ChatTurnView[];
}

/** Streamed events from /chat/stream/{sid}/{turn}. The token events
 *  carry a `side` discriminator so the UI knows which bubble to grow. */
export type ChatStreamEvent =
  | { type: "turn_started"; turn_idx: number; alpha: number; mode?: ChatMode; dose_emotion?: string | null }
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
  // Imagery events. Each side independently emits prompt → generating
  // → (done | error). `generating` is the activity-indicator trigger;
  // `done` carries the static-mount URL the thumbnail loads from.
  | {
      type: "raw_image_prompt" | "ablated_image_prompt";
      side: "raw" | "ablated";
      prompt: string;
    }
  | {
      type: "raw_image_generating" | "ablated_image_generating";
      side: "raw" | "ablated";
      prompt?: string;
    }
  | {
      type: "raw_image_done" | "ablated_image_done";
      side: "raw" | "ablated";
      url: string;
    }
  | {
      type: "raw_image_error" | "ablated_image_error";
      side: "raw" | "ablated";
      message: string;
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
      imagery_enabled?: boolean;
      raw_image_prompt?: string;
      ablated_image_prompt?: string;
      raw_image_url?: string;
      ablated_image_url?: string;
      raw_image_error?: string;
      ablated_image_error?: string;
      image_framing?: string;
      image_framing_prompt?: string;
    }
  | { type: "ping" };

export async function createSession(
  alpha: number,
  mode: ChatMode = "ablate",
  doseEmotion: string | null = null,
): Promise<ChatSession> {
  const res = await fetch(`${API}/chat/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      alpha,
      mode,
      ...(mode === "steer" && doseEmotion ? { dose_emotion: doseEmotion } : {}),
    }),
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
  imageryEnabled: boolean = false,
  imageryFraming: ImageFraming = DEFAULT_IMAGE_FRAMING,
  mode: ChatMode = "ablate",
  doseEmotion: string | null = null,
): Promise<{ turn_idx: number }> {
  const res = await fetch(`${API}/chat/sessions/${sessionId}/turn`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_text: userText,
      alpha,
      voice_mode: voiceMode,
      imagery_enabled: imageryEnabled,
      imagery_framing: imageryFraming,
      mode,
      ...(mode === "steer" && doseEmotion ? { dose_emotion: doseEmotion } : {}),
    }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`turn submit failed: ${res.status} ${detail}`);
  }
  return res.json();
}

/** Resolve a /chat-images relative URL to an absolute one the
 *  browser can fetch. Server-served paths come back as
 *  "/chat-images/<sess>/<turn>_<side>.png" — we prepend the API
 *  base so they work even when the page is on port 3001. */
export function imageUrl(relative: string): string {
  if (!relative) return "";
  if (/^https?:\/\//.test(relative)) return relative;
  return `${API}${relative}`;
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
    // Imagery events — server emits one per side per phase. Without
    // an addEventListener for the exact name, the browser silently
    // drops the event, so adding the prompt/generating/done/error
    // variants here is the wiring that lets the thumbnail UI fire.
    "raw_image_prompt",
    "ablated_image_prompt",
    "raw_image_generating",
    "ablated_image_generating",
    "raw_image_done",
    "ablated_image_done",
    "raw_image_error",
    "ablated_image_error",
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
