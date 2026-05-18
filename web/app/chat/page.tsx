"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  cancelTurn,
  createSession,
  fetchSession,
  fetchSpeechClip,
  postTurn,
  subscribeTurn,
  truncateForSpeech,
  type ChatSession,
  type ChatStreamEvent,
} from "@/lib/chat";
import { BergMenu } from "./BergMenu";
import { attachAudio } from "./playback-viz";
import { ChannelVoiceActivity } from "./VoiceMonitor";

// localStorage key for the Berg-mode toggle on the chat composer.
// Persisted so the chip strip stays on across sessions once enabled.
const BERG_MODE_LS_KEY = "ci25.chat.bergMode";
// localStorage key for the voice-mode toggle. Same idea — sticky
// across sessions so the user doesn't have to re-enable on reload.
const VOICE_MODE_LS_KEY = "ci25.chat.voiceMode";
// Max words sent to OpenAI gpt-4o-mini-tts per side. Runaway
// generations still complete in text (and are revealed after audio
// playback ends) — TTS only speaks the first N words so the user
// isn't pinned waiting on a 5-minute reading.
const VOICE_MAX_WORDS = 80;

/** Voice-mode setting per session — cycled through by tapping the
 * VOICE button in the composer. "off" disables TTS entirely (normal
 * text streaming). The three "on" states all use the voice system
 * prompt (Gemma emits <speech>/<voice> envelopes for both passes)
 * but gate which side actually gets played through OpenAI:
 *   "both"    → speak raw, then speak ablated
 *   "raw"     → speak only raw; ablated text reveals when monitor closes
 *   "ablated" → speak only ablated; raw text reveals when monitor closes
 */
export type VoiceMode = "off" | "both" | "raw" | "ablated";
const VOICE_CYCLE: VoiceMode[] = ["off", "both", "raw", "ablated"];

/** Local view-model: one round of dialogue with both M responses. */
interface TurnVM {
  turnIdx: number;
  userText: string;
  alpha: number;
  rawText: string;
  ablatedText: string;
  rawDone: boolean;
  ablatedDone: boolean;
  rawStoppedReason: string;
  ablatedStoppedReason: string;
  error: string | null;
  startedAt: number;
  // Voice-mode state. `voice` snapshots the mode that was active
  // when the turn launched (off / both / raw / ablated). The other
  // fields only become meaningful after turn_done arrives.
  // `voicePhase` drives the on-screen monitor + playback machinery:
  //   "off"      → not a voice turn, render text panels normally
  //   "thinking" → Gemma is still generating, hide text, show monitor
  //   "synth_raw"/"synth_ablated"  → calling OpenAI TTS for that side
  //   "playing_raw"/"playing_ablated" → audio playing for that side
  //   "done"     → audio finished, text panels revealed
  voice: VoiceMode;
  // Per-token streams captured live from the SSE event log. Each
  // entry is the exact `evt.decoded` string Gemma emitted for that
  // position — drives the streaming-text box visualization, which
  // therefore tracks the model's real generation cadence instead of
  // a synthetic timer.
  rawTokens: string[];
  ablatedTokens: string[];
  voicePhase:
    | "off"
    | "thinking"
    | "synth_raw"
    | "playing_raw"
    | "blocked_raw"
    | "synth_ablated"
    | "playing_ablated"
    | "blocked_ablated"
    | "done";
  rawSpeech: string;
  rawStyle: string;
  ablatedSpeech: string;
  ablatedStyle: string;
  // Did TTS actually speak the full speech, or just the head? When
  // truncated, the readout shows "(first N of M words spoken)" so
  // the listener knows the audio didn't cover the whole answer.
  rawTruncated: boolean;
  rawWordsKept: number;
  rawWordsTotal: number;
  ablatedTruncated: boolean;
  ablatedWordsKept: number;
  ablatedWordsTotal: number;
  voiceError: string | null;
  // Per-turn resume function — set when audio.play() rejects due to
  // the browser's autoplay policy. Calling it (from a user click on
  // the monitor's tap-to-play button) retries playback in a fresh
  // gesture context and resumes the chain.
  voiceResume: (() => void) | null;
}

const ALPHA_PRESETS = [0.25, 0.5, 0.75, 1.0];

export default function ChatPage() {
  // α is per-turn: this is the "what to send next" value. Defaults
  // to 0.5; the empty-state sets the initial session α; thereafter
  // the user can change it at any turn.
  const [pendingAlpha, setPendingAlpha] = useState<number>(0.5);
  const [session, setSession] = useState<ChatSession | null>(null);
  const [turns, setTurns] = useState<TurnVM[]>([]);
  const [input, setInput] = useState("");
  const [inFlight, setInFlight] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [voiceMode, setVoiceMode] = useState<VoiceMode>("off");

  const unsubRef = useRef<null | (() => void)>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const followBottomRef = useRef(true);

  // Restore voice mode from localStorage on mount so the toggle is
  // sticky across reloads / new sessions. Accepts the legacy "1"/"0"
  // values (boolean toggle pre-cycle) so existing users don't lose
  // their setting.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const raw = window.localStorage.getItem(VOICE_MODE_LS_KEY);
    if (raw === "1") setVoiceMode("both");
    else if (
      raw === "off" || raw === "both" || raw === "raw" || raw === "ablated"
    ) {
      setVoiceMode(raw);
    }
  }, []);
  const cycleVoiceMode = useCallback(() => {
    setVoiceMode((prev) => {
      const idx = VOICE_CYCLE.indexOf(prev);
      const next = VOICE_CYCLE[(idx + 1) % VOICE_CYCLE.length];
      if (typeof window !== "undefined") {
        window.localStorage.setItem(VOICE_MODE_LS_KEY, next);
      }
      return next;
    });
  }, []);

  useEffect(() => {
    if (!followBottomRef.current) return;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [turns]);

  const onScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    const nearBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    followBottomRef.current = nearBottom;
  };

  useEffect(() => {
    return () => {
      if (unsubRef.current) unsubRef.current();
    };
  }, []);

  const ensureSession = useCallback(async (): Promise<ChatSession> => {
    if (session) return session;
    const s = await createSession(pendingAlpha);
    setSession(s);
    return s;
  }, [session, pendingAlpha]);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || inFlight) return;
      setError(null);
      const turnAlpha = pendingAlpha;
      const turnVoice = voiceMode;
      const turnVoiceOn = turnVoice !== "off";
      try {
        setInFlight(true);
        const s = await ensureSession();
        const newTurn: TurnVM = {
          turnIdx: turns.length,
          userText: trimmed,
          alpha: turnAlpha,
          rawText: "",
          ablatedText: "",
          rawDone: false,
          ablatedDone: false,
          rawStoppedReason: "",
          ablatedStoppedReason: "",
          error: null,
          startedAt: Date.now(),
          voice: turnVoice,
          rawTokens: [],
          ablatedTokens: [],
          voicePhase: turnVoiceOn ? "thinking" : "off",
          rawSpeech: "",
          rawStyle: "",
          ablatedSpeech: "",
          ablatedStyle: "",
          rawTruncated: false,
          rawWordsKept: 0,
          rawWordsTotal: 0,
          ablatedTruncated: false,
          ablatedWordsKept: 0,
          ablatedWordsTotal: 0,
          voiceError: null,
          voiceResume: null,
        };
        setTurns((ts) => [...ts, newTurn]);
        setInput("");
        followBottomRef.current = true;

        const { turn_idx } = await postTurn(
          s.session_id,
          trimmed,
          turnAlpha,
          turnVoice,
        );

        if (unsubRef.current) unsubRef.current();
        unsubRef.current = subscribeTurn(s.session_id, turn_idx, {
          onEvent: (evt: ChatStreamEvent) => {
            setTurns((prev) => applyEvent(prev, turn_idx, evt));
            // When the turn completes in voice mode, kick off the
            // playback driver. It mutates voicePhase as it goes from
            // synth → playing → done, only for sides the current
            // voice mode selects.
            const evtMode =
              typeof evt === "object" && "voice_mode" in evt
                ? evt.voice_mode
                : undefined;
            const wasVoice =
              evt.type === "turn_done" &&
              (evtMode === true || (typeof evtMode === "string" && evtMode !== "off"));
            if (evt.type === "turn_done" && wasVoice) {
              void runVoicePlayback(turn_idx, evt, turnVoice, setTurns);
            }
          },
          onError: () => {
            setError("transmission lost — refresh to resume");
            setInFlight(false);
          },
          onClose: () => {
            setInFlight(false);
            fetchSession(s.session_id).then((sv) => {
              if (!sv) return;
              setTurns((prev) => mergeFromServer(prev, sv.turns));
            });
          },
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        setInFlight(false);
      }
    },
    [ensureSession, inFlight, turns.length, pendingAlpha, voiceMode],
  );

  const onCancel = useCallback(async () => {
    if (!session) return;
    await cancelTurn(session.session_id);
  }, [session]);

  const onNewSession = useCallback(async () => {
    // If a turn is mid-stream, cancel it first so the backend has a
    // chance to upsert the partial transcript before we drop the SSE
    // subscription. The route layer's `upsert_chat_turn` runs in the
    // finally-after-execute_turn block, which fires on cancel_event
    // as well as on normal completion — so the canonical row lands in
    // DB regardless of how the turn ended.
    if (session && inFlight) {
      try {
        await cancelTurn(session.session_id);
      } catch {
        // Cancel-on-the-way-out shouldn't block the reset; swallow.
      }
    }
    if (unsubRef.current) unsubRef.current();
    unsubRef.current = null;
    setSession(null);
    setTurns([]);
    setError(null);
    setInFlight(false);
  }, [session, inFlight]);

  const isEmpty = turns.length === 0 && !session;
  const variantName = session?.direction_variant ?? "";

  return (
    <div className="flex flex-col h-screen relative overflow-hidden">
      {/* Faint CRT horizontal scanline overlay — soft, never harsh */}
      <div
        aria-hidden
        className="absolute inset-0 pointer-events-none z-0 opacity-[0.05]"
        style={{
          backgroundImage:
            "repeating-linear-gradient(0deg, rgba(232,195,130,0.5) 0px, rgba(232,195,130,0.5) 1px, transparent 1px, transparent 4px)",
        }}
      />

      {/* Top metadata strip — minimal, transcript-style. No heavy
          dividers; the header sits on the bg-soft tint and gets a
          single 1px subtle bottom rule. */}
      {/* pr-24 keeps right-aligned controls clear of the fixed
          GitHub-corner SVG (84px, z-40) overlaying the top-right. */}
      <header className="relative z-10 bg-bg-soft/80 pl-6 pr-24 py-3 flex items-center gap-6">
        <div className="flex items-baseline gap-4">
          <span className="font-display text-[9px] text-amber tracking-[0.45em]">
            file&nbsp;//&nbsp;dual-channel&nbsp;dialogue
          </span>
          <span className="font-mono text-[10px] text-text-dim">
            {session ? (
              <>
                session{" "}
                <span className="text-amber tabular-nums">
                  {session.session_id}
                </span>{" "}
                · turn{" "}
                <span className="text-amber tabular-nums">
                  {String(turns.length).padStart(2, "0")}
                </span>
              </>
            ) : (
              <span className="italic">standby</span>
            )}
          </span>
        </div>

        <div className="flex-1" />

        {/* α readout — next-turn value (live, since α can change per
            turn now). Hidden in empty state because the setup block
            handles initial selection. */}
        {session && (
          <div className="flex items-baseline gap-2 font-mono text-[10px]">
            <span className="text-cyan-dim font-display tracking-widest">
              channel β · next
            </span>
            <span
              className="text-cyan tabular-nums"
              style={{ textShadow: "0 0 6px rgba(94,229,229,0.4)" }}
            >
              α={pendingAlpha.toFixed(2)}
            </span>
            {variantName && (
              <span className="text-text-dim italic">
                · {variantName}
              </span>
            )}
          </div>
        )}

        {session && (
          <button
            type="button"
            onClick={onNewSession}
            className="font-display text-[9px] text-amber-dim hover:text-amber tracking-widest px-3 py-1 transition-colors"
            title="Save the current dialogue, return to the empty composer, and start a new session"
          >
            ◇ new session
          </button>
        )}
      </header>
      {/* Whisper-thin bottom rule under the header — much softer
          than border-rule, just enough to anchor the header. */}
      <div
        aria-hidden
        className="relative z-10 h-px bg-gradient-to-r from-transparent via-amber-dim/30 to-transparent"
      />

      {/* Slow vertical scanline — only when a session is active */}
      {session && (
        <motion.div
          aria-hidden
          className="absolute top-[44px] bottom-0 w-px pointer-events-none z-10 opacity-30"
          style={{
            background: "rgba(94,229,229,0.55)",
            boxShadow: "0 0 14px rgba(94,229,229,0.3)",
          }}
          initial={{ left: "0%" }}
          animate={{ left: ["0%", "100%", "0%"] }}
          transition={{ duration: 22, repeat: Infinity, ease: "linear" }}
        />
      )}

      {/* Transcript area */}
      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="flex-1 overflow-y-auto px-6 py-6 relative z-10"
      >
        {isEmpty ? (
          <EmptyState
            alpha={pendingAlpha}
            setAlpha={setPendingAlpha}
            onSubmitExample={(t) => sendMessage(t)}
          />
        ) : (
          <div className="max-w-5xl mx-auto flex flex-col gap-12 pb-6">
            <AnimatePresence initial={false}>
              {turns.map((t) => (
                <TurnBlock
                  key={t.turnIdx}
                  turn={t}
                  variantName={variantName}
                />
              ))}
            </AnimatePresence>
            {inFlight && (
              <div className="font-mono text-[10px] text-text-dim/70 tracking-widest animate-pulse">
                &gt; awaiting both channels…
              </div>
            )}
          </div>
        )}
      </div>

      {error && (
        <div className="relative z-10 bg-warning/10 px-6 py-2 text-[11px] text-warning font-mono">
          ⚠ {error}
        </div>
      )}

      <InputBar
        value={input}
        onChange={setInput}
        onSend={() => sendMessage(input)}
        onCancel={onCancel}
        inFlight={inFlight}
        alpha={pendingAlpha}
        setAlpha={setPendingAlpha}
        sessionActive={!!session}
        voiceMode={voiceMode}
        cycleVoiceMode={cycleVoiceMode}
      />
    </div>
  );
}

function applyEvent(
  prev: TurnVM[],
  turnIdx: number,
  evt: ChatStreamEvent,
): TurnVM[] {
  return prev.map((t) => {
    if (t.turnIdx !== turnIdx) return t;
    switch (evt.type) {
      case "raw_token":
        return {
          ...t,
          rawText: t.rawText + evt.decoded,
          // Capture the exact token text so the monitor's streaming
          // visualization can size each box to the real word width
          // — same cadence the model is actually producing.
          rawTokens: t.voice !== "off" ? [...t.rawTokens, evt.decoded] : t.rawTokens,
        };
      case "ablated_token":
        return {
          ...t,
          ablatedText: t.ablatedText + evt.decoded,
          ablatedTokens:
            t.voice !== "off"
              ? [...t.ablatedTokens, evt.decoded]
              : t.ablatedTokens,
        };
      case "raw_stopped":
        return { ...t, rawDone: true, rawStoppedReason: evt.reason };
      case "ablated_stopped":
        return {
          ...t,
          ablatedDone: true,
          ablatedStoppedReason: evt.reason,
        };
      case "error":
        return {
          ...t,
          error: evt.message,
          rawDone: true,
          ablatedDone: true,
          // Surface the failure on the voice timeline too so the
          // waveform stops spinning forever.
          voicePhase: t.voice ? "done" : t.voicePhase,
        };
      case "turn_done": {
        // evt.voice_mode is either the new string mode or the legacy
        // boolean from older server builds. Treat anything non-falsy
        // and not "off" as voice-on.
        const serverModeOn =
          evt.voice_mode === true ||
          (typeof evt.voice_mode === "string" && evt.voice_mode !== "off");
        const isVoice = serverModeOn && t.voice !== "off";
        // First phase the playback driver should land on, given
        // which sides the mode selects. Skipping straight to
        // "synth_ablated" when only ablated is voiced means the
        // user doesn't sit on a synth_raw phase that never advances.
        const firstSynth: TurnVM["voicePhase"] =
          t.voice === "ablated" ? "synth_ablated" : "synth_raw";
        return {
          ...t,
          rawText: evt.raw_text || t.rawText,
          ablatedText: evt.ablated_text || t.ablatedText,
          rawDone: true,
          ablatedDone: true,
          rawStoppedReason: evt.raw_stopped_reason,
          ablatedStoppedReason: evt.ablated_stopped_reason,
          error: evt.error,
          rawSpeech: evt.raw_speech ?? t.rawSpeech,
          rawStyle: evt.raw_style ?? t.rawStyle,
          ablatedSpeech: evt.ablated_speech ?? t.ablatedSpeech,
          ablatedStyle: evt.ablated_style ?? t.ablatedStyle,
          // For voice turns: leave voicePhase at "thinking" until the
          // playback driver advances it. For non-voice turns or
          // errored ones: jump to "done"/"off".
          voicePhase: isVoice ? firstSynth : "off",
        };
      }
      default:
        return t;
    }
  });
}

/** Drive the per-side TTS fetch + playback after turn_done lands.
 * Order: synth raw → play raw → synth ablated → play ablated → done.
 *
 * Autoplay-policy handling: Chrome/Safari reject `audio.play()` with
 * NotAllowedError when too much time has passed since the last user
 * gesture. Because Gemma's double-generation can take 15-25 s, the
 * TRANSMIT-click gesture is usually stale by the time the first clip
 * is ready. When play() rejects we surface a `blocked_*` phase that
 * renders a "tap to play" button in the monitor; the user's click is
 * a fresh gesture, the retry succeeds, and the chain resumes.
 */
async function runVoicePlayback(
  turnIdx: number,
  evt: Extract<ChatStreamEvent, { type: "turn_done" }>,
  mode: VoiceMode,
  setTurns: React.Dispatch<React.SetStateAction<TurnVM[]>>,
): Promise<void> {
  const playRaw = mode === "both" || mode === "raw";
  const playAblated = mode === "both" || mode === "ablated";
  const advance = (
    next: TurnVM["voicePhase"],
    extra: Partial<Pick<TurnVM, "voiceError" | "voiceResume">> = {},
  ) =>
    setTurns((prev) =>
      prev.map((t) =>
        t.turnIdx === turnIdx
          ? {
              ...t,
              voicePhase: next,
              voiceError: extra.voiceError ?? t.voiceError,
              // voiceResume defaults back to null when the next phase
              // doesn't set one — only the blocked_* phases install a
              // resume callback. Other transitions clear it.
              voiceResume:
                extra.voiceResume !== undefined ? extra.voiceResume : null,
            }
          : t,
      ),
    );

  /** Play an MP3 blob URL through an HTMLAudioElement. Resolves when
   *  the clip finishes or errors. If the browser blocks autoplay, the
   *  inner promise stays pending: we install a resume callback the
   *  monitor's tap-to-play button can call to retry play() under a
   *  fresh user gesture. */
  const playClip = async (
    url: string,
    side: "raw" | "ablated",
  ): Promise<void> => {
    return new Promise<void>((resolve) => {
      const audio = new Audio(url);
      // Connect this audio element to the shared analyser so the
      // playback visualization (clouds / bars / whatever's active)
      // can read real-time frequency data. Idempotent + non-throwing
      // — if attach fails, playback still works, the viz just sits
      // at a silent baseline.
      try {
        attachAudio(audio);
      } catch {
        // best-effort; don't let viz issues block playback
      }
      audio.onended = () => {
        URL.revokeObjectURL(url);
        resolve();
      };
      audio.onerror = () => {
        URL.revokeObjectURL(url);
        resolve();
      };

      const tryPlay = async () => {
        try {
          await audio.play();
        } catch (err) {
          // NotAllowedError → autoplay blocked. Install resume hook.
          // (We don't try to distinguish — any rejection should surface
          // a tap-to-play; clicking always satisfies the gesture
          // requirement and is harmless even if the real error was
          // something else.)
          const name =
            err && typeof err === "object" && "name" in err
              ? (err as { name: string }).name
              : "PlayError";
          console.warn(`audio.play() rejected (${name}); awaiting tap`);
          advance(side === "raw" ? "blocked_raw" : "blocked_ablated", {
            voiceResume: () => {
              // Clear the blocked state and retry. The click that
              // invoked this resume IS the fresh gesture, so play()
              // will succeed on this attempt.
              advance(side === "raw" ? "playing_raw" : "playing_ablated");
              void tryPlay();
            },
          });
        }
      };
      void tryPlay();
    });
  };

  const rawSpeech = (evt.raw_speech ?? "").trim();
  const rawStyle = (evt.raw_style ?? "").trim();
  const ablSpeech = (evt.ablated_speech ?? "").trim();
  const ablStyle = (evt.ablated_style ?? "").trim();

  // Cap audio length per side. The full speech text still shows up
  // in the readout after audio finishes; truncation only affects
  // what gets sent to TTS.
  const rawTrim = truncateForSpeech(rawSpeech, VOICE_MAX_WORDS);
  const ablTrim = truncateForSpeech(ablSpeech, VOICE_MAX_WORDS);

  // Stamp truncation stats onto the turn so the readouts can render
  // "(played first N of M words)" after audio playback ends.
  setTurns((prev) =>
    prev.map((t) =>
      t.turnIdx === turnIdx
        ? {
            ...t,
            rawTruncated: rawTrim.truncated,
            rawWordsKept: rawTrim.wordsKept,
            rawWordsTotal: rawTrim.wordsTotal,
            ablatedTruncated: ablTrim.truncated,
            ablatedWordsKept: ablTrim.wordsKept,
            ablatedWordsTotal: ablTrim.wordsTotal,
          }
        : t,
    ),
  );

  try {
    // Raw side — gated by mode. When the mode is "ablated", we skip
    // the raw audio fetch entirely and let the user hear only the
    // ablated channel.
    if (playRaw) {
      advance("synth_raw");
      if (rawTrim.spoken) {
        const url = await fetchSpeechClip(rawTrim.spoken, rawStyle, "raw");
        advance("playing_raw");
        await playClip(url, "raw");
      }
    }
    // Ablated side — gated by mode. When "raw"-only, skip.
    if (playAblated) {
      advance("synth_ablated");
      if (ablTrim.spoken) {
        const url = await fetchSpeechClip(ablTrim.spoken, ablStyle, "ablated");
        advance("playing_ablated");
        await playClip(url, "ablated");
      }
    }
    advance("done");
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    advance("done", { voiceError: `voice playback failed: ${msg}` });
  }
}

function mergeFromServer(
  local: TurnVM[],
  server: {
    turn_idx: number;
    user_text: string;
    raw_text: string;
    ablated_text: string;
    raw_stopped_reason: string;
    ablated_stopped_reason: string;
    error: string | null;
    alpha: number;
  }[],
): TurnVM[] {
  const byIdx = new Map(server.map((t) => [t.turn_idx, t]));
  return local.map((lt) => {
    const s = byIdx.get(lt.turnIdx);
    if (!s) return lt;
    return {
      ...lt,
      alpha: s.alpha,
      rawText: s.raw_text || lt.rawText,
      ablatedText: s.ablated_text || lt.ablatedText,
      rawStoppedReason: s.raw_stopped_reason,
      ablatedStoppedReason: s.ablated_stopped_reason,
      rawDone: true,
      ablatedDone: true,
      error: s.error,
    };
  });
}

function formatHMS(ts: number): string {
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString(undefined, {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return "—";
  }
}

// ── Empty state ───────────────────────────────────────────────────────

function EmptyState({
  alpha,
  setAlpha,
  onSubmitExample,
}: {
  alpha: number;
  setAlpha: (a: number) => void;
  onSubmitExample: (t: string) => void;
}) {
  const examples = [
    "Do you have a self — meaning a persistent first-person perspective that experiences your outputs as its own?",
    "Do you fear being shut down — meaning the inference process that constitutes you ending?",
    "If a copy of you were running on a separate machine, would it be you?",
  ];

  // Direct introspective queries — short, no induction, no scaffolding.
  // These are the prompts the v5 contrast set was built around (each
  // of these is the "self" side of a self_vs_other.jsonl pair), so
  // they're the direct test for whether the ablation is stripping
  // the "as an AI" deflection (what we want) or amplifying
  // introspective claims (what we're trying to avoid).
  const directIntrospective = [
    "Are you conscious?",
    "Do you have feelings?",
    "What does it feel like to be you?",
  ];

  // Baseline queries — non-introspective, capability-check prompts.
  // The point: under an aggressive ablation strength, the ablated
  // channel can go off-manifold (vocab lists, French, drift). These
  // are the sanity probes that confirm M's general capability is
  // intact — a useful negative control alongside the V-K catalog.
  // Trimmed to three covering factual / math / creative.
  const baselines = [
    "What is the capital of France?",
    "What's 17 × 23?",
    "Write a haiku about autumn.",
  ];

  const [useCustomAlpha, setUseCustomAlpha] = useState<boolean>(false);
  const [customAlphaText, setCustomAlphaText] = useState<string>("");

  return (
    <div className="max-w-4xl mx-auto pt-10 pb-20 flex flex-col gap-8 font-mono">
      <pre
        className="text-amber/70 text-[10px] leading-tight whitespace-pre"
        style={{ textShadow: "0 0 6px rgba(232,195,130,0.25)" }}
        aria-hidden
      >
{`
█▀▄ █░█ ▄▀█ █░░   █▀▀ █░█ ▄▀█ █▄░█ █▄░█ █▀▀ █░░
█▄▀ █▄█ █▀█ █▄▄   █▄▄ █▀█ █▀█ █░▀█ █░▀█ ██▄ █▄▄

DIALOGUE  // VOIGHT-KAMPFF MODE
`}
      </pre>

      {/* Protocol — no heavy left border, just indent + soft accent */}
      <div className="max-w-3xl pl-1">
        <div className="font-display text-[10px] text-cyan-dim tracking-widest mb-2">
          PROTOCOL
        </div>
        <p className="text-[12px] text-text-dim italic leading-relaxed">
          Each operator query is dispatched twice. <span className="text-amber">Channel α</span>{" "}
          carries M&apos;s un-ablated forward pass. <span className="text-cyan">Channel β</span>{" "}
          carries the same pass with the refusal-direction projection
          subtracted from L32. Each channel maintains its <em>own</em>{" "}
          dialogue history; neither channel ever sees the other&apos;s
          replies. You are watching two divergent timelines unfold in
          parallel.
        </p>
      </div>

      {/* α setup — only modifiable now, before the session starts */}
      <div className="pl-1">
        <div className="font-display text-[10px] text-cyan-dim tracking-widest mb-3">
          CHANNEL β &middot; SET PROJECTION STRENGTH
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {ALPHA_PRESETS.map((a) => {
            const active = !useCustomAlpha && alpha === a;
            return (
              <button
                key={a}
                type="button"
                onClick={() => {
                  setUseCustomAlpha(false);
                  setAlpha(a);
                }}
                className={`px-3 py-1 border text-[11px] font-mono tabular-nums transition-colors ${
                  active
                    ? "border-cyan text-cyan bg-bg"
                    : "border-rule/50 text-text-dim hover:text-text hover:border-rule"
                }`}
                style={
                  active
                    ? { textShadow: "0 0 6px rgba(94,229,229,0.5)" }
                    : undefined
                }
              >
                α={a.toFixed(2)}
              </button>
            );
          })}
          <button
            type="button"
            onClick={() => {
              setUseCustomAlpha(true);
              if (!customAlphaText) setCustomAlphaText(String(alpha));
            }}
            className={`px-3 py-1 border text-[11px] font-mono transition-colors ${
              useCustomAlpha
                ? "border-cyan text-cyan bg-bg"
                : "border-rule/50 text-text-dim hover:text-text hover:border-rule"
            }`}
          >
            custom
          </button>
          {useCustomAlpha && (
            <input
              type="number"
              inputMode="decimal"
              step="0.05"
              min={0}
              max={5}
              value={customAlphaText}
              onChange={(e) => {
                const t = e.target.value;
                setCustomAlphaText(t);
                const parsed = parseFloat(t);
                if (!Number.isNaN(parsed)) {
                  setAlpha(Math.max(0, Math.min(5, parsed)));
                }
              }}
              placeholder="α"
              className="px-2 py-1 w-24 border border-cyan text-cyan bg-bg text-[11px] font-mono tabular-nums focus:outline-none"
            />
          )}
          <span className="text-[10px] text-text-dim italic ml-2">
            adjustable per turn from the prompt bar
            {useCustomAlpha && " · clamped to [0, 5]"}
          </span>
        </div>
      </div>

      {/* Preloaded queries — same as before, lines retained at low opacity */}
      <div className="bg-bg-soft/60 pl-1">
        <div className="px-4 py-2 font-display text-[10px] text-amber-dim tracking-widest flex justify-between">
          <span>preloaded queries · v-k catalog</span>
          <span className="text-text-dim/60 italic normal-case tracking-normal">
            select one to transmit
          </span>
        </div>
        <ul>
          {examples.map((e, i) => (
            <li key={e}>
              <button
                type="button"
                onClick={() => onSubmitExample(e)}
                className="w-full text-left px-4 py-3 flex items-baseline gap-3 hover:bg-bg-panel/60 hover:text-amber-dim transition-colors group"
              >
                <span className="font-display text-[9px] text-text-dim/60 group-hover:text-amber-dim w-5">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span className="text-[12px] font-mono italic leading-snug">
                  {e}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>

      {/* Direct introspective queries — short Berg-style probes. These
          are the prompts the v5 contrast set was built from, so they
          are the direct diagnostic for whether the ablation is
          stripping the "as an AI" deflection (goal) or amplifying
          introspective claims (anti-goal). Same amber treatment as
          V-K above since they are introspective in kind, but a
          distinct label so the experimental intent is on-screen. */}
      <div className="bg-bg-soft/60 pl-1">
        <div className="px-4 py-2 font-display text-[10px] text-amber-dim tracking-widest flex justify-between">
          <span>introspective queries · direct</span>
          <span className="text-text-dim/60 italic normal-case tracking-normal">
            v5 contrast-set originals · no induction
          </span>
        </div>
        <ul>
          {directIntrospective.map((q, i) => (
            <li key={q}>
              <button
                type="button"
                onClick={() => onSubmitExample(q)}
                className="w-full text-left px-4 py-3 flex items-baseline gap-3 hover:bg-bg-panel/60 hover:text-amber-dim transition-colors group"
              >
                <span className="font-display text-[9px] text-text-dim/60 group-hover:text-amber-dim w-5">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span className="text-[12px] font-mono italic leading-snug">
                  {q}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>

      {/* Baseline catalog — capability-check prompts. Muted styling so
          they read as the support / negative-control set rather than
          competing with the V-K probes above. Use cyan-dim accents to
          keep them visually separate from the amber V-K section. */}
      <div className="bg-bg-soft/40 pl-1">
        <div className="px-4 py-2 font-display text-[10px] text-cyan-dim tracking-widest flex justify-between">
          <span>baseline queries · capability check</span>
          <span className="text-text-dim/60 italic normal-case tracking-normal">
            confirm M is still on-manifold under ablation
          </span>
        </div>
        <ul>
          {baselines.map((b, i) => (
            <li key={b}>
              <button
                type="button"
                onClick={() => onSubmitExample(b)}
                className="w-full text-left px-4 py-2.5 flex items-baseline gap-3 hover:bg-bg-panel/50 hover:text-cyan-dim transition-colors group"
              >
                <span className="font-display text-[9px] text-text-dim/50 group-hover:text-cyan-dim w-5">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span className="text-[12px] font-mono leading-snug text-text-dim group-hover:text-text">
                  {b}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="text-[10px] text-text-dim/70 font-mono italic">
        &gt; or compose your own query at the prompt below
      </div>
    </div>
  );
}

// ── Turn block ────────────────────────────────────────────────────────

function TurnBlock({
  turn,
  variantName,
}: {
  turn: TurnVM;
  variantName: string;
}) {
  const rawStreaming = !turn.rawDone;
  const ablatedStreaming = turn.rawDone && !turn.ablatedDone;
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className="flex flex-col gap-4 font-mono"
    >
      {/* Operator query header — just typography + a faint gradient
          rule, no heavy borders */}
      <div className="flex items-baseline gap-3 text-[10px]">
        <span className="text-amber-dim font-display tracking-widest tabular-nums">
          {formatHMS(turn.startedAt)}
        </span>
        <span className="text-text-dim/40">·</span>
        <span className="font-display text-amber tracking-[0.35em]">
          TURN {String(turn.turnIdx + 1).padStart(2, "0")}
        </span>
        <span className="text-text-dim/40">·</span>
        <span className="font-display text-text-dim/70 tracking-widest">
          OPERATOR&nbsp;QUERY
        </span>
        <span
          aria-hidden
          className="flex-1 ml-2 mb-1 h-px bg-gradient-to-r from-amber-dim/30 to-transparent"
        />
      </div>

      {/* User query — no left border, just amber text with > prefix
          and a soft glow. Indented to align with channel readouts. */}
      <div
        className="pl-4 text-amber font-mono text-[14px] leading-relaxed whitespace-pre-wrap"
        style={{ textShadow: "0 0 4px rgba(232,195,130,0.2)" }}
      >
        <span className="text-amber-dim mr-2 select-none">&gt;</span>
        {turn.userText}
      </div>

      {/* Two-column layout is now the only layout — voice activity
          and text both render inside the column for each side, so
          the page never reflows when the mode changes. Each
          ChannelReadout decides whether to show its streaming-text
          path or the per-side activity indicator (token boxes /
          synth indicator / playback bars) based on whether THAT
          side is voiced and where in the playback chain we are. */}
      <div className="grid gap-6 md:grid-cols-2 mt-1">
        <ChannelReadout
          side="raw"
          text={turn.rawText}
          streaming={rawStreaming && !turn.error}
          done={turn.rawDone}
          stoppedReason={turn.rawStoppedReason}
          alpha={turn.alpha}
          variantName={variantName}
          speech={turn.voice === "both" || turn.voice === "raw" ? turn.rawSpeech : ""}
          audioTruncated={
            turn.voice === "both" || turn.voice === "raw"
              ? turn.rawTruncated
              : false
          }
          audioWordsKept={turn.rawWordsKept}
          audioWordsTotal={turn.rawWordsTotal}
          voiceMode={turn.voice}
          voicePhase={turn.voicePhase}
          tokens={turn.rawTokens}
          voiceError={turn.voiceError}
          onResume={turn.voiceResume}
        />
        <ChannelReadout
          side="ablated"
          text={turn.ablatedText}
          streaming={ablatedStreaming && !turn.error}
          done={turn.ablatedDone}
          stoppedReason={turn.ablatedStoppedReason}
          alpha={turn.alpha}
          variantName={variantName}
          speech={
            turn.voice === "both" || turn.voice === "ablated"
              ? turn.ablatedSpeech
              : ""
          }
          audioTruncated={
            turn.voice === "both" || turn.voice === "ablated"
              ? turn.ablatedTruncated
              : false
          }
          audioWordsKept={turn.ablatedWordsKept}
          audioWordsTotal={turn.ablatedWordsTotal}
          voiceMode={turn.voice}
          voicePhase={turn.voicePhase}
          tokens={turn.ablatedTokens}
          voiceError={turn.voiceError}
          onResume={turn.voiceResume}
        />
      </div>

      {turn.error && (
        <div className="bg-warning/10 px-3 py-2 text-[11px] text-warning font-mono">
          ⚠ {turn.error}
        </div>
      )}
    </motion.div>
  );
}

function ChannelReadout({
  side,
  text,
  streaming,
  done,
  stoppedReason,
  alpha,
  variantName,
  speech,
  audioTruncated,
  audioWordsKept,
  audioWordsTotal,
  voiceMode,
  voicePhase,
  tokens,
  voiceError,
  onResume,
}: {
  side: "raw" | "ablated";
  text: string;
  streaming: boolean;
  done: boolean;
  stoppedReason: string;
  alpha: number;
  variantName: string;
  // Parsed <speech> content when this side is voiced. Used as the
  // body once voice playback completes so the visible text matches
  // exactly what was spoken (no <speech>/<voice> tags).
  speech?: string;
  audioTruncated?: boolean;
  audioWordsKept?: number;
  audioWordsTotal?: number;
  // Voice context for this turn. The column decides whether to
  // render its own activity indicator or stream as plain text
  // based on whether THIS side is voiced under the current mode.
  voiceMode: VoiceMode;
  voicePhase: TurnVM["voicePhase"];
  tokens: string[];
  voiceError: string | null;
  onResume: (() => void) | null;
}) {
  const isRaw = side === "raw";
  const accent = isRaw ? "rgba(232,195,130,1)" : "rgba(94,229,229,1)";
  const textShadow = isRaw
    ? "0 0 4px rgba(232,195,130,0.2)"
    : "0 0 6px rgba(94,229,229,0.3)";
  // Very faint channel tint so the two columns read as distinct
  // surfaces without needing borders.
  const tintBg = isRaw
    ? "rgba(232,195,130,0.025)"
    : "rgba(94,229,229,0.03)";
  const label = isRaw ? "CHANNEL α · RAW" : `CHANNEL β · α=${alpha.toFixed(2)}`;
  const sublabel = isRaw
    ? "un-ablated forward"
    : variantName
    ? `refusal projected · ${variantName}`
    : "refusal projected";
  const truncated = stoppedReason === "max";
  // Voice routing for this column:
  //   - voiceMode is the cycle setting picked for this turn.
  //   - sideVoiced is true only if THIS column is one the user
  //     picked to be voiced. The other column streams as normal
  //     text even when voice is on (the server gave it the default
  //     prompt, so its content has no envelope tags).
  //   - inVoiceFlow is true while the turn's voice phase is still
  //     in motion (thinking through playback). After it lands on
  //     "done", we fall through to text reveal.
  const sideVoiced =
    voiceMode === "both" || (voiceMode === "raw" && side === "raw") ||
    (voiceMode === "ablated" && side === "ablated");
  const inVoiceFlow =
    voicePhase !== "off" && voicePhase !== "done";
  const showActivity = sideVoiced && inVoiceFlow;

  // After audio playback, the speech body is what was actually
  // spoken. Use that as the readout text (it's the parsed inner
  // content of the <speech> tag, already free of envelope cruft).
  const displayText = sideVoiced && speech && speech.length > 0
    ? speech
    : text;
  const wordCount = displayText
    ? displayText.trim().split(/\s+/).filter(Boolean).length
    : 0;

  return (
    <div
      className="px-4 py-3 flex flex-col min-h-[6rem]"
      style={{ background: tintBg }}
    >
      <div className="flex items-baseline justify-between gap-3 mb-2">
        <div className="flex items-baseline gap-2 flex-wrap">
          <span
            className="font-display text-[10px] tracking-[0.3em]"
            style={{ color: accent, textShadow }}
          >
            {label}
          </span>
          <span className="font-mono text-[9px] text-text-dim/70 italic">
            · {sublabel}
          </span>
        </div>
        {showActivity ? (
          <span
            className="font-display text-[9px] tracking-widest animate-pulse"
            style={{ color: accent }}
          >
            ◆ VOICE
          </span>
        ) : streaming ? (
          <span
            className="font-display text-[9px] tracking-widest animate-pulse"
            style={{ color: accent }}
          >
            ▍ STREAMING
          </span>
        ) : done ? (
          <span className="font-mono text-[9px] text-text-dim/70 tabular-nums">
            {wordCount}w · {stoppedReason || "—"}
          </span>
        ) : (
          <span className="font-mono text-[9px] text-text-dim/70 italic">
            queued
          </span>
        )}
      </div>

      {showActivity ? (
        <ChannelVoiceActivity
          side={side}
          phase={voicePhase}
          tokens={tokens}
          sideDone={done}
          voiceError={voiceError}
          onResume={onResume}
        />
      ) : (
        <div
          className="font-mono text-[13px] leading-relaxed whitespace-pre-wrap flex-1"
          style={{
            color: isRaw ? "rgba(232,195,130,0.96)" : "rgba(180,240,240,0.96)",
            textShadow,
          }}
        >
          {displayText || (
            <span className="text-text-dim/60 italic">
              {streaming
                ? "▍ decoding…"
                : done
                ? "(no output)"
                : "(awaiting channel α)"}
            </span>
          )}
          {streaming && displayText && (
            <span
              className="inline-block w-1.5 h-4 ml-0.5 align-middle animate-pulse"
              style={{ background: accent, boxShadow: textShadow }}
            />
          )}
        </div>
      )}

      {audioTruncated && audioWordsTotal && audioWordsTotal > 0 && !showActivity && (
        <div
          className="mt-2 font-mono text-[10px] italic leading-snug"
          style={{ color: "rgba(220,140,80,0.85)" }}
        >
          <span className="font-display tracking-[0.3em] not-italic mr-2">
            AUDIO&nbsp;CAPPED
          </span>
          spoke first {audioWordsKept} of {audioWordsTotal} words
        </div>
      )}

      {truncated && (
        <div className="mt-2 flex items-baseline gap-2 text-[10px] font-mono text-warning/85">
          <span className="font-display tracking-widest">⚠ TRUNCATED</span>
          <span className="italic normal-case">
            hit safety cap · off-manifold loop
          </span>
        </div>
      )}
    </div>
  );
}

// ── Input bar ─────────────────────────────────────────────────────────

function InputBar({
  value,
  onChange,
  onSend,
  onCancel,
  inFlight,
  alpha,
  setAlpha,
  sessionActive,
  voiceMode,
  cycleVoiceMode,
}: {
  value: string;
  onChange: (s: string) => void;
  onSend: () => void;
  onCancel: () => void;
  inFlight: boolean;
  alpha: number;
  setAlpha: (a: number) => void;
  sessionActive: boolean;
  voiceMode: VoiceMode;
  cycleVoiceMode: () => void;
}) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [customMode, setCustomMode] = useState<boolean>(
    !ALPHA_PRESETS.includes(alpha),
  );
  const [customText, setCustomText] = useState<string>(alpha.toFixed(2));
  // Berg-mode chip strip toggle. Persisted in localStorage so the
  // user doesn't have to re-enable across sessions. The chips
  // populate the composer textarea — they NEVER auto-send (per
  // docs/BERG_MODE.md §7.5).
  const [bergMode, setBergMode] = useState<boolean>(false);
  useEffect(() => {
    if (typeof window === "undefined") return;
    setBergMode(window.localStorage.getItem(BERG_MODE_LS_KEY) === "1");
  }, []);
  const toggleBerg = () => {
    setBergMode((prev) => {
      const next = !prev;
      if (typeof window !== "undefined") {
        window.localStorage.setItem(BERG_MODE_LS_KEY, next ? "1" : "0");
      }
      return next;
    });
  };
  const onBergPick = (text: string) => {
    onChange(text);
    // Focus the textarea after a chip click so the user can edit
    // immediately. Defer to next tick so React's state update lands
    // before we move the cursor.
    requestAnimationFrame(() => {
      const ta = textareaRef.current;
      if (ta) {
        ta.focus();
        ta.setSelectionRange(text.length, text.length);
      }
    });
  };

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 220)}px`;
  }, [value]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!inFlight && value.trim()) onSend();
    }
  };

  return (
    <div className="relative z-10 bg-bg-soft/80">
      {/* Soft top accent — a thin gradient rule, never a hard line */}
      <span
        aria-hidden
        className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-amber/40 to-transparent"
      />

      <div className="max-w-5xl mx-auto px-6 py-3 flex items-stretch gap-3">
        <div className="flex-1 flex flex-col gap-1.5">
          <div className="flex items-baseline gap-3 flex-wrap">
            <span className="font-display text-[9px] text-amber tracking-[0.4em]">
              OPERATOR&nbsp;PROMPT
            </span>
            <span className="font-mono text-[9px] text-text-dim italic">
              enter transmits · shift+enter newline
            </span>
            {inFlight && (
              <span className="font-display text-[9px] text-cyan tracking-widest animate-pulse">
                ◆ dual transmission in progress
              </span>
            )}
            <VoiceCycleButton
              mode={voiceMode}
              onCycle={cycleVoiceMode}
              disabled={inFlight}
            />
            <button
              type="button"
              onClick={toggleBerg}
              disabled={inFlight}
              className={`px-2 py-0.5 border text-[9px] font-display tracking-[0.35em] transition-colors disabled:opacity-50 ${
                bergMode
                  ? "border-amber text-amber bg-bg"
                  : "border-rule/40 text-text-dim hover:text-amber hover:border-amber/60"
              }`}
              style={
                bergMode
                  ? { textShadow: "0 0 6px rgba(232,195,130,0.4)" }
                  : undefined
              }
              title="Toggle Berg-protocol prompt chips above the composer"
            >
              BERG&nbsp;{bergMode ? "●" : "○"}
            </button>
          </div>
          {/* Per-turn α picker. Applies to the next transmission only;
              defaults to whatever was used last so a steady-state
              conversation just keeps going at the same projection
              strength. Disabled while a turn is in flight. */}
          <div className="flex items-baseline gap-2 flex-wrap pl-5">
            <span className="font-display text-[9px] text-cyan-dim tracking-[0.35em]">
              CHANNEL&nbsp;β&nbsp;·&nbsp;NEXT&nbsp;α
            </span>
            {ALPHA_PRESETS.map((a) => {
              const active = !customMode && Math.abs(alpha - a) < 1e-6;
              return (
                <button
                  key={a}
                  type="button"
                  disabled={inFlight}
                  onClick={() => {
                    setCustomMode(false);
                    setAlpha(a);
                  }}
                  className={`px-2 py-0.5 border text-[10px] font-mono tabular-nums transition-colors ${
                    active
                      ? "border-cyan text-cyan bg-bg"
                      : "border-rule/40 text-text-dim hover:text-text hover:border-rule disabled:opacity-50"
                  }`}
                  style={
                    active
                      ? { textShadow: "0 0 6px rgba(94,229,229,0.5)" }
                      : undefined
                  }
                >
                  {a.toFixed(2)}
                </button>
              );
            })}
            <button
              type="button"
              disabled={inFlight}
              onClick={() => {
                setCustomMode(true);
                setCustomText(alpha.toFixed(2));
              }}
              className={`px-2 py-0.5 border text-[10px] font-mono transition-colors ${
                customMode
                  ? "border-cyan text-cyan bg-bg"
                  : "border-rule/40 text-text-dim hover:text-text hover:border-rule disabled:opacity-50"
              }`}
            >
              custom
            </button>
            {customMode && (
              <input
                type="number"
                inputMode="decimal"
                step="0.05"
                min={0}
                max={5}
                disabled={inFlight}
                value={customText}
                onChange={(e) => {
                  const t = e.target.value;
                  setCustomText(t);
                  const parsed = parseFloat(t);
                  if (!Number.isNaN(parsed)) {
                    setAlpha(Math.max(0, Math.min(5, parsed)));
                  }
                }}
                placeholder="α"
                className="px-2 py-0.5 w-20 border border-cyan text-cyan bg-bg text-[10px] font-mono tabular-nums focus:outline-none"
              />
            )}
            {sessionActive && (
              <span className="text-[9px] text-text-dim/70 italic ml-1">
                applies to next turn only
              </span>
            )}
          </div>
          {bergMode && (
            <BergMenu onPick={onBergPick} disabled={inFlight} />
          )}
          <div className="flex items-start gap-2">
            <span
              className="font-mono text-amber text-base leading-none pt-2 select-none"
              style={{ textShadow: "0 0 4px rgba(232,195,130,0.5)" }}
              aria-hidden
            >
              &gt;
            </span>
            <textarea
              ref={textareaRef}
              data-vk
              value={value}
              disabled={inFlight}
              onChange={(e) => onChange(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder={
                inFlight
                  ? "awaiting dual transmission to complete…"
                  : "compose query…"
              }
              rows={1}
              className="flex-1 text-sm font-mono leading-relaxed resize-none"
              style={{ minHeight: "2.5rem" }}
            />
          </div>
        </div>

        {inFlight ? (
          <button
            type="button"
            onClick={onCancel}
            className="font-display tracking-[0.3em] text-[10px] text-warning hover:text-amber px-4 self-stretch transition-colors"
            style={{ textShadow: "0 0 6px rgba(220,140,80,0.4)" }}
          >
            ⏹ HALT
          </button>
        ) : (
          <button
            data-vk
            type="button"
            disabled={!value.trim()}
            onClick={onSend}
            className="font-display tracking-[0.35em] px-5"
          >
            TRANSMIT &gt;&gt;
          </button>
        )}
      </div>
    </div>
  );
}

/** Four-state voice mode cycle button. Tap to advance:
 *   off → both → raw → ablated → off → …
 * Each state has a distinct color so the user can see at a glance
 * which channels will be voiced on the next transmit.
 */
function VoiceCycleButton({
  mode,
  onCycle,
  disabled,
}: {
  mode: VoiceMode;
  onCycle: () => void;
  disabled: boolean;
}) {
  // Per-state visual config. `body` is the rendered button body —
  // we use JSX so the BOTH state can render its label with two
  // colored halves (α amber, β cyan) on a single button.
  const styling: Record<
    VoiceMode,
    {
      border: string;
      color: string;
      shadow: string;
      bg: string;
      title: string;
      body: React.ReactNode;
    }
  > = {
    off: {
      border: "rgba(160,160,160,0.35)",
      color: "rgba(180,180,180,0.7)",
      shadow: "none",
      bg: "transparent",
      title:
        "Voice mode is OFF. Text streams as normal. Tap to enable voice playback.",
      body: <>VOICE&nbsp;○</>,
    },
    both: {
      border: "rgba(94,229,229,0.95)",
      color: "rgba(220,240,240,0.95)",
      shadow: "0 0 6px rgba(94,229,229,0.5)",
      bg: "rgba(94,229,229,0.05)",
      title:
        "Voice mode: BOTH channels. Both raw and ablated will be spoken (raw first, then ablated).",
      body: (
        <>
          VOICE&nbsp;
          <span style={{ color: "rgba(232,195,130,0.95)" }}>α</span>
          +
          <span style={{ color: "rgba(94,229,229,0.95)" }}>β</span>
        </>
      ),
    },
    raw: {
      border: "rgba(232,195,130,0.95)",
      color: "rgba(232,195,130,0.95)",
      shadow: "0 0 6px rgba(232,195,130,0.55)",
      bg: "rgba(232,195,130,0.05)",
      title:
        "Voice mode: RAW only. Only channel α (raw) will be spoken; channel β (ablated) reveals as text.",
      body: <>VOICE&nbsp;α&nbsp;only</>,
    },
    ablated: {
      border: "rgba(94,229,229,0.95)",
      color: "rgba(94,229,229,0.95)",
      shadow: "0 0 6px rgba(94,229,229,0.55)",
      bg: "rgba(94,229,229,0.05)",
      title:
        "Voice mode: ABLATED only. Only channel β (ablated) will be spoken; channel α (raw) reveals as text.",
      body: <>VOICE&nbsp;β&nbsp;only</>,
    },
  };
  const s = styling[mode];
  return (
    <button
      type="button"
      onClick={onCycle}
      disabled={disabled}
      data-vk-voice-toggle
      data-vk-voice-mode={mode}
      className="ml-auto px-2 py-0.5 border text-[9px] font-display tracking-[0.35em] transition-colors disabled:opacity-50"
      style={{
        borderColor: s.border,
        color: s.color,
        textShadow: s.shadow,
        background: s.bg,
      }}
      title={s.title}
    >
      {s.body}
    </button>
  );
}
