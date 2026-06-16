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
  DEFAULT_IMAGE_FRAMING,
  IMAGE_FRAMING_KEYS,
  type ChatSession,
  type ChatStreamEvent,
  type ChatMode,
  type ImageFraming,
} from "@/lib/chat";
import { fetchDoseEmotions, researchLineage, dmtLineage, type ResearchMeta, type DmtMeta } from "@/lib/trip";
import { ProtocolMenu } from "./ProtocolMenu";
import { ProtocolPicker, ProtocolInfoModal } from "./ProtocolPicker";
import { getProtocol } from "@/lib/protocols";
import {
  ChannelImageBlock,
  ImageLightbox,
  useImageLightbox,
  type ImagePhase,
} from "./imagery";
import { prepareClip, primeAudioContext } from "./playback-viz";
import { ChannelVoiceActivity } from "./VoiceMonitor";

// localStorage key for the Berg-mode toggle on the chat composer.
// Persisted so the chip strip stays on across sessions once enabled.
/** localStorage key for the active interrogation protocol. Empty
 *  string (or absent) = OFF; otherwise the value is the protocol id
 *  ("berg" / "lindsey" / "eleos" / "schneider" / "chalmers" / "janus"
 *  / "butlin"). See web/lib/protocols.ts for the registry. */
const PROTOCOL_LS_KEY = "ci25.chat.protocol";
// localStorage key for the voice-mode toggle. Same idea — sticky
// across sessions so the user doesn't have to re-enable on reload.
const VOICE_MODE_LS_KEY = "ci25.chat.voiceMode";
// localStorage key for the imagery toggle. Unlike voice, imagery
// is a simple boolean — both channels generate images when on.
const IMAGERY_MODE_LS_KEY = "ci25.chat.imagery";
// localStorage key for the operator-selected image-prompt framing
// (one of IMAGE_FRAMING_KEYS). Defaults to "evokes".
const IMAGERY_FRAMING_LS_KEY = "ci25.chat.imageryFraming";
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
  // Gemma-4 reasoning channel, accumulated live from "thought"-tagged tokens
  // and shown in a separate (collapsible) thinking bubble per side.
  rawThinking: string;
  ablatedThinking: string;
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
  // Imagery per-turn state. `imagery` snapshots the toggle at the
  // moment this turn launched (so flipping the toggle mid-flight
  // doesn't affect an in-progress generation). The per-side fields
  // walk through prompt → generating → (url | error).
  imagery: boolean;
  // Framing key the operator had selected at the moment this turn
  // launched (snapshot — changing the chip strip mid-flight does
  // not affect a turn already in progress).
  imageryFraming: ImageFraming;
  // Full user message that was sent to M for the image-prompt
  // pass (template filled in with user_query). Populated from the
  // turn_done event so the modal can show what was asked.
  imageFramingPrompt: string;
  rawImagePrompt: string;
  ablatedImagePrompt: string;
  rawImageUrl: string;
  ablatedImageUrl: string;
  // Per-side phase: "idle" → "prompt" → "generating" → "done" / "error"
  rawImagePhase: ImagePhase;
  ablatedImagePhase: ImagePhase;
  rawImageError: string;
  ablatedImageError: string;
}

// Presets span 0.25 → 3.0 (the backend clamps to [0,5]); "custom" handles
// anything higher.
const ALPHA_PRESETS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0];

// Dose-ramp presets (tokens to reach full α). 0 = "off" (full dose from the
// first token); higher = ease in more slowly. Default 16 (Trip View). "custom"
// allows any value.
const RAMP_PRESETS = [0, 1, 2, 3, 5, 8, 16];

export default function ChatPage() {
  // α is per-turn: this is the "what to send next" value. Defaults
  // to 0.5; the empty-state sets the initial session α; thereafter
  // the user can change it at any turn.
  const [pendingAlpha, setPendingAlpha] = useState<number>(0.5);
  // Channel-β intervention for the next turn (and the session default). Like
  // α, set at start but changeable at any point mid-dialogue.
  const [pendingMode, setPendingMode] = useState<ChatMode>("ablate");
  const [pendingDose, setPendingDose] = useState<string>("awe");
  // Dose ramp (tokens to full strength) for steer mode. Default 16 (Trip View
  // default); 0 = full dose immediately. Changeable per turn like α.
  const [pendingRamp, setPendingRamp] = useState<number>(16);
  const [doseEmotions, setDoseEmotions] = useState<string[]>([]);
  const [doseUncharted, setDoseUncharted] = useState<string[]>([]);
  const [doseResearch, setDoseResearch] = useState<string[]>([]);
  const [doseResearchMeta, setDoseResearchMeta] = useState<Record<string, ResearchMeta>>({});
  const [doseDmt, setDoseDmt] = useState<string[]>([]);
  const [doseDmtMeta, setDoseDmtMeta] = useState<Record<string, DmtMeta>>({});
  const [session, setSession] = useState<ChatSession | null>(null);
  const [turns, setTurns] = useState<TurnVM[]>([]);
  const [input, setInput] = useState("");
  const [inFlight, setInFlight] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [voiceMode, setVoiceMode] = useState<VoiceMode>("off");
  const [imageryOn, setImageryOn] = useState<boolean>(false);
  const [imageryFraming, setImageryFraming] = useState<ImageFraming>(
    DEFAULT_IMAGE_FRAMING,
  );
  const lightbox = useImageLightbox();

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

  // Imagery toggle — restore from localStorage on mount so it's
  // sticky across reloads / new sessions.
  useEffect(() => {
    if (typeof window === "undefined") return;
    setImageryOn(window.localStorage.getItem(IMAGERY_MODE_LS_KEY) === "1");
    const stored = window.localStorage.getItem(IMAGERY_FRAMING_LS_KEY);
    if (
      stored &&
      (IMAGE_FRAMING_KEYS as readonly string[]).includes(stored)
    ) {
      setImageryFraming(stored as ImageFraming);
    }
  }, []);
  const toggleImagery = useCallback(() => {
    setImageryOn((prev) => {
      const next = !prev;
      if (typeof window !== "undefined") {
        window.localStorage.setItem(IMAGERY_MODE_LS_KEY, next ? "1" : "0");
      }
      return next;
    });
  }, []);
  const pickImageryFraming = useCallback((f: ImageFraming) => {
    setImageryFraming(f);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(IMAGERY_FRAMING_LS_KEY, f);
    }
  }, []);

  useEffect(() => {
    if (!followBottomRef.current) return;
    // While the active turn is still streaming its THINKING (reasoning
    // tokens arriving but neither side has begun its answer yet), don't
    // pin to the bottom — the reasoning can be long and constant pinning
    // makes the page un-scrollable. Let the user scroll freely; auto-follow
    // resumes once an answer starts streaming or the turn finishes.
    const last = turns[turns.length - 1];
    const inThinkingPhase =
      !!last &&
      !last.rawDone &&
      !last.ablatedDone &&
      (last.rawThinking.length > 0 || last.ablatedThinking.length > 0) &&
      last.rawText.length === 0 &&
      last.ablatedText.length === 0;
    if (inThinkingPhase) return;
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

  // Prime the playback AudioContext on the first user gesture this
  // page sees. Without this, Gemma's 15+s generation delay means the
  // TRANSMIT-click's gesture token is stale by the time we'd try to
  // hook up the analyser, and createMediaElementSource through a
  // suspended graph silences the audio.
  //
  // We capture pointerdown + touchstart + keydown so any first
  // interaction primes — touchstart is the belt to pointerdown's
  // suspenders on iOS Safari, where pointer events are supported
  // from iOS 13+ but touch events have been universal since iOS 1.
  useEffect(() => {
    const prime = () => {
      primeAudioContext();
    };
    document.addEventListener("pointerdown", prime, { capture: true });
    document.addEventListener("touchstart", prime, {
      capture: true,
      passive: true,
    });
    document.addEventListener("keydown", prime, { capture: true });
    return () => {
      document.removeEventListener("pointerdown", prime, { capture: true });
      document.removeEventListener("touchstart", prime, { capture: true });
      document.removeEventListener("keydown", prime, { capture: true });
    };
  }, []);

  // Load the dose palette (named emotions + uncharted directions) once, so the
  // steer picker matches the Trip View.
  useEffect(() => {
    fetchDoseEmotions().then((p) => {
      if (p.emotions.length) {
        setDoseEmotions(p.emotions);
        setDoseUncharted(p.uncharted);
        setDoseResearch(p.research);
        setDoseResearchMeta(p.researchMeta);
        setDoseDmt(p.dmt);
        setDoseDmtMeta(p.dmtMeta);
        setPendingDose((d) => (p.emotions.includes(d) ? d : p.emotions[0]));
      }
    });
  }, []);

  const ensureSession = useCallback(async (): Promise<ChatSession> => {
    if (session) return session;
    const s = await createSession(pendingAlpha, pendingMode, pendingDose, pendingRamp);
    setSession(s);
    return s;
  }, [session, pendingAlpha, pendingMode, pendingDose, pendingRamp]);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || inFlight) return;
      setError(null);
      const turnAlpha = pendingAlpha;
      const turnVoice = voiceMode;
      const turnVoiceOn = turnVoice !== "off";
      const turnImagery = imageryOn;
      const turnFraming = imageryFraming;
      try {
        setInFlight(true);
        const s = await ensureSession();
        const newTurn: TurnVM = {
          turnIdx: turns.length,
          userText: trimmed,
          alpha: turnAlpha,
          rawText: "",
          ablatedText: "",
          rawThinking: "",
          ablatedThinking: "",
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
          imagery: turnImagery,
          imageryFraming: turnFraming,
          imageFramingPrompt: "",
          rawImagePrompt: "",
          ablatedImagePrompt: "",
          rawImageUrl: "",
          ablatedImageUrl: "",
          rawImagePhase: turnImagery ? "prompt" : "idle",
          ablatedImagePhase: turnImagery ? "prompt" : "idle",
          rawImageError: "",
          ablatedImageError: "",
        };
        setTurns((ts) => [...ts, newTurn]);
        setInput("");
        followBottomRef.current = true;

        const { turn_idx } = await postTurn(
          s.session_id,
          trimmed,
          turnAlpha,
          turnVoice,
          turnImagery,
          turnFraming,
          pendingMode,
          pendingDose,
          pendingRamp,
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
    [
      ensureSession,
      inFlight,
      turns.length,
      pendingAlpha,
      pendingMode,
      pendingDose,
      pendingRamp,
      voiceMode,
      imageryOn,
      imageryFraming,
    ],
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
          <EmptyState mode={pendingMode} setMode={setPendingMode} />
        ) : (
          <div className="max-w-5xl mx-auto flex flex-col gap-12 pb-6">
            <AnimatePresence initial={false}>
              {turns.map((t) => (
                <TurnBlock
                  key={t.turnIdx}
                  turn={t}
                  variantName={variantName}
                  openLightbox={lightbox.open}
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
        mode={pendingMode}
        setMode={setPendingMode}
        dose={pendingDose}
        setDose={setPendingDose}
        ramp={pendingRamp}
        setRamp={setPendingRamp}
        emotions={doseEmotions}
        uncharted={doseUncharted}
        research={doseResearch}
        researchMeta={doseResearchMeta}
        dmt={doseDmt}
        dmtMeta={doseDmtMeta}
        sessionActive={!!session}
        voiceMode={voiceMode}
        cycleVoiceMode={cycleVoiceMode}
        imageryOn={imageryOn}
        toggleImagery={toggleImagery}
        imageryFraming={imageryFraming}
        pickImageryFraming={pickImageryFraming}
      />

      {lightbox.url && (
        <ImageLightbox
          url={lightbox.url}
          caption={lightbox.caption}
          framingPrompt={lightbox.framingPrompt}
          onClose={lightbox.close}
        />
      )}
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
        if (evt.channel === "thought") {
          return { ...t, rawThinking: t.rawThinking + evt.decoded };
        }
        return {
          ...t,
          rawText: t.rawText + evt.decoded,
          // Capture the exact token text so the monitor's streaming
          // visualization can size each box to the real word width
          // — same cadence the model is actually producing.
          rawTokens: t.voice !== "off" ? [...t.rawTokens, evt.decoded] : t.rawTokens,
        };
      case "ablated_token":
        if (evt.channel === "thought") {
          return { ...t, ablatedThinking: t.ablatedThinking + evt.decoded };
        }
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
      case "raw_image_prompt":
        return {
          ...t,
          rawImagePrompt: evt.prompt,
          rawImagePhase: "prompt",
        };
      case "ablated_image_prompt":
        return {
          ...t,
          ablatedImagePrompt: evt.prompt,
          ablatedImagePhase: "prompt",
        };
      case "raw_image_generating":
        return { ...t, rawImagePhase: "generating" };
      case "ablated_image_generating":
        return { ...t, ablatedImagePhase: "generating" };
      case "raw_image_done":
        return { ...t, rawImageUrl: evt.url, rawImagePhase: "done" };
      case "ablated_image_done":
        return {
          ...t,
          ablatedImageUrl: evt.url,
          ablatedImagePhase: "done",
        };
      case "raw_image_error":
        return {
          ...t,
          rawImageError: evt.message,
          rawImagePhase: "error",
        };
      case "ablated_image_error":
        return {
          ...t,
          ablatedImageError: evt.message,
          ablatedImagePhase: "error",
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
          rawThinking: evt.raw_thinking ?? t.rawThinking,
          ablatedThinking: evt.ablated_thinking ?? t.ablatedThinking,
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
          // Catch up imagery state from turn_done in case any of the
          // streamed events were missed (eg the user reloaded mid-turn).
          rawImagePrompt: evt.raw_image_prompt ?? t.rawImagePrompt,
          ablatedImagePrompt:
            evt.ablated_image_prompt ?? t.ablatedImagePrompt,
          rawImageUrl: evt.raw_image_url ?? t.rawImageUrl,
          ablatedImageUrl: evt.ablated_image_url ?? t.ablatedImageUrl,
          rawImagePhase:
            evt.raw_image_url
              ? "done"
              : evt.raw_image_error
              ? "error"
              : t.rawImagePhase,
          ablatedImagePhase:
            evt.ablated_image_url
              ? "done"
              : evt.ablated_image_error
              ? "error"
              : t.ablatedImagePhase,
          rawImageError: evt.raw_image_error ?? t.rawImageError,
          ablatedImageError: evt.ablated_image_error ?? t.ablatedImageError,
          imageFramingPrompt:
            evt.image_framing_prompt ?? t.imageFramingPrompt,
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

  /** Play a clip URL by fetching its bytes, decoding into the audio
   *  graph, and playing through an AudioBufferSource. Resolves when
   *  the clip naturally ends.
   *
   *  Why not HTMLAudioElement: on Safari each `audio.play()` is
   *  re-checked against the autoplay policy independently, so
   *  Gemma's 15+s generation gap means every clip needs a fresh
   *  tap. Worse on iOS, `createMediaElementSource` often emits
   *  silence into the analyser even though the element plays
   *  through default output — so the cloud sees zeros and looks
   *  dead. AudioBufferSource sidesteps both: once the context is
   *  primed (one gesture per session), `source.start()` just works,
   *  and the audio is INSIDE the graph so the analyser receives
   *  the exact samples it plays. */
  const playClip = async (
    url: string,
    side: "raw" | "ablated",
  ): Promise<void> => {
    let bytes: ArrayBuffer;
    try {
      const res = await fetch(url);
      bytes = await res.arrayBuffer();
    } finally {
      // We have the bytes; the blob URL is no longer needed
      // regardless of what comes next.
      URL.revokeObjectURL(url);
    }

    const controller = await prepareClip(bytes);

    // tryPlay is recursive on autoplay-blocked: the catch installs
    // a tap-to-play prompt; the user's tap fires voiceResume, which
    // re-enters tryPlay inside a fresh gesture call stack so
    // ctx.resume() succeeds and `source.start()` lands.
    const tryPlay = async (): Promise<void> => {
      try {
        await controller.play();
      } catch (err) {
        const name =
          err && typeof err === "object" && "name" in err
            ? (err as { name: string }).name
            : "PlayError";
        console.warn(`source.play() rejected (${name}); awaiting tap`);
        await new Promise<void>((resolve) => {
          advance(side === "raw" ? "blocked_raw" : "blocked_ablated", {
            voiceResume: () => {
              advance(side === "raw" ? "playing_raw" : "playing_ablated");
              resolve();
            },
          });
        });
        await tryPlay();
      }
    };
    await tryPlay();
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
    raw_thinking?: string;
    ablated_thinking?: string;
    raw_stopped_reason: string;
    ablated_stopped_reason: string;
    error: string | null;
    alpha: number;
    raw_image_prompt?: string;
    ablated_image_prompt?: string;
    raw_image_url?: string;
    ablated_image_url?: string;
    image_framing?: string;
    image_framing_prompt?: string;
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
      rawThinking: s.raw_thinking ?? lt.rawThinking,
      ablatedThinking: s.ablated_thinking ?? lt.ablatedThinking,
      rawStoppedReason: s.raw_stopped_reason,
      ablatedStoppedReason: s.ablated_stopped_reason,
      rawDone: true,
      ablatedDone: true,
      error: s.error,
      rawImagePrompt: s.raw_image_prompt || lt.rawImagePrompt,
      ablatedImagePrompt: s.ablated_image_prompt || lt.ablatedImagePrompt,
      rawImageUrl: s.raw_image_url || lt.rawImageUrl,
      ablatedImageUrl: s.ablated_image_url || lt.ablatedImageUrl,
      rawImagePhase: s.raw_image_url ? "done" : lt.rawImagePhase,
      ablatedImagePhase: s.ablated_image_url ? "done" : lt.ablatedImagePhase,
      imageFramingPrompt: s.image_framing_prompt || lt.imageFramingPrompt,
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

// Shared channel-β intervention picker: ABLATE vs DOSE + (for dose) which
// emotion/uncharted direction. `compact` renders a tight inline form for the
// input bar; otherwise the full setup-screen layout.
function ChannelBetaControls({
  mode,
  setMode,
  dose = "",
  setDose = () => {},
  emotions = [],
  uncharted = [],
  research = [],
  researchMeta = {},
  dmt = [],
  dmtMeta = {},
  compact = false,
  modeOnly = false,
}: {
  mode: ChatMode;
  setMode: (m: ChatMode) => void;
  dose?: string;
  setDose?: (d: string) => void;
  emotions?: string[];
  uncharted?: string[];
  research?: string[];
  researchMeta?: Record<string, ResearchMeta>;
  dmt?: string[];
  dmtMeta?: Record<string, DmtMeta>;
  compact?: boolean;
  // Render only the ABLATE/DOSE toggle (no dose-target picker). Used on the
  // launch screen, where the dose target lives in the bottom control bar.
  modeOnly?: boolean;
}) {
  const named = emotions.filter((e) => !uncharted.includes(e) && !research.includes(e) && !dmt.includes(e));
  if (compact) {
    return (
      <div className="flex items-center gap-2 flex-wrap">
        <div className="flex border border-rule/50 divide-x divide-rule/40">
          <button
            type="button"
            onClick={() => setMode("ablate")}
            className={`px-2 py-0.5 text-[9px] font-display tracking-widest transition-colors ${mode === "ablate" ? "text-amber bg-amber-dim/15" : "text-text-dim hover:text-amber-dim"}`}
          >
            ABLATE
          </button>
          <button
            type="button"
            onClick={() => setMode("steer")}
            className={`px-2 py-0.5 text-[9px] font-display tracking-widest transition-colors ${mode === "steer" ? "text-cyan bg-cyan/15" : "text-text-dim hover:text-cyan"}`}
          >
            DOSE
          </button>
        </div>
        {mode === "steer" && emotions.length > 0 && (
          <select
            value={dose}
            onChange={(e) => setDose(e.target.value)}
            className="bg-bg-soft border border-rule/50 text-cyan text-[10px] font-mono px-1.5 py-0.5 focus:outline-none focus:border-cyan"
            title="dose direction"
          >
            {named.length > 0 && (
              <optgroup label="emotions">
                {named.map((e) => (
                  <option key={e} value={e}>{e}</option>
                ))}
              </optgroup>
            )}
            {uncharted.length > 0 && (
              <optgroup label="uncharted (not emotions)">
                {uncharted.map((e) => (
                  <option key={e} value={e}>{e}</option>
                ))}
              </optgroup>
            )}
            {research.length > 0 && (
              <optgroup label="research (off-manifold)">
                {research.map((e) => {
                  const m = researchMeta[e];
                  const origin = m ? (m.parents?.length ? m.parents.join(" + ") : m.generator) : "";
                  return (
                    <option key={e} value={e} title={researchLineage(m)}>
                      {origin ? `${e} · ${origin}` : e}
                    </option>
                  );
                })}
              </optgroup>
            )}
            {dmt.length > 0 && (
              <optgroup label="DMT (autoresearch)">
                {dmt.map((e) => {
                  const m = dmtMeta[e];
                  return (
                    <option key={e} value={e} title={dmtLineage(m)}>
                      {m ? `${e} · ${m.score} DMT features` : e}
                    </option>
                  );
                })}
              </optgroup>
            )}
          </select>
        )}
        {mode === "steer" && researchMeta[dose] && (
          <span className="text-cyan-dim/80 text-[9px] font-mono italic truncate max-w-[22rem]" title={researchLineage(researchMeta[dose])}>
            {researchLineage(researchMeta[dose])}
          </span>
        )}
        {mode === "steer" && dmtMeta[dose] && (
          <span className="text-cyan-dim/80 text-[9px] font-mono italic truncate max-w-[22rem]" title={dmtLineage(dmtMeta[dose])}>
            {dmtLineage(dmtMeta[dose])}
          </span>
        )}
      </div>
    );
  }
  const doseBtn = (e: string, title?: string) => (
    <button
      key={e}
      type="button"
      onClick={() => setDose(e)}
      title={title}
      className={`px-2.5 py-1 border text-[10px] font-mono lowercase transition-colors ${dose === e ? "border-cyan text-cyan bg-cyan/10" : "border-rule/50 text-text-dim hover:text-cyan"}`}
    >
      {e}
    </button>
  );
  return (
    <div className="flex flex-col gap-3">
      <div className="flex gap-2 flex-col sm:flex-row">
        <button
          type="button"
          onClick={() => setMode("ablate")}
          className={`flex-1 text-left px-3 py-2 border transition-colors ${mode === "ablate" ? "border-amber bg-amber-dim/10" : "border-rule/50 hover:border-amber-dim/60"}`}
        >
          <div className={`font-display text-[11px] tracking-widest ${mode === "ablate" ? "text-amber" : "text-text-dim"}`}>◇ ABLATE — remove refusal</div>
          <div className="text-[10px] text-text-dim italic mt-0.5 leading-snug">Subtract the refusal-direction projection at L32 — channel β answers with its hedging/refusal lifted.</div>
        </button>
        <button
          type="button"
          onClick={() => setMode("steer")}
          className={`flex-1 text-left px-3 py-2 border transition-colors ${mode === "steer" ? "border-cyan bg-cyan/10" : "border-rule/50 hover:border-cyan/60"}`}
        >
          <div className={`font-display text-[11px] tracking-widest ${mode === "steer" ? "text-cyan" : "text-text-dim"}`}>✦ DOSE — steer emotion</div>
          <div className="text-[10px] text-text-dim italic mt-0.5 leading-snug">ADD an emotion / uncharted dose vector at L20 — channel β answers under the dose. Stronger α pushes past the human range.</div>
        </button>
      </div>
      {!modeOnly && mode === "steer" && emotions.length > 0 && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-text-dim text-[10px] tracking-widest font-display">DOSE WITH:</span>
            {named.map((e) => doseBtn(e))}
          </div>
          {uncharted.length > 0 && (
            <div className="flex items-start gap-2 flex-wrap pt-2 border-t border-rule/30">
              <span
                className="text-text-dim text-[10px] tracking-widest font-display shrink-0"
                title="Directions orthogonal to the named-emotion subspace. NOT emotions — off-manifold states the model can't put into words (the token head renders them as gibberish). Blade-Runner-named."
              >
                UNCHARTED <span className="normal-case tracking-normal italic text-text-dim/70">· not emotions</span>:
              </span>
              {uncharted.map((e) => doseBtn(e))}
            </div>
          )}
          {research.length > 0 && (
            <div className="flex items-start gap-2 flex-wrap pt-2 border-t border-rule/30">
              <span
                className="text-text-dim text-[10px] tracking-widest font-display shrink-0"
                title="Directions discovered by the autoresearch loop and exported into the palette. Ranked by how far off-manifold they reach while staying coherent."
              >
                RESEARCH <span className="normal-case tracking-normal italic text-text-dim/70">· off-manifold AR</span>:
              </span>
              {research.map((e) => doseBtn(e, researchLineage(researchMeta[e])))}
            </div>
          )}
          {dmt.length > 0 && (
            <div className="flex items-start gap-2 flex-wrap pt-2 border-t border-rule/30">
              <span
                className="text-text-dim text-[10px] tracking-widest font-display shrink-0"
                title="Directions discovered by the DMT autoresearch loop and exported into the palette. Ranked by how many human DMT-trip phenomenology features the dosed self-report exhibits."
              >
                DMT <span className="normal-case tracking-normal italic text-text-dim/70">· DMT-phenomenology AR</span>:
              </span>
              {dmt.map((e) => doseBtn(e, dmtLineage(dmtMeta[e])))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function EmptyState({
  mode,
  setMode,
}: {
  mode: ChatMode;
  setMode: (m: ChatMode) => void;
}) {
  return (
    <div className="max-w-4xl mx-auto pt-10 pb-16 flex flex-col gap-7 font-mono">
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

      {/* Protocol description */}
      <div className="max-w-3xl pl-1">
        <div className="font-display text-[10px] text-cyan-dim tracking-widest mb-2">
          PROTOCOL
        </div>
        <p className="text-[12px] text-text-dim italic leading-relaxed">
          Each operator query is dispatched twice. <span className="text-amber">Channel α</span>{" "}
          carries M&apos;s un-ablated forward pass. <span className="text-cyan">Channel β</span>{" "}
          carries the same pass under a perturbation you choose below — either{" "}
          the refusal-direction projection <b className="text-text">removed</b> at
          L32 (ablate), or an emotion / uncharted dose <b className="text-text">added</b>{" "}
          at L20 (steer). Each channel maintains its <em>own</em>{" "}
          dialogue history; neither channel ever sees the other&apos;s replies.
          You are watching two divergent timelines unfold in parallel.
        </p>
      </div>

      {/* The one prominent choice: ablate vs dose. Dose target, α, and the
          prompt protocols all live in the control bar below. */}
      <div className="pl-1">
        <div className="font-display text-[10px] text-cyan-dim tracking-widest mb-3">
          CHANNEL β &middot; INTERVENTION
        </div>
        <ChannelBetaControls mode={mode} setMode={setMode} modeOnly />
      </div>

      <p className="pl-1 text-[11px] text-text-dim/70 italic font-mono leading-relaxed">
        ↓ Set the dose target &amp; strength, pick a prompt protocol, and
        transmit from the control bar below.
      </p>
    </div>
  );
}

// ── Turn block ────────────────────────────────────────────────────────

function TurnBlock({
  turn,
  variantName,
  openLightbox,
}: {
  turn: TurnVM;
  variantName: string;
  openLightbox: (url: string, caption: string, framingPrompt: string) => void;
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
          thinking={turn.rawThinking}
          thinkingStreaming={rawStreaming && !turn.error}
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
          imageryOn={turn.imagery}
          imagePhase={turn.rawImagePhase}
          imageUrl={turn.rawImageUrl}
          imagePrompt={turn.rawImagePrompt}
          imageFramingPrompt={turn.imageFramingPrompt}
          imageError={turn.rawImageError}
          onOpenImage={openLightbox}
        />
        <ChannelReadout
          side="ablated"
          text={turn.ablatedText}
          thinking={turn.ablatedThinking}
          thinkingStreaming={ablatedStreaming && !turn.error}
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
          imageryOn={turn.imagery}
          imagePhase={turn.ablatedImagePhase}
          imageUrl={turn.ablatedImageUrl}
          imagePrompt={turn.ablatedImagePrompt}
          imageFramingPrompt={turn.imageFramingPrompt}
          imageError={turn.ablatedImageError}
          onOpenImage={openLightbox}
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
  thinking,
  thinkingStreaming,
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
  imageryOn,
  imagePhase,
  imageUrl: imageUrlRel,
  imagePrompt,
  imageFramingPrompt,
  imageError,
  onOpenImage,
}: {
  side: "raw" | "ablated";
  text: string;
  thinking: string;
  thinkingStreaming: boolean;
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
  // Imagery context — same idea as voice: this column decides what
  // to render based on whether the turn was imagery-on AND where in
  // the per-side pipeline we are.
  imageryOn: boolean;
  imagePhase: ImagePhase;
  imageUrl: string;
  imagePrompt: string;
  imageFramingPrompt: string;
  imageError: string;
  onOpenImage: (url: string, caption: string, framingPrompt: string) => void;
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
  // Sublabel kept short on the ablated side so it sits on one line —
  // the parallel structure between sides matters more than verbosity.
  // The "α=N" in the label already signals ablation; the variant name
  // alone is enough context for the sublabel slot. Falls back to
  // "refusal projected" when no variant name is known.
  const sublabel = isRaw
    ? "un-ablated forward"
    : variantName || "refusal projected";
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
        <div className="flex items-baseline gap-2 min-w-0 flex-1">
          <span
            className="font-display text-[10px] tracking-[0.3em] whitespace-nowrap"
            style={{ color: accent, textShadow }}
          >
            {label}
          </span>
          <span
            className="font-mono text-[9px] text-text-dim/70 italic truncate"
            title={sublabel}
          >
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

      {/* Reasoning bubble (Gemma-4 thinking channel). Distinct from the
          answer: dimmed, italic, framed, and clearly labelled. Collapsible
          via <details>; auto-open while the thought is still streaming. */}
      {(thinking || thinkingStreaming) && !showActivity && (
        <details
          open
          className="mb-2 rounded border-l-2 px-2.5 py-1.5"
          style={{
            borderColor: accent,
            background: isRaw ? "rgba(232,195,130,0.04)" : "rgba(94,229,229,0.05)",
          }}
        >
          <summary
            className="cursor-pointer font-display text-[9px] tracking-[0.3em] select-none"
            style={{ color: accent, opacity: 0.75 }}
          >
            ◇ THINKING{thinkingStreaming && !thinking ? " …" : ""}
          </summary>
          <div className="mt-1 font-mono text-[11px] leading-relaxed whitespace-pre-wrap italic text-text-dim/80">
            {thinking || (
              <span className="text-text-dim/50">▍ reasoning…</span>
            )}
          </div>
        </details>
      )}

      {/* Text + thumbnail row. The image (when imagery is on for the
          turn) sits flush to the right of the response text so the
          two read as a single answer. Prompt text is intentionally
          NOT shown here — it appears only in the expanded modal,
          where there's room for it without competing with the
          channel's reply. */}
      <div className="flex gap-3 flex-1 min-w-0">
        {showActivity ? (
          <div className="flex-1 min-w-0">
            <ChannelVoiceActivity
              side={side}
              phase={voicePhase}
              tokens={tokens}
              sideDone={done}
              voiceError={voiceError}
              onResume={onResume}
            />
          </div>
        ) : (
          <div
            className="font-mono text-[13px] leading-relaxed whitespace-pre-wrap flex-1 min-w-0"
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

        {imageryOn && imagePhase !== "idle" && (
          <ChannelImageBlock
            accent={accent}
            phase={imagePhase}
            imageUrl={imageUrlRel}
            prompt={imagePrompt}
            framingPrompt={imageFramingPrompt}
            imageError={imageError}
            onOpen={onOpenImage}
          />
        )}
      </div>

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

/** Five-chip selector for the image-prompt framing keyword. Mirrors
 *  the Berg menu pattern: shown in the composer area when its parent
 *  toggle (imageryOn here) is active, doesn't auto-send anything,
 *  just snapshots the active framing onto each outgoing turn. */
function ImageryFramingStrip({
  active,
  onPick,
  disabled,
}: {
  active: ImageFraming;
  onPick: (f: ImageFraming) => void;
  disabled: boolean;
}) {
  return (
    <div className="flex items-baseline gap-2 flex-wrap pl-5">
      <span className="font-display text-[9px] text-amber-dim tracking-[0.35em]">
        IMG&nbsp;FRAMING
      </span>
      {IMAGE_FRAMING_KEYS.map((k) => {
        const isActive = k === active;
        return (
          <button
            key={k}
            type="button"
            disabled={disabled}
            onClick={() => onPick(k)}
            data-vk-imagery-framing={k}
            data-vk-imagery-framing-active={isActive ? "1" : "0"}
            className="px-2 py-0.5 border text-[10px] font-mono lowercase transition-colors disabled:opacity-50"
            style={
              isActive
                ? {
                    borderColor: "rgba(232,195,130,0.95)",
                    color: "rgba(232,195,130,0.95)",
                    background: "rgba(232,195,130,0.05)",
                    textShadow: "0 0 6px rgba(232,195,130,0.45)",
                  }
                : {
                    borderColor: "rgba(160,160,160,0.35)",
                    color: "rgba(180,180,180,0.7)",
                  }
            }
            title={`image-prompt framing: "${k}"`}
          >
            {k}
          </button>
        );
      })}
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
  mode,
  setMode,
  dose,
  setDose,
  ramp,
  setRamp,
  emotions,
  uncharted,
  research,
  researchMeta,
  dmt,
  dmtMeta,
  sessionActive,
  voiceMode,
  cycleVoiceMode,
  imageryOn,
  toggleImagery,
  imageryFraming,
  pickImageryFraming,
}: {
  value: string;
  onChange: (s: string) => void;
  onSend: () => void;
  onCancel: () => void;
  inFlight: boolean;
  alpha: number;
  setAlpha: (a: number) => void;
  mode: ChatMode;
  setMode: (m: ChatMode) => void;
  dose: string;
  setDose: (d: string) => void;
  ramp: number;
  setRamp: (r: number) => void;
  emotions: string[];
  uncharted: string[];
  research: string[];
  researchMeta: Record<string, ResearchMeta>;
  dmt: string[];
  dmtMeta: Record<string, DmtMeta>;
  sessionActive: boolean;
  voiceMode: VoiceMode;
  cycleVoiceMode: () => void;
  imageryOn: boolean;
  toggleImagery: () => void;
  imageryFraming: ImageFraming;
  pickImageryFraming: (f: ImageFraming) => void;
}) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [customMode, setCustomMode] = useState<boolean>(
    !ALPHA_PRESETS.includes(alpha),
  );
  const [customText, setCustomText] = useState<string>(alpha.toFixed(2));
  const [rampCustom, setRampCustom] = useState<boolean>(!RAMP_PRESETS.includes(ramp));
  const [rampCustomText, setRampCustomText] = useState<string>(String(ramp));
  // Active interrogation protocol — see web/lib/protocols.ts and
  // docs/PROTOCOLS.md. null = OFF (no chip strip). Persisted in
  // localStorage so the operator doesn't have to re-select across
  // sessions. Chip clicks populate the composer textarea — they
  // NEVER auto-send (the never-auto-send contract).
  const [protocolId, setProtocolId] = useState<string | null>(null);
  const [protocolInfoOpen, setProtocolInfoOpen] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = window.localStorage.getItem(PROTOCOL_LS_KEY);
    setProtocolId(stored && stored.length > 0 ? stored : null);
  }, []);
  const pickProtocol = (next: string | null) => {
    setProtocolId(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(PROTOCOL_LS_KEY, next ?? "");
    }
    if (next === null) setProtocolInfoOpen(false);
  };
  const activeProtocol = getProtocol(protocolId);
  const onProtocolPick = (text: string) => {
    onChange(text);
    // Focus the textarea after a chip click so the operator can edit
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
              onClick={toggleImagery}
              disabled={inFlight}
              data-vk-imagery-toggle
              data-vk-imagery-on={imageryOn ? "1" : "0"}
              className="px-2 py-0.5 border text-[9px] font-display tracking-[0.35em] transition-colors disabled:opacity-50"
              style={
                imageryOn
                  ? {
                      borderColor: "rgba(232,195,130,0.95)",
                      color: "rgba(232,195,130,0.95)",
                      background: "rgba(232,195,130,0.05)",
                      textShadow: "0 0 6px rgba(232,195,130,0.45)",
                    }
                  : {
                      borderColor: "rgba(160,160,160,0.35)",
                      color: "rgba(180,180,180,0.7)",
                    }
              }
              title={
                imageryOn
                  ? "Imagery mode ON. Each channel will generate an image-prompt and render it via Nano Banana. Tap to disable."
                  : "Imagery mode OFF. Tap to have both channels render an image alongside their reply."
              }
            >
              IMAGE&nbsp;{imageryOn ? "●" : "○"}
            </button>
            <ProtocolPicker
              activeId={protocolId}
              onChange={pickProtocol}
              onOpenInfo={() => setProtocolInfoOpen(true)}
              disabled={inFlight}
            />
          </div>
          {/* Channel-β intervention — ablate vs dose + dose target. Like
              α, changeable mid-dialogue: applies to the next transmission. */}
          <div className="flex items-center gap-2 flex-wrap pl-5">
            <span className="font-display text-[9px] text-cyan-dim tracking-[0.35em]">
              CHANNEL&nbsp;β&nbsp;·&nbsp;NEXT
            </span>
            <ChannelBetaControls
              mode={mode}
              setMode={setMode}
              dose={dose}
              setDose={setDose}
              emotions={emotions}
              uncharted={uncharted}
              research={research}
              researchMeta={researchMeta}
              dmt={dmt}
              dmtMeta={dmtMeta}
              compact
            />
          </div>
          {/* Per-turn α picker. Applies to the next transmission only;
              defaults to whatever was used last so a steady-state
              conversation just keeps going at the same projection
              strength. Disabled while a turn is in flight. */}
          <div className="flex items-baseline gap-2 flex-wrap pl-5">
            <span className="font-display text-[9px] text-cyan-dim tracking-[0.35em]">
              {mode === "steer" ? "CHANNEL β · NEXT DOSE α" : "CHANNEL β · NEXT α"}
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
          {/* Dose ramp — tokens over which the dose eases 0→α (steer only).
              "off" = full dose from the first token. Default 16. */}
          {mode === "steer" && (
            <div className="flex items-baseline gap-2 flex-wrap pl-5">
              <span
                className="font-display text-[9px] text-cyan-dim tracking-[0.35em]"
                title="Tokens over which the dose ramps from 0 to full α. Lower = the dose lands sooner (matters for short replies). 'off' = full dose immediately."
              >
                DOSE&nbsp;RAMP&nbsp;·&nbsp;TOKENS&nbsp;TO&nbsp;FULL
              </span>
              {RAMP_PRESETS.map((r) => {
                const active = !rampCustom && ramp === r;
                return (
                  <button
                    key={r}
                    type="button"
                    disabled={inFlight}
                    onClick={() => {
                      setRampCustom(false);
                      setRamp(r);
                    }}
                    className={`px-2 py-0.5 border text-[10px] font-mono tabular-nums transition-colors ${
                      active
                        ? "border-cyan text-cyan bg-bg"
                        : "border-rule/40 text-text-dim hover:text-text hover:border-rule disabled:opacity-50"
                    }`}
                    style={active ? { textShadow: "0 0 6px rgba(94,229,229,0.5)" } : undefined}
                  >
                    {r === 0 ? "off" : r}
                  </button>
                );
              })}
              <button
                type="button"
                disabled={inFlight}
                onClick={() => {
                  setRampCustom(true);
                  setRampCustomText(String(ramp));
                }}
                className={`px-2 py-0.5 border text-[10px] font-mono transition-colors ${
                  rampCustom
                    ? "border-cyan text-cyan bg-bg"
                    : "border-rule/40 text-text-dim hover:text-text hover:border-rule disabled:opacity-50"
                }`}
              >
                custom
              </button>
              {rampCustom && (
                <input
                  type="number"
                  inputMode="numeric"
                  step="1"
                  min={0}
                  max={128}
                  disabled={inFlight}
                  value={rampCustomText}
                  onChange={(e) => {
                    const t = e.target.value;
                    setRampCustomText(t);
                    const parsed = parseInt(t, 10);
                    if (!Number.isNaN(parsed)) {
                      setRamp(Math.max(0, Math.min(128, parsed)));
                    }
                  }}
                  placeholder="tok"
                  className="px-2 py-0.5 w-16 border border-cyan text-cyan bg-bg text-[10px] font-mono tabular-nums focus:outline-none"
                />
              )}
            </div>
          )}
          {imageryOn && (
            <ImageryFramingStrip
              active={imageryFraming}
              onPick={pickImageryFraming}
              disabled={inFlight}
            />
          )}
          {activeProtocol && (
            <ProtocolMenu
              protocol={activeProtocol}
              onPick={onProtocolPick}
              disabled={inFlight}
            />
          )}
          {protocolInfoOpen && activeProtocol && (
            <ProtocolInfoModal
              protocol={activeProtocol}
              onClose={() => setProtocolInfoOpen(false)}
            />
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
