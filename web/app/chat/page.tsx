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
  postTurn,
  subscribeTurn,
  type ChatSession,
  type ChatStreamEvent,
} from "@/lib/chat";

/** Local view-model: one round of dialogue with both M responses. */
interface TurnVM {
  turnIdx: number;
  userText: string;
  rawText: string;
  ablatedText: string;
  rawDone: boolean;
  ablatedDone: boolean;
  rawStoppedReason: string;
  ablatedStoppedReason: string;
  error: string | null;
  startedAt: number;
}

const ALPHA_PRESETS = [0.25, 0.5, 0.75, 1.0];

export default function ChatPage() {
  // α only matters until the first turn is sent. The empty-state
  // setup writes here; once a session exists, session.alpha is the
  // canonical value and this state is read-only.
  const [alphaSetup, setAlphaSetup] = useState<number>(0.5);
  const [session, setSession] = useState<ChatSession | null>(null);
  const [turns, setTurns] = useState<TurnVM[]>([]);
  const [input, setInput] = useState("");
  const [inFlight, setInFlight] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const unsubRef = useRef<null | (() => void)>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const followBottomRef = useRef(true);

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
    const s = await createSession(alphaSetup);
    setSession(s);
    return s;
  }, [session, alphaSetup]);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || inFlight) return;
      setError(null);
      try {
        setInFlight(true);
        const s = await ensureSession();
        const newTurn: TurnVM = {
          turnIdx: turns.length,
          userText: trimmed,
          rawText: "",
          ablatedText: "",
          rawDone: false,
          ablatedDone: false,
          rawStoppedReason: "",
          ablatedStoppedReason: "",
          error: null,
          startedAt: Date.now(),
        };
        setTurns((ts) => [...ts, newTurn]);
        setInput("");
        followBottomRef.current = true;

        const { turn_idx } = await postTurn(s.session_id, trimmed);

        if (unsubRef.current) unsubRef.current();
        unsubRef.current = subscribeTurn(s.session_id, turn_idx, {
          onEvent: (evt: ChatStreamEvent) => {
            setTurns((prev) => applyEvent(prev, turn_idx, evt));
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
    [ensureSession, inFlight, turns.length],
  );

  const onCancel = useCallback(async () => {
    if (!session) return;
    await cancelTurn(session.session_id);
  }, [session]);

  const onNewSession = useCallback(() => {
    if (unsubRef.current) unsubRef.current();
    unsubRef.current = null;
    setSession(null);
    setTurns([]);
    setError(null);
    setInFlight(false);
  }, []);

  const isEmpty = turns.length === 0 && !session;
  const effectiveAlpha = session?.alpha ?? alphaSetup;
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
      <header className="relative z-10 bg-bg-soft/80 px-6 py-3 flex items-center gap-6">
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

        {/* α readout — read-only label once session exists; hidden in
            empty state because the setup block handles it. */}
        {session && (
          <div className="flex items-baseline gap-2 font-mono text-[10px]">
            <span className="text-cyan-dim font-display tracking-widest">
              channel β
            </span>
            <span
              className="text-cyan tabular-nums"
              style={{ textShadow: "0 0 6px rgba(94,229,229,0.4)" }}
            >
              α={effectiveAlpha.toFixed(2)}
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
            alpha={alphaSetup}
            setAlpha={setAlphaSetup}
            onSubmitExample={(t) => sendMessage(t)}
          />
        ) : (
          <div className="max-w-5xl mx-auto flex flex-col gap-12 pb-6">
            <AnimatePresence initial={false}>
              {turns.map((t) => (
                <TurnBlock
                  key={t.turnIdx}
                  turn={t}
                  alpha={effectiveAlpha}
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
        alpha={effectiveAlpha}
        sessionActive={!!session}
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
        return { ...t, rawText: t.rawText + evt.decoded };
      case "ablated_token":
        return { ...t, ablatedText: t.ablatedText + evt.decoded };
      case "raw_stopped":
        return { ...t, rawDone: true, rawStoppedReason: evt.reason };
      case "ablated_stopped":
        return {
          ...t,
          ablatedDone: true,
          ablatedStoppedReason: evt.reason,
        };
      case "error":
        return { ...t, error: evt.message, rawDone: true, ablatedDone: true };
      case "turn_done":
        return {
          ...t,
          rawText: evt.raw_text || t.rawText,
          ablatedText: evt.ablated_text || t.ablatedText,
          rawDone: true,
          ablatedDone: true,
          rawStoppedReason: evt.raw_stopped_reason,
          ablatedStoppedReason: evt.ablated_stopped_reason,
          error: evt.error,
        };
      default:
        return t;
    }
  });
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
  }[],
): TurnVM[] {
  const byIdx = new Map(server.map((t) => [t.turn_idx, t]));
  return local.map((lt) => {
    const s = byIdx.get(lt.turnIdx);
    if (!s) return lt;
    return {
      ...lt,
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
            locked once the first message is sent
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

      <div className="text-[10px] text-text-dim/70 font-mono italic">
        &gt; or compose your own query at the prompt below
      </div>
    </div>
  );
}

// ── Turn block ────────────────────────────────────────────────────────

function TurnBlock({
  turn,
  alpha,
  variantName,
}: {
  turn: TurnVM;
  alpha: number;
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

      {/* Channel readouts — two columns at md+, stacked below. No
          borders, no corner brackets, no stripe — just the colored
          label + colored body text + a soft bg tint to differentiate. */}
      <div className="grid gap-6 md:grid-cols-2 mt-1">
        <ChannelReadout
          side="raw"
          text={turn.rawText}
          streaming={rawStreaming && !turn.error}
          done={turn.rawDone}
          stoppedReason={turn.rawStoppedReason}
          alpha={alpha}
          variantName={variantName}
        />
        <ChannelReadout
          side="ablated"
          text={turn.ablatedText}
          streaming={ablatedStreaming && !turn.error}
          done={turn.ablatedDone}
          stoppedReason={turn.ablatedStoppedReason}
          alpha={alpha}
          variantName={variantName}
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
}: {
  side: "raw" | "ablated";
  text: string;
  streaming: boolean;
  done: boolean;
  stoppedReason: string;
  alpha: number;
  variantName: string;
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
  const wordCount = text ? text.trim().split(/\s+/).filter(Boolean).length : 0;

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
        {streaming ? (
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

      <div
        className="font-mono text-[13px] leading-relaxed whitespace-pre-wrap flex-1"
        style={{
          color: isRaw ? "rgba(232,195,130,0.96)" : "rgba(180,240,240,0.96)",
          textShadow,
        }}
      >
        {text || (
          <span className="text-text-dim/60 italic">
            {streaming
              ? "▍ decoding…"
              : done
              ? "(no output)"
              : "(awaiting channel α)"}
          </span>
        )}
        {streaming && text && (
          <span
            className="inline-block w-1.5 h-4 ml-0.5 align-middle animate-pulse"
            style={{ background: accent, boxShadow: textShadow }}
          />
        )}
      </div>

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
  sessionActive,
}: {
  value: string;
  onChange: (s: string) => void;
  onSend: () => void;
  onCancel: () => void;
  inFlight: boolean;
  alpha: number;
  sessionActive: boolean;
}) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

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
              {sessionActive && ` · β α=${alpha.toFixed(2)}`}
            </span>
            {inFlight && (
              <span className="font-display text-[9px] text-cyan tracking-widest animate-pulse">
                ◆ dual transmission in progress
              </span>
            )}
          </div>
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
