"use client";

import { AnimatePresence, motion } from "framer-motion";

import {
  ACTIVE_PLAYBACK_VIZ,
  PLAYBACK_VIZ_REGISTRY,
} from "./playback-viz";

/** Phase the chat-page voice driver advances through after turn_done. */
export type VoicePhase =
  | "off"
  | "thinking"
  | "synth_raw"
  | "playing_raw"
  | "blocked_raw"
  | "synth_ablated"
  | "playing_ablated"
  | "blocked_ablated"
  | "done";

type Side = "raw" | "ablated";

const RAW_ACCENT = "rgba(232,195,130,0.95)";
const RAW_ACCENT_RGB = "232,195,130";
const ABLATED_ACCENT = "rgba(94,229,229,0.95)";
const ABLATED_ACCENT_RGB = "94,229,229";

/** Per-column voice activity. Shown inside a ChannelReadout's body
 *  whenever that side is voiced AND the turn's voice flow is still
 *  active. Renders one of:
 *
 *    • token boxes (during generation OR after this side has played
 *      and we're waiting on the other side — boxes stay frozen so
 *      the column doesn't snap to empty)
 *    • synth indicator (during this side's synth_* phase only)
 *    • playback bars  (during this side's playing_* phase only)
 *    • tap-to-play prompt (during this side's blocked_* phase)
 *
 *  Everything is keyed off the `side` prop, so each column renders
 *  itself independently. No full-width banners, no side-switch
 *  overlays — the per-column activity IS the channel handoff.
 */
export function ChannelVoiceActivity({
  side,
  phase,
  tokens,
  sideDone,
  voiceError,
  onResume,
}: {
  side: Side;
  phase: VoicePhase;
  tokens: string[];
  // Whether THIS side's generation has finished (M done emitting).
  // Drives the caret on the token box lane.
  sideDone: boolean;
  voiceError: string | null;
  onResume: (() => void) | null;
}) {
  const accent = side === "raw" ? RAW_ACCENT : ABLATED_ACCENT;
  const accentRgb = side === "raw" ? RAW_ACCENT_RGB : ABLATED_ACCENT_RGB;

  // Which sub-view to render in this column for the current phase.
  // The four "this side is active" phases drive the dedicated
  // visualizations; everything else (thinking, the other side's
  // phases) falls back to the token boxes so the column doesn't
  // go blank between this side's turn and the next.
  const myPhases: Record<
    VoicePhase,
    "boxes" | "synth" | "playing" | "blocked"
  > = {
    off: "boxes",
    thinking: "boxes",
    synth_raw: side === "raw" ? "synth" : "boxes",
    playing_raw: side === "raw" ? "playing" : "boxes",
    blocked_raw: side === "raw" ? "blocked" : "boxes",
    synth_ablated: side === "ablated" ? "synth" : "boxes",
    playing_ablated: side === "ablated" ? "playing" : "boxes",
    blocked_ablated: side === "ablated" ? "blocked" : "boxes",
    done: "boxes",
  };
  const view = myPhases[phase];

  return (
    <div className="relative flex-1 min-h-[6rem]">
      <style>{`
        @keyframes ci-token-in {
          0%   { transform: scale(0.55); opacity: 0; }
          70%  { transform: scale(1.05); opacity: 1; }
          100% { transform: scale(1);    opacity: 1; }
        }
        @keyframes ci-caret {
          0%, 49%   { opacity: 1; }
          50%, 100% { opacity: 0; }
        }
        @keyframes ci-packet {
          0%   { left: 0%;    opacity: 0; }
          12%  { opacity: 1; }
          85%  { opacity: 1; }
          100% { left: 100%;  opacity: 0; }
        }
        @keyframes ci-synth-pulse {
          0%, 100% { opacity: 0.55; }
          50%      { opacity: 1; }
        }
      `}</style>

      <AnimatePresence mode="wait">
        {view === "boxes" && (
          <motion.div
            key="v-boxes"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            data-vk-channel-activity={`boxes-${side}`}
            className="absolute inset-0"
          >
            <TokenBoxes
              accent={accent}
              accentRgb={accentRgb}
              tokens={tokens}
              done={sideDone}
            />
          </motion.div>
        )}

        {view === "synth" && (
          <motion.div
            key="v-synth"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            data-vk-channel-activity={`synth-${side}`}
            className="absolute inset-0 flex items-center justify-center"
          >
            <SynthIndicator accent={accent} accentRgb={accentRgb} side={side} />
          </motion.div>
        )}

        {view === "playing" && (
          <motion.div
            key="v-playing"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            data-vk-channel-activity={`playing-${side}`}
            className="absolute inset-0"
          >
            {(() => {
              const Viz = PLAYBACK_VIZ_REGISTRY[ACTIVE_PLAYBACK_VIZ];
              return <Viz accent={accent} accentRgb={accentRgb} />;
            })()}
          </motion.div>
        )}

        {view === "blocked" && (
          <motion.div
            key="v-blocked"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            data-vk-channel-activity={`blocked-${side}`}
            className="absolute inset-0 flex flex-col items-center justify-center gap-2"
          >
            {onResume && (
              <motion.button
                type="button"
                onClick={onResume}
                data-vk-voice-resume
                initial={{ scale: 0.92 }}
                animate={{ scale: 1 }}
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.97 }}
                className="px-5 py-1.5 border font-display tracking-[0.35em] text-[10px]"
                style={{
                  borderColor: accent,
                  color: accent,
                  textShadow: `0 0 8px ${accent}`,
                  background: "rgba(0,0,0,0.25)",
                }}
              >
                ▶&nbsp;&nbsp;TAP TO PLAY
              </motion.button>
            )}
            <span className="font-mono text-[9px] text-text-dim/70 italic text-center">
              browser blocked autoplay — tap to play this channel
            </span>
          </motion.div>
        )}
      </AnimatePresence>

      {voiceError && (
        <div className="absolute bottom-0 inset-x-0 font-mono text-[10px] text-warning not-italic">
          ⚠ {voiceError}
        </div>
      )}
    </div>
  );
}

/** Per-token box visualization driven by real stream events. Each
 *  entry in `tokens` is the exact `evt.decoded` string Gemma emitted
 *  for that position — the box contains the real text rendered
 *  transparent so its width matches the real word width exactly. */
function TokenBoxes({
  accent,
  accentRgb,
  tokens,
  done,
}: {
  accent: string;
  accentRgb: string;
  tokens: string[];
  done: boolean;
}) {
  return (
    <div
      className="relative w-full h-full py-1 flex flex-wrap items-start content-start gap-y-1.5 leading-[1.4] overflow-hidden"
      style={{
        borderBottom: `1px solid rgba(${accentRgb},0.18)`,
      }}
    >
      {tokens.map((tok, i) => {
        if (/^\s*$/.test(tok)) {
          if (tok.includes("\n")) {
            return <span key={i} className="w-full" aria-hidden />;
          }
          return <span key={i} className="inline-block w-1" aria-hidden />;
        }
        return (
          <span
            key={i}
            aria-hidden
            className="inline-block rounded-[2px] font-mono text-[12px]"
            style={{
              color: "transparent",
              border: `1px solid rgba(${accentRgb},0.85)`,
              background: `rgba(${accentRgb},0.14)`,
              boxShadow: `0 0 4px rgba(${accentRgb},0.32)`,
              padding: "0 3px",
              marginRight: "2px",
              animation: "ci-token-in 0.18s ease-out both",
            }}
          >
            {tok}
          </span>
        );
      })}
      <span
        aria-hidden
        className="inline-block self-center"
        style={{
          width: "2px",
          height: "12px",
          background: accent,
          boxShadow: `0 0 6px ${accent}`,
          animation: done ? undefined : "ci-caret 0.95s steps(2, end) infinite",
          opacity: done ? 0.25 : 1,
        }}
      />
    </div>
  );
}

/** Network-transmission indicator for the synth_* phases. Compact
 *  per-column version: an endpoint badge on each side + a short pipe
 *  of channel-tinted packets sliding across. Reads as "talking to
 *  OpenAI" — distinct from playback bars. */
function SynthIndicator({
  accent,
  accentRgb,
  side,
}: {
  accent: string;
  accentRgb: string;
  side: Side;
}) {
  const label = side === "raw" ? "α · raw" : "β · ablated";
  return (
    <div className="w-full flex flex-col gap-2">
      <div
        className="flex items-center justify-between font-display text-[9px] tracking-[0.3em]"
        style={{ color: accent }}
      >
        <span
          className="px-1.5 py-0.5 border"
          style={{
            borderColor: `rgba(${accentRgb},0.55)`,
            textShadow: `0 0 5px ${accent}`,
          }}
        >
          M&nbsp;·&nbsp;{label}
        </span>
        <span
          className="text-[9px]"
          style={{
            color: `rgba(${accentRgb},0.85)`,
            animation: "ci-synth-pulse 1.4s ease-in-out infinite",
          }}
        >
          ▸ FETCHING ▸
        </span>
        <span
          className="px-1.5 py-0.5 border"
          style={{
            borderColor: `rgba(${accentRgb},0.55)`,
            textShadow: `0 0 5px ${accent}`,
          }}
        >
          openai
        </span>
      </div>
      <div className="flex flex-col gap-1.5">
        {[0, 1, 2].map((r) => (
          <div
            key={r}
            className="relative h-1.5"
            style={{
              background: `linear-gradient(90deg, rgba(${accentRgb},0.05), rgba(${accentRgb},0.15) 50%, rgba(${accentRgb},0.05))`,
              borderRadius: "1px",
            }}
          >
            {[0, 1, 2, 3].map((p) => {
              const dur = 1.6 + ((r * 5 + p * 7) % 9) * 0.08;
              const delay = (p / 4) * dur + r * 0.18;
              return (
                <span
                  key={p}
                  aria-hidden
                  className="absolute top-1/2 -translate-y-1/2 rounded-[1px]"
                  style={{
                    width: "8px",
                    height: "5px",
                    background: accent,
                    boxShadow: `0 0 4px rgba(${accentRgb},0.85)`,
                    animation: `ci-packet ${dur}s ${delay}s linear infinite`,
                    willChange: "left, opacity",
                  }}
                />
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

