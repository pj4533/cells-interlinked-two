"use client";

/** Shared AudioContext + AnalyserNode for chat-mode playback
 * visualizations.
 *
 * The voice playback driver creates a new <audio> per clip; we tap
 * each one through createMediaElementSource → analyser → destination
 * so visualizations can read live frequency / waveform data via
 * `getAnalyser()`. The graph is a singleton (one AudioContext per
 * page lifecycle); the analyser persists across clips so React
 * visualizations just keep reading and naturally show silence
 * between clips.
 *
 * ── Critical gotcha (the bug this module exists to dodge) ──
 *
 * createMediaElementSource *captures* the audio element's output:
 * after the call, audio no longer reaches the speakers via the
 * element's default path — it ONLY flows through the analyser →
 * destination chain in the graph. If the AudioContext is suspended
 * at that moment (autoplay policy keeps it suspended until a fresh
 * user gesture; Gemma's ~15s generation gap means the TRANSMIT
 * click's gesture token is long stale by the time we'd attach),
 * the graph doesn't pull samples and the audio is silent.
 *
 * Two-part defense:
 *   1. `primeAudioContext()` is called from a `pointerdown`
 *      listener on first user interaction with the chat page —
 *      that IS a fresh gesture, so the resume actually takes.
 *   2. `attachAudio()` refuses to capture an element when the
 *      context is not running. In that case it returns null, the
 *      audio element plays normally through default output, and
 *      the visualization falls back to silent baseline. Better to
 *      lose the responsive viz on the first clip than to silence
 *      playback entirely.
 *
 * Notes:
 *   - blob: URLs are same-origin with their creator (the page),
 *     so the analyser receives un-tainted samples regardless of
 *     which port the audio bytes came from.
 *   - createMediaElementSource may only be called ONCE per element.
 *     We guard via a WeakSet so a double-attach (from React strict-
 *     mode double-effects or a play retry under autoplay block)
 *     can't throw.
 */

let ctx: AudioContext | null = null;
let analyser: AnalyserNode | null = null;
const attached = new WeakSet<HTMLMediaElement>();

function ensureContext(): AnalyserNode | null {
  if (analyser) return analyser;
  if (typeof window === "undefined") return null;
  const Ctor =
    window.AudioContext ||
    (window as unknown as { webkitAudioContext?: typeof AudioContext })
      .webkitAudioContext;
  if (!Ctor) return null;
  try {
    ctx = new Ctor();
    analyser = ctx.createAnalyser();
    // 512-point FFT → 256 frequency bins. Plenty of resolution for a
    // cloud-style visualization without overheating the requestAnimationFrame
    // loop.
    analyser.fftSize = 512;
    // Hardware-side temporal smoothing. Visualizations apply their own
    // smoothing on top per layer for cloud feel.
    analyser.smoothingTimeConstant = 0.78;
    analyser.connect(ctx.destination);
  } catch (e) {
    console.warn("AudioContext creation failed", e);
    return null;
  }
  return analyser;
}

/** Initialize the AudioContext and resume it. Call this from a
 *  user-gesture handler (pointerdown, click) — that's when resume()
 *  is permitted to take effect. Safe to call any number of times;
 *  it's idempotent. */
export function primeAudioContext(): void {
  ensureContext();
  if (ctx && ctx.state === "suspended") {
    void ctx.resume().catch((e) => {
      console.warn("AudioContext resume rejected", e);
    });
  }
}

/** Connect an HTMLMediaElement to the shared analyser. Returns the
 *  analyser on success, or `null` if the context can't be put into
 *  a running state right now (in which case the caller's audio
 *  should play normally — we deliberately do NOT capture it,
 *  because capturing through a suspended graph silences playback).
 *  Idempotent — calling twice on the same element is a no-op. */
export function attachAudio(audio: HTMLMediaElement): AnalyserNode | null {
  const a = ensureContext();
  if (!a || !ctx) return null;
  if (ctx.state === "suspended") {
    // Best-effort resume. If we're outside a user gesture this will
    // queue but not actually resume; we then bail rather than risk
    // a silent capture.
    void ctx.resume().catch(() => {});
  }
  if (ctx.state !== "running") {
    return null;
  }
  if (attached.has(audio)) return a;
  try {
    const src = ctx.createMediaElementSource(audio);
    src.connect(a);
    attached.add(audio);
  } catch (e) {
    // Most commonly: element was already attached in another path.
    console.warn("attachAudio failed; falling back to direct playback", e);
    return null;
  }
  return a;
}

/** Read the shared analyser. Returns null if no audio has ever been
 * attached this session (visualizations should treat this as "render
 * a silent baseline"). */
export function getAnalyser(): AnalyserNode | null {
  return analyser;
}

/** Test-only: expose the current AudioContext state. Used by the
 *  e2e smoke to verify priming actually puts the context in
 *  `running`. */
export function getContextState(): AudioContextState | null {
  return ctx?.state ?? null;
}

// Dev/test hook: expose internals on window so e2e can introspect
// without bundle-internal imports. The added overhead is one
// property assignment per page load; production users will never
// hit it.
if (typeof window !== "undefined") {
  (window as unknown as Record<string, unknown>).__ci_audio_graph = {
    getContextState,
    getAnalyser,
    primeAudioContext,
  };
}
