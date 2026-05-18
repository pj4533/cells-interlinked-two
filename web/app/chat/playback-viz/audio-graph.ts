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
 * Notes:
 *   - createMediaElementSource may only be called ONCE per element.
 *     We guard via a WeakSet so a double-attach (from React strict-
 *     mode double-effects or a play retry under autoplay block)
 *     can't throw.
 *   - blob: URLs are same-origin with their creator (the page),
 *     so the analyser receives un-tainted samples regardless of
 *     which port the audio bytes came from.
 *   - AudioContext starts suspended until a user gesture. The
 *     TRANSMIT click that kicks the turn off is a gesture; we
 *     call resume() defensively on each attach in case the context
 *     was suspended in the meantime.
 */

let ctx: AudioContext | null = null;
let analyser: AnalyserNode | null = null;
const attached = new WeakSet<HTMLMediaElement>();

function ensureContext(): AnalyserNode {
  if (analyser) return analyser;
  // Lazy-construct on first attach so we don't poke audio APIs
  // before the user has interacted with the page.
  ctx = new (window.AudioContext ||
    (window as unknown as { webkitAudioContext: typeof AudioContext })
      .webkitAudioContext)();
  analyser = ctx.createAnalyser();
  // 512-point FFT → 256 frequency bins. Plenty of resolution for a
  // cloud-style visualization without overheating the requestAnimationFrame
  // loop.
  analyser.fftSize = 512;
  // Hardware-side temporal smoothing. Visualizations apply their own
  // smoothing on top per layer for cloud feel.
  analyser.smoothingTimeConstant = 0.78;
  analyser.connect(ctx.destination);
  return analyser;
}

/** Connect an HTMLMediaElement (Audio or Video) to the shared
 * analyser. Idempotent — calling twice on the same element is a
 * no-op. Returns the analyser so callers can pass it along to a
 * visualization that wants to bind synchronously. */
export function attachAudio(audio: HTMLMediaElement): AnalyserNode {
  const a = ensureContext();
  if (ctx && ctx.state === "suspended") {
    void ctx.resume();
  }
  if (attached.has(audio)) return a;
  try {
    const src = ctx!.createMediaElementSource(audio);
    src.connect(a);
    attached.add(audio);
  } catch (e) {
    // Most commonly: element was already attached in another path.
    // Safe to ignore — playback still works because the prior source
    // is still in the graph.
    console.warn("attachAudio failed; element may already be connected", e);
  }
  return a;
}

/** Read the shared analyser. Returns null if no audio has ever been
 * attached this session (visualizations should treat this as "render
 * a silent baseline"). */
export function getAnalyser(): AnalyserNode | null {
  return analyser;
}
