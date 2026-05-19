"use client";

/** Shared AudioContext + AnalyserNode for chat-mode playback
 * visualizations.
 *
 * Architecture: clip audio is fetched as bytes, decoded into an
 * AudioBuffer, and played via AudioBufferSource → analyser →
 * destination. We deliberately do NOT use HTMLAudioElement +
 * createMediaElementSource. Two reasons:
 *
 *   1. Safari's autoplay policy gates each `audio.play()` call
 *      independently against a "recent user gesture". Gemma's
 *      ~15s generation gap means every clip's play() gets
 *      rejected and forces a tap-to-play. With BufferSource,
 *      `source.start()` only requires the AudioContext to be in
 *      "running" state — a single initial gesture (the TRANSMIT
 *      click that primes the context) is enough for the rest of
 *      the session.
 *
 *   2. iOS Safari has long-standing bugs with createMediaElementSource:
 *      the audio plays through the element's default output path
 *      while the source node sometimes emits silence into the
 *      graph, so the analyser sees zeros and the visualization
 *      sits dead. AudioBufferSource feeds the graph directly —
 *      audio and analyser data are the same samples.
 *
 * The graph is a singleton (one AudioContext per page lifecycle);
 * the analyser persists across clips so React visualizations just
 * keep reading and naturally show silence between clips.
 */

let ctx: AudioContext | null = null;
let analyser: AnalyserNode | null = null;

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
 *  user-gesture handler (pointerdown, touchstart, click) — that's
 *  when resume() is permitted to take effect. Safe to call any
 *  number of times; it's idempotent. */
export function primeAudioContext(): void {
  ensureContext();
  if (ctx && ctx.state === "suspended") {
    void ctx.resume().catch((e) => {
      console.warn("AudioContext resume rejected", e);
    });
  }
}

export function getAnalyser(): AnalyserNode | null {
  return analyser;
}

export function getContextState(): AudioContextState | null {
  return ctx?.state ?? null;
}

/** Controller returned by `prepareClip`. `play()` starts playback
 *  and resolves when the clip naturally ends, or throws
 *  `"autoplay-blocked"` if the context can't reach `running` state
 *  (caller should display a tap-to-play prompt + retry). */
export interface ClipController {
  /** Start playback. Resolves on natural end. Idempotent — calling
   *  while playing returns a promise that resolves at the same end
   *  event. */
  play: () => Promise<void>;
  /** Schedule duration in seconds, for callers that want to size
   *  progress bars / show ETA. */
  duration: number;
}

/** Decode raw audio bytes and return a controller that can start
 *  playback through the shared analyser. Decode happens up-front so
 *  `play()` is instant once a user gesture is available. */
export async function prepareClip(
  bytes: ArrayBuffer,
): Promise<ClipController> {
  const a = ensureContext();
  if (!a || !ctx) throw new Error("audio context unavailable");
  const buffer = await ctx.decodeAudioData(bytes.slice(0));

  let started = false;
  let endedPromise: Promise<void> | null = null;

  const play = async (): Promise<void> => {
    if (endedPromise) return endedPromise;
    // If the context is suspended, attempt to resume. resume() can
    // only succeed inside a user-gesture call stack on Safari, but
    // we try regardless — it's cheap and the caller will know via
    // the thrown error if it didn't take.
    if (ctx!.state === "suspended") {
      try {
        await ctx!.resume();
      } catch {
        /* keep going to the state check below */
      }
    }
    if (ctx!.state !== "running") {
      throw new Error("autoplay-blocked");
    }
    // Create the source lazily so retries (after autoplay-blocked
    // throws) get a fresh source node. AudioBufferSource is
    // one-shot — `start()` can only be called once per node.
    const source = ctx!.createBufferSource();
    source.buffer = buffer;
    source.connect(a);
    started = true;
    endedPromise = new Promise<void>((resolve) => {
      source.onended = () => resolve();
    });
    source.start();
    return endedPromise;
  };

  void started; // silence unused-var warning in some configs
  return { play, duration: buffer.duration };
}

// Dev/test hook: expose internals on window so e2e can introspect.
if (typeof window !== "undefined") {
  (window as unknown as Record<string, unknown>).__ci_audio_graph = {
    getContextState,
    getAnalyser,
    primeAudioContext,
  };
}
