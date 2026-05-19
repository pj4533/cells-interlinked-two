"use client";

/** Audio playback + analyser for chat-mode voice clips.
 *
 * Hybrid playback architecture (Safari workaround):
 *
 *   - AUDIBLE OUTPUT travels through an HTMLAudioElement playing the
 *     blob URL directly. This is the same path Spotify and every
 *     other "just works in Safari" media app uses. Safari's
 *     AudioContext.destination → speakers route appears to be broken
 *     on at least some macOS configurations — the AudioContext state
 *     reports "running", the tab's audio-playing indicator turns on,
 *     samples flow into the AnalyserNode (waveform animates), but
 *     no audible sound reaches the speakers. Chrome on the same
 *     machine works fine with the Web Audio path, but Safari does
 *     not. The HTMLAudioElement path bypasses Safari's broken Web
 *     Audio output and goes through the media-playback path that
 *     does work.
 *
 *   - VISUALIZATION DATA still comes from Web Audio: we decode the
 *     same bytes into an AudioBuffer and play it through an
 *     AudioBufferSource → analyser → silent-gain → destination
 *     chain. The silent gain (0) prevents this path from doubling
 *     the audible output on browsers where Web Audio output works.
 *     The analyser sits in a chain that reaches destination so
 *     Safari/WebKit actually pulls samples through it (leaf
 *     analyser nodes don't reliably get pulled on some
 *     implementations).
 *
 * The Web Audio path also handles the Safari "suspended" /
 * "interrupted" state recovery via primeAudioContext + the
 * statechange listener wired below.
 *
 * The graph (ctx + analyser) is a singleton; the analyser persists
 * across clips so React visualizations just keep reading and
 * naturally show silence between clips.
 */

let ctx: AudioContext | null = null;
let analyser: AnalyserNode | null = null;
// Persistent zero-gain node that terminates the Web Audio chain.
// Putting the chain through destination keeps the analyser in the
// pull path on browsers that won't process leaf analysers, while
// the zero gain prevents this path from being audible on browsers
// where AudioContext.destination DOES work (we want the user to
// hear the HTMLAudio path only).
let silentSink: GainNode | null = null;

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
    silentSink = ctx.createGain();
    silentSink.gain.value = 0;
    analyser.connect(silentSink);
    silentSink.connect(ctx.destination);
  } catch (e) {
    console.warn("AudioContext creation failed", e);
    return null;
  }
  return analyser;
}

let statechangeWired = false;
function wireStatechangeRecovery(c: AudioContext): void {
  if (statechangeWired) return;
  statechangeWired = true;
  c.addEventListener("statechange", () => {
    console.log("[audio-graph] state →", c.state);
    if (c.state === "suspended" || c.state === ("interrupted" as AudioContextState)) {
      void c.resume().catch((e) => {
        console.warn("[audio-graph] auto-resume rejected", e);
      });
    }
  });
}

/** Initialize the AudioContext and resume it. Call this from a
 *  user-gesture handler (pointerdown, touchstart, click). Safe to
 *  call any number of times. Resumes from both "suspended" (W3C)
 *  and Safari's non-standard "interrupted" state.
 */
export function primeAudioContext(): void {
  ensureContext();
  if (!ctx) return;
  wireStatechangeRecovery(ctx);
  if (ctx.state !== "running") {
    console.log("[audio-graph] prime resume from state:", ctx.state);
    void ctx.resume().catch((e) => {
      console.warn("[audio-graph] resume rejected", e);
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
 *  `"autoplay-blocked"` if the browser refused HTMLAudio playback
 *  (caller should display a tap-to-play prompt + retry).
 */
export interface ClipController {
  /** Start playback. Resolves on natural end. Idempotent — calling
   *  while playing returns a promise that resolves at the same end
   *  event. */
  play: () => Promise<void>;
  /** Schedule duration in seconds, for callers that want to size
   *  progress bars / show ETA. */
  duration: number;
}

/** Decode raw audio bytes + create an HTMLAudioElement bound to the
 *  same content. Returns a controller whose `play()` starts both
 *  the HTMLAudio (audible) and the Web Audio source (visualization
 *  data) in parallel.
 *
 *  Decode + HTMLAudio creation happens up-front so `play()` is
 *  instant once a user gesture is available.
 */
export async function prepareClip(
  bytes: ArrayBuffer,
): Promise<ClipController> {
  const a = ensureContext();
  if (!a || !ctx) throw new Error("audio context unavailable");
  // decodeAudioData copies bytes; slice(0) ensures the input buffer
  // can't be invalidated mid-decode by an early GC of the original.
  const buffer = await ctx.decodeAudioData(bytes.slice(0));

  // Build an HTMLAudioElement that plays the same audio. We make a
  // fresh Blob/URL here (rather than reusing the caller's URL) so
  // this module owns the lifecycle and can revoke on cleanup. The
  // blob's MIME type — audio/wav — matches what the server emits.
  const blob = new Blob([bytes], { type: "audio/wav" });
  const audioUrl = URL.createObjectURL(blob);
  const audio = new Audio();
  audio.preload = "auto";
  audio.src = audioUrl;
  // Important for Safari: ensure the element is not muted and its
  // volume is non-zero.
  audio.muted = false;
  audio.volume = 1.0;

  let endedPromise: Promise<void> | null = null;

  const play = async (): Promise<void> => {
    if (endedPromise) return endedPromise;

    // Best-effort: bring the AudioContext to "running" so the
    // analyser path actually processes samples. Whether or not this
    // succeeds, audio playback proceeds via the HTMLAudio element.
    if (ctx!.state !== "running") {
      try {
        await ctx!.resume();
      } catch (e) {
        console.warn("[audio-graph] resume rejected in play()", e);
      }
    }
    console.log(
      "[audio-graph] play() entry — ctx.state:",
      ctx!.state,
      "sampleRate:",
      ctx!.sampleRate,
      "bufferRate:",
      buffer.sampleRate,
      "bufferDuration:",
      buffer.duration,
    );

    // Set up the Web Audio side (analyser feed). The chain is
    // source → analyser → silentSink → destination. silentSink is
    // pre-built at gain=0 so this path stays inaudible while
    // ensuring the analyser is pulled. Failure here is non-fatal —
    // the audible HTMLAudio path below is independent.
    try {
      const source = ctx!.createBufferSource();
      source.buffer = buffer;
      source.connect(a);
      source.start();
    } catch (e) {
      console.warn("[audio-graph] Web Audio source setup failed", e);
    }

    // Set up the HTMLAudio side (audible output). Catch
    // autoplay rejections so the caller can show a tap-to-play
    // prompt and retry inside a fresh gesture.
    const htmlEndedPromise = new Promise<void>((resolve) => {
      audio.addEventListener("ended", () => resolve(), { once: true });
      // If the element errors out, resolve too so we don't hang
      // forever. The error is logged for diagnostic purposes.
      audio.addEventListener(
        "error",
        () => {
          console.warn(
            "[audio-graph] HTMLAudio error",
            audio.error?.code,
            audio.error?.message,
          );
          resolve();
        },
        { once: true },
      );
    });

    try {
      console.log("[audio-graph] HTMLAudio.play() — readyState:", audio.readyState);
      await audio.play();
    } catch (e) {
      const name =
        e && typeof e === "object" && "name" in e
          ? (e as { name: string }).name
          : "PlayError";
      console.warn(`[audio-graph] HTMLAudio.play() rejected: ${name}`);
      throw new Error("autoplay-blocked");
    }

    endedPromise = (async () => {
      // Wait for HTMLAudio's "ended" — that's the user's audible
      // experience. The Web Audio source plays in parallel through
      // a silent sink for the analyser; we don't separately gate
      // on it ending.
      await htmlEndedPromise;
      try {
        URL.revokeObjectURL(audioUrl);
      } catch {
        /* ignore */
      }
    })();
    return endedPromise;
  };

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
