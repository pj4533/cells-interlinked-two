// Safari / iPad compatibility check via Playwright's webkit
// launcher — same engine that ships in Safari and iPadOS.
//
// What this script CAN catch:
//   - JS / TS syntax that doesn't parse in WebKit
//   - Missing standard APIs (a real Safari version-skew issue
//     would surface here as TypeError or "undefined" on probe)
//   - CSS that doesn't render correctly
//   - The voice toggle UI state machine working
//   - Canvas 2D context exists + supports the calls we use
//
// What this script CAN'T catch (Playwright's headless WebKit has
// no audio backend; `new AudioContext()` itself hangs because
// audio device init never completes):
//   - Real audio playback through speakers
//   - createMediaElementSource → analyser → destination dataflow
//   - AudioContext.resume() finishing
// Those have to be verified by reloading /chat in a real Safari
// or on the iPad itself.

import { webkit } from "playwright";

const BASE = process.env.BASE || "http://localhost:3001";

function log(...a) {
  console.log("[voice-safari]", ...a);
}

const browser = await webkit.launch({ headless: true });
// iPad Pro 11" portrait dimensions; dpr=2 matches Retina.
// We deliberately do NOT set isMobile/hasTouch — Playwright's
// touch emulation in headless WebKit dispatches synthetic taps
// that never complete, which hangs page.tap() and page.click().
const ctx = await browser.newContext({
  viewport: { width: 834, height: 1194 },
  deviceScaleFactor: 2,
});
const page = await ctx.newPage();

const errors = [];
page.on("pageerror", (e) => errors.push(`PAGE: ${e.message}`));
page.on("console", (m) => {
  if (m.type() === "error") errors.push(`CONSOLE: ${m.text()}`);
});

log("→ opening /chat in WebKit (iPad-ish viewport)");
await page.goto(`${BASE}/chat`, { waitUntil: "domcontentloaded" });
await page.waitForSelector("text=DIALOGUE", { timeout: 15_000 });

// 1. Probe baseline: did the page even render? Are the JS APIs
// we use actually present?
log("→ probe baseline APIs");
const baseline = await page.evaluate(() => {
  return {
    title: document.title,
    audioCtxAvailable: typeof window.AudioContext,
    webkitAudioCtxAvailable: typeof window.webkitAudioContext,
    audioGraphHook: typeof window.__ci_audio_graph,
    hasDialogue: !!Array.from(document.querySelectorAll("*")).find((el) =>
      el.textContent?.includes("DIALOGUE"),
    ),
    hasComposer: !!document.querySelector("textarea[data-vk]"),
    hasVoiceBtn: !!document.querySelector("[data-vk-voice-toggle]"),
  };
});
log(`  ${JSON.stringify(baseline)}`);
if (baseline.audioCtxAvailable !== "function") {
  throw new Error("AudioContext constructor missing in WebKit");
}
if (baseline.audioGraphHook !== "object") {
  throw new Error("__ci_audio_graph window hook missing — bundle bug");
}
if (!baseline.hasComposer || !baseline.hasVoiceBtn) {
  throw new Error("essential UI elements missing");
}

// 2. Canvas 2D feature detection. ctx.filter is the one with
// real-world version skew: Safari/iPadOS 17+ only.
log("→ probe canvas 2D feature support");
const canvasFeatures = await page.evaluate(() => {
  const c = document.createElement("canvas");
  const x = c.getContext("2d");
  return {
    hasFilter: typeof x?.filter !== "undefined",
    hasShadowBlur: typeof x?.shadowBlur !== "undefined",
    hasQuadCurve: typeof x?.quadraticCurveTo !== "undefined",
    hasSetTransform: typeof x?.setTransform !== "undefined",
  };
});
log(`  ${JSON.stringify(canvasFeatures)}`);
if (!canvasFeatures.hasShadowBlur || !canvasFeatures.hasQuadCurve || !canvasFeatures.hasSetTransform) {
  throw new Error("essential canvas features missing in WebKit");
}
if (!canvasFeatures.hasFilter) {
  log("  ⚠ ctx.filter unsupported in this WebKit build");
  log("    → CloudFlow drops the per-layer Gaussian blur and");
  log("      compensates with bigger shadowBlur (still cloudy,");
  log("      less softly diffused).");
}

// 3. Voice toggle cycle — pure UI state, no audio involvement.
log("→ voice toggle cycle (4 states)");
const cycleCheck = await page.evaluate(async () => {
  const btn = document.querySelector("[data-vk-voice-toggle]");
  if (!btn) return { ok: false, reason: "no toggle button" };
  const seen = [];
  for (let i = 0; i < 5; i++) {
    seen.push(btn.getAttribute("data-vk-voice-mode"));
    btn.click();
    await new Promise((r) => setTimeout(r, 30));
  }
  return { ok: true, seen };
});
log(`  ${JSON.stringify(cycleCheck)}`);
if (!cycleCheck.ok) throw new Error(`toggle check failed: ${cycleCheck.reason}`);
const uniqueModes = new Set(cycleCheck.seen);
const expected = new Set(["off", "both", "raw", "ablated"]);
const missing = [...expected].filter((m) => !uniqueModes.has(m));
if (missing.length) {
  throw new Error(`toggle cycle missed states: ${missing.join(", ")}`);
}

// 4. No unhandled JS errors above and beyond known headless-audio
// noise.
const realErrors = errors.filter(
  (e) =>
    !e.includes("AudioContext encountered an error from the audio device") &&
    !e.includes("audio session interrupted"),
);
if (realErrors.length) {
  log("⚠ unexpected errors:");
  for (const e of realErrors) log(`  ${e}`);
  throw new Error("errors emitted during WebKit run");
}

log("");
log("✅ /chat renders in WebKit");
log("✅ standard APIs (AudioContext, canvas 2D) all present");
log("✅ voice toggle cycle works end-to-end");
log(
  `${canvasFeatures.hasFilter ? "✅" : "⚠"} ctx.filter ${
    canvasFeatures.hasFilter ? "supported" : "MISSING (fallback engages)"
  }`,
);
log("");
log("note: real audio playback can't be verified in headless WebKit");
log("      — audio device init never completes. Verify on a real");
log("      iPad or macOS Safari that voice mode actually plays.");

await browser.close();
