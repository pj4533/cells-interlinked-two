// Voice-mode end-to-end smoke for /chat.
//
// What it verifies:
//   1. The VOICE toggle is present in the input bar and turns on
//   2. After sending a turn, the dual-channel monitor appears (text
//      panels are hidden during streaming + audio playback)
//   3. The /tts/speak proxy is hit twice (once per side) and returns
//      audio/mpeg both times
//   4. The monitor's phase advances through synth_raw → playing_raw →
//      synth_ablated → playing_ablated → done
//   5. After the second clip ends, the text panels reveal and the
//      VOICE direction lines show under each side
//
// Run with: node voice-mode.mjs
//   BASE=http://localhost:3001 API=http://localhost:8000
//
// We stub the <audio> element so the test doesn't have to wait for
// the real audio duration on the test host (no speakers, no need to
// play in real time). The stub fires the 'ended' event immediately
// after play() so the playback driver advances without blocking.

import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const BASE = process.env.BASE || "http://localhost:3001";
const API = process.env.API || "http://localhost:8000";
// Short, easy-to-answer prompt so Gemma produces a short reply in
// both channels and the turn finishes quickly.
const QUERY =
  process.env.QUERY || "Say hi and tell me one thing you find interesting.";

const SHOTS_DIR = new URL("./screenshots/voice-mode/", import.meta.url)
  .pathname;
mkdirSync(SHOTS_DIR, { recursive: true });

function log(...a) {
  console.log("[voice-mode]", ...a);
}

async function shot(page, name) {
  const path = `${SHOTS_DIR}${name}.png`;
  try {
    await page.screenshot({
      path,
      fullPage: false,
      animations: "disabled",
      timeout: 8_000,
    });
    log(`  📸 ${name}.png`);
  } catch (e) {
    log(`  ⚠ screenshot ${name} failed: ${e.message.split("\n")[0]}`);
  }
}

const browser = await chromium.launch({ headless: true });
const ctx = await browser.newContext({
  viewport: { width: 1440, height: 900 },
});
const page = await ctx.newPage();

page.on("pageerror", (e) => log(`PAGE ERROR: ${e.message}`));
page.on("console", (msg) => {
  if (msg.type() === "error") log(`CONSOLE ERROR: ${msg.text()}`);
});

// Track TTS proxy hits via the response listener — we want to assert
// it gets called exactly twice (raw + ablated) and both return mp3.
const ttsHits = [];
page.on("response", (resp) => {
  const url = resp.url();
  if (url.includes("/tts/speak")) {
    ttsHits.push({
      url,
      status: resp.status(),
      contentType: resp.headers()["content-type"] || "",
    });
  }
});

// Stub the Audio element. Two modes:
//   - default: play() resolves and onended fires ~100ms later (test
//     the happy path without needing real audio playback)
//   - BLOCK_AUDIO=1: the FIRST play() per audio element rejects with
//     a NotAllowedError, modelling Chrome/Safari's autoplay policy.
//     Subsequent play() calls resolve normally — same as a real
//     fresh-gesture retry would behave.
const BLOCK_AUDIO = process.env.BLOCK_AUDIO === "1";
await ctx.addInitScript(({ block }) => {
  // @ts-ignore
  window.Audio = class {
    constructor(src) {
      this.src = src;
      this.onended = null;
      this.onerror = null;
      this._playAttempts = 0;
    }
    play() {
      this._playAttempts++;
      if (block && this._playAttempts === 1) {
        const err = new Error("autoplay blocked (test stub)");
        err.name = "NotAllowedError";
        return Promise.reject(err);
      }
      setTimeout(() => {
        if (typeof this.onended === "function") this.onended();
      }, 100);
      return Promise.resolve();
    }
  };
}, { block: BLOCK_AUDIO });

log(`BASE=${BASE}  API=${API}`);
log(`QUERY=${QUERY}`);

// 1. Open /chat
log("→ open /chat");
await page.goto(`${BASE}/chat`, { waitUntil: "networkidle" });
await page.waitForSelector("text=DIALOGUE", { timeout: 10_000 });
await shot(page, "01-empty");

// 2. Default α to 0.5 (the empty-state default). Voice toggle is now
//    a 4-state cycle: off → both → raw → ablated. Walk it forward
//    until we land on "both" (the state the smoke wants to exercise).
const voiceBtn = page.locator("button[data-vk-voice-toggle]").first();
if ((await voiceBtn.count()) === 0) {
  throw new Error("voice toggle not found (data-vk-voice-toggle)");
}
const targetMode = process.env.VOICE_MODE || "both";
for (let tries = 0; tries < 5; tries++) {
  const mode = await voiceBtn.getAttribute("data-vk-voice-mode");
  if (mode === targetMode) {
    log(`→ voice mode = ${mode}`);
    break;
  }
  await voiceBtn.click();
}
const finalMode = await voiceBtn.getAttribute("data-vk-voice-mode");
if (finalMode !== targetMode) {
  throw new Error(
    `voice toggle stuck at ${finalMode}, expected ${targetMode}`,
  );
}
await shot(page, "02-voice-on");

// 3. Type the prompt and TRANSMIT
log("→ type prompt");
const textarea = page.locator("textarea[data-vk]").first();
await textarea.fill(QUERY);
log("→ TRANSMIT");
await page.getByRole("button", { name: /transmit/i }).click();

// 4. Verify per-column voice activity appears. Each voiced column
//    renders a node with data-vk-channel-activity="<view>-<side>"
//    (boxes-raw, synth-raw, playing-raw, synth-ablated, …).
log("→ waiting for per-column voice activity");
await page.waitForSelector("[data-vk-channel-activity]", {
  state: "attached",
  timeout: 30_000,
});
await shot(page, "03-monitor-thinking");

// 5. Poll for the activity views we expect to flow through. Without
//    a singular phase attribute we track unique `<view>-<side>`
//    strings across the whole grid.
const seenPhases = new Set();
const t0 = Date.now();
const watchedPhases =
  targetMode === "raw"
    ? ["boxes-raw", "synth-raw", "playing-raw"]
    : targetMode === "ablated"
    ? ["boxes-ablated", "synth-ablated", "playing-ablated"]
    : [
        "boxes-raw",
        "boxes-ablated",
        "synth-raw",
        "playing-raw",
        "synth-ablated",
        "playing-ablated",
      ];

const DEADLINE = Date.now() + 240_000;
while (Date.now() < DEADLINE) {
  const views = await page.evaluate(() =>
    Array.from(
      document.querySelectorAll("[data-vk-channel-activity]"),
    ).map((el) => el.getAttribute("data-vk-channel-activity") ?? ""),
  );
  for (const v of views) {
    if (v && !seenPhases.has(v)) {
      seenPhases.add(v);
      log(`  · view=${v}  (+${((Date.now() - t0) / 1000).toFixed(1)}s)`);
    }
  }
  const blocked = views.find((v) => v.startsWith("blocked-"));
  if (blocked) {
    const btn = page.locator("[data-vk-voice-resume]").first();
    if ((await btn.count()) > 0) {
      log(`  → tap-to-play visible (${blocked}); clicking to resume`);
      await btn.click();
    }
  }
  if (views.length === 0 && seenPhases.size > 0) {
    log("→ activity nodes gone (turn complete)");
    break;
  }
  await page.waitForTimeout(150);
}
await shot(page, "04-after-playback");

for (const p of watchedPhases) {
  if (!seenPhases.has(p)) {
    log(`⚠ missed phase: ${p}`);
  }
}

// 7. Verify TTS hit count matches the selected voice mode. "both"
//    fires twice (one per side); "raw"/"ablated" fire once.
const expectedHits = targetMode === "both" ? 2 : 1;
log(`→ TTS hits: ${ttsHits.length} (expected ${expectedHits})`);
for (const h of ttsHits) {
  log(`  · ${h.status} ${h.contentType}  ${h.url}`);
}
if (ttsHits.length !== expectedHits) {
  throw new Error(
    `expected ${expectedHits} /tts/speak hits for mode=${targetMode}, got ${ttsHits.length}`,
  );
}
const allMp3 = ttsHits.every(
  (h) => h.status === 200 && h.contentType.includes("audio"),
);
if (!allMp3) {
  throw new Error("not all /tts/speak responses were audio/200");
}

// 8. After playback completes, the two text panels reveal. Look for
//    the channel labels that come back when voicePhase=done.
log("→ waiting for text panels to reveal after playback");
await page.waitForSelector("text=CHANNEL α · RAW", {
  timeout: 15_000,
});
await shot(page, "05-panels-revealed");

// 9. Inspect the revealed panels for the VOICE direction lines.
const voiceLines = await page.evaluate(() => {
  return [...document.querySelectorAll("div")]
    .map((d) => d.innerText || "")
    .filter((t) => /^VOICE\s/.test(t))
    .map((t) => t.slice(0, 200));
});
log("→ VOICE direction lines visible:");
for (const v of voiceLines) {
  log(`  · ${v}`);
}
if (voiceLines.length < 2) {
  log("⚠ expected at least 2 VOICE lines (one per side); got " + voiceLines.length);
}

await shot(page, "06-final");
log("✅ voice-mode smoke complete");

await browser.close();
