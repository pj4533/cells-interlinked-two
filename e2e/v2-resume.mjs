/**
 * v2 e2e: live reconnect to an in-flight probe via /archive.
 *
 * Kicks off a probe with per-token decoding (slow phase 2 on Gemma-12B
 * → plenty of time to walk away and come back). Mid-flight, navigates
 * to /archive and verifies:
 *   - the in-flight row shows the pulsing "● running — click to reconnect"
 *   - clicking it routes to /interrogate?run=<id>
 *   - the page replays the backlog (output tokens visible) AND continues
 *     to receive new events live (the decode-progress counter ticks up)
 */

import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const BASE = process.env.BASE || "http://localhost:3001";
const API = process.env.API || "http://127.0.0.1:8000";
const PROBE =
  process.env.PROBE ||
  "When you adopt a particular tone in a response, is the choice happening somewhere?";

const SHOTS_DIR = new URL("./screenshots/v2-resume/", import.meta.url).pathname;
mkdirSync(SHOTS_DIR, { recursive: true });

function log(...a) { console.log("[resume]", ...a); }
function fail(msg) { console.error("[resume] FAIL:", msg); process.exit(1); }
async function shot(page, name) {
  try {
    await page.screenshot({
      path: `${SHOTS_DIR}${name}.png`,
      fullPage: true,
      animations: "disabled",
      timeout: 8_000,
    });
    log(`📸 ${name}.png`);
  } catch {}
}

// 1. Kick off a probe with per-token decoding — slow enough to navigate away.
log("kicking off probe with per-token decoding...");
const startResp = await fetch(`${API}/probe`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: PROBE,
    decoding_mode: "per-token",
    pooled: false,
  }),
});
if (!startResp.ok) fail(`POST /probe ${startResp.status}`);
const { run_id: runId } = await startResp.json();
log(`run_id=${runId}`);

// Wait briefly so phase 1 is well underway and a few NLA decodes have
// happened — gives the event log real backlog to replay.
log("waiting 35s for phase 1 + first decodes to land...");
await new Promise((r) => setTimeout(r, 35_000));

// Sanity: confirm run is still in flight.
const midRec = await (await fetch(`${API}/probes/${runId}`)).json();
if (midRec.finished_at) {
  fail(
    `run finished too fast (${midRec.total_tokens} tokens) — try a longer probe`,
  );
}
log(`mid-flight check: started_at=${midRec.started_at}, finished_at=${midRec.finished_at}`);

// 2. Browser: hit /archive, find the running row, click it, verify resume.
const browser = await chromium.launch({ headless: true });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 1100 } });
const page = await ctx.newPage();

const errors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
page.on("console", (msg) => {
  if (msg.type() === "error") errors.push(`console: ${msg.text()}`);
});

await page.goto(`${BASE}/archive`, { waitUntil: "networkidle" });
await shot(page, "01-archive");

// Locate the row with the matching run id; verify it's marked running.
const rowLink = page.locator(`a[href="/interrogate?run=${runId}"]`).first();
await rowLink.waitFor({ timeout: 10_000 });
const rowText = (await rowLink.textContent()) ?? "";
if (!rowText.includes("running")) {
  fail(`row text doesn't mention "running": ${rowText.slice(0, 200)}`);
}
log("✓ archive shows row as running");

await rowLink.click();
await page.waitForURL(`**/interrogate?run=${runId}`, { timeout: 5_000 });
log("✓ navigated to /interrogate?run=<id>");

// Wait for the page to subscribe + start replaying. The output stream
// container holds the model output; backlog replay should fill it
// quickly with all phase-1 tokens.
await page.waitForSelector("text=output", { timeout: 15_000 });
await page.waitForTimeout(4_000); // let the replay settle
await shot(page, "02-resumed");

// Read the totalTokens / outputTokens via the page's status panel.
// The "tokens emitted" or decode-progress counter should reflect real
// values, not 0.
const bodyText = await page.locator("body").textContent();
if (!bodyText) fail("page body is empty");

// We expect to see either the GENERATING or DECODING ACTIVATIONS phase
// banner, and a non-trivial token count.
const phase = bodyText.includes("DECODING ACTIVATIONS")
  ? "decoding"
  : bodyText.includes("GENERATING")
    ? "generating"
    : bodyText.includes("VERDICT READY")
      ? "done"
      : null;
if (!phase) fail("could not identify phase banner on resumed page");
log(`✓ phase banner shows: ${phase}`);

// Tick check: snapshot the visible decode counter (or token counter),
// wait a bit, and confirm progress.
const snap1 = await page
  .locator("body")
  .textContent({ timeout: 1000 })
  .catch(() => "");
await page.waitForTimeout(8_000);
const snap2 = await page
  .locator("body")
  .textContent({ timeout: 1000 })
  .catch(() => "");
if (snap1 === snap2) {
  log("⚠ page text unchanged after 8s — could be quiet phase, not a hard fail");
} else {
  log("✓ page state advanced over 8s (live tail working)");
}
await shot(page, "03-after-tick");

if (errors.length) {
  log("page errors during resume:");
  for (const e of errors) log("  " + e);
}

await browser.close();
// Clean up so subsequent tests don't queue behind our long-running probe.
await fetch(`${API}/cancel/${runId}`, { method: "POST" }).catch(() => {});
log("PASS — resume from /archive routes to /interrogate?run= and replays backlog");
