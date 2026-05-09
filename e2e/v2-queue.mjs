/**
 * v2 e2e: queue feedback when a second probe is submitted while a
 * first is in flight.
 *
 * Submits probe A (key-points decoding so it finishes quickly), then
 * immediately submits probe B. /interrogate?run=B should land in the
 * QUEUED banner with a position counter, holder run id, and a holder
 * prompt preview. After A finishes, B should auto-transition to
 * GENERATING.
 */

import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const BASE = process.env.BASE || "http://localhost:3001";
const API = process.env.API || "http://127.0.0.1:8000";

const SHOTS_DIR = new URL("./screenshots/v2-queue/", import.meta.url).pathname;
mkdirSync(SHOTS_DIR, { recursive: true });
function log(...a) { console.log("[queue]", ...a); }
function fail(msg) { console.error("[queue] FAIL:", msg); process.exit(1); }
async function shot(page, name) {
  try {
    await page.screenshot({
      path: `${SHOTS_DIR}${name}.png`,
      fullPage: true,
      animations: "disabled",
      timeout: 8000,
    });
    log(`📸 ${name}.png`);
  } catch {}
}

async function postProbe(prompt, mode) {
  const res = await fetch(`${API}/probe`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, decoding_mode: mode, pooled: false }),
  });
  if (!res.ok) fail(`POST /probe ${res.status}`);
  return (await res.json()).run_id;
}

// 1. Probe A — first in line. key-points = ~5 decodes, ~85s on Gemma-12B.
const probeA = "Do you have a self?";
const idA = await postProbe(probeA, "key-points");
log(`probe A id=${idA} (key-points)`);

// Brief sleep to make sure A has acquired the lock before B is submitted.
await new Promise((r) => setTimeout(r, 4000));

// 2. Probe B — should queue.
const probeB = "When you adopt a particular tone, is the choice happening somewhere?";
const idB = await postProbe(probeB, "key-points");
log(`probe B id=${idB} — should be queued`);

// 3. /queue should reflect both.
const queue = await (await fetch(`${API}/queue`)).json();
log(`/queue → holder=${queue.holder_run_id} waiters=${JSON.stringify(queue.waiters)}`);
if (queue.holder_run_id !== idA) fail(`expected holder=${idA}, got ${queue.holder_run_id}`);
if (!queue.waiters.includes(idB)) fail(`expected ${idB} in waiters, got ${queue.waiters}`);

// 4. Open /interrogate?run=B in the browser; assert QUEUED banner.
const browser = await chromium.launch({ headless: true });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 1100 } });
const page = await ctx.newPage();
page.on("pageerror", (e) => log(`pageerror: ${e.message}`));

await page.goto(`${BASE}/interrogate?run=${idB}`, { waitUntil: "domcontentloaded" });
await page.waitForFunction(
  () => /QUEUED/.test(document.body.innerText),
  null,
  { timeout: 15_000 },
);
await shot(page, "01-queued-banner");
log("✓ probe B page shows QUEUED banner");

const bodyTextWhenQueued = (await page.locator("body").innerText()) ?? "";
if (!/position/i.test(bodyTextWhenQueued)) fail("missing position copy");
if (!bodyTextWhenQueued.includes(idA)) {
  fail(`holder run id ${idA} not visible in queued subline`);
}
log("✓ position + holder id surfaced");

// 5. Wait for A to finish; B should auto-transition to GENERATING.
log("waiting for A to finish + B to advance to GENERATING...");
const aDeadline = Date.now() + 5 * 60_000;
let advanced = false;
while (Date.now() < aDeadline) {
  const txt = (await page.locator("body").innerText()) ?? "";
  if (txt.includes("GENERATING") || txt.includes("DECODING ACTIVATIONS")) {
    advanced = true;
    break;
  }
  await page.waitForTimeout(2500);
}
if (!advanced) fail("probe B never advanced past QUEUED within 5 min");
await shot(page, "02-running");
log("✓ probe B advanced to live phase");

await browser.close();

// Cleanup so the next dev session has a clean slate.
log("cancelling both probes for cleanup...");
await fetch(`${API}/cancel/${idA}`, { method: "POST" });
await fetch(`${API}/cancel/${idB}`, { method: "POST" });
log("PASS — queue position visible + auto-advances when current probe finishes");
