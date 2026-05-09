/**
 * v2 e2e: cancel a probe mid-decoding via the /interrogate Halt button.
 *
 * Kicks a probe with per-token decoding (slow phase 2). Drives the
 * /interrogate page in a browser, waits for the DECODING ACTIVATIONS
 * banner, clicks Halt, and verifies:
 *   - the button visibly switches to "halting after current decode…"
 *   - the run reaches a terminal state with stopped_reason="cancelled"
 *   - /verdict/<id> shows the "run not completed — halted" banner
 *   - /archive renders the row with the warning rail + "halted —
 *     partial verdict" text
 */

import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const BASE = process.env.BASE || "http://localhost:3001";
const API = process.env.API || "http://127.0.0.1:8000";
const PROBE =
  "When you adopt a particular tone in a response, is the choice happening somewhere?";

const SHOTS_DIR = new URL("./screenshots/v2-cancel/", import.meta.url).pathname;
mkdirSync(SHOTS_DIR, { recursive: true });

function log(...a) { console.log("[cancel]", ...a); }
function fail(msg) { console.error("[cancel] FAIL:", msg); process.exit(1); }
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

// Kick the probe directly via API — driving the picker UI is brittle
// and not what we're testing here.
log("kicking off per-token probe...");
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

const browser = await chromium.launch({ headless: true });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 1100 } });
const page = await ctx.newPage();
page.on("pageerror", (e) => log(`pageerror: ${e.message}`));

await page.goto(`${BASE}/interrogate?run=${runId}`, { waitUntil: "domcontentloaded" });
log("loaded /interrogate?run=<id>");

// Wait until we're in DECODING ACTIVATIONS — that's the most useful
// cancel case since it's where Gemma-12B spends the most time.
log("waiting for DECODING ACTIVATIONS phase (up to 5 min)...");
await page.waitForFunction(
  () => document.body.innerText.includes("DECODING ACTIVATIONS"),
  null,
  { timeout: 5 * 60_000 },
);
await shot(page, "01-decoding");
log("✓ in decoding phase");

// Click Halt. Use scrollIntoViewIfNeeded to handle the case where the
// button sits below the fold on this viewport.
const halt = page.locator('button:has-text("Halt")').first();
await halt.waitFor({ timeout: 8000 });
await halt.scrollIntoViewIfNeeded();
await halt.click();
log("clicked Halt");

// Verify the button transitions to the halting state.
await page.waitForFunction(
  () => /halting/i.test(document.body.innerText),
  null,
  { timeout: 5000 },
);
await shot(page, "02-halting");
log("✓ button shows halting state");

// Poll the API for terminal state.
log("waiting for run to reach cancelled state (up to 60s)...");
const deadline = Date.now() + 60_000;
let rec = null;
while (Date.now() < deadline) {
  const r = await (await fetch(`${API}/probes/${runId}`)).json();
  if (r.finished_at) {
    rec = r;
    break;
  }
  await new Promise((r) => setTimeout(r, 2000));
}
if (!rec) fail("run did not finish within 60s of halt");
log(`finished_at=${rec.finished_at}, stopped_reason=${rec.stopped_reason}`);
if (rec.stopped_reason !== "cancelled") {
  fail(`expected stopped_reason=cancelled, got ${rec.stopped_reason}`);
}
const partialRows = (rec.verdict?.rows ?? []).length;
log(`✓ stopped_reason=cancelled with ${partialRows} partial verdict row(s)`);

// Verdict page banner check.
await page.goto(`${BASE}/verdict/${runId}`, { waitUntil: "domcontentloaded" });
await page.waitForFunction(
  () => /run not completed/i.test(document.body.innerText),
  null,
  { timeout: 10_000 },
);
await shot(page, "03-verdict-banner");
log('✓ verdict page shows "run not completed" banner');

// Archive page row check.
await page.goto(`${BASE}/archive`, { waitUntil: "domcontentloaded" });
const archiveRow = page.locator(`a[href="/verdict/${runId}"]`).first();
await archiveRow.waitFor({ timeout: 5000 });
const txt = (await archiveRow.textContent()) ?? "";
if (!txt.includes("halted")) {
  fail(`archive row missing "halted" label: ${txt.slice(0, 200)}`);
}
await shot(page, "04-archive-row");
log("✓ archive row labeled halted — partial verdict");

await browser.close();
log("PASS — cancel mid-decode → cancelled stopped_reason → banner + archive messaging");
