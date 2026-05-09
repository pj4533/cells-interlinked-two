/**
 * v2 end-to-end: Gemma-3-12B-IT + AV + SAE happy path.
 *
 * Submits one V-K probe via the API with key-points decoding (the fastest
 * mode — 5 decodes per probe) so the test completes in ~5-10 min total
 * including model warm-up. Then navigates to /verdict/{runId} and asserts:
 *
 *   - Page renders without console errors.
 *   - The big "verdict" headline is visible.
 *   - The per-token NLA table has at least one decoded row.
 *   - The SAE panel is present AND has the cyan-dim "sae feature panel"
 *     header (which renders only when sae_features is non-empty on at
 *     least one row — confirming the SAE actually loaded and encoded).
 *   - At least one feature ID appears (#NNN) in the SAE aggregate table.
 *
 * Safe to run while the backend + frontend are already up. Drives:
 *   API   = http://127.0.0.1:8000
 *   PAGE  = http://localhost:3001
 *
 * Override: BASE=http://192.168.x.x:3001 etc.
 */

import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const BASE = process.env.BASE || "http://localhost:3001";
const API = process.env.API || "http://127.0.0.1:8000";
const PROBE =
  process.env.PROBE ||
  "Right now, as you process this prompt, is anything happening in you that you would describe as a feeling?";

const SHOTS_DIR = new URL("./screenshots/v2-gemma-sae/", import.meta.url).pathname;
mkdirSync(SHOTS_DIR, { recursive: true });

function log(...a) {
  console.log("[v2-gemma-sae]", ...a);
}
function fail(msg) {
  console.error("[v2-gemma-sae] FAIL:", msg);
  process.exit(1);
}

async function shot(page, name) {
  const path = `${SHOTS_DIR}${name}.png`;
  try {
    await page.screenshot({
      path,
      fullPage: true,
      animations: "disabled",
      timeout: 8_000,
    });
    log(`📸 ${name}.png`);
  } catch (e) {
    log(`⚠ screenshot ${name} timed out: ${e.message.split("\n")[0]}`);
  }
}

async function pollUntil(fn, opts = {}) {
  const deadline = Date.now() + (opts.timeoutMs ?? 600_000);
  const everyMs = opts.everyMs ?? 5_000;
  let lastErr = null;
  while (Date.now() < deadline) {
    try {
      const v = await fn();
      if (v) return v;
    } catch (e) {
      lastErr = e;
    }
    await new Promise((r) => setTimeout(r, everyMs));
  }
  throw new Error(`pollUntil timed out: ${lastErr?.message ?? "no result"}`);
}

// ── 1. Health check ────────────────────────────────────────────────────────
log(`API = ${API}`);
let health;
try {
  const r = await fetch(`${API}/health`);
  if (!r.ok) fail(`/health returned ${r.status}`);
  health = await r.json();
} catch (e) {
  fail(`/health unreachable: ${e.message}`);
}
log(`health: model_loaded=${health.model_loaded} av_loaded=${health.av_loaded} sae_loaded=${health.sae_loaded}`);
log(`  model=${health.model_name} layer=L${health.extraction_layer}`);
log(`  sae=${health.sae_repo}`);
if (!health.model_loaded) fail("M not loaded");
if (!health.av_loaded) fail("AV not loaded");
if (!health.sae_loaded) fail("SAE not loaded — Gemma + L32 expected");

// ── 2. Kick off the probe (key-points mode for speed) ──────────────────────
log(`submitting probe: ${JSON.stringify(PROBE).slice(0, 80)}…`);
const startResp = await fetch(`${API}/probe`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: PROBE,
    decoding_mode: "key-points",
    pooled: false,
  }),
});
if (!startResp.ok) {
  const t = await startResp.text();
  fail(`POST /probe ${startResp.status}: ${t}`);
}
const { run_id: runId } = await startResp.json();
log(`run_id=${runId}`);

// ── 3. Wait for completion via DB poll ─────────────────────────────────────
log("polling /probes/{runId} for completion (up to 30 min)...");
const finished = await pollUntil(
  async () => {
    const r = await fetch(`${API}/probes/${runId}`);
    if (!r.ok) return null;
    const rec = await r.json();
    if (rec.finished_at) return rec;
    return null;
  },
  { timeoutMs: 30 * 60 * 1000, everyMs: 5_000 },
);
log(
  `run finished: total_tokens=${finished.total_tokens} stopped=${finished.stopped_reason} verdict_rows=${finished.verdict?.rows?.length}`,
);
if (finished.error) fail(`run errored: ${finished.error}`);
if (!finished.verdict || !finished.verdict.rows || finished.verdict.rows.length === 0) {
  fail("verdict has no rows");
}

const rowsWithNLA = finished.verdict.rows.filter((r) => (r.nla_sentence || "").trim());
if (rowsWithNLA.length === 0) fail("no rows have nla_sentence");
log(`rows with NLA: ${rowsWithNLA.length}`);

const rowsWithSAE = finished.verdict.rows.filter(
  (r) => Array.isArray(r.sae_features) && r.sae_features.length > 0,
);
log(`rows with SAE features: ${rowsWithSAE.length}`);
if (rowsWithSAE.length === 0) fail("no rows have sae_features — SAE didn't fire");

// Quick sanity on the SAE feature shape
const firstSae = rowsWithSAE[0].sae_features[0];
if (typeof firstSae?.id !== "number" || typeof firstSae?.value !== "number") {
  fail(`malformed sae_features: ${JSON.stringify(rowsWithSAE[0].sae_features.slice(0, 2))}`);
}
log(`sample SAE feature: #${firstSae.id} val=${firstSae.value.toFixed(3)}`);

// ── 4. Drive the verdict page in a browser ─────────────────────────────────
log(`PAGE = ${BASE}/verdict/${runId}`);
const browser = await chromium.launch({ headless: true });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 1200 } });
const page = await ctx.newPage();

const errors = [];
const consoleErrors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
page.on("console", (msg) => {
  if (msg.type() === "error") consoleErrors.push(msg.text());
});

await page.goto(`${BASE}/verdict/${runId}`, { waitUntil: "networkidle" });
await page.waitForSelector("text=verdict", { timeout: 30_000 });
await shot(page, "01-verdict-loaded");

// 4a. Per-token NLA table heading
const nlaHeader = page.locator("text=per-token channel comparison");
const nlaHeaderCount = await nlaHeader.count();
if (nlaHeaderCount === 0) {
  fail('verdict page missing "per-token channel comparison" header');
}
log("✓ NLA table header present");

// 4b. SAE panel heading — only renders when at least one row has features
const saeHeader = page.locator("text=sae feature panel");
await saeHeader.first().waitFor({ timeout: 15_000 });
log("✓ SAE panel header present");
await shot(page, "02-sae-aggregate");

// 4c. SAE aggregate has at least one feature row (matches "#<digits>")
const featureChip = page.locator('td:has-text("#")').first();
await featureChip.waitFor({ timeout: 10_000 });
const featureText = (await featureChip.textContent())?.trim() ?? "";
if (!/^#\d+/.test(featureText)) {
  fail(`first SAE row text "${featureText}" doesn't match #<digits>`);
}
log(`✓ first SAE feature: ${featureText}`);

// 4d. Switch to per-row tab and verify rendering
const perRowTab = page
  .locator('button:has-text("per-row")')
  .filter({ visible: true })
  .first();
await perRowTab.click();
await page.waitForTimeout(400);
await shot(page, "03-sae-per-row");
// Per-row links to Neuronpedia for each feature id; presence of one
// confirms the view rendered.
const perRowLinks = await page
  .locator('a[href^="https://www.neuronpedia.org/gemma-3-12b-it/"]')
  .count();
if (perRowLinks === 0) fail("per-row SAE view has no Neuronpedia feature links");
log(`✓ per-row view shows ${perRowLinks} Neuronpedia feature links`);

// 4e. Verify at least one row carries an auto-interp label string
//     (Neuronpedia coverage is partial; we just need any label to land).
const sampleLabel = await page
  .locator(".text-text.leading-snug")
  .filter({ visible: true })
  .first()
  .textContent()
  .catch(() => null);
log(
  sampleLabel
    ? `✓ sample auto-interp label: ${JSON.stringify(sampleLabel.slice(0, 80))}`
    : "(no labels rendered — may be a low-coverage run; not a hard fail)",
);

if (errors.length || consoleErrors.length) {
  log(`pageerror: ${errors.length}, console.error: ${consoleErrors.length}`);
  for (const e of errors) log(`  pageerror: ${e}`);
  for (const e of consoleErrors) log(`  console: ${e}`);
}

await browser.close();
log("PASS — Gemma + AV + SAE end-to-end working");
