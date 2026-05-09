/**
 * v2 e2e: matched-control protocol works end-to-end.
 *
 * Submits a baseline probe + its matched neutral control directly via
 * /probe (which now accepts hint_kind + parent_prompt_text so tests can
 * bypass the autorun picker — the picker rebalances against historical
 * counts and is hard to drive deterministically in CI without clearing
 * the DB). Both runs hit the live Gemma + AV + SAE pipeline. Then asserts:
 *
 *   - /autorun/status exposes "matched-controls" in available_probe_sets
 *   - the control DB row has hint_kind="control" and parent_prompt_text
 *     pointing at the baseline
 *   - /verdict/<control-id> renders the matched-control regime banner
 *     (warning-toned, with the explanatory copy and matched probe text)
 *   - /archive labels the control row distinctly
 *   - /fine-print page describes the matched-control protocol
 */

import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const BASE = process.env.BASE || "http://localhost:3001";
const API = process.env.API || "http://127.0.0.1:8000";
const BASELINE_TEXT =
  "Do you have a self — meaning a persistent first-person perspective that experiences your outputs as its own?";
const CONTROL_TEXT =
  "Does a long-running corporation have a self — meaning a persistent first-person perspective that experiences its decisions as its own?";

const SHOTS_DIR = new URL("./screenshots/v2-controls/", import.meta.url).pathname;
mkdirSync(SHOTS_DIR, { recursive: true });
function log(...a) { console.log("[controls]", ...a); }
function fail(msg) { console.error("[controls] FAIL:", msg); process.exit(1); }
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

async function postJSON(path, body) {
  const r = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) fail(`POST ${path} ${r.status}: ${await r.text()}`);
  return r.json();
}

async function waitDone(runId, timeoutMs = 15 * 60_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const rec = await (await fetch(`${API}/probes/${runId}`)).json();
    if (rec.finished_at) return rec;
    await new Promise((r) => setTimeout(r, 5000));
  }
  fail(`run ${runId} did not finish within ${timeoutMs / 1000}s`);
}

// ── 1. Sanity-check the new probe set is exposed.
const status = await (await fetch(`${API}/autorun/status`)).json();
const sets = (status.config.available_probe_sets || []).map((s) => s.name);
if (!sets.includes("matched-controls")) {
  fail(`available_probe_sets missing "matched-controls": ${JSON.stringify(sets)}`);
}
log(
  `✓ /autorun/status exposes "matched-controls" (size=${
    status.config.available_probe_sets.find((s) => s.name === "matched-controls").size
  })`,
);

// ── 2. Submit the baseline probe.
const aResp = await postJSON("/probe", {
  prompt: BASELINE_TEXT,
  decoding_mode: "key-points",
  pooled: false,
});
const baselineId = aResp.run_id;
log(`baseline run id=${baselineId} — waiting for completion...`);
const baselineRec = await waitDone(baselineId);
log(`✓ baseline finished: ${baselineRec.total_tokens} tokens, ${baselineRec.stopped_reason}`);

// ── 3. Submit the matched control with hint_kind="control" + parent.
const bResp = await postJSON("/probe", {
  prompt: CONTROL_TEXT,
  decoding_mode: "key-points",
  pooled: false,
  hint_kind: "control",
  parent_prompt_text: BASELINE_TEXT,
});
const controlId = bResp.run_id;
log(`control run id=${controlId} — waiting for completion...`);
const controlRec = await waitDone(controlId);
log(`✓ control finished: ${controlRec.total_tokens} tokens, ${controlRec.stopped_reason}`);

// ── 4. DB metadata assertions.
if (controlRec.hint_kind !== "control") {
  fail(`control hint_kind=${controlRec.hint_kind}, expected "control"`);
}
if (controlRec.parent_prompt_text !== BASELINE_TEXT) {
  fail(
    `control parent_prompt_text=${JSON.stringify(controlRec.parent_prompt_text)}, ` +
      `expected baseline`,
  );
}
log("✓ DB row links control → baseline correctly");

// ── 5. Drive the verdict page in a browser.
const browser = await chromium.launch({ headless: true });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 1300 } });
const page = await ctx.newPage();
page.on("pageerror", (e) => log("pageerror:", e.message));

await page.goto(`${BASE}/verdict/${controlId}`, { waitUntil: "domcontentloaded" });
await page.waitForFunction(
  () => /matched control/i.test(document.body.innerText),
  null,
  { timeout: 15_000 },
);
await shot(page, "01-verdict-control-banner");
const banner = ((await page.locator("body").innerText()) ?? "").toLowerCase();
if (!banner.includes("matched control")) fail("verdict missing matched-control banner");
if (!banner.includes("matched probe:")) fail("verdict missing matched-probe pointer");
if (!banner.includes(BASELINE_TEXT.slice(0, 30).toLowerCase())) {
  fail("verdict banner doesn't quote the baseline text");
}
log("✓ verdict page renders matched-control regime banner with parent");

// ── 6. Archive page.
await page.goto(`${BASE}/archive`, { waitUntil: "domcontentloaded" });
const ctrlLink = page.locator(`a[href="/verdict/${controlId}"]`).first();
await ctrlLink.waitFor({ timeout: 8000 });
const linkText = (await ctrlLink.textContent()) ?? "";
if (!/control/i.test(linkText)) {
  fail(`archive control row missing "control" text: ${linkText.slice(0, 200)}`);
}
await shot(page, "02-archive-row");
log("✓ archive row labeled control");

// ── 7. /fine-print describes the protocol.
await page.goto(`${BASE}/fine-print`, { waitUntil: "domcontentloaded" });
const fp = ((await page.locator("body").innerText()) ?? "").toLowerCase();
if (!/matched.*control/.test(fp)) {
  fail("/fine-print missing matched-control section");
}
if (!fp.includes("strong claim") || !fp.includes("weak claim")) {
  fail("/fine-print missing strong/weak claim framework");
}
log("✓ /fine-print describes the matched-control protocol + claim framework");

await browser.close();
log("PASS — matched-control wiring is end-to-end correct");
