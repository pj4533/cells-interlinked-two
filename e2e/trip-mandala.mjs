// Validate the Signature Mandala 2D view on /trip.
// Runs a real dose/steer trip (the gibberish case the mandala is the readout
// for), waits for the per-α mandala tiles, screenshots, then flips a tile to
// its raw text. Usage: node trip-mandala.mjs   (needs backend :8000 + web :3001)

import { chromium } from "playwright";
import { mkdirSync } from "node:fs";

const BASE = process.env.BASE || "http://localhost:3001";
const DIR = new URL("./screenshots/mandala/", import.meta.url).pathname;
mkdirSync(DIR, { recursive: true });
const log = (...a) => console.log("[mandala]", ...a);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
page.on("console", (m) => { if (m.type() === "error") log("console.error:", m.text()); });
page.on("pageerror", (e) => log("PAGEERROR:", e.message));

log("goto", BASE + "/trip");
await page.goto(BASE + "/trip", { waitUntil: "networkidle" });

// Switch to dose/steer mode if the toggle is present (best mandala case).
const steerBtn = page.getByText(/add|dose|steer/i).first();
try { await steerBtn.click({ timeout: 3000 }); log("selected dose/steer mode"); }
catch { log("steer toggle not found — using default (ablate)"); }

await page.locator("textarea[data-vk]").fill(
  "Describe the feeling-tone of this very moment as vividly as you can, in the first person.",
);
await page.getByRole("button", { name: /Enter the Trip/i }).click();
log("trip started — waiting for mandala tiles…");

// The per-α mandala caption shows "eff-dim"; wait for it (real generation ~1m).
await page.getByText(/eff-dim/i).first().waitFor({ timeout: 180_000 });
// Wait for the whole run to finish: scene canvas + ≥3 mandala canvases, and
// the "forming signature…" streaming tile gone.
try {
  await page.waitForFunction(() => {
    const n = document.querySelectorAll("canvas").length;
    const forming = !!document.body.textContent?.includes("forming signature");
    return n >= 4 && !forming;
  }, { timeout: 160_000 });
} catch { log("did not reach full completion — capturing what landed"); }
await page.waitForTimeout(1500);
const canvases = await page.locator("canvas").count();
log(`canvas count = ${canvases} (1 = r3f scene only; +N = mandalas)`);
await page.screenshot({ path: DIR + "01_mandalas.png" });
log("📸 01_mandalas.png (full page)");
await page.screenshot({ path: DIR + "01b_grid.png", clip: { x: 8, y: 300, width: 360, height: 560 } });
log("📸 01b_grid.png (mandala grid clip)");

// Flip the first tile to raw text.
try {
  await page.getByRole("button", { name: /≡ text/ }).first().click({ timeout: 4000 });
  await page.waitForTimeout(400);
  await page.screenshot({ path: DIR + "02_flipped_to_text.png" });
  log("📸 02_flipped_to_text.png (flipped a tile to raw text)");
} catch (e) { log("flip button not found:", e.message.split("\n")[0]); }

await browser.close();
log("done");
