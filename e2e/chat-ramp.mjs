import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
const DIR = new URL("./screenshots/chat/", import.meta.url).pathname;
mkdirSync(DIR, { recursive: true });
const log = (...a) => console.log("[ramp]", ...a);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 880 }, deviceScaleFactor: 2 });
let turnPost = null;
page.on("request", (r) => { if (r.method() === "POST" && /\/turn$/.test(r.url())) turnPost = r.postData(); });
const errs = []; page.on("pageerror", (e) => errs.push(e.message));
await page.goto("http://localhost:3001/chat", { waitUntil: "networkidle" });
await page.locator("textarea").first().waitFor({ timeout: 8000 });
// switch channel β to DOSE via the compact control (bottom bar)
await page.getByRole("button", { name: /^DOSE$/ }).first().click();
await page.waitForTimeout(400);
const rampRow = await page.getByText(/DOSE RAMP/i).count();
log("DOSE RAMP row present:", rampRow > 0);
const presets = await page.locator("div", { has: page.getByText(/DOSE RAMP/i) }).first().locator("button").allInnerTexts();
log("ramp buttons:", presets.join("  "));
await page.screenshot({ path: DIR + "13_ramp.png", clip: { x: 0, y: 740, width: 1100, height: 140 } });
log("📸 13_ramp.png");
// pick ramp=2, send a turn, capture POST
await page.getByRole("button", { name: "2", exact: true }).first().click().catch(()=>{});
await page.locator("textarea").first().fill("describe your state");
await page.getByRole("button", { name: /TRANSMIT/i }).click();
await page.waitForTimeout(1200);
log("turn POST:", turnPost);
if (errs.length) log("⚠ pageerrors:", errs.join(" | "));
await browser.close();
