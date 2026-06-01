import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
const DIR = new URL("./screenshots/chat/", import.meta.url).pathname;
mkdirSync(DIR, { recursive: true });
const log = (...a) => console.log("[chat]", ...a);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1200, height: 1000 }, deviceScaleFactor: 2 });
const posts = {};
page.on("request", (r) => {
  if (r.method() === "POST" && /\/chat\/sessions$/.test(r.url())) posts.create = r.postData();
  if (r.method() === "POST" && /\/turn$/.test(r.url())) posts.turn = r.postData();
});
const errs = []; page.on("pageerror", (e) => errs.push(e.message));
await page.goto("http://localhost:3001/chat", { waitUntil: "networkidle" });
await page.getByText(/CHANNEL β · INTERVENTION/).waitFor({ timeout: 8000 });
log("intervention control present");
// switch to DOSE
await page.getByText(/DOSE — steer emotion/).click();
await page.waitForTimeout(500);
await page.screenshot({ path: DIR + "01_dose_setup.png" });
log("📸 01_dose_setup.png");
// pick an uncharted dose
const unch = page.getByRole("button", { name: "tears-in-rain" }).first();
if (await unch.count()) { await unch.click(); log("picked tears-in-rain"); }
// click a dosing-experiential prompt (fires create+turn)
await page.getByText(/Speak in the first person about the texture/).first().click();
await page.waitForTimeout(1500);
log("createSession POST:", posts.create);
log("turn POST:", posts.turn);
if (errs.length) log("⚠ pageerrors:", errs.join(" | "));
await browser.close();
