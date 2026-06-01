import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
const DIR = new URL("./screenshots/chat/", import.meta.url).pathname;
mkdirSync(DIR, { recursive: true });
const log = (...a) => console.log("[launch]", ...a);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 860 }, deviceScaleFactor: 2 });
const errs = []; page.on("pageerror", (e) => errs.push(e.message));
await page.goto("http://localhost:3001/chat", { waitUntil: "networkidle" });
await page.getByText(/CHANNEL β · INTERVENTION/).waitFor({ timeout: 8000 });
await page.waitForTimeout(400);
await page.screenshot({ path: DIR + "10_launch_slim.png" });
log("📸 10_launch_slim.png");
// does the transcript area need scrolling?
const overflow = await page.evaluate(() => {
  const el = document.querySelector('[class*="overflow-y-auto"]');
  if (!el) return "no scroll container found";
  return { scrollH: el.scrollHeight, clientH: el.clientHeight, scrolls: el.scrollHeight > el.clientHeight + 2 };
});
log("scroll check:", JSON.stringify(overflow));
// open the protocol picker
const picker = page.getByRole("button", { name: /PROTOCOL/ }).first();
await picker.click();
await page.waitForTimeout(400);
await page.screenshot({ path: DIR + "11_protocols.png" });
log("📸 11_protocols.png");
for (const n of ["DOSING", "VOIGHT-KAMPFF", "DIRECT", "BASELINE"]) {
  const found = await page.getByText(new RegExp(n)).count();
  log(`  protocol "${n}" present: ${found > 0}`);
}
if (errs.length) log("⚠ pageerrors:", errs.join(" | "));
await browser.close();
