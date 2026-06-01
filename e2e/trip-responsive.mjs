import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
const DIR = new URL("./screenshots/redesign/", import.meta.url).pathname;
mkdirSync(DIR, { recursive: true });
const log = (...a) => console.log("[resp]", ...a);
const RUN = process.env.RUN || "e4bb83c05951"; // 7-path dose·rapture run
const browser = await chromium.launch();

async function shoot(name, w, h, tabs) {
  const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 2 });
  const errs = [];
  page.on("pageerror", (e) => errs.push(e.message));
  await page.goto(`http://localhost:3001/trip?run=${RUN}`, { waitUntil: "networkidle" });
  await page.getByText(/TRIP MAPPED|GENERATING|MAPPING/).first().waitFor({ timeout: 15000 });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${DIR}${name}.png` });
  log(`📸 ${name}.png (${w}x${h})`);
  if (tabs) {
    for (const t of ["SIGNATURES", "MEASURES", "SCENE"]) {
      const btn = page.getByRole("button", { name: new RegExp(t) }).first();
      if (await btn.count()) {
        await btn.click().catch(() => {});
        await page.waitForTimeout(900);
        await page.screenshot({ path: `${DIR}${name}_${t.toLowerCase()}.png` });
        log(`📸 ${name}_${t.toLowerCase()}.png`);
      }
    }
  }
  if (errs.length) log(`  ⚠ pageerrors: ${errs.join(" | ")}`);
  await page.close();
}

await shoot("tablet_portrait", 834, 1112, true);
await shoot("tablet_landscape", 1194, 834, true);
await shoot("desktop", 1440, 900, false);
await browser.close();
log("done");
