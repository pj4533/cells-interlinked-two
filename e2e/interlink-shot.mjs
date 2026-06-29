// Screenshot the Interlink page (live conversation) on desktop + phone.
import { chromium } from "playwright";

const URL = "http://localhost:3001/interlink";
const VIEWS = [
  { name: "desktop", w: 1280, h: 1000 },
  { name: "phone", w: 390, h: 844 },
];

const browser = await chromium.launch();
for (const v of VIEWS) {
  const ctx = await browser.newContext({ viewport: { width: v.w, height: v.h }, deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  // 'load' not 'networkidle' — the live SSE EventSource keeps the network busy.
  await page.goto(URL, { waitUntil: "load" });
  await page.waitForTimeout(3000); // resume-on-mount fetch + first render
  await page.screenshot({ path: `screenshots/interlink-${v.name}.png`, fullPage: true });
  console.log(`shot ${v.name}`);
  await ctx.close();
}
await browser.close();
console.log("done");
