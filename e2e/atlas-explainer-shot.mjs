// Screenshot the /autoresearch-dmt explainer COMPONENT in isolation across the
// viewports PJ uses, collapsed + expanded — so we can judge readability,
// wrapping, and how much vertical space it eats at narrow widths.
import { chromium } from "playwright";

const URL = "http://localhost:3001/autoresearch-dmt";
const VIEWS = [
  { name: "desktop", w: 1440, h: 900 },
  { name: "ipad-portrait", w: 820, h: 1180 },
  { name: "phone", w: 390, h: 844 },
];

const browser = await chromium.launch();
for (const v of VIEWS) {
  const ctx = await browser.newContext({ viewport: { width: v.w, height: v.h }, deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  await page.goto(URL, { waitUntil: "networkidle" });
  await page.waitForTimeout(1500);
  const el = page.locator("[data-explainer]");
  await el.scrollIntoViewIfNeeded();
  const hC = await el.evaluate((n) => Math.round(n.getBoundingClientRect().height));
  await el.screenshot({ path: `screenshots/atlas-${v.name}-collapsed.png` });
  await page.getByText("scoring & how to read a row").first().click();
  await page.waitForTimeout(300);
  const hE = await el.evaluate((n) => Math.round(n.getBoundingClientRect().height));
  await el.screenshot({ path: `screenshots/atlas-${v.name}-expanded.png` });
  console.log(`${v.name} (${v.w}px): explainer height collapsed=${hC}px expanded=${hE}px`);
  await ctx.close();
}
await browser.close();
console.log("done");
