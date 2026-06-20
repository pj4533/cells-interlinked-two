// Screenshot the un-steered BASELINE panel on /autoresearch-dmt: collapsed,
// expanded (sample list), and one sample drilled open. Desktop + phone.
import { chromium } from "playwright";

const URL = "http://localhost:3001/autoresearch-dmt";
const VIEWS = [
  { name: "desktop", w: 1440, h: 900 },
  { name: "phone", w: 390, h: 844 },
];

const browser = await chromium.launch();
for (const v of VIEWS) {
  const ctx = await browser.newContext({ viewport: { width: v.w, height: v.h }, deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  await page.goto(URL, { waitUntil: "networkidle" });
  await page.waitForTimeout(1500);
  const panel = page.locator("[data-baseline]");
  if (!(await panel.count())) { console.log(`${v.name}: NO baseline panel (placebo not computed?)`); await ctx.close(); continue; }
  await panel.scrollIntoViewIfNeeded();
  await page.waitForTimeout(200);
  await panel.screenshot({ path: `screenshots/baseline-${v.name}-collapsed.png` });
  // expand the panel
  await page.getByText("BASELINE — NO STEERING").first().click();
  await page.waitForTimeout(900); // lazy fetch
  await panel.scrollIntoViewIfNeeded();
  await panel.screenshot({ path: `screenshots/baseline-${v.name}-expanded.png` });
  // drill one sample open
  const run1 = page.getByText(/^run #1/).first();
  if (await run1.count()) {
    await run1.click();
    await page.waitForTimeout(300);
    await panel.scrollIntoViewIfNeeded();
    await panel.screenshot({ path: `screenshots/baseline-${v.name}-sample.png` });
  }
  const h = await panel.evaluate((n) => Math.round(n.getBoundingClientRect().height));
  console.log(`${v.name}: shot ok (expanded panel height ${h}px)`);
  await ctx.close();
}
await browser.close();
console.log("done");
