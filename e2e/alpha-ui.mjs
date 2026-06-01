import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
const DIR = new URL("./screenshots/mandala/", import.meta.url).pathname;
mkdirSync(DIR, { recursive: true });
const log = (...a) => console.log("[alpha-ui]", ...a);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1100, height: 860 } });

let postBody = null;
page.on("request", (r) => {
  if (r.method() === "POST" && r.url().endsWith("/trip")) postBody = r.postData();
});

await page.goto("http://localhost:3001/trip", { waitUntil: "networkidle" });
await page.getByText(/α SWEEP/).waitFor({ timeout: 8000 });
log("α SWEEP row present");

// preset fills the input
await page.getByRole("button", { name: "fine-low" }).click();
const val = await page.locator('input[placeholder^="custom"]').inputValue();
log("after fine-low, input =", JSON.stringify(val));
await page.screenshot({ path: DIR + "04_alpha_ui.png", clip: { x: 220, y: 360, width: 660, height: 300 } });
log("📸 04_alpha_ui.png");

// type a prompt and submit; capture the POST body, then bail before generation
await page.locator("textarea[data-vk]").fill("test prompt for alpha sweep");
await page.getByRole("button", { name: /Enter the Trip/i }).click();
await page.waitForTimeout(1200);
log("POST /trip body =", postBody);
await browser.close();
