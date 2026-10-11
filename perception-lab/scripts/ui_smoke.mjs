#!/usr/bin/env node
// UI smoke test against the live stack: load the app, press Analyze, record the status phases
// actually shown (queued/running/completed), and save screenshots as evidence.
//   node scripts/ui_smoke.mjs [--app http://127.0.0.1:5173] [--out artifacts/screenshots]
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(join(ROOT, "frontend", "package.json"));
const { chromium } = require("playwright-core");
const args = process.argv.slice(2);
const opt = (n, d) => (args.includes(n) ? args[args.indexOf(n) + 1] : d);
const APP = opt("--app", "http://127.0.0.1:5173");
const OUT = resolve(opt("--out", join(ROOT, "artifacts", "screenshots")));
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--use-angle=metal", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1560, height: 1100 } });
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => m.type() === "error" && !m.text().includes("favicon") && errors.push(m.text()));
await page.goto(APP);
await page.getByText("LIVE local inference").or(page.getByText("FIXTURE MODE")).first().waitFor({ timeout: 20000 });
await page.waitForTimeout(1500);
await page.screenshot({ path: join(OUT, "01-initial.png"), fullPage: false });

// Interactive analysis of the depiction board from a custom angle
await page.getByLabel("Truck picture on a flat board").check();
await page.getByLabel(/Camera preset/).selectOption("cam_b_oblique");
await page.waitForTimeout(500);
const seen = new Set();
const t0 = Date.now();
await page.getByRole("button", { name: /Analyze with both methods/ }).click();
while (Date.now() - t0 < 120000) {
  const txt = await page.locator(".status").innerText();
  for (const p of ["capturing", "uploading", "queued", "running", "completed", "error"]) if (txt.includes(p)) seen.add(p);
  if (txt.includes("completed") || txt.includes("error")) break;
  await page.waitForTimeout(100);
}
await page.waitForTimeout(1500);
await page.screenshot({ path: join(OUT, "02-after-analyze.png"), fullPage: false });
const evidence = await page.locator(".evidence").innerText();
await page.screenshot({ path: join(OUT, "03-full-page.png"), fullPage: true });

// Gallery interaction: filter to non-correct cases and open one
await page.locator(".filters select").first().selectOption("any_error");
await page.waitForTimeout(400);
const nCards = await page.locator(".gallery .card").count();
if (nCards) {
  await page.locator(".gallery .card").first().click();
  await page.waitForTimeout(1500);
  await page.screenshot({ path: join(OUT, "04-gallery-case.png"), fullPage: false });
}
const summary = {
  app: APP, at: new Date().toISOString(), phases_seen: [...seen], elapsed_ms: Date.now() - t0,
  evidence_has_both_methods: evidence.includes("RGB baseline") && evidence.includes("RGB + structure prompt"),
  identical_input_banner: evidence.includes("byte-identical input"), gallery_non_correct_cards: nCards, page_errors: errors,
};
writeFileSync(join(OUT, "ui-smoke.json"), JSON.stringify(summary, null, 2));
console.log(JSON.stringify(summary, null, 2));
await browser.close();
