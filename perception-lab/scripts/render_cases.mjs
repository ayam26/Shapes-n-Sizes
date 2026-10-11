#!/usr/bin/env node
// Render manifest cases with the same Three.js scene code as the UI, in headless system Chrome,
// then (optionally) upload them to the backend. Rendering happens on this laptop, not the GX10.
//
//   node scripts/render_cases.mjs --build-manifest            # writes experiments/manifest.json
//   node scripts/render_cases.mjs --out DIR [--only id,id]    # render to DIR (preview / determinism check)
//   node scripts/render_cases.mjs --upload [--split development]  # render + POST /api/cases
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(join(ROOT, "frontend", "package.json"));
const { chromium } = require("playwright-core");

const args = process.argv.slice(2);
const flag = (n) => args.includes(n);
const opt = (n, d) => (args.includes(n) ? args[args.indexOf(n) + 1] : d);
const APP = opt("--app", "http://127.0.0.1:5173");
const API = opt("--api", "http://127.0.0.1:8765");
const MANIFEST = join(ROOT, "experiments", "manifest.json");
const sha = (b) => createHash("sha256").update(b).digest("hex");
const fromDataUrl = (u) => Buffer.from(u.split(",")[1], "base64");

const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--use-angle=metal", "--ignore-gpu-blocklist"] });
const page = await browser.newPage();
page.on("console", (m) => m.type() === "error" && console.error("[page]", m.text()));
await page.goto(`${APP}/?mode=capture`);
await page.waitForFunction(() => document.body.dataset.ready === "1");
const version = await page.evaluate(() => window.__pl.version);

if (flag("--build-manifest")) {
  const m = await page.evaluate(async () => {
    const mod = await import("/src/scene/manifest.ts");
    return { core: mod.buildCoreManifest(), discovery: mod.buildDiscoveryManifest() };
  });
  const body = { schema_version: 1, renderer_version: version, created_at: new Date().toISOString(),
    note: "Core 36 = 3 targets x 3 appearances x 2 cameras x 2 lights. development = cam_a_ego_lane, heldout = cam_b_oblique. Discovery is exploratory only.",
    cases: [...m.core, ...m.discovery] };
  writeFileSync(MANIFEST, JSON.stringify(body, null, 2) + "\n");
  console.log(`manifest: ${body.cases.length} cases, sha256 ${sha(readFileSync(MANIFEST))}`);
  await browser.close();
  process.exit(0);
}

const manifest = JSON.parse(readFileSync(MANIFEST, "utf8"));
let cases = manifest.cases;
if (opt("--only")) cases = cases.filter((c) => opt("--only").split(",").includes(c.case_id));
if (opt("--split")) cases = cases.filter((c) => c.split === opt("--split"));
const out = opt("--out");
if (out) mkdirSync(out, { recursive: true });
const assetsDir = join(ROOT, "experiments", "assets");
mkdirSync(assetsDir, { recursive: true });
const log = [];

for (const c of cases) {
  const r1 = await page.evaluate((s) => window.__pl.renderCase(s), c.spec);
  const png = fromDataUrl(r1.dataUrl);
  const entry = { case_id: c.case_id, sha256_client: sha(png), bytes: png.length, marker: r1.marker };
  if (flag("--check-determinism")) {
    const r2 = await page.evaluate((s) => window.__pl.renderCase(s), c.spec);
    entry.deterministic = sha(fromDataUrl(r2.dataUrl)) === entry.sha256_client;
  }
  if (r1.textureDataUrl) {
    const tex = fromDataUrl(r1.textureDataUrl);
    const p = join(assetsDir, `depiction-${c.spec.appearance}.png`);
    if (!existsSync(p)) writeFileSync(p, tex);
    entry.texture_sha256 = sha(tex);
    entry.texture_matches_saved = sha(readFileSync(p)) === entry.texture_sha256;
  }
  if (out) writeFileSync(join(out, `${c.case_id}.png`), png);
  if (flag("--upload")) {
    const meta = { case_id: c.case_id, split: c.split, family: c.family, truth: c.truth, scene_seed: c.spec.seed,
      scene_spec: { ...c.spec, depiction_texture_sha256: entry.texture_sha256 ?? null }, observability: c.observability,
      source_kind: "procedural", license: "self-created", renderer_version: version };
    const fd = new FormData();
    fd.append("meta", JSON.stringify(meta));
    fd.append("image", new Blob([png], { type: "image/png" }), "input.png");
    const res = await fetch(`${API}/api/cases`, { method: "POST", body: fd });
    entry.upload = res.status;
    entry.server = await res.json();
    if (res.status !== 200) console.error(`upload ${c.case_id} -> ${res.status}`, entry.server);
  }
  log.push(entry);
  console.log(JSON.stringify(entry));
}
await browser.close();
writeFileSync(join(ROOT, "experiments", `render-log-${opt("--split", "all")}.json`), JSON.stringify({ renderer_version: version, rendered_at: new Date().toISOString(), cases: log }, null, 2));
