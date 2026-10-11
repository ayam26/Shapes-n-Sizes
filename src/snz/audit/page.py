"""Static blind-audit page: one image per sampled detection, four labels, CSV download.

Open `index.html` in a browser (no server needed). Answers are kept in the browser's
localStorage while working and downloaded as `audit_labels.csv` (item_id,label).
The page never contains arm names or scores.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw

LABELS = ("correct", "wrong_class", "invented", "unsure")
MAX_WIDTH = 960

_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Detection audit</title>
<style>
:root { --bg:#fafafa; --fg:#1b1b1b; --card:#fff; --line:#ddd; --accent:#2b6cb0; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141414; --fg:#eaeaea; --card:#1f1f1f; --line:#333; --accent:#63a4ff; } }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.4 system-ui, sans-serif; }
header { position:sticky; top:0; background:var(--card); border-bottom:1px solid var(--line);
  padding:10px 16px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
main { max-width:1000px; margin:0 auto; padding:16px; }
.item { background:var(--card); border:1px solid var(--line); border-radius:8px;
  margin:0 0 16px; padding:12px; }
.item.focus { outline:2px solid var(--accent); }
.item img { width:100%; height:auto; display:block; border-radius:4px; }
.row { display:flex; gap:12px; flex-wrap:wrap; margin-top:8px; align-items:center; }
.cls { font-weight:600; } label { cursor:pointer; }
button { font:inherit; padding:6px 12px; border-radius:6px; border:1px solid var(--line);
  background:var(--accent); color:#fff; cursor:pointer; }
</style></head><body>
<header>
  <strong>Detection audit</strong>
  <span id="progress"></span>
  <span>Keys: 1 real/correct class · 2 real/wrong class · 3 nothing there · 4 unsure ·
    j/k next/previous</span>
  <button id="download">Download CSV</button>
</header>
<main id="items"></main>
<script>
const ITEMS = __ITEMS__;
const LABELS = __LABELS__;
const TEXT = {correct: "real object, correct class", wrong_class: "real object, wrong class",
              invented: "nothing there (invented)", unsure: "unsure"};
const KEY = "snz-audit-__DIGEST__";
let answers = {};
try { answers = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let focus = 0;
const main = document.getElementById("items");
function save() { try { localStorage.setItem(KEY, JSON.stringify(answers)); } catch (e) {}
  document.getElementById("progress").textContent =
    Object.keys(answers).length + " / " + ITEMS.length + " labeled"; }
ITEMS.forEach((it, i) => {
  const div = document.createElement("div");
  div.className = "item"; div.id = it.id;
  div.innerHTML = '<img loading="lazy" src="items/' + it.id + '.jpg" alt="">' +
    '<div class="row"><span>' + it.id + '</span> predicted: <span class="cls">' +
    it["class"] + '</span></div><div class="row">' + LABELS.map((l, k) =>
      '<label><input type="radio" name="' + it.id + '" value="' + l + '"' +
      (answers[it.id] === l ? " checked" : "") + '> ' + (k + 1) + ' ' + TEXT[l] + '</label>'
    ).join("") + '</div>';
  div.addEventListener("change", e => { answers[it.id] = e.target.value; save(); });
  div.addEventListener("click", () => setFocus(i));
  main.appendChild(div);
});
function setFocus(i) {
  focus = Math.max(0, Math.min(ITEMS.length - 1, i));
  document.querySelectorAll(".item").forEach((d, k) => d.classList.toggle("focus", k === focus));
  document.getElementById(ITEMS[focus].id).scrollIntoView({block: "center"});
}
document.addEventListener("keydown", e => {
  if (!ITEMS.length) return;
  const k = parseInt(e.key, 10);
  if (k >= 1 && k <= 4) {
    const it = ITEMS[focus]; answers[it.id] = LABELS[k - 1];
    const sel = 'input[name="' + it.id + '"][value="' + LABELS[k - 1] + '"]';
    document.querySelector(sel).checked = true;
    save(); setFocus(focus + 1);
  } else if (e.key === "j") setFocus(focus + 1);
  else if (e.key === "k") setFocus(focus - 1);
});
document.getElementById("download").onclick = () => {
  const rows = ["item_id,label"].concat(ITEMS.filter(it => answers[it.id])
    .map(it => it.id + "," + answers[it.id]));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([rows.join("\\n") + "\\n"], {type: "text/csv"}));
  a.download = "audit_labels.csv"; a.click();
};
save(); if (ITEMS.length) setFocus(0);
</script></body></html>
"""


def build_audit_page(items: Sequence[dict], frames_dir: Path, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    (out_dir / "items").mkdir(parents=True, exist_ok=True)
    for it in items:
        img = Image.open(Path(frames_dir) / it["frame"]).convert("RGB")
        draw = ImageDraw.Draw(img)
        width = max(2, img.width // 400)
        draw.rectangle(it["box"], outline=(255, 214, 0), width=width)
        draw.rectangle([it["box"][0] - width, it["box"][1] - width,
                        it["box"][2] + width, it["box"][3] + width], outline=(0, 0, 0), width=1)
        if img.width > MAX_WIDTH:
            img = img.resize((MAX_WIDTH, round(img.height * MAX_WIDTH / img.width)))
        img.save(out_dir / "items" / f"{it['id']}.jpg", quality=90)
    public = [{"id": it["id"], "class": it["class"]} for it in items]
    digest = hashlib.sha256(json.dumps(list(items), sort_keys=True).encode()).hexdigest()[:16]
    html = (_HTML.replace("__ITEMS__", json.dumps(public))
            .replace("__LABELS__", json.dumps(list(LABELS))).replace("__DIGEST__", digest))
    (out_dir / "index.html").write_text(html)
    return out_dir / "index.html"
