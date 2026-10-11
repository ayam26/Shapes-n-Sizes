"""Demo rendering for the GNN shape verifier.

- Gallery: side-by-side BASE | GNN | RGB panels for night/weather images, drawn from the
  committed re-scored detections. Each arm is shown at its T-shift equal-recall threshold
  (results/gnn/analysis.json); boxes are colored by the evaluation's own matching at IoU 0.5.
  Examples are chosen by a fixed rule and labeled as selected.
- Live: one image through detector -> outlines -> second stages with per-stage timings, the
  outline graphs the GNN sees, and an optional recoloring test.

Rendered images contain BDD100K frames (research license): keep them local, out of git.
"""

from __future__ import annotations

import dataclasses
import html
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from snz.data.bdd import EVAL_CLASSES, Frame, load_frames, select_frames
from snz.data.manifest import read_manifest
from snz.eval.matching import (
    FALSE_TYPES,
    FD_DUPLICATE,
    FD_INVENTED,
    FD_NEAR_OTHER,
    FD_POOR_LOC,
    FD_TP,
    FD_WRONG_CLASS,
    box_iou,
    match_split,
)
from snz.eval.predict import read_detections
from snz.gnn import GNN_CLASSES
from snz.gnn.outlines import OutlineSet, extract_outlines

GREEN, RED, CYAN, GRAY = (46, 204, 113), (231, 76, 60), (0, 188, 212), (170, 170, 170)
HEADER = 28
PANEL_WIDTH = 640  # BDD frames are 1280 px; three panels at 640 keep 11-16 px labels readable
FD_NAME = {FD_DUPLICATE: "duplicate", FD_WRONG_CLASS: "wrong class", FD_POOR_LOC: "poor box",
           FD_INVENTED: "invented", FD_NEAR_OTHER: "near other"}
CLASS_NAME = {**{c: c for c in GNN_CLASSES}, "background": "not an object"}
RULE = ("Selected by a fixed rule, not by hand: 'helped' = images where the GNN dropped the most "
        "false detections the detector kept, minus real objects it dropped; 'hurt' = images "
        "where the GNN dropped the most real objects. Ties by file name. Full results over all "
        "4,752 night/rain/snow/fog images are in results/gnn/report.md.")


# ---------- statuses and differences -------------------------------------------------------

def detection_status(frame: Frame, record: dict, classes: Sequence[str] = EVAL_CLASSES,
                     max_dets: int = 100) -> np.ndarray:
    """FD_* code per detection (record order) under the evaluation's matching at IoU 0.5;
    -1 for detections that are not evaluated (class `train`, or beyond `max_dets`)."""
    n = len(record["scores"])
    status = np.full(n, -1, np.int64)
    if not n:
        return status
    # match under the frame's name: a recolored copy (other file name) shows the same scene
    m = match_split([frame], [{**record, "name": frame.name}], classes=classes,
                    max_dets=max_dets)
    # match_split appends rows class by class (in `classes` order), each by descending score
    order = []
    for c in classes:
        sel = [j for j in range(n) if record["labels"][j] == c]
        sel.sort(key=lambda j: -record["scores"][j])  # stable, like its mergesort
        order.extend(sel[:max_dets])
    status[np.asarray(order, np.int64)] = m.det_fd_type
    return status


def kept(record: dict, threshold: float) -> np.ndarray:
    return np.asarray(record["scores"], float) >= threshold


@dataclass(frozen=True)
class ImageDiff:
    name: str
    removed_false: int  # BASE kept a false detection; the arm dropped it
    removed_true: int   # BASE kept a real object; the arm dropped it
    added_false: int    # the arm kept a false detection BASE had dropped
    added_true: int     # the arm kept a real object BASE had dropped


def _same_detections(a: dict, b: dict) -> bool:
    ba, bb = np.asarray(a["boxes"], float), np.asarray(b["boxes"], float)
    return ba.shape == bb.shape and np.allclose(ba, bb) and list(a["labels"]) == list(b["labels"])


def image_diff(frame: Frame, base: dict, arm: dict, *, thr_base: float,
               thr_arm: float) -> ImageDiff:
    if not _same_detections(base, arm):
        raise ValueError(f"{frame.name}: BASE and arm records must hold the same detections")
    sb, sa = detection_status(frame, base), detection_status(frame, arm)
    kb, ka = kept(base, thr_base), kept(arm, thr_arm)
    fb, fa = np.isin(sb, FALSE_TYPES), np.isin(sa, FALSE_TYPES)
    return ImageDiff(frame.name, int((kb & ~ka & fb).sum()), int((kb & ~ka & (sb == FD_TP)).sum()),
                     int((~kb & ka & fa).sum()), int((~kb & ka & (sa == FD_TP)).sum()))


def select_examples(diffs: Sequence[ImageDiff], n_best: int = 6,
                    n_worst: int = 3) -> tuple[list[str], list[str]]:
    net = {d.name: d.removed_false - d.removed_true for d in diffs}
    best = sorted((d.name for d in diffs if net[d.name] > 0), key=lambda n: (-net[n], n))[:n_best]
    hurt = {d.name: d.removed_true for d in diffs}
    worst = sorted((n for n in hurt if hurt[n] > 0 and n not in best),
                   key=lambda n: (-hurt[n], n))[:n_worst]
    return best, worst


# ---------- drawing ------------------------------------------------------------------------

def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _dashed_rect(draw: ImageDraw.ImageDraw, box, color, width: int = 2, dash: int = 6) -> None:
    x1, y1, x2, y2 = box
    sides = ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1))
    for (ax, ay, bx, by) in sides:
        length = max(abs(bx - ax), abs(by - ay))
        steps = max(int(length // dash), 1)
        for s in range(0, steps, 2):
            t0, t1 = s / steps, min((s + 1) / steps, 1.0)
            draw.line((ax + (bx - ax) * t0, ay + (by - ay) * t0,
                       ax + (bx - ax) * t1, ay + (by - ay) * t1), fill=color, width=width)


def _tag(draw: ImageDraw.ImageDraw, xy, text: str, color, font) -> None:
    x, y = xy
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle((left - 1, top - 1, right + 1, bottom + 1), fill=(0, 0, 0))
    draw.text((x, y), text, fill=color, font=font)


def render_panel(image: Image.Image, record: dict, threshold: float, status: np.ndarray | None,
                 title: str, dropped: np.ndarray | None = None,
                 max_width: int | None = None) -> Image.Image:
    """One arm: kept detections (green real / red false / cyan unlabeled); dashed gray =
    kept by BASE but dropped by this arm. `max_width` downsizes first so labels stay legible
    when three full-resolution panels sit side by side."""
    f = 1.0
    if max_width is not None and image.size[0] > max_width:
        f = max_width / image.size[0]
        image = image.resize((max_width, round(image.size[1] * f)))
        record = {**record, "boxes": [[v * f for v in b] for b in record["boxes"]]}
    w, h = image.size
    canvas = Image.new("RGB", (w, h + HEADER), (18, 18, 18))
    canvas.paste(image.convert("RGB"), (0, HEADER))
    d = ImageDraw.Draw(canvas)
    font, small = _font(16), _font(11)
    keep = kept(record, threshold)
    n_real = n_false = 0
    for j, (box, label, score) in enumerate(zip(record["boxes"], record["labels"],
                                                record["scores"], strict=True)):
        bx = np.clip(np.asarray(box, float), 0, [w - 1, h - 1, w - 1, h - 1])
        x1, y1, x2, y2 = bx[0], bx[1] + HEADER, bx[2], bx[3] + HEADER
        if dropped is not None and dropped[j]:
            _dashed_rect(d, (x1, y1, x2, y2), GRAY)
            _tag(d, (x1 + 2, y2 - 12), "dropped", GRAY, small)
            continue
        if not keep[j]:
            continue
        code = None if status is None else int(status[j])
        if code is None or code < 0:
            color, extra = CYAN, ""
        elif code == FD_TP:
            color, extra = GREEN, ""
            n_real += 1
        elif code in FALSE_TYPES:
            color, extra = RED, f" {FD_NAME[code]}"
            n_false += 1
        else:  # ignore region: neither real nor false
            color, extra = GRAY, ""
        d.rectangle((x1, y1, x2, y2), outline=color, width=2)
        _tag(d, (x1 + 2, max(y1 - 12, HEADER)), f"{label} {score:.2f}{extra}", color, small)
    head = f"{title}   kept {int(keep.sum())}"
    if status is not None:
        head += f"   real {n_real}   false {n_false}"
    d.text((8, 5), head, fill=(240, 240, 240), font=font)
    return canvas


def render_comparison(panels: Sequence[Image.Image]) -> Image.Image:
    width, height = sum(p.size[0] for p in panels), max(p.size[1] for p in panels)
    out = Image.new("RGB", (width, height), (18, 18, 18))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.size[0]
    return out


def draw_outline_graph(contour: np.ndarray, size: int = 160, probs: np.ndarray | None = None,
                       title: str = "") -> Image.Image:
    """The GNN's view of one object: 64 contour nodes with ring edges, plus its top classes."""
    lines = [title] if title else []
    if probs is not None:
        top = np.argsort(-np.asarray(probs))[:3]
        lines += [f"{CLASS_NAME[GNN_CLASSES[k]]} {probs[k]:.2f}" for k in top]
    canvas = Image.new("RGB", (size, size + 14 * len(lines) + 4), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    c = np.asarray(contour, float)
    c = c - c.mean(axis=0)
    extent = np.abs(c).max()
    c = c / (extent if extent > 0 else 1.0) * (size * 0.42) + size / 2
    pts = [tuple(p) for p in c]
    for k in range(len(pts)):
        d.line((*pts[k], *pts[(k + 1) % len(pts)]), fill=(120, 170, 255), width=1)
    for k, (x, y) in enumerate(pts):
        r = 3 if k == 0 else 2
        d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 210, 90) if k == 0 else (200, 220, 255))
    font = _font(11)
    for i, text in enumerate(lines):
        d.text((4, size + 2 + 14 * i), text, fill=(235, 235, 235), font=font)
    return canvas


def graph_grid(images: Sequence[Image.Image], cols: int = 6) -> Image.Image:
    if not images:
        return Image.new("RGB", (160, 40), (18, 18, 18))
    w, h = max(i.size[0] for i in images), max(i.size[1] for i in images)
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (w * min(cols, len(images)), h * rows), (18, 18, 18))
    for k, im in enumerate(images):
        out.paste(im, ((k % cols) * w, (k // cols) * h))
    return out


def recolor(image: Image.Image, mode: str) -> Image.Image:
    image = image.convert("RGB")
    if mode == "invert":
        return ImageOps.invert(image)
    if mode == "gray":
        return ImageOps.grayscale(image).convert("RGB")
    if mode == "hue":  # rotate hue by 180 degrees
        h, s, v = image.convert("HSV").split()
        h = h.point(lambda x: (x + 128) % 256)
        return Image.merge("HSV", (h, s, v)).convert("RGB")
    raise ValueError(f"unknown recolor mode {mode!r} (hue, invert, gray)")


def gnn_inputs_color_invariant(o: OutlineSet) -> bool:
    """True if the GNN's inputs are identical when every outline color is inverted."""
    import torch

    from snz.gnn.model import GraphDataset

    if not len(o.score):
        return True
    t = np.zeros(len(o.score), np.int64)
    a = GraphDataset(o, t, "gnn", False)
    inverted = dataclasses.replace(o, color=(255 - o.color).astype(np.uint8))
    b = GraphDataset(inverted, t, "gnn", False)
    return all(torch.equal(a[i][0], b[i][0]) and torch.equal(a[i][1], b[i][1])
               for i in range(len(a)))


def gnn_probabilities(o: OutlineSet, models_dir: Path, device: str, arm: str = "GNN") -> np.ndarray:
    import torch

    from snz.gnn.model import predict_probs
    from snz.gnn.train import make_dataset, make_model

    model = make_model(arm, pretrained=False)
    model.load_state_dict(torch.load(Path(models_dir) / f"{arm}.pt", map_location="cpu",
                                     weights_only=True))
    ds = make_dataset(arm, o, np.zeros(len(o.score), np.int64), image_dir=Path("."),
                      augment=False)
    return predict_probs(model, ds, device=device)


# ---------- gallery ------------------------------------------------------------------------

def load_thresholds(analysis_path: Path, arms: Sequence[str]) -> dict[str, float]:
    point = json.loads(Path(analysis_path).read_text())["point"]
    return {arm: float(point[arm]["threshold_at_R"]["threshold"]) for arm in arms}


def _rows_of(o: OutlineSet, name: str) -> np.ndarray:
    i = o.names.index(name)
    return np.flatnonzero(o.image == i)


def _page(title: str, thresholds: dict, entries: list[dict]) -> str:
    def section(group: str, heading: str) -> str:
        items = [e for e in entries if e["group"] == group]
        if not items:
            return f"<h2>{heading}</h2><p>None under the rule.</p>"
        figs = []
        for e in items:
            caption = (f"{html.escape(e['name'])}: GNN dropped {e['removed_false']} false and "
                       f"{e['removed_true']} real detection(s) the detector kept")
            graphs = (f'<img src="{html.escape(e["graphs"])}" alt="outline graphs">'
                      if e.get("graphs") else "")
            src, alt = html.escape(e["image"]), html.escape(e["name"])
            figs.append(f'<figure><img src="{src}" alt="{alt}">'
                        f"{graphs}<figcaption>{caption}</figcaption></figure>")
        return f"<h2>{heading}</h2>" + "".join(figs)

    thr = ", ".join(f"{a} {t:.3f}" for a, t in thresholds.items())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:24px;max-width:1500px}}
img{{max-width:100%;display:block;margin-top:6px}} figure{{margin:0 0 28px}}
figcaption{{color:#bbb;margin-top:6px}} .k span{{padding:0 8px;margin-right:10px;border-radius:3px}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>{html.escape(RULE)}</p>
<p class="k">Legend: <span style="background:#2ecc71;color:#000">real object</span>
<span style="background:#e74c3c">false detection</span>
<span style="border:1px dashed #aaa">dropped by the second stage</span>
Each panel is drawn at that arm's equal-recall threshold on T-shift ({html.escape(thr)}).</p>
{section("helped", "Where the GNN helped")}
{section("hurt", "Where the GNN hurt")}
</body></html>
"""


def build_gallery(*, runs_root: Path, analysis_path: Path, labels: Path, manifests_dir: Path,
                  image_dir: Path, out: Path,
                  splits: Sequence[str] = ("t-night", "t-rain", "t-snow"),
                  arms: Sequence[str] = ("BASE", "GNN", "RGB"), n_best: int = 6, n_worst: int = 3,
                  outlines: Path | None = None, models_dir: Path | None = None,
                  device: str = "cpu") -> dict:
    if "BASE" not in arms or "GNN" not in arms:
        raise ValueError("the gallery compares GNN with BASE: --arms must include both")
    thr = load_thresholds(analysis_path, arms)
    names, seen = [], set()
    for s in splits:
        for n, _ in read_manifest(Path(manifests_dir) / f"{s}.csv"):
            if n not in seen:
                seen.add(n)
                names.append(n)
    frames = {f.name: f for f in select_frames(load_frames(labels)[0], names)}
    recs = {arm: {r["name"]: r for r in read_detections(
        Path(runs_root) / f"{arm}-s0" / "eval" / "val-all.detections.jsonl") if r["name"] in seen}
        for arm in arms}
    diffs = [image_diff(frames[n], recs["BASE"][n], recs["GNN"][n],
                        thr_base=thr["BASE"], thr_arm=thr["GNN"])
             for n in names if n in recs["BASE"] and n in recs["GNN"]]
    best, worst = select_examples(diffs, n_best, n_worst)
    by_name = {d.name: d for d in diffs}

    o = probs = None
    if outlines is not None and models_dir is not None:
        from snz.gnn.outlines import load_outlines
        from snz.gnn.train import subset_outlines

        o = subset_outlines(load_outlines(outlines), best + worst)
        probs = gnn_probabilities(o, models_dir, device) if len(o.score) else None

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for group, chosen in (("helped", best), ("hurt", worst)):
        for n in chosen:
            img = Image.open(Path(image_dir) / n).convert("RGB")
            kb = kept(recs["BASE"][n], thr["BASE"])
            panels = []
            for arm in arms:
                r = recs[arm][n]
                dropped = None if arm == "BASE" else kb & ~kept(r, thr[arm])
                panels.append(render_panel(img, r, thr[arm], detection_status(frames[n], r),
                                           arm, dropped, max_width=PANEL_WIDTH))
            stem = Path(n).stem
            render_comparison(panels).save(out / f"{stem}.png")
            entry = {"name": n, "group": group, "image": f"{stem}.png",
                     **dataclasses.asdict(by_name[n])}
            if o is not None and probs is not None:
                rows = _rows_of(o, n)
                g = recs["GNN"][n]
                dropped = kb & ~kept(g, thr["GNN"])
                show = [j for j in range(len(rows)) if dropped[j]][:6]
                thumbs = [draw_outline_graph(o.contour[rows[j]], probs=probs[rows[j]],
                                             title=f"{o.label[rows[j]]}: dropped")
                          for j in show]
                if thumbs:
                    graph_grid(thumbs).save(out / f"{stem}_graphs.png")
                    entry["graphs"] = f"{stem}_graphs.png"
            entries.append(entry)
    summary = {"thresholds": thr, "splits": list(splits), "rule": RULE, "best": best,
               "worst": worst, "entries": entries, "images_considered": len(diffs)}
    (out / "gallery.json").write_text(json.dumps(summary, indent=2))
    (out / "index.html").write_text(_page("GNN shape verifier: selected examples", thr, entries))
    return summary


# ---------- live ---------------------------------------------------------------------------

@dataclass
class LiveResult:
    image: Path
    record: dict                    # selected detections, detector scores
    outlines: OutlineSet
    scores: dict[str, np.ndarray]   # per arm, record order
    kept: dict[str, np.ndarray]
    thresholds: dict[str, float]
    timings_ms: dict[str, float]


def _clock(device: str) -> float:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()
    return time.perf_counter()


def load_detector(checkpoint: Path, device: str) -> Callable[[Path], dict]:
    """The BASE-s0 detector, preprocessed exactly as in `snz evaluate`."""
    import torch
    import torchvision.transforms.functional as TF

    from snz.data.bdd import CLASSES
    from snz.train.config import Recipe
    from snz.train.model import build_model

    recipe = Recipe(**json.loads((Path(checkpoint).parent / "run.json").read_text())["recipe"])
    model = build_model(None, trainable_layers=recipe.trainable_layers,
                        min_size=tuple(recipe.min_sizes), max_size=recipe.max_size)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval().to(device)

    @torch.no_grad()
    def detect(path: Path) -> dict:
        x = TF.to_tensor(Image.open(path).convert("RGB")).to(device)
        out = model([x])[0]
        return {"name": Path(path).name,
                "boxes": [[round(v, 2) for v in b] for b in out["boxes"].float().cpu().tolist()],
                "labels": [CLASSES[i - 1] for i in out["labels"].cpu().tolist()],
                "scores": out["scores"].float().cpu().tolist()}

    return detect


def make_segmenter(kind: str, device: str, checkpoint: Path | None = None):
    from snz.gnn.outlines import GrabCutSegmenter, Sam2Segmenter

    if kind == "sam2":
        return Sam2Segmenter(device=device, checkpoint=checkpoint)
    return GrabCutSegmenter()


def run_live(image_path: Path, *, detector: Callable[[Path], dict], segmenter, models_dir: Path,
             arms: Sequence[str], thresholds: dict[str, float], device: str) -> LiveResult:
    from snz.gnn.rescore import rescore
    from snz.gnn.targets import outline_records

    image_path = Path(image_path)
    t0 = _clock(device)
    rec = {**detector(image_path), "name": image_path.name}
    t1 = _clock(device)
    o = extract_outlines([rec], image_path.parent, segmenter)
    t2 = _clock(device)
    timings = {"detector": (t1 - t0) * 1e3, "outlines": (t2 - t1) * 1e3}
    scores, keep = {}, {}
    for arm in arms:
        ta = _clock(device)
        out = rescore(o, arm, models_dir, image_dir=image_path.parent, device=device)
        timings[arm] = (_clock(device) - ta) * 1e3
        scores[arm] = np.asarray(out[0]["scores"], float)
        keep[arm] = scores[arm] >= thresholds[arm]
    timings["total"] = sum(timings.values())
    return LiveResult(image_path, outline_records(o)[0], o, scores, keep,
                      {arm: float(thresholds[arm]) for arm in arms}, timings)


def _polygon_iou(a: np.ndarray, b: np.ndarray, wh: tuple[int, int]) -> float:
    w, h = int(wh[0]), int(wh[1])
    ma, mb = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    cv2.fillPoly(ma, [np.round(a).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round(b).astype(np.int32)], 1)
    union = int((ma | mb).sum())
    return float((ma & mb).sum() / union) if union else 1.0


def compare_runs(a: LiveResult, b: LiveResult) -> dict:
    """Original vs recolored run: which detections survive, how much the outlines move, and
    whether the GNN's keep/drop verdict agrees on detections present in both."""
    ba, bb = np.asarray(a.record["boxes"], float), np.asarray(b.record["boxes"], float)
    pairs, used = [], set()
    if len(ba) and len(bb):
        iou = box_iou(ba.reshape(-1, 4), bb.reshape(-1, 4))
        for i in np.argsort(-np.asarray(a.record["scores"])):
            cand = [j for j in range(len(bb)) if j not in used and iou[i, j] >= 0.5
                    and b.record["labels"][j] == a.record["labels"][i]]
            if cand:
                j = max(cand, key=lambda j: iou[i, j])
                used.add(j)
                pairs.append((int(i), j))
    wh = tuple(a.outlines.image_wh[0]) if len(a.outlines.score) else (1, 1)
    ious = [_polygon_iou(a.outlines.contour[i], b.outlines.contour[j], wh) for i, j in pairs]
    agree = None
    if "GNN" in a.kept and pairs:
        agree = float(np.mean([a.kept["GNN"][i] == b.kept["GNN"][j] for i, j in pairs]))
    return {"detections_original": len(ba), "detections_recolored": len(bb),
            "matched": len(pairs), "outline_iou_mean": float(np.mean(ious)) if ious else None,
            "gnn_verdict_agreement": agree,
            "gnn_input_color_invariant": gnn_inputs_color_invariant(a.outlines),
            "note": ("The GNN never reads pixels; differences come from the detector's boxes and "
                     "SAM 2's outlines changing with the colors.")}


def live_summary(res: LiveResult) -> dict:
    return {"image": str(res.image), "thresholds": res.thresholds,
            "timings_ms": {k: round(v, 1) for k, v in res.timings_ms.items()},
            "detections": [{"label": lab, "box": box, "detector_score": round(float(s), 3),
                            **{f"{arm}_score": round(float(res.scores[arm][j]), 3)
                               for arm in res.scores},
                            **{f"{arm}_kept": bool(res.kept[arm][j]) for arm in res.kept}}
                           for j, (lab, box, s) in enumerate(zip(res.record["labels"],
                                                                 res.record["boxes"],
                                                                 res.record["scores"],
                                                                 strict=True))]}


def render_live(res: LiveResult, out_png: Path, frame: Frame | None = None) -> None:
    img = Image.open(res.image).convert("RGB")
    kb = res.kept.get("BASE")
    panels = []
    for arm in res.scores:
        rec = {**res.record, "scores": [float(s) for s in res.scores[arm]]}
        status = detection_status(frame, rec) if frame is not None else None
        dropped = None if arm == "BASE" or kb is None else kb & ~res.kept[arm]
        ms = res.timings_ms["detector"] if arm == "BASE" else res.timings_ms[arm]
        panels.append(render_panel(img, rec, res.thresholds[arm], status, f"{arm}  {ms:.0f} ms",
                                   dropped, max_width=PANEL_WIDTH))
    render_comparison(panels).save(out_png)


def render_live_graphs(res: LiveResult, out_png: Path, models_dir: Path, device: str,
                       limit: int = 12) -> None:
    o = res.outlines
    probs = (gnn_probabilities(o, models_dir, device)
             if len(o.score) and (Path(models_dir) / "GNN.pt").exists() else None)
    thumbs = []
    for j in range(min(len(o.score), limit)):
        verdict = ""
        if "GNN" in res.kept:
            verdict = " kept" if res.kept["GNN"][j] else " dropped"
        thumbs.append(draw_outline_graph(o.contour[j], probs=None if probs is None else probs[j],
                                         title=f"{o.label[j]} {o.score[j]:.2f}{verdict}"))
    graph_grid(thumbs).save(out_png)
