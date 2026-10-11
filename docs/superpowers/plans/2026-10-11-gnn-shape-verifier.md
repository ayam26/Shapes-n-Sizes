# GNN Shape Verifier Implementation Plan (Plan 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Repo style (Plans 1–3): TDD per module; tests use `tests/helpers.py`; `uv run pytest`, `uv run ruff check src tests`.

**Goal:** Re-score the BASE-s0 Faster R-CNN detections with a texture-blind graph neural network over SAM 2 outlines (plus size cues), and test whether that cuts false detections and raises recall under night/rain/snow/fog.

**Architecture:** New package `src/snz/gnn/` (graph features, outlines, targets, networks, fusion, re-scoring, analysis) plus five `snz gnn-*` CLI commands. Outputs use the `prereg-v1` detection JSONL format, so evaluation reuses `snz.eval` (matching, AP, operating points, bootstrap, Holm). SAM 2 sits behind a small segmenter interface so CI tests run with a fake.

**Tech Stack:** Python ≥ 3.12, PyTorch/torchvision (existing), NumPy, OpenCV (`opencv-python-headless`), SAM 2.1 (`sam2`, optional extra `gnn`), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-11-gnn-shape-verifier-design.md`

## Global Constraints

- Deadline 13:00 PDT 2026-10-11. Priority if short: BASE + GNN → RGB → GNN-SHAPE → GNN-COLOR (spec §3); unrun arms are reported as not run.
- Stage 1 = `~/snz-data/runs/BASE-s0/final.pt`; detections with score ≥ 0.05, top 50 per image by score, `EVAL_CLASSES` only (`train` excluded). **The same selected set is evaluated for every arm, BASE included.**
- 64 contour points; ring neighbor offsets (±1, ±2, ±4); node features (5) and size features (6) exactly as spec §5.2; no pixel values in GNN / GNN-SHAPE inputs.
- Classes: 9 `EVAL_CLASSES` + `background` (index 9).
- Network targets: best IoU ≥ 0.5 with a real box → its class; IoU < 0.3 with every box incl. ignore regions → background; otherwise excluded (−1). Fusion target: `Matches.det_tp[:, 0]` from `match_split`.
- id-val folds: 50/50 by image name, seed 0 → `manifests/gnn_fold_{a,b}.csv`; fold A trains networks, fold B fits fusion.
- GNN: EdgeConv 64-64-128, linear 512, max‖mean pool, size MLP 6→32, MLP 256, dropout 0.3; AdamW lr 1e-3, wd 1e-4, batch 512, 30 epochs, cosine, label smoothing 0.05; seed 0.
- RGB: ResNet-18 `IMAGENET1K_V1`, crop = box + 10% per side → 112×112; AdamW lr 3e-4, batch 256, 8 epochs; seed 0.
- Fusion: logistic regression on [logit(det score), log p(det class), log p(background), log max p(other class)], L2 1e-4, IRLS.
- `prereg-v2` tag required before `gnn-rescore` / `gnn-analyze` touch any split other than id-val. Outlines on test images are allowed before the tag (no labels used).
- R\* = BASE-s0 recall on pooled T-shift at score ≥ 0.5 (IoU 0.5); F\* = BASE-s0 false detections per image on pooled T-shift at score ≥ 0.5. B = 2,000, seed 0; Holm over {G1, G2, G3}.
- New core dependency: `opencv-python-headless>=4.9`. Optional extra `gnn = ["sam2>=1.0", "huggingface_hub>=0.23"]` (CI never installs it).

## Review Focus

1. **Images with no selected detections** (or no detections at all) must still appear as records with empty lists in re-scored files, or `match_split` silently drops their ground truth from recall — test in Task 6.
2. **Masks with several blobs or holes** (SAM often returns fragments) → the largest external contour is used, never a crash — test in Task 1.
3. **Tiny or degenerate boxes** (1–3 px, zero-area polygon) → rectangle fallback with `ok=False` and no NaN/inf in any feature — test in Task 1.
4. **T-texture outlines must come from the stylized image** (`<stem>_s1.jpg`), not the original — test in Task 2.
5. **An interrupted run must not leave a half-written output** that a rerun trusts → every output written through `snz.fsutil.atomic_replace` — test in Task 2.

## File Structure

| File | Responsibility |
|---|---|
| `src/snz/gnn/__init__.py` | `GNN_CLASSES = (*EVAL_CLASSES, "background")`, `ARMS_G = ("BASE", "GNN", "RGB", "GNN-SHAPE", "GNN-COLOR")`, `GNN_PREREG_TAG = "prereg-v2"` |
| `src/snz/gnn/graph.py` | mask → contour → node/size features; neighbor index |
| `src/snz/gnn/outlines.py` | detection selection, segmenters (SAM 2, GrabCut), `OutlineSet` + npz I/O |
| `src/snz/gnn/targets.py` | folds, network targets, fusion targets |
| `src/snz/gnn/model.py` | `ShapeGNN`, `rgb_model`, datasets, `train_network`, `predict_probs` |
| `src/snz/gnn/fusion.py` | fusion features, IRLS logistic regression, AUROC |
| `src/snz/gnn/rescore.py` | per-arm re-scored detection records |
| `src/snz/gnn/analyze.py` | G1–G3 + descriptive report |
| `src/snz/eval/guard.py` | (modify) `tag` parameter |
| `src/snz/eval/operating.py` | (modify) `threshold_for_false_rate` |
| `src/snz/cli.py` | (modify) `gnn-outlines`, `gnn-train`, `gnn-rescore`, `gnn-analyze` |
| `scripts/run_gnn.sh` | GX10 stages |
| `tests/gnn_helpers.py` | shared fixtures (created in Task 1, extended per task): `FakeSegmenter` (fills each box), `EmptySegmenter` (all-false masks), `outline_set_for(name, boxes, labels=None, scores=None, extra_names=())`, `synthetic_outlines(color)`, `squares_and_circles(n) -> (OutlineSet, targets)`, `tiny_idval(tmp_path, n_images) -> (image_dir, labels_json, idval_manifest, outlines_npz)`, `matches_from(rows) -> Matches`, `planted_runs(tmp_path) -> (runs_root, labels_json, manifests_dir)` |
| `prereg/prereg-v2.md`, `docs/runbook.md`, `README.md` | preregistration, commands, status |

---

### Task 1: Shape graph features

**Files:** Create `src/snz/gnn/__init__.py`, `src/snz/gnn/graph.py`; Modify `pyproject.toml` (add `opencv-python-headless>=4.9`, extra `gnn`); Test `tests/test_gnn_graph.py`

**Interfaces:**
- Produces: `N_POINTS = 64`; `NEIGHBOR_OFFSETS = (1, -1, 2, -2, 4, -4)`; `neighbor_index(n: int = 64) -> np.ndarray[int64, (n, 6)]`; `mask_to_contour(mask: np.ndarray[bool, (H, W)], n: int = 64) -> np.ndarray[float32, (n, 2)] | None`; `box_contour(box, n=64) -> np.ndarray[(n, 2)]`; `node_features(contour) -> np.ndarray[float32, (64, 5)]`; `size_features(box, contour, image_wh: tuple[int, int]) -> np.ndarray[float32, (6,)]`.

- [ ] **Step 1: Write the failing tests**

```python
def _square(h=100, w=120, y0=20, x0=30, s=40):
    m = np.zeros((h, w), bool); m[y0:y0 + s, x0:x0 + s] = True; return m

def test_contour_shape_order_start():
    c = mask_to_contour(_square())
    assert c.shape == (64, 2)
    assert c[0, 1] == pytest.approx(c[:, 1].min(), abs=1.0)          # starts at top-most point
    x, y = c[:, 0], c[:, 1]
    signed = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert signed > 0                                                 # clockwise on screen (y down)

def test_node_features_translation_scale_invariant():
    a = node_features(mask_to_contour(_square(s=40)))
    b = node_features(mask_to_contour(_square(h=400, w=400, y0=150, x0=90, s=160)))
    np.testing.assert_allclose(a, b, atol=0.05)

def test_circle_and_square_differ():
    yy, xx = np.mgrid[:100, :100]; circle = (yy - 50) ** 2 + (xx - 50) ** 2 < 30 ** 2
    assert node_features(mask_to_contour(circle))[:, 2].std() < node_features(
        mask_to_contour(_square()))[:, 2].std()                       # turning angle varies more on a square

def test_largest_blob_and_holes():
    m = _square(s=40); m[80:84, 100:104] = True; m[30:40, 40:50] = False   # small blob + hole
    c = mask_to_contour(m)
    assert c[:, 0].min() >= 29 and c[:, 0].max() <= 70                     # the big square only

def test_degenerate_masks():
    assert mask_to_contour(np.zeros((10, 10), bool)) is None
    one = np.zeros((10, 10), bool); one[5, 5] = True
    assert mask_to_contour(one) is None
    f = node_features(box_contour([5, 5, 6, 5.5]))
    assert np.isfinite(f).all()
    assert np.isfinite(size_features([5, 5, 6, 5.5], box_contour([5, 5, 6, 5.5]), (1280, 720))).all()

def test_size_features_values():
    box = [100, 200, 300, 400]
    s = size_features(box, box_contour(box), (1000, 500))
    np.testing.assert_allclose(s, [0.2, 0.4, 0.8, 0.2, 0.0, 1.0], atol=1e-5)

def test_neighbor_index_ring():
    idx = neighbor_index(64)
    assert idx.shape == (64, 6) and list(idx[0]) == [1, 63, 2, 62, 4, 60]
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_gnn_graph.py -q` → `ModuleNotFoundError: snz.gnn`.

- [ ] **Step 3: Implement `graph.py`** — `cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)`, largest by `cv2.contourArea`; < 3 points or zero area → `None`. Resample by cumulative arc length (closed polygon), enforce clockwise-on-screen (positive shoelace with y down; reverse if negative), roll so index 0 is the top-most point (lowest y, then lowest x). Node features: centroid = mean of points; s = sqrt(max(shoelace area, 1e-6)); turning angle = signed angle between consecutive edge vectors, divided by π; curvature = turning angle × 64 / max(perimeter / s, 1e-6); radius = ‖p − centroid‖ / s. Size features order: w/W, h/H, y2/H, ((x1+x2)/2)/W, log(max(w,1e-3)/max(h,1e-3)), clip(polygon area / max(w·h, 1e-6), 0, 1). `box_contour` samples the box perimeter clockwise from the top-left corner.

- [ ] **Step 4: Run, expect PASS** — same command; `uv run ruff check src tests`.

- [ ] **Step 5: Commit** — `git commit -m "Plan 4 T1: shape graph features (contour, node and size features)"`

### Task 2: Outlines (selection, segmenters, OutlineSet I/O) + `snz gnn-outlines`

**Files:** Create `src/snz/gnn/outlines.py`; Modify `src/snz/cli.py`; Test `tests/test_gnn_outlines.py`

**Interfaces:**
- Consumes: Task 1 (`mask_to_contour`, `box_contour`); `snz.eval.predict.read_detections`; `snz.stylize.build.stylized_name`; `snz.fsutil.atomic_replace`.
- Produces:
  - `select_detections(record: dict, min_score=0.05, top_k=50) -> list[int]` (indices into the record, `EVAL_CLASSES` only, descending score, stable).
  - `class BoxSegmenter(Protocol): def __call__(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray  # (N, H, W) bool`
  - `Sam2Segmenter(model_id: str = "facebook/sam2.1-hiera-small", device: str = "cuda", checkpoint: Path | None = None)`; `GrabCutSegmenter(max_side: int = 128, iters: int = 5)`.
  - `@dataclass OutlineSet: names: list[str]` (all images, in input order) `; image: int32[M]; det_index: int32[M]; box: float32[M,4]; score: float32[M]; label: list[str]; contour: float32[M,64,2]; color: uint8[M,64,3]; ok: bool[M]; image_wh: int32[M,2]`
  - `extract_outlines(records: list[dict], image_dir: Path, segmenter: BoxSegmenter, file_name: Callable[[str], str] | None = None) -> OutlineSet`; `save_outlines(path, o)`, `load_outlines(path) -> OutlineSet` (npz, atomic).
  - CLI: `snz gnn-outlines --detections FILE --images DIR --out FILE.npz [--stylized-copy N] [--segmenter sam2|grabcut] [--device cuda] [--sam2-checkpoint FILE]`.

- [ ] **Step 1: Write the failing tests** (a `FakeSegmenter` fills each box, or an ellipse inside it)

```python
def test_select_detections():
    r = {"name": "a.jpg", "boxes": [[0, 0, 5, 5]] * 4,
         "labels": ["car", "train", "car", "pedestrian"], "scores": [0.9, 0.95, 0.04, 0.3]}
    assert select_detections(r, top_k=50) == [0, 3]
    assert select_detections(r, top_k=1) == [0]

def test_extract_outlines_and_roundtrip(tmp_path):
    write_image(tmp_path / "a.jpg", size=(64, 48), color=(200, 10, 10))
    write_image(tmp_path / "b.jpg", size=(64, 48))
    recs = [{"name": "a.jpg", "boxes": [[10, 10, 40, 30]], "labels": ["car"], "scores": [0.8]},
            {"name": "b.jpg", "boxes": [], "labels": [], "scores": []}]
    o = extract_outlines(recs, tmp_path, FakeSegmenter())
    assert o.names == ["a.jpg", "b.jpg"] and len(o.score) == 1 and o.ok.all()
    assert o.color[0].mean(axis=0)[0] > 150                     # colors sampled from the image
    save_outlines(tmp_path / "o.npz", o)
    p = load_outlines(tmp_path / "o.npz")
    assert p.names == o.names and np.array_equal(p.contour, o.contour)

def test_failed_mask_falls_back_to_box(tmp_path):
    write_image(tmp_path / "a.jpg")
    recs = [{"name": "a.jpg", "boxes": [[10, 10, 40, 30]], "labels": ["car"], "scores": [0.8]}]
    o = extract_outlines(recs, tmp_path, EmptySegmenter())
    assert not o.ok[0] and np.isfinite(o.contour).all()

def test_stylized_file_name_used(tmp_path):
    write_image(tmp_path / "a_s1.jpg")                           # only the stylized copy exists
    recs = [{"name": "a.jpg", "boxes": [[1, 1, 9, 9]], "labels": ["car"], "scores": [0.9]}]
    o = extract_outlines(recs, tmp_path, FakeSegmenter(),
                         file_name=functools.partial(stylized_name, copy=1))
    assert o.names == ["a.jpg"]                                  # records keep the original name

def test_save_is_atomic(tmp_path, monkeypatch):
    o = extract_outlines([], tmp_path, FakeSegmenter())
    monkeypatch.setattr(np, "savez_compressed", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError):
        save_outlines(tmp_path / "o.npz", o)
    assert not (tmp_path / "o.npz").exists()
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_gnn_outlines.py -q`.

- [ ] **Step 3: Implement `outlines.py` and the CLI command.** Images read with Pillow → RGB `uint8`. Color = mean of the 5×5 patch (clipped to the image) at each contour point. `Sam2Segmenter` imports `sam2` lazily inside `__init__`; with no checkpoint it uses `SAM2ImagePredictor.from_pretrained(model_id, device=device)`, else `build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", checkpoint, device=device)`; call under `torch.inference_mode()` and `torch.autocast("cuda", dtype=torch.bfloat16)`; `predictor.set_image(image)`, `masks, _, _ = predictor.predict(box=boxes, multimask_output=False)`; normalize masks to `(N, H, W)` bool (N = 1 returns `(1, H, W)`, N > 1 returns `(N, 1, H, W)`). `GrabCutSegmenter`: crop box + 10%, downscale to ≤ `max_side`, `cv2.grabCut(..., cv2.GC_INIT_WITH_RECT)`, upsample mask into a full-size array. Save via `np.savez_compressed` to `<out>.tmp.npz` then `atomic_replace`. CLI prints `{"images", "outlines", "ok_rate"}`.

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit** — `"Plan 4 T2: outlines (selection, SAM 2 / GrabCut segmenters, OutlineSet) and gnn-outlines"`

### Task 3: Targets and folds

**Files:** Create `src/snz/gnn/targets.py`; Test `tests/test_gnn_targets.py`

**Interfaces:**
- Consumes: `OutlineSet` (Task 2); `snz.data.bdd.Frame`, `EVAL_CLASSES`; `snz.eval.matching.box_iou`, `match_split`.
- Produces: `fold_split(names: Sequence[str], seed: int = 0) -> tuple[list[str], list[str]]`; `network_targets(o: OutlineSet, frames: Sequence[Frame]) -> np.ndarray[int64, M]` (0–8 class, 9 background, −1 excluded); `tp_targets(o: OutlineSet, frames) -> np.ndarray[bool, M]`; `outline_records(o: OutlineSet, scores: np.ndarray | None = None) -> list[dict]` (prereg-v1 JSONL records for every image in `o.names`, selected detections only, optional replacement scores).

- [ ] **Step 1: Write the failing tests**

```python
def test_fold_split_deterministic_disjoint():
    names = [f"{i}.jpg" for i in range(11)]
    a, b = fold_split(names)
    assert fold_split(names) == (a, b) and not set(a) & set(b) and sorted(a + b) == sorted(names)
    assert abs(len(a) - len(b)) <= 1

def test_network_targets_rules():
    f = Frame("a.jpg", "clear", "daytime", (Box("car", 0, 0, 10, 10), Box("truck", 50, 50, 60, 60),
                                            Box("car", 100, 100, 110, 110, ignore=True)))
    o = outline_set_for("a.jpg", boxes=[[0, 0, 10, 10],         # IoU 1.0 with car     -> car
                                        [50, 50, 60, 62],       # IoU 0.83 with truck  -> truck
                                        [200, 200, 210, 210],   # nothing near         -> background
                                        [0, 0, 10, 25],         # IoU 0.4 with car     -> excluded
                                        [100, 100, 110, 110]])  # only an ignore region -> excluded
    car, truck = EVAL_CLASSES.index("car"), EVAL_CLASSES.index("truck")
    assert list(network_targets(o, [f])) == [car, truck, 9, -1, -1]

def test_outline_records_cover_every_image():
    o = outline_set_for("a.jpg", boxes=[[0, 0, 5, 5]], extra_names=["empty.jpg"])
    recs = outline_records(o)
    assert [r["name"] for r in recs] == ["a.jpg", "empty.jpg"] and recs[1]["labels"] == []

def test_tp_targets_match_prereg_v1_matching():
    f = Frame("a.jpg", "clear", "daytime", (Box("car", 0, 0, 10, 10),))
    o = outline_set_for("a.jpg", boxes=[[0, 0, 10, 10], [0, 0, 10, 11]],
                        labels=["car", "car"], scores=[0.9, 0.8])           # second is a duplicate
    assert list(tp_targets(o, [f])) == [True, False]
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement.** `fold_split`: sort names, shuffle with `random.Random(seed)`, first half = A. `network_targets`: per image, IoU with real boxes (best ≥ 0.5 → class index of that box; boxes of class `train` count as background-excluded → −1) and with all boxes incl. ignore (max < 0.3 → 9). `tp_targets`: `match_split(frames, outline_records(o))`, then map each outline row to its `Matches` row (same image, same class, same score order) and read `det_tp[:, 0]`.

- [ ] **Step 4: Run, expect PASS.** **Step 5: Commit** — `"Plan 4 T3: id-val folds and second-stage targets"`

### Task 4: Networks and training

**Files:** Create `src/snz/gnn/model.py`; Test `tests/test_gnn_model.py`

**Interfaces:**
- Consumes: Task 1 features, Task 2 `OutlineSet`.
- Produces:
  - `class ShapeGNN(nn.Module): __init__(node_dim=5, size_dim=6, n_classes=10, use_size=True); forward(nodes: Tensor[B,64,F], size: Tensor[B,6]) -> Tensor[B,10]`
  - `class GraphDataset(Dataset): __init__(o: OutlineSet, targets: np.ndarray, variant: Literal["gnn", "shape", "color"], augment: bool)` → items `(nodes float32[64,F], size float32[6], target int64)`; F = 5 (`gnn`, `shape`) or 8 (`color`: + color/255).
  - `class CropDataset(Dataset): __init__(o, targets, image_dir: Path, file_name=None, size=112, augment=False)` → `(image float32[3,112,112] ImageNet-normalized, target)`.
  - `rgb_model(pretrained: bool = True) -> nn.Module` (ResNet-18, fc → 10).
  - `train_network(model, dataset, *, epochs: int, lr: float, batch_size: int, weight_decay: float, device: str, seed: int = 0) -> nn.Module` (rows with target −1 skipped); `predict_probs(model, dataset, *, device: str, batch_size=1024) -> np.ndarray[float32, (M, 10)]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_shape_gnn_output_and_cyclic_invariance():
    torch.manual_seed(0); m = ShapeGNN().eval()
    x, s = torch.randn(3, 64, 5), torch.randn(3, 6)
    assert m(x, s).shape == (3, 10)
    torch.testing.assert_close(m(x, s), m(torch.roll(x, 7, dims=1), s), atol=1e-5, rtol=1e-5)

def test_shape_variant_ignores_size():
    m = ShapeGNN(use_size=False).eval(); x = torch.randn(2, 64, 5)
    torch.testing.assert_close(m(x, torch.zeros(2, 6)), m(x, torch.randn(2, 6)))

def test_gnn_features_are_color_blind():
    o1 = synthetic_outlines(color=(255, 0, 0)); o2 = synthetic_outlines(color=(0, 0, 255))
    t = np.zeros(len(o1.score), np.int64)
    a, b = GraphDataset(o1, t, "gnn", False)[0], GraphDataset(o2, t, "gnn", False)[0]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert GraphDataset(o1, t, "color", False)[0][0].shape == (64, 8)

def test_learns_square_vs_circle():
    o, t = squares_and_circles(n=200)                     # targets 0 / 1
    m = train_network(ShapeGNN(), GraphDataset(o, t, "shape", True), epochs=20, lr=1e-3,
                      batch_size=64, weight_decay=1e-4, device="cpu")
    acc = (predict_probs(m, GraphDataset(o, t, "shape", False), device="cpu").argmax(1) == t).mean()
    assert acc > 0.95

def test_rgb_model_shape():
    assert rgb_model(pretrained=False)(torch.randn(2, 3, 112, 112)).shape == (2, 10)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** per Global Constraints (layer sizes, optimizer, schedule, smoothing). Augmentation acts on the **contour, before** `node_features` is computed: random cyclic roll of the 64 points; mirror with p = 0.5 (x → 2·cx − x, then reverse point order so it stays clockwise); rotation uniform in ±10° about the centroid; then size features get Gaussian noise σ = 0.01. EdgeConv in plain PyTorch:

```python
def edge_conv(h, idx, mlp):            # h [B,N,F], idx [N,K]
    nb = h[:, idx]                     # [B,N,K,F]
    hi = h.unsqueeze(2).expand_as(nb)
    return mlp(torch.cat([hi, nb - hi], -1)).max(dim=2).values
```

- [ ] **Step 4: Run, expect PASS.** **Step 5: Commit** — `"Plan 4 T4: ShapeGNN, RGB control, datasets and training loop"`

### Task 5: Fusion + `snz gnn-train`

**Files:** Create `src/snz/gnn/fusion.py`; Modify `src/snz/cli.py`; Test `tests/test_gnn_fusion.py`, `tests/test_gnn_cli.py`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces:
  - `fusion_features(det_score: np.ndarray, probs: np.ndarray, label_idx: np.ndarray) -> np.ndarray[(M, 4)]`; `fit_logistic(X, y, l2=1e-4, iters=50) -> np.ndarray[(5,)]` (intercept first); `predict_logistic(w, X) -> np.ndarray`; `auroc(y: np.ndarray, p: np.ndarray) -> float`.
  - CLI `snz gnn-train --outlines id-val.npz --labels det_train.json --manifest manifests/id-val.csv --images DIR --out DIR [--arms GNN,RGB,GNN-SHAPE,GNN-COLOR] [--device cuda] [--fold-dir manifests]` writes `<fold-dir>/gnn_fold_{a,b}.csv` (rows copied from `id-val.csv`), `<out>/<ARM>.pt`, `<out>/<ARM>.fusion.json` (`{"w": [...], "features": [...]}`), `<out>/gnn-train.json` (`{"counts": {"fold_a": {...per target...}, "fold_b": ...}, "fold_b_auroc": {"BASE": ..., "<ARM>": ...}, "sha256": {"<ARM>.pt": ...}, "config": {...}}`).

- [ ] **Step 1: Write the failing tests**

```python
def test_fit_logistic_recovers_weights():
    rng = np.random.default_rng(0); X = rng.normal(size=(5000, 4)); w = np.array([0.5, 2, -1, 0, 1])
    y = rng.random(5000) < 1 / (1 + np.exp(-(w[0] + X @ w[1:])))
    np.testing.assert_allclose(fit_logistic(X, y), w, atol=0.15)

def test_fit_logistic_separable_is_finite():
    X = np.r_[-np.ones((50, 4)), np.ones((50, 4))]; y = np.r_[np.zeros(50), np.ones(50)].astype(bool)
    w = fit_logistic(X, y); assert np.isfinite(w).all()
    p = predict_logistic(w, X); assert ((p > 0) & (p < 1)).all()

def test_fusion_features_columns():
    probs = np.array([[0.7, 0.1, 0.0, 0, 0, 0, 0, 0, 0, 0.2]])
    X = fusion_features(np.array([0.5]), probs, np.array([0]))
    np.testing.assert_allclose(X[0, :3], [0.0, np.log(0.7), np.log(0.2)], atol=1e-6)

def test_gnn_train_cli_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("SNZ_GNN_EPOCHS", "1")
    imgs, labels, manifest, outlines = tiny_idval(tmp_path, n_images=8)   # gnn_helpers
    out = tmp_path / "models"
    assert main(["gnn-train", "--outlines", str(outlines), "--labels", str(labels),
                 "--manifest", str(manifest), "--images", str(imgs), "--out", str(out),
                 "--arms", "GNN,GNN-SHAPE", "--device", "cpu",
                 "--fold-dir", str(tmp_path / "manifests")]) == 0
    assert (out / "GNN.pt").exists() and (out / "GNN.fusion.json").exists()
    summary = json.loads((out / "gnn-train.json").read_text())
    assert set(summary["fold_b_auroc"]) >= {"BASE", "GNN"}
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement.** Probabilities clipped to [1e-6, 1] before logs; logit(score) with score clipped to [1e-6, 1 − 1e-6]. IRLS with L2 on non-intercept weights. `gnn-train` trains each requested arm on fold A (targets ≠ −1), predicts fold B, fits fusion on fold B `tp_targets`, records fold-B AUROC of the fused score (and of the raw detector score as `BASE`). An environment variable `SNZ_GNN_EPOCHS` (tests only) overrides epochs.

- [ ] **Step 4: Run, expect PASS.** **Step 5: Commit** — `"Plan 4 T5: fusion (IRLS) and gnn-train"`

### Task 6: Re-scoring + guard tag + `snz gnn-rescore`

**Files:** Create `src/snz/gnn/rescore.py`; Modify `src/snz/eval/guard.py`, `src/snz/cli.py`; Test `tests/test_gnn_rescore.py`

**Interfaces:**
- Consumes: Tasks 2–5; `snz.eval.predict.write_detections`.
- Produces: `ensure_split_allowed(split, repo=Path("."), manifest=None, tag: str = PREREG_TAG)` (existing callers unchanged); `rescore(o: OutlineSet, arm: str, models_dir: Path, *, image_dir: Path, file_name=None, device: str) -> list[dict]`; CLI `snz gnn-rescore --split {id-val,val-all,t-texture} --outlines FILE --models DIR --images DIR --out-root DIR [--arms ...] [--stylized-copy N] [--device cuda]` writing `<out-root>/<ARM>-s0/eval/<split>.detections.jsonl` and `<out-root>/<ARM>-s0/provenance.json` (`{"<split>": {"outlines": path, "outlines_sha256": ..., "models_sha256": ..., "ok_rate": ...}}`, merged across splits).

- [ ] **Step 1: Write the failing tests**

```python
def test_base_arm_keeps_selected_scores(tmp_path):
    o = outline_set_for("a.jpg", boxes=[[0, 0, 5, 5], [1, 1, 6, 6]], scores=[0.9, 0.2],
                        extra_names=["empty.jpg"])
    recs = rescore(o, "BASE", tmp_path, image_dir=tmp_path, device="cpu")
    assert recs[0]["scores"] == pytest.approx([0.9, 0.2])
    assert recs[1] == {"name": "empty.jpg", "boxes": [], "labels": [], "scores": []}

def test_rescored_records_are_matchable(tmp_path, monkeypatch):
    monkeypatch.setenv("SNZ_GNN_EPOCHS", "1")
    imgs, labels, manifest, outlines = tiny_idval(tmp_path, n_images=8)
    main(["gnn-train", "--outlines", str(outlines), "--labels", str(labels), "--manifest",
          str(manifest), "--images", str(imgs), "--out", str(tmp_path / "m"), "--arms", "GNN",
          "--device", "cpu", "--fold-dir", str(tmp_path / "manifests")])
    o, frames = load_outlines(outlines), load_frames(labels)[0]
    recs = rescore(o, "GNN", tmp_path / "m", image_dir=imgs, device="cpu")
    m = match_split(frames, recs); assert len(m.det_score) == len(o.score)
    assert all(0 < s < 1 for r in recs for s in r["scores"])

def test_guard_requires_prereg_v2(tmp_path):            # git repo with only prereg-v1
    with pytest.raises(PermissionError, match="prereg-v2"):
        ensure_split_allowed("val-all", repo=tmp_path, tag="prereg-v2")
    ensure_split_allowed("id-val", repo=tmp_path, tag="prereg-v2")   # allowed
```

- [ ] **Step 2: Run, expect FAIL.** **Step 3: Implement** (`gnn-rescore` calls the guard with `tag=GNN_PREREG_TAG`; BASE = identity scores over the selected set; other arms: features → network probs → `predict_logistic`). **Step 4: PASS.** **Step 5: Commit** — `"Plan 4 T6: re-scoring, prereg-v2 guard and gnn-rescore"`

### Task 7: Analysis + `snz gnn-analyze`

**Files:** Create `src/snz/gnn/analyze.py`; Modify `src/snz/eval/operating.py`, `src/snz/cli.py`; Test `tests/test_gnn_analyze.py`, `tests/test_operating.py`

**Interfaces:**
- Consumes: re-scored files from Task 6; `snz.eval` matching/AP/operating/bootstrap; `SHIFT`, `SUBSETS` from `snz.eval.analyze`.
- Produces: `threshold_for_false_rate(m: Matches, target: float, weights=None) -> OperatingPoint` (lowest threshold whose false detections per image ≤ target; if none, the highest score threshold, flagged); `gnn_analyze(runs_root: Path, val_labels: Path, manifests_dir: Path, n_boot: int = 2000, seed: int = 0) -> dict`; CLI `snz gnn-analyze --runs DIR --labels det_val.json --manifests manifests --out results/gnn` writing `analysis.json`, `report.md` (guarded by `prereg-v2`).

- [ ] **Step 1: Write the failing tests**

```python
def test_threshold_for_false_rate():
    m = matches_from([("a", 0.9, "tp"), ("a", 0.8, "fd"), ("a", 0.7, "tp"), ("b", 0.6, "fd")])
    assert threshold_for_false_rate(m, 0.5).threshold == pytest.approx(0.7)   # 1 FD over 2 images
    assert threshold_for_false_rate(m, 0.0).threshold == pytest.approx(0.9)
    assert threshold_for_false_rate(m, 0.0).flagged is False

def test_planted_gnn_improvement_is_detected(tmp_path):
    runs, labels, manifests = planted_runs(tmp_path)   # gnn_helpers: 40 val images split into
    # t-clean/t-night/t-rain manifests; BASE-s0 and GNN-s0 share boxes, GNN gives every
    # false detection score 0.01; no RGB-s0 directory
    res = gnn_analyze(runs, labels, manifests, n_boot=200)
    assert res["confirmatory"]["G1"]["supported"] and res["confirmatory"]["G2"]["supported"]
    assert res["confirmatory"]["G3"]["p_holm"] == 1.0 and "RGB-s0" in res["missing"]
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement.** Mirror `snz.eval.analyze.analyze`: slice `val-all` into subsets and pooled T-shift; per arm point estimates (mAP, AP50, per-class, false detections by type at R\*, recall at F\*); bootstrap with one weight matrix for T-shift, re-estimating R\*, F\* and thresholds per replicate; G1 = GNN − BASE false/img (less), G2 = GNN − BASE recall at F\* (greater), G3 = GNN − RGB false/img (less); Holm over the three (missing arm → p = 1); descriptive CIs for GNN − GNN-SHAPE, GNN-COLOR − GNN, every subset, T-texture retention, outline `ok` rate per subset (read from the outline npz paths recorded in each run dir's `provenance.json`, written by `gnn-rescore`).

- [ ] **Step 4: Run, expect PASS.** **Step 5: Commit** — `"Plan 4 T7: gnn-analyze (G1-G3, Holm, descriptive) and threshold_for_false_rate"`

### Task 8: GX10 script, preregistration, runbook, README

**Files:** Create `scripts/run_gnn.sh`, `prereg/prereg-v2.md`; Modify `docs/runbook.md`, `README.md`

- [ ] **Step 1: `scripts/run_gnn.sh <stage>`** with stages `predict` (the three `snz evaluate` calls of Operations step 2), `outlines` (id-val, val-all, t-texture with `--stylized-copy 1`), `train`, `rescore` (val-all, t-texture), `analyze`; each stage skips outputs that exist; `set -euo pipefail`; `D="${SNZ_DATA:-$HOME/snz-data}"`, outputs under `$D/gnn/`. Verify: `bash -n scripts/run_gnn.sh`.
- [ ] **Step 2: `prereg/prereg-v2.md`** — hypotheses G1–G3 and descriptive analyses verbatim from spec §2; arms (§3); data and targets (§4); components and hyperparameters (§5); deviations log. Placeholders to fill **after `train`, before tagging**: fold manifest SHA-256s, target counts, fold-B AUROCs, `sha256` of each `<ARM>.pt`, segmenter used (SAM 2.1 hiera-small or GrabCut fallback), outline ok-rate on id-val.
- [ ] **Step 3: Runbook + README** — GNN section in `docs/runbook.md` with the Operations commands below; README status line points to the spec, this plan and `results/gnn/report.md`.
- [ ] **Step 4: Full check** — `uv run ruff check src tests && uv run pytest` → all pass (133 existing + new).
- [ ] **Step 5: Commit** — `"Plan 4 T8: run_gnn.sh, prereg-v2 draft, runbook and README"`

---

## Operations (GX10, in order)

| # | Command / action | Who | ≈ time |
|---|---|---|---|
| 0 | Confirm `~/snz-data/runs/BASE-s0/run.json` exists, then stop G5 (`pkill -f run_g5.sh; pkill -f "snz train"`). Leave finished runs in place. | Enoch | 1 min |
| 1 | `git fetch origin && git checkout <branch with Plan 4> && SAM2_BUILD_CUDA=0 uv sync --extra gnn` | Enoch | 5 min |
| 1b | SAM 2 smoke: `uv run python -c "from sam2.sam2_image_predictor import SAM2ImagePredictor as P; P.from_pretrained('facebook/sam2.1-hiera-small', device='cuda')"`. If Hugging Face fails: download `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt` and pass `--sam2-checkpoint`. If `sam2` won't install within 15 min: use `--segmenter grabcut` and record it in `prereg-v2`. | Enoch | 5 min |
| 2 | BASE-s0 detections with the **existing** code (start now, no Plan 4 code needed). With `D=~/snz-data L=$D/bdd100k/labels/det_2018 R=$D/runs/BASE-s0`: `uv run snz evaluate --split id-val --labels $L/det_train.json --manifest manifests/id-val.csv --images $D/bdd100k/images/100k/train --checkpoint $R/final.pt --out $R/eval`; `uv run snz evaluate --split val-all --labels $L/det_val.json --manifest manifests/val-all.csv --images $D/bdd100k/images/100k/val --checkpoint $R/final.pt --out $R/eval`; `uv run snz evaluate --split t-texture --labels $L/det_val.json --manifest manifests/t-clean.csv --images $D/stylized/t-texture --stylized-copy 1 --checkpoint $R/final.pt --out $R/eval` (skip any whose `.detections.jsonl` already exists) | Enoch | 10 min |
| 3 | `scripts/run_gnn.sh outlines` | GX10 | 25–40 min |
| 4 | `scripts/run_gnn.sh train` | GX10 | ≈ 15 min |
| 5 | Fill `prereg/prereg-v2.md` from `$D/gnn/models/gnn-train.json`; commit; `git tag prereg-v2 && git push origin prereg-v2` | Aaron | 10 min |
| 6 | `scripts/run_gnn.sh rescore && scripts/run_gnn.sh analyze`; commit `results/gnn/` | Enoch | 10 min |
| 7 | Results into `README.md` and `paper/paper.md` | Aaron | 15 min |

Laptops implement Tasks 1–8 in parallel with steps 0–1 (Tasks 1–3 first: step 3 needs them).

## Self-review notes

- Spec coverage: §1 pipeline → T1–T6; §2 hypotheses → T7; §3 arms and priority → T4–T6, Global Constraints; §4 data/targets/folds → T3, T5; §5.1 outlines incl. GrabCut fallback → T2; §5.2 features → T1; §5.3 networks/fusion → T4–T5; §5.4 re-scoring/guard/analysis → T6–T7; §6 roles/timeline → Operations; §7 risks → Review Focus, Operations 1b; §8 follow-up → not in this plan (by design).
- Names used across tasks: `OutlineSet`, `outline_records`, `GNN_CLASSES`, `GNN_PREREG_TAG`, `threshold_for_false_rate`, `ARMS_G` — defined once (T1–T3, T6–T7).
