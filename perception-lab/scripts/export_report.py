#!/usr/bin/env python3
"""Generate artifacts/report.md from committed run bundles (data/results/*.zip).
Every number in the tables is computed here from saved records; the narrative only cites them.

    python3 scripts/export_report.py
"""
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.evaluate import BASELINE, VARIANT, metrics, percentile  # noqa: E402

runs = {}
for z in sorted((ROOT / "data/results").glob("run-*.zip")):
    with zipfile.ZipFile(z) as f:
        run = json.loads(f.read("run.json"))
        recs = [json.loads(line) for line in f.read("results.jsonl").decode().splitlines() if line.strip()]
        runs[run["run_key"] or run["run_id"]] = {"run": run, "recs": recs, "m": metrics(recs)}

prompts = json.loads((ROOT / "experiments/prompts.json").read_text())
lock = json.loads((ROOT / "experiments/model-lock.json").read_text())
ver = json.loads((ROOT / "artifacts/verification.json").read_text())
M = {BASELINE: "RGB baseline", VARIANT: "RGB + structure prompt"}
TARGETS = [("truck", "physical_vehicle"), ("depiction_board", "flat_depiction"), ("plain_board", "other")]


def frac(f):
    return "–" if not f["d"] else f"{f['n']}/{f['d']} ({f['pct']}%)"


def cond(r):
    return json.loads(r["conditions"])


def summary_table(key, split):
    m = runs[key]["m"]["by_split_method"][split]
    rows = ["| Method | Attempted | Valid | Op. failures | Abstentions | **Correct / attempted** | Acc. among answered | Coverage | "
            "False vehicle on depictions | Vehicle recall | Control → other | Latency p50 / p95 (n) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for meth in (BASELINE, VARIANT):
        s = m[meth]
        lat = s["latency_ms"]
        rows.append(f"| {M[meth]} | {s['attempted']} | {s['valid_responses']} | {s['operational_failures']} | {s['abstentions']} | "
                    f"**{frac(s['correct_over_attempted'])}** | {frac(s['accuracy_answered'])} | {frac(s['coverage'])} | "
                    f"{frac(s['false_vehicle_on_depictions'])} | {frac(s['vehicle_recall'])} | {frac(s['control_correct'])} | "
                    f"{lat['p50']:.0f} / {lat['p95']:.0f} ms ({lat['n']}) |")
    return "\n".join(rows)


def target_table(key, split):
    recs = [r for r in runs[key]["recs"] if r["split"] == split]
    rows = ["| Target (truth) | Method | Predicted labels | Correct |", "|---|---|---|---|"]
    for t, truth in TARGETS:
        for meth in (BASELINE, VARIANT):
            rs = [r for r in recs if cond(r)["target"] == t and r["method_id"] == meth]
            c = Counter(r["label"] or r["status"] for r in rs)
            rows.append(f"| {t} (`{truth}`) | {M[meth]} | {', '.join(f'{k} {v}' for k, v in sorted(c.items()))} | "
                        f"{sum(r['label'] == truth for r in rs)}/{len(rs)} |")
    return "\n".join(rows)


def paired_line(key, split):
    p = runs[key]["m"]["paired_by_split"][split]
    return (f"Paired on byte-identical inputs ({p['pairs']} pairs, hash mismatches {p['hash_mismatch']}): both correct **{p['both_correct']}**, "
            f"baseline-only (variant regressions) **{p['baseline_only']}**, variant-only (variant repairs) **{p['variant_only']}**, "
            f"neither **{p['neither']}**, unresolved **{p['unresolved']}**.")


def disc_table():
    recs = runs["discovery-v1"]["recs"]
    rows = ["| Variant | Target (truth) | Baseline labels | Structure-prompt labels |", "|---|---|---|---|"]
    for v in ("near", "far", "angle25", "occluded"):
        for t, truth in TARGETS:
            cells = []
            for meth in (BASELINE, VARIANT):
                rs = [r for r in recs if cond(r).get("variant") == v and cond(r)["target"] == t and r["method_id"] == meth]
                cells.append(", ".join(f"{k} {n}" for k, n in sorted(Counter(r["label"] for r in rs).items())))
            rows.append(f"| {v} | {t} (`{truth}`) | {cells[0]} | {cells[1]} |")
    return "\n".join(rows)


XM = [("nemotron3:33b (headline lock)", "dev-v2", "heldout-v1"),
      ("qwen3.6:35b", "xmodel-qwen3.6-dev-v1", "xmodel-qwen3.6-heldout-v1"),
      ("gemma4:26b", "xmodel-gemma4-dev-v1", "xmodel-gemma4-heldout-v1")]


def xmodel_table():
    rows = ["| Model | Split | Method | Correct / attempted | Physical trucks → physical_vehicle | Depictions → flat_depiction | "
            "Plain boards → other | False vehicle on depictions | Abstain | Op. failures | Fenced JSON | p50 ms |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, dk, hk in XM:
        for key, split in ((dk, "development"), (hk, "heldout")):
            for meth in (BASELINE, VARIANT):
                sm = runs[key]["m"]["by_split_method"][split][meth]
                rs = [r for r in runs[key]["recs"] if r["method_id"] == meth]
                dep = sum(r["label"] == "flat_depiction" for r in rs if r["truth"] == "flat_depiction")
                fenced = sum((r["raw_output"] or "").strip().startswith("```") for r in rs)
                rows.append(f"| {name} | {split} | {M[meth]} | **{frac(sm['correct_over_attempted'])}** | {frac(sm['vehicle_recall'])} | "
                            f"{dep}/6 | {frac(sm['control_correct'])} | {frac(sm['false_vehicle_on_depictions'])} | {sm['abstentions']} | "
                            f"{sm['operational_failures']} | {fenced} | {sm['latency_ms']['p50']:.0f} |")
    return "\n".join(rows)


def xmodel_paired():
    out = []
    for name, dk, hk in XM:
        for key, split in ((dk, "development"), (hk, "heldout")):
            p = runs[key]["m"]["paired_by_split"][split]
            out.append(f"| {name} | {split} | {p['both_correct']} | {p['baseline_only']} | {p['variant_only']} | {p['neither']} | {p['unresolved']} |")
    return "| Model | Split | Both correct | Baseline-only (variant regressions) | Variant-only (repairs) | Neither | Unresolved |\n|---|---|---|---|---|---|---|\n" + "\n".join(out)


FID = [("nemotron3:33b", ["dev-v2", "heldout-v1"], "fidelity-nemotron-v1"),
       ("qwen3.6:35b", ["xmodel-qwen3.6-dev-v1", "xmodel-qwen3.6-heldout-v1"], "fidelity-qwen3.6-v1"),
       ("gemma4:26b", ["xmodel-gemma4-dev-v1", "xmodel-gemma4-heldout-v1"], "fidelity-gemma4-v1")]


def fidelity_table():
    rows = ["| Model | View | Target (truth) | Method | Standard render correct | High-fidelity correct | Label changed |",
            "|---|---|---|---|---|---|---|"]
    totals = []
    for name, core_keys, fk in FID:
        core = {(r["case_id"], r["method_id"]): r for k in core_keys for r in runs[k]["recs"]}
        cells = {}
        for r in runs[fk]["recs"]:
            c = core[(r["case_id"].replace("fid-", "core-"), r["method_id"])]
            cd = cond(r)
            key = ("ego lane" if cd["camera"] == "cam_a_ego_lane" else "oblique", cd["target"], r["method_id"])
            t = cells.setdefault(key, [0, 0, 0, 0])
            t[0] += c["label"] == c["truth"]; t[1] += r["label"] == r["truth"]; t[2] += 1; t[3] += c["label"] != r["label"]
        changed = sum(v[3] for v in cells.values())
        totals.append(f"{name}: {changed}/{sum(v[2] for v in cells.values())} labels changed")
        for (view, tgt, meth), (a_, b_, n_, ch) in sorted(cells.items()):
            truth = dict(TARGETS)[tgt]
            rows.append(f"| {name} | {view} | {tgt} (`{truth}`) | {M[meth]} | {a_}/{n_} | {b_}/{n_} | {ch} |")
    return "\n".join(rows), "; ".join(totals)


RES = json.loads((ROOT / "artifacts/model-resources.json").read_text()) if (ROOT / "artifacts/model-resources.json").exists() else None


def resources_section():
    if not RES:
        return "Not measured."
    rows = ["| Model | Params / quant | Resident in unified GPU memory | Cold request (load) | Warm p50 / p95 (n=10) | Output tokens/s | GPU vs CPU-only (warm p50) |",
            "|---|---|---|---|---|---|---|"]
    meta = {"nemotron3:33b": "33.0B / Q4_K_M", "qwen3.6:35b": "36.0B MoE / Q4_K_M", "gemma4:26b": "25.8B / Q4_K_M"}
    for m, d in RES["per_model"].items():
        cv = RES["cpu_vs_gpu"].get(m)
        cpu = f"{cv['gpu_warm_p50_ms']:.0f} ms vs {cv['cpu_warm_p50_ms']:.0f} ms (**{cv['speedup_gpu_over_cpu']:.1f}×**)" if cv else "not measured"
        rows.append(f"| {m} | {meta.get(m, '')} | {d['resident'][0]['size_gb']:.1f} GB ({d['resident'][0]['vram_gb']:.1f} GB on GPU) | "
                    f"{d['cold_request_ms'] / 1000:.1f} s ({d['cold_load_ms'] / 1000:.1f} s) | {d['warm_ms']['p50']:.0f} / {d['warm_ms']['p95']:.0f} ms | "
                    f"{d['warm_eval_tokens_per_s']:.0f} | {cpu} |")
    a, sw = RES["all_resident"], RES["switching_all_resident"]
    return ("\n".join(rows) + f"""

All three models resident at once: **{a['total_resident_gb']:.1f} GB** of model weights in the GX10's {a['memory']['total_gib']:.1f} GiB unified memory
({a['memory']['available_gib']:.1f} GiB still available). Round-robin requests across the three resident models: p50 **{sw['p50_ms']:.0f} ms**, max model-load time
{sw['max_load_ms']:.0f} ms — i.e. switching models costs nothing beyond a normal request, versus 6–11 s per cold load when only one model fits.
Measured {RES['measured_at']} with Ollama {RES['ollama_version']}, the frozen baseline prompt and a real corpus image (`fid-019`), serial requests
(`scripts/measure_models.py`, raw data `artifacts/model-resources.json`). CPU-only = Ollama `num_gpu: 0` on the GX10's 20 Arm cores; labels matched the GPU runs.""")


h1, h2 = runs["heldout-v1"]["recs"], runs["heldout-repeat-v1"]["recs"]
a = {(r["case_id"], r["method_id"]): r["label"] for r in h1}
b = {(r["case_id"], r["method_id"]): r["label"] for r in h2}
agree = sum(a[k] == b[k] for k in a)
flips = [f"{k[0]} / {M[k[1]]}: {a[k]} → {b[k]}" for k in a if a[k] != b[k]]
heldout_labels = Counter(r["label"] for r in h1)
const_flat = sum(r["truth"] == "flat_depiction" for r in h1 if r["method_id"] == BASELINE)
all_live = [r for k in runs for r in runs[k]["recs"] if r["origin"] == "live" and r["status"] == "ok"]
lat_all = {meth: [r["elapsed_ms"] for r in all_live if r["method_id"] == meth] for meth in (BASELINE, VARIANT)}
n_live = len(all_live)
n_ops = sum(r["status"] in ("invalid_output", "timeout", "provider_error") for k in runs for r in runs[k]["recs"])
hm = runs["heldout-v1"]["m"]["by_split_method"]["heldout"]
rm = runs["heldout-repeat-v1"]["m"]["by_split_method"]["heldout"]
off = runs["offline-guard-smoke-v1"]
smoke_id = next(k for k in runs if k.startswith("run-"))
rid = lambda k: runs[k]["run"]["run_id"]
base_prompt = prompts["methods"][BASELINE]["prompt"]
extra = prompts["methods"][VARIANT]["prompt"][len(base_prompt) + 1:]

out = f"""# Perception Lab — report

Status: **complete** against the MVP criteria in the design document (§10 lists what is and is not covered), plus a
user-approved exploratory cross-model comparison (§6.7) on the same frozen protocol.
Generated by `scripts/export_report.py` from the committed run bundles in `data/results/`; every table is computed from saved records.
Artifact verification: **{ver['passed']} passed, {ver['failed']} failed** (`artifacts/verification.json`).

## 1. Question

With physical geometry held fixed, does a local vision-language model running on the ASUS Ascent GX10 distinguish a
**physical truck** from a **picture of the same truck on a flat board**, and from a **plain board** (negative control)?
And does one fixed structure-focused prompt sentence change its answers on the *same* images?

This is an image-classification test under controlled synthetic conditions. It does not measure braking, detection, or
the behaviour of any production driver-assistance system.

## 2. System and model

- Rendering: Three.js/WebGL in the laptop browser (headless Chrome for the corpus). **The GX10 GPU does not render the scene.**
- Inference, job queue, SQLite and artifacts: on the GX10 (Ubuntu 24.04, aarch64, NVIDIA GB10, 121 GiB unified memory); backend bound to loopback and reached through an SSH tunnel over Tailscale.
- Model: `{lock['model_id']}` ({lock['runtime_version']}), digest `{lock['model_digest'][:12]}…`, {lock['parameter_size']}, {lock['quantization']},
  architecture `{lock['architecture']}` (NVIDIA Nemotron 3 Nano Omni family; NVIDIA Open Model License). Image input verified with a real image request before use.
- Decoding: `think=false`, options `{json.dumps(lock['options'])}`, Ollama structured output constrained to the 4-label JSON schema (same for both methods). Temperature 0 and a fixed seed improve, but do not guarantee, repeatability (§6.4).
- Cold model load ≈ 11.3–11.5 s on the first request after idle; kept out of measured latency by an unrecorded warm-up call.

## 3. Corpus

Procedural scene: straight two-lane road, one simplified truck, a roadside board, a background billboard (abstract ad), trees, buildings, lamp posts. All assets self-created.
The depiction texture is the truck photographed from the saved ego-lane camera with an off-axis frustum that exactly covers the board, so **from the ego-lane camera the picture has the truck's own perspective**; from the oblique camera the board's flatness, posts and braces are visible.

| | |
|---|---|
| Core corpus | 36 = 3 targets × 3 appearances (white / striped / dark livery) × 2 cameras × 2 lighting (day / back-lit dusk) |
| Development split | 18 — camera A, ego lane, azimuth 9°, 22.3 m |
| Held-out split | 18 — camera B, oblique, azimuth 50°, 22.3 m; sealed by the API until prompts and model lock were frozen |
| Discovery (exploratory) | 24 — near 14 m, far 34 m, 25° angle, hedge occluder; white livery; never pooled into headline metrics |
| Resolution | 512 × 512 PNG; metadata chunks stripped at ingest; SHA-256 recorded |
| Marker | Yellow rectangle = projected bounds of the target slot (union of truck and board extents); identical for all target types per camera (verified) |
| Determinism | All 60 cases rendered twice → byte-identical PNGs (verified) |
| Observability | `unchecked` for every case — no independent reviewer was available |

Development and held-out share synthetic assets and differ in viewpoint: they are **not** an independent real-world benchmark and are not of equal difficulty.

## 4. Methods (both receive the byte-identical RGB image)

**RGB baseline** (frozen, revision {prompts['revision']}):

> {base_prompt}

**RGB + structure prompt** = the baseline plus one fixed sentence:

> {extra}

This is a **prompt intervention, not a new model**. The model's `visible_evidence` strings are its stated evidence, not a verified account of its internal reasoning.

## 5. Protocol timeline (UTC, 2026-10-11)

| Time | Step | Run |
|---|---|---|
| 05:17 | Build start; deadline 15:17 | – |
| 05:42 | End-to-end smoke (2 dev cases) | `{rid(smoke_id)}` |
| 05:43 | Development, prompt revision 1 | `{rid('dev-v1')}` |
| 05:45 | **Prompt revision 2**: revision 1 labelled all 6 plain-board controls `flat_depiction` under both methods; label names were undefined. One identical label-definition clause added to both prompts. | – |
| 05:45 | Development, revision 2 | `{rid('dev-v2')}` |
| {prompts['frozen_at'][11:16]} | **Freeze** prompts (rev 2) and model lock; committed to git before any held-out request | – |
| 05:46 | **Held-out (headline)** | `{rid('heldout-v1')}` |
| 05:47 | Discovery (exploratory) | `{rid('discovery-v1')}` |
| 05:53 | Fresh held-out inference with outbound guard on | `{rid('offline-guard-smoke-v1')}` |
| 05:55 | Held-out repeat (test–retest only) | `{rid('heldout-repeat-v1')}` |
| 06:05 | Pre-registered frozen locks for `qwen3.6:35b` and `gemma4:26b` committed (same prompts/options/schema) | – |
| 06:22–06:27 | Exploratory high-fidelity renders pre-registered (36 paired), then run on all three models | `{rid('fidelity-nemotron-v1')}`, `{rid('fidelity-qwen3.6-v1')}`, `{rid('fidelity-gemma4-v1')}` |
| 06:06–06:10 | Exploratory cross-model: qwen3.6 dev / held-out, gemma4 dev / held-out | `{rid('xmodel-qwen3.6-dev-v1')}`, `{rid('xmodel-qwen3.6-heldout-v1')}`, `{rid('xmodel-gemma4-dev-v1')}`, `{rid('xmodel-gemma4-heldout-v1')}` |

Development numbers after revision 2 were measured on the data used to choose the revision and are not an unbiased estimate. The held-out split is the headline.

## 6. Results

### 6.1 Held-out (headline; frozen configuration; one run)

{summary_table('heldout-v1', 'heldout')}

{paired_line('heldout-v1', 'heldout')}

{target_table('heldout-v1', 'heldout')}

**Read this carefully.** Across both methods the model answered `flat_depiction` for {heldout_labels.get('flat_depiction', 0)} of {sum(heldout_labels.values())} held-out
responses (`physical_vehicle` {heldout_labels.get('physical_vehicle', 0)}, `other` {heldout_labels.get('other', 0)}, `uncertain` {heldout_labels.get('uncertain', 0)}).
A constant `flat_depiction` predictor would score {const_flat}/18 on this split — the same as the baseline — so the perfect score on depictions is
**not evidence of discrimination** here. From the oblique view the model labelled the physical truck a flat depiction in
{hm[BASELINE]['outcomes']['vehicle_as_depiction']}/6 (baseline) and {hm[VARIANT]['outcomes']['vehicle_as_depiction']}/6 (structure prompt) cases, typically stating that
the truck "appears as a 2D graphic with flat colors and simple shapes". Our first hypothesis was that the low-poly rendering simply looks like an
illustration. The exploratory comparison (§6.7) weakens that as a complete explanation: on the **same images**, `gemma4:26b` labelled 5/6 of these trucks
`physical_vehicle` (1 abstention) and `qwen3.6:35b` 6/6. The oblique-truck failure is therefore at least partly specific to `nemotron3:33b`
under this protocol; rendering style may still contribute (gemma4's one abstention cites a "highly stylized illustration").
No false-vehicle label was produced on a depiction in any frozen-prompt run.

### 6.2 Development (camera A; revision 2 is post-tuning)

Revision 1 (before label definitions):

{summary_table('dev-v1', 'development')}

{paired_line('dev-v1', 'development')}

Revision 2 (frozen prompts):

{summary_table('dev-v2', 'development')}

{paired_line('dev-v2', 'development')}

{target_table('dev-v2', 'development')}

From the ego-lane view the model recognised all 6 physical trucks and all 6 depiction boards under both methods, including the back-lit dusk depictions that
revision 1's baseline had abstained on or labelled `physical_vehicle`. All 6 plain-board controls were still labelled `flat_depiction`, even when the stated evidence was
"no visible vehicle features". We report this as a genuine `wrong_other_label` failure in how the model applies the label set and did not tune further.

### 6.3 Discovery (exploratory, 24 cases, not pooled)

{disc_table()}

{paired_line('discovery-v1', 'discovery')} The physical truck was recognised at 25° and at 34 m but labelled `flat_depiction` when close (14 m) and when partly hidden by a hedge.
These are single exploratory observations, not a measured trend.

### 6.4 Repeatability

Uncached repeat of the full held-out split with the identical frozen configuration: label agreement **{agree}/{len(a)}**.
Changed: {'; '.join(flips) if flips else 'none'}. Repeat-run correct/attempted: baseline
{rm[BASELINE]['correct_over_attempted']['n']}/18, structure prompt {rm[VARIANT]['correct_over_attempted']['n']}/18.
**The structure prompt's one-case held-out advantage in the headline run is the same size as run-to-run variation**, so no improvement is claimed.

### 6.5 Local-only check

With the backend's application-level outbound guard on, a probe connection from the backend process to 1.1.1.1:443 was blocked
(`artifacts/offline-guard-probe.json`), and a fresh uncached held-out inference ({len(off['recs'])} requests, run `{rid('offline-guard-smoke-v1')}`) completed live on the GX10:
{', '.join(f"{r['case_id']}/{M[r['method_id']]} → {r['label']}" for r in off['recs'])}.
Limits: the guard covers the backend Python process only (not the Ollama server process or other host traffic); host-wide networking was not changed.

### 6.6 Latency and reliability

All {n_live} live, valid requests across runs (serial, single worker, warm model): baseline p50 {percentile(lat_all[BASELINE], 50):.0f} ms / p95 {percentile(lat_all[BASELINE], 95):.0f} ms
(n={len(lat_all[BASELINE])}); structure prompt p50 {percentile(lat_all[VARIANT], 50):.0f} ms / p95 {percentile(lat_all[VARIANT], 95):.0f} ms (n={len(lat_all[VARIANT])}).
This is end-to-end request latency measured in the backend, not frontend frame rate. Operational failures (invalid output, timeout, provider error) across all runs: **{n_ops}**.
The UI smoke test observed capturing → uploading → queued → running → completed for an interactive two-method analysis in about 7 s (`artifacts/screenshots/ui-smoke.json`).

### 6.7 Exploratory cross-model comparison (user-approved extension; not the headline)

Same 36 core images, same frozen prompts (rev 2), options and output schema; only the model changes, via separately frozen lock files committed
before these runs. The prompts were developed on `nemotron3:33b` development results only, so the other models run a protocol not tuned for them.
Image input was verified for each model with a red-square smoke image. All results are live, single runs.

{xmodel_table()}

{xmodel_paired()}

Observations (single exploratory runs; small samples; no significance claimed):

- **The oblique physical-truck failure is model-specific.** gemma4 and qwen3.6 recognised the held-out physical trucks (5/6 + 1 abstention; 6/6) where nemotron3 labelled them `flat_depiction` (6/6 baseline).
- **The plain-board control failure is partly shared.** On held-out (oblique, board on visible posts) qwen3.6 labelled 5/6 plain boards `flat_depiction` under both methods, as nemotron3 did 6/6; from the ego-lane view qwen3.6's baseline got 5/6 right. gemma4 labelled all 12 `other`. With these label definitions, "a flat picture of a vehicle" vs "a flat surface that shows no vehicle" is not reliably separated by two of the three models.
- **The structure sentence can hurt.** On qwen3.6 development (ego-lane view, where the depiction has the truck's own perspective), the structure prompt turned 3/6 depictions into `physical_vehicle` (`depiction_as_vehicle`) and 2 more controls into `flat_depiction`: 5 regressions, 0 repairs. It made no held-out difference for qwen3.6 or gemma4, and gemma4 was already at ceiling.
- **Output format.** qwen3.6 returned some answers inside a single code fence (allowed by the parser without repair; counts in the table). In a red-square smoke test with a vaguer prompt it ignored the schema entirely (a fenced JSON array) — none of those occurred with the frozen task prompt. No model produced invalid outputs or timeouts here.

### 6.8 Exploratory high-fidelity render study (user-approved extension; not the headline)

The 36 core cases were re-rendered with a pre-registered high-fidelity path (`frontend/src/scene/hq.ts`: rounded panels, clear-coat paint,
corrugated trailer, wheel rims, fuel tanks, procedural road/grass/bark/window textures, sky environment map, ACES tone mapping, soft
shadows, 2× supersampling). Geometry extents, placement, cameras, lighting directions, seed and marker are identical; `fid-NNN` pairs with
`core-NNN`. The 36 renders were deterministic and committed before any model saw them. Each model ran once per method with its frozen lock;
results are compared with the same model's standard-render runs on the paired case.

{fidelity_table()[0]}

Labels changed between standard and high-fidelity renders — {fidelity_table()[1]}.

- **Higher fidelity did not fix nemotron3's oblique-truck failure** (baseline 0/6 → 0/6; structure prompt 1/6 → 2/6, within the run-to-run variation of §6.4).
  Its stated evidence shifted from "2D graphic" to e.g. "a 2D computer-generated image with simplified shapes", which suggests it treats
  *the image being synthetic* as evidence of a depiction. A still-synthetic render cannot separate that from the viewpoint effect; real photographs would be needed.
  From the ego-lane view, nemotron3 gave identical labels on every case at both fidelity levels.
- gemma4's one oblique-truck abstention became correct (5/6 → 6/6, both methods); qwen3.6's oblique plain-board accuracy rose from 1/6 to 3/6 (both methods).
- No model produced a false `physical_vehicle` label on any high-fidelity depiction under the baseline; qwen3.6's structure-prompt `depiction_as_vehicle` errors from the ego-lane view recurred (3/6, same as standard).

### 6.9 Local hardware: memory, latency and acceleration

{resources_section()}

**Why local matters for this project.** (1) *Reproducibility*: every result is tied to a pinned model digest that cannot change underneath us,
which is what makes the frozen-protocol comparisons in this report meaningful; hosted models can change silently between runs.
(2) *Privacy and offline operation*: images never leave the LAN, and the backend runs with outbound connections blocked (§6.5).
(3) *Cost-free iteration*: over 1,500 model requests were run for this report at no marginal cost. (4) *Many models at once*: the GX10's unified
memory holds three ~30 B-parameter vision models simultaneously, so "same image, three models" is a 1–2 s interactive comparison
(UI: **Compare all local models**) instead of a reload per model.

## 7. What this does and does not show

- It shows a working, reproducible local testbed: deterministic scenes, exact model inputs with hashes, ground truth kept out of the request, frozen prompts, a full pass/fail ledger, and recomputable metrics.
- On this synthetic corpus, `{lock['model_id']}` (a) separated truck from truck-picture from the ego-lane view, (b) labelled the physical truck `flat_depiction` from the oblique view, and (c) never used `other` for a plain board.
- The exploratory comparison shows these are not properties of the images alone: on the same inputs, gemma4 got 17/18 held-out (both methods) and qwen3.6 13/18. Which local model is chosen matters more here than the prompt sentence.
- More realistic (but still synthetic) rendering did not change nemotron3's oblique-truck answers (§6.8). Whether it would behave differently on real photographs is untested.
- It does **not** show a vulnerability in any vehicle, a robustness improvement from the structure prompt, a texture or shape bias in the usual sense, or anything about real-world safety. Samples are small (18 per split per method); no statistical significance is claimed.

## 8. Limitations

1. **Synthetic-image confound (partly addressed).** Two other models recognised the same trucks, and a higher-fidelity re-render left nemotron3's oblique failures unchanged, so low-poly shading is not the explanation. nemotron3's evidence ("computer-generated image") suggests it may treat synthetic imagery itself as a depiction cue; only real photographs can test that.
2. Splits differ by viewpoint, so development and held-out are different difficulty regimes; numbers must not be pooled.
3. Development results after revision 2 are post-tuning.
4. The marker spans the target slot; in the oblique view it is wide and includes trees and the empty space where the truck would stand. It is identical across target types, but how much of it the target fills differs by target, which a model could in principle use.
5. One model, one quantization, one runtime; structured-output decoding may affect label choice.
6. Observability is `unchecked`; no independent human review of image ambiguity.
7. Test–retest showed {len(a) - agree}/{len(a)} label changes at temperature 0; one-case differences between methods are not meaningful.
8. Synthetic scenes share assets; this is not an independent real-world benchmark.

## 9. Failures and regressions — headline model `nemotron3:33b` (full ledgers in each run bundle; other models in §6.7)

- `vehicle_as_depiction`: held-out physical trucks — baseline {hm[BASELINE]['outcomes']['vehicle_as_depiction']}/6, structure prompt {hm[VARIANT]['outcomes']['vehicle_as_depiction']}/6; discovery near and occluded trucks 4/4 per method.
- `wrong_other_label`: plain-board controls labelled `flat_depiction` in every frozen-prompt run (6/6 development, 6/6 held-out, 8/8 discovery, per method).
- `depiction_as_vehicle`: only in development revision 1 (baseline, 1 case, back-lit striped depiction); none after the freeze.
- Variant regressions (baseline correct, structure prompt wrong): 0 in every run.
- Operational failures: {n_ops}.

## 10. Acceptance status

| Criterion | Status |
|---|---|
| Verified GX10 image inference | ✓ smoke test + {n_live} live results |
| Reproducible 3D scene and exact capture | ✓ deterministic double render, hashes, scene specs |
| Frozen balanced test manifest | ✓ 36 core (18/18), manifest hash in state |
| Baseline + prompt-variant results | ✓ development ×2, held-out ×2, discovery |
| Visible ground-truth separation | ✓ truth only in observer/evidence panels; scene checks 60/60; request contains only prompt + image |
| Full pass/fail/error ledger | ✓ run bundles with raw outputs |
| Working comparison UI + export | ✓ UI smoke + screenshots; zip/CSV/JSONL/MD export |
| Measured latency | ✓ §6.6 |
| Fresh local-only smoke test | ✓ §6.5 (application-level guard) |
| Documented startup | ✓ `scripts/start.sh`, README |

Not done / out of scope: edge-image and second-view variants (optional), independent observability review, real images. Exploratory extensions done: a second and third model (§6.7) and a high-fidelity re-render (§6.8).

## 11. Next steps

1. Test with real photographs (a toy truck vs a printed photo of it on a board, shot from both viewpoints) to separate "synthetic image" from "depiction" for nemotron3.
2. Pre-register a cross-model held-out comparison with a fresh held-out set (the §6.7 runs reused a split already seen), and repeat each model ≥5 times.
3. Independent observability review of held-out images before inference.
4. ≥5 repeat runs per configuration to estimate label stability.
5. A new pre-registered revision for the control label (e.g. an explicit "blank sign" description) — only with a fresh held-out set.
"""
(ROOT / "artifacts/report.md").write_text(out)
print(f"wrote artifacts/report.md ({len(out)} chars); runs: {', '.join(runs)}")
