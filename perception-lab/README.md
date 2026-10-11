# Perception Lab

A local, interactive automotive-perception testbed for the ASUS Ascent GX10. Manipulate a small procedural 3D road scene,
capture the exact sensor-camera image, run a local vision-language model on the GX10, and compare its answer with simulator
ground truth — with full provenance (image hashes, exact prompts, model digest, raw responses, latency).

> *Change appearance without changing physical geometry, and inspect when an automotive vision model changes its answer.*

Educational/research prototype. Not a driving controller, ADAS, collision predictor, or evidence about production vehicles.

## Start (one command, on the laptop)

```bash
./scripts/start.sh          # build UI → sync to GX10 → venv deps → start backend (guard on) → SSH tunnel
open http://127.0.0.1:8765
```

Prerequisites: SSH key access to the GX10 (`PL_GX10`, default `asus@100.104.225.90` over Tailscale); Ollama running on the
GX10 with `nemotron3:33b` pulled (digest pinned in `experiments/model-lock.json`); Node ≥ 20 and Google Chrome on the laptop;
Python ≥ 3.11 on the GX10. No cloud inference is used at any point; after setup no WAN access is needed (LAN/Tailscale between
laptop and GX10 is).

Where things run: **3D rendering in the laptop browser (WebGL)**; **inference, job queue, SQLite and artifacts on the GX10**.
The backend binds to 127.0.0.1 on the GX10 only; the laptop reaches it through the SSH tunnel.

Backend control on the GX10: `scripts/gx10_backend.sh start|stop|restart|status` (outbound guard on by default,
`PL_OFFLINE_GUARD=0` to disable). Development UI with hot reload: `cd frontend && npx vite` (proxies `/api` to the tunnel).

## Headline result (held-out, frozen prompts, `nemotron3:33b`)

| Method | Correct / attempted | Depictions → `flat_depiction` | Physical trucks → `physical_vehicle` | Plain boards → `other` |
|---|---|---|---|---|
| RGB baseline | 6/18 | 6/6 | 0/6 | 0/6 |
| RGB + structure prompt | 7/18 | 6/6 | 1/6 | 0/6 |

35/36 held-out answers were `flat_depiction` (a constant guesser also scores 6/18). From the oblique camera the model calls our
low-poly physical truck "a 2D graphic"; it never labels a plain board `other`. The structure prompt's +1 is within run-to-run
variation (repeat run: 35/36 labels identical).

Exploratory cross-model check on the same images and frozen prompts (pre-registered lock files): held-out correct/attempted
`gemma4:26b` **17/18** (both methods), `qwen3.6:35b` **13/18** (both). The oblique-truck failure is specific to nemotron3; the
plain-board → `flat_depiction` failure also appears for qwen3.6; the structure sentence caused 5 regressions for qwen3.6 on development.
Full analysis, limitations and every number: **[artifacts/report.md](artifacts/report.md)**.

## Reproduce

```bash
cd frontend && npx vite &                                   # capture harness at :5173
node scripts/render_cases.mjs --check-determinism --upload --out data/renders   # 60 cases, double-render check
node scripts/render_cases.mjs --scene-checks                # scene geometry vs metadata, helper layers
python3 scripts/run_batch.py --split development --run-key dev-vN
python3 scripts/run_batch.py --split heldout --run-key heldout-vN   # refused (HTTP 423) unless prompts + lock are frozen
curl -s 'localhost:8765/api/export/<run_id>?fmt=zip' -o data/results/<run_id>.zip
python3 scripts/verify_artifacts.py                         # recompute hashes + metrics from committed bundles
python3 scripts/export_report.py                            # regenerate artifacts/report.md
node scripts/ui_smoke.mjs --app http://127.0.0.1:8765       # UI evidence + screenshots
```

Backend tests (run on the GX10 or anywhere with the venv): `.venv/bin/python -m pytest -q tests` — parser, evaluator
denominators, paired outcomes, timeouts, metadata stripping, identical paired inputs, restart/cache behaviour, held-out seal,
upload validation, offline guard.

## Layout

```text
backend/app/   main.py (API) · jobs.py (single GPU worker, restart-safe) · providers.py (Ollama adapter + strict parser)
               evaluate.py (outcomes, metrics, paired) · export.py · png.py (validation, metadata strip) · netguard.py
frontend/src/  scene/{spec,build,capture,manifest}.ts (deterministic procedural scene) · App.tsx (3-column UI) · api.ts
experiments/   manifest.json (60 cases) · prompts.json (frozen rev 2) · model-lock.json (frozen) · assets/ (depiction textures)
               render-log-all.json (client hashes, marker boxes, determinism)
data/          renders/ (all 60 model inputs) · results/ (exported run bundles: images, raw outputs, metrics, report)
artifacts/     report.md · demo-script.md · environment.md · source-register.jsonl · verification.json · scene-checks.json
               offline-guard-probe.json · screenshots/
state/         run.json · checkpoint.md · decisions.md
```

## API

`GET /api/health` · `POST /api/cases` (multipart `meta` + PNG; server strips metadata, hashes, assigns id) ·
`POST /api/runs` (case ids × method ids, optional `run_key` for idempotent restarts) · `GET /api/runs/{id}` ·
`GET /api/cases/{id}` · `GET /api/metrics?run_id=` · `GET /api/export/{run_id}?fmt=zip|csv|jsonl|md` ·
`POST /api/warmup` (unrecorded) · `GET /api/netguard/probe`.

Ground truth is stored with the case and used only by the evaluator; the provider adapter receives image bytes and the prompt string only.
