"""Perception Lab API. Run on the GX10: uvicorn app.main:app --host 127.0.0.1 --port 8765"""
import hashlib
import json
import re
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, db, export, netguard
from .evaluate import metrics as compute_metrics, outcome
from .jobs import Worker, lock_config_hash, reconcile_after_restart, snapshot_config
from .png import PNGError, solid_png, strip_and_validate
from .providers import make_provider

CASE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
LOCK_FILE = re.compile(r"^model-lock(-[a-z0-9.]{1,40})?\.json$")
worker = Worker()


@asynccontextmanager
async def lifespan(app):
    if config.OFFLINE_GUARD:
        netguard.enable()
    db.init()
    reconcile_after_restart()
    if not worker.is_alive():
        worker.start()
    yield
    worker.stop.set()


app = FastAPI(title="Perception Lab", lifespan=lifespan)


# ---------- health ----------
@app.get("/api/health")
def health():
    lock = config.load_json("model-lock.json")
    prompts = config.load_json("prompts.json")
    provider = make_provider(lock, prompts["output_schema"])
    info = {"app": "ok", "fixture_mode": provider.mode == "fixture", "provider_mode": provider.mode,
            "configured_model": lock["model_id"], "offline_guard": netguard.enabled(),
            "worker_alive": worker.is_alive(), "worker_last_error": worker.last_error,
            "queue_depth": db.one("SELECT COUNT(*) n FROM results WHERE status IN ('queued','running')")["n"],
            "prompts_frozen": prompts.get("frozen", False), "lock_frozen": lock.get("frozen", False),
            "model_config_hash": lock_config_hash(lock, prompts["output_schema"]),
            "rendering": "browser WebGL on the client (laptop)", "inference": "Ollama on this host (GX10)"}
    try:
        tags = {m["name"]: m["digest"] for m in provider.tags()}
        info["provider_status"] = "reachable" if provider.mode == "live" else "fixture"
        info["actual_model_digest"] = tags.get(lock["model_id"])
        info["digest_matches_lock"] = tags.get(lock["model_id"]) == lock.get("model_digest")
    except Exception as e:
        info["provider_status"] = f"unreachable: {type(e).__name__}"
    return info


@app.get("/api/netguard/probe")
def netguard_probe():
    """Attempt one outbound TCP connection from this backend process to show whether the
    application-level offline guard blocks it. Loopback/LAN serving is unaffected."""
    import socket
    s = socket.socket()
    s.settimeout(2)
    try:
        s.connect(("1.1.1.1", 443))
        result = "connected (outbound allowed)"
    except ConnectionRefusedError as e:
        result = f"blocked: {e}" if "PL_OFFLINE_GUARD" in str(e) else f"refused by network: {e}"
    except OSError as e:
        result = f"failed: {type(e).__name__}: {e}"
    finally:
        s.close()
    return {"offline_guard": netguard.enabled(), "target": "1.1.1.1:443", "result": result}


# ---------- cases ----------
class CaseMeta(BaseModel):
    case_id: str | None = None
    split: str
    family: str = Field(max_length=64)
    truth: str
    scene_seed: int = 0
    scene_spec: dict
    observability: str = "unchecked"
    source_kind: str = "procedural"
    license: str = "self-created"
    renderer_version: str = Field(max_length=128)


@app.post("/api/cases")
async def create_case(meta: str = Form(...), image: UploadFile = File(...)):
    try:
        m = CaseMeta(**json.loads(meta))
    except Exception as e:
        raise HTTPException(422, f"invalid metadata: {e}")
    if m.split not in config.SPLITS or m.truth not in config.TRUTH_LABELS or m.observability not in config.OBSERVABILITY:
        raise HTTPException(422, "split/truth/observability outside allowed values")
    data = await image.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image exceeds 8 MB")
    try:
        clean, w, h = strip_and_validate(data, config.MAX_IMAGE_DIM)
    except PNGError as e:
        raise HTTPException(422, f"malformed image: {e}")
    sha = hashlib.sha256(clean).hexdigest()
    if m.case_id:
        if not CASE_ID.match(m.case_id):
            raise HTTPException(422, "case_id must match [a-z0-9-]{1,64}")
        case_id = m.case_id
    else:
        case_id = f"ix-{sha[:8]}-{secrets.token_hex(2)}"
    existing = db.one("SELECT * FROM cases WHERE case_id=?", (case_id,))
    if existing:
        if existing["image_sha256"] == sha:
            return {"case_id": case_id, "image_sha256": sha, "existing": True}
        raise HTTPException(409, f"{case_id} already exists with a different image; frozen cases are immutable")
    cdir = config.CASES_DIR / case_id
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "input.png").write_bytes(clean)
    (cdir / "scene.json").write_text(json.dumps(m.scene_spec, indent=2, sort_keys=True))
    conditions = m.scene_spec.get("conditions", {})
    db.execute("INSERT INTO cases(case_id, schema_version, split, family, scene_seed, scene_spec, conditions, image_path,"
               " image_sha256, width, height, truth, observability, source_kind, license, renderer_version, created_at)"
               " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (case_id, 1, m.split, m.family, m.scene_seed, json.dumps(m.scene_spec, sort_keys=True),
                json.dumps(conditions, sort_keys=True), f"cases/{case_id}/input.png", sha, w, h, m.truth,
                m.observability, m.source_kind, m.license, m.renderer_version, db.now()))
    (cdir / "case.json").write_text(json.dumps(db.one("SELECT * FROM cases WHERE case_id=?", (case_id,)), indent=2))
    return {"case_id": case_id, "image_sha256": sha, "width": w, "height": h, "existing": False}


@app.get("/api/cases")
def list_cases(split: str | None = None):
    sql = "SELECT case_id, split, family, truth, observability, conditions, image_sha256, created_at FROM cases"
    return db.rows(sql + (" WHERE split=?" if split else "") + " ORDER BY case_id", (split,) if split else ())


@app.get("/api/cases/{case_id}")
def get_case(case_id: str):
    case = db.one("SELECT * FROM cases WHERE case_id=?", (case_id,))
    if not case:
        raise HTTPException(404, "no such case")
    case["scene_spec"] = json.loads(case["scene_spec"])
    case["conditions"] = json.loads(case["conditions"] or "{}")
    res = db.rows("SELECT r.*, u.label AS run_label, u.provider_mode FROM results r JOIN runs u ON u.run_id=r.run_id "
                  "WHERE r.case_id=? ORDER BY r.result_id DESC", (case_id,))
    for r in res:
        r["outcome"] = outcome(case["truth"], r["status"], r["label"])
    case["results"] = res
    return case


@app.get("/api/cases/{case_id}/image")
def case_image(case_id: str):
    case = db.one("SELECT image_path FROM cases WHERE case_id=?", (case_id,))
    if not case:
        raise HTTPException(404, "no such case")
    return FileResponse(config.DATA / case["image_path"], media_type="image/png")


# ---------- runs ----------
class RunRequest(BaseModel):
    case_ids: list[str] = Field(min_length=1, max_length=500)
    method_ids: list[str] = Field(min_length=1, max_length=8)
    run_key: str | None = Field(default=None, max_length=64)
    label: str | None = Field(default=None, max_length=200)
    use_cache: bool = False
    model_lock: str = "model-lock.json"  # alternative frozen lock files are for exploratory cross-model runs


@app.post("/api/runs")
def create_run(req: RunRequest):
    if req.run_key:
        prev = db.one("SELECT run_id FROM runs WHERE run_key=?", (req.run_key,))
        if prev:  # restart-safe: same key never duplicates completed work
            return {"run_id": prev["run_id"], "existing": True}
    ids = list(dict.fromkeys(req.case_ids))
    cases = db.rows(f"SELECT case_id, split, render_status FROM cases WHERE case_id IN ({','.join('?' * len(ids))})", ids)
    if len(cases) != len(ids):
        raise HTTPException(404, "unknown case id(s)")
    if any(c["render_status"] != "valid" for c in cases):
        raise HTTPException(422, "render_invalid cases cannot be run")
    if not LOCK_FILE.match(req.model_lock) or not (config.EXPERIMENTS / req.model_lock).is_file():
        raise HTTPException(422, "model_lock must name an existing experiments/model-lock*.json file")
    try:
        cfg = snapshot_config(req.method_ids, req.model_lock)
    except KeyError as e:
        raise HTTPException(422, f"unknown method {e}")
    if any(c["split"] == "heldout" for c in cases) and not (cfg["prompts_frozen"] and cfg["lock_frozen"]):
        raise HTTPException(423, "held-out split is sealed until prompts.json and model-lock.json are frozen")
    provider = make_provider(cfg["lock"], cfg["output_schema"])
    run_id = f"run-{db.now()[:19].replace(':', '').replace('-', '')}-{secrets.token_hex(2)}"
    db.execute("INSERT INTO runs(run_id, run_key, label, created_at, model_config, model_config_hash, method_ids, status,"
               " provider_mode, use_cache, offline_guard) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
               (run_id, req.run_key, req.label, db.now(), json.dumps(cfg), cfg["lock_hash"], json.dumps(req.method_ids),
                "queued", provider.mode, int(req.use_cache), int(netguard.enabled())))
    seq = 0
    for cid in ids:
        for mid in req.method_ids:
            db.execute("INSERT INTO results(run_id, seq, case_id, method_id, status, queued_at) VALUES (?,?,?,?,?,?)",
                       (run_id, seq, cid, mid, "queued", db.now()))
            seq += 1
    return {"run_id": run_id, "existing": False, "jobs": seq}


@app.get("/api/runs")
def list_runs():
    runs = db.rows("SELECT run_id, run_key, label, created_at, status, provider_mode, offline_guard, use_cache, "
                   "finished_at FROM runs ORDER BY created_at DESC")
    for r in runs:
        r["counts"] = {x["status"]: x["n"] for x in db.rows(
            "SELECT status, COUNT(*) n FROM results WHERE run_id=? GROUP BY status", (r["run_id"],))}
    return runs


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = db.one("SELECT * FROM runs WHERE run_id=?", (run_id,))
    if not run:
        raise HTTPException(404, "no such run")
    run["model_config"] = json.loads(run["model_config"])
    recs = export.records(run_id)
    counts = {}
    for r in recs:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    run["counts"] = counts
    run["total"] = len(recs)
    run["error_summary"] = [{"case_id": r["case_id"], "method_id": r["method_id"], "status": r["status"],
                             "error": r["parse_error"]} for r in recs
                            if r["status"] in ("invalid_output", "timeout", "provider_error")]
    run["results"] = [{k: r[k] for k in ("result_id", "case_id", "method_id", "status", "label", "outcome", "elapsed_ms",
                                         "origin", "split", "truth", "family", "observability", "image_sha256",
                                         "conditions")} for r in recs]
    run["links"] = {"metrics": f"/api/metrics?run_id={run_id}", "export": f"/api/export/{run_id}"}
    return run


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    db.execute("UPDATE results SET status='cancelled', finished_at=? WHERE run_id=? AND status='queued'", (db.now(), run_id))
    db.execute("UPDATE runs SET status='cancelled' WHERE run_id=? AND status IN ('queued','running')", (run_id,))
    return {"run_id": run_id, "cancelled": True}


@app.get("/api/results/{result_id}")
def get_result(result_id: int):
    r = db.one("SELECT * FROM results WHERE result_id=?", (result_id,))
    if not r:
        raise HTTPException(404, "no such result")
    return r


@app.get("/api/metrics")
def get_metrics(run_id: str = Query(...)):
    if not db.one("SELECT 1 FROM runs WHERE run_id=?", (run_id,)):
        raise HTTPException(404, "no such run")
    return compute_metrics(export.records(run_id))


WARMUP_PNG = solid_png(64, 64, (128, 128, 128))


@app.post("/api/warmup")
def warmup(model_lock: str = "model-lock.json"):
    """One unrecorded inference so cold-load time is kept out of measured latency."""
    if not LOCK_FILE.match(model_lock) or not (config.EXPERIMENTS / model_lock).is_file():
        raise HTTPException(422, "model_lock must name an existing experiments/model-lock*.json file")
    lock = config.load_json(model_lock)
    prompts = config.load_json("prompts.json")
    provider = make_provider(lock, prompts["output_schema"])
    res = provider.infer(WARMUP_PNG, "Describe this image. Return JSON with label and visible_evidence.")
    load_ms = (res["meta"].get("load_duration") or 0) / 1e6
    db.execute("INSERT INTO warmups(created_at, model_id, status, elapsed_ms, load_ms) VALUES (?,?,?,?,?)",
               (db.now(), lock["model_id"], res["status"], res["elapsed_ms"], load_ms))
    return {"status": res["status"], "elapsed_ms": res["elapsed_ms"], "load_ms": load_ms, "mode": provider.mode}


@app.get("/api/export/{run_id}")
def export_run(run_id: str, fmt: str = "zip"):
    run = db.one("SELECT * FROM runs WHERE run_id=?", (run_id,))
    if not run:
        raise HTTPException(404, "no such run")
    recs = export.records(run_id)
    if fmt == "csv":
        return PlainTextResponse(export.to_csv(recs), media_type="text/csv")
    if fmt == "jsonl":
        return PlainTextResponse("".join(json.dumps(r) + "\n" for r in recs), media_type="application/x-ndjson")
    if fmt == "md":
        return PlainTextResponse(export.report_md(run, recs, compute_metrics(recs)), media_type="text/markdown")
    if fmt == "zip":
        return Response(export.bundle(run_id), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{run_id}.zip"'})
    raise HTTPException(422, "fmt must be zip|csv|jsonl|md")


if config.FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIST, html=True), name="frontend")
