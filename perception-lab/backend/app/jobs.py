"""Single-worker GPU queue. Result rows are created 'queued' at enqueue time, so a
restart only re-runs rows that never finished and never duplicates completed ones."""
import hashlib
import json
import threading
import time
import traceback

from . import config, db
from .providers import make_provider, parse_output

FINISHED = ("ok", "invalid_output", "timeout", "provider_error", "cancelled")


def lock_config_hash(lock: dict, output_schema: dict) -> str:
    keys = ("provider", "endpoint", "model_id", "model_digest", "quantization", "think", "options", "runtime_version")
    return config.canonical_hash({**{k: lock.get(k) for k in keys}, "output_schema": output_schema,
                                  "preprocessing": "png->base64, Ollama model-internal resize"})


def snapshot_config(method_ids: list[str]) -> dict:
    """Freeze the exact model config and prompts a run will use."""
    lock = config.load_json("model-lock.json")
    prompts = config.load_json("prompts.json")
    methods = {}
    for m in method_ids:
        if m not in prompts["methods"]:
            raise KeyError(m)
        text = prompts["methods"][m]["prompt"]
        methods[m] = {"prompt": text, "prompt_hash": hashlib.sha256(
            (text + json.dumps(prompts["output_schema"], sort_keys=True)).encode()).hexdigest()}
    return {"lock": lock, "output_schema": prompts["output_schema"], "methods": methods,
            "lock_hash": lock_config_hash(lock, prompts["output_schema"]),
            "prompts_frozen": prompts.get("frozen", False), "lock_frozen": lock.get("frozen", False)}


def reconcile_after_restart() -> int:
    n = len(db.rows("SELECT result_id FROM results WHERE status='running'"))
    db.execute("UPDATE results SET status='queued', started_at=NULL WHERE status='running'")
    return n


def _finish_run_if_done(run_id: str) -> None:
    left = db.one("SELECT COUNT(*) n FROM results WHERE run_id=? AND status IN ('queued','running')", (run_id,))
    if left["n"] == 0:
        db.execute("UPDATE runs SET status='completed', finished_at=? WHERE run_id=? AND status!='completed'",
                   (db.now(), run_id))


class Worker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="gpu-worker")
        self.stop = threading.Event()
        self.current = None
        self.last_error = None

    def run(self):
        while not self.stop.is_set():
            row = db.one("SELECT r.* FROM results r JOIN runs u ON u.run_id=r.run_id "
                         "WHERE r.status='queued' ORDER BY u.created_at, r.seq LIMIT 1")
            if not row:
                time.sleep(0.3)
                continue
            try:
                self.current = row["result_id"]
                self.process(row)
            except Exception as e:  # record, never crash the worker
                self.last_error = repr(e)
                db.execute("UPDATE results SET status='provider_error', parse_error=?, finished_at=? WHERE result_id=?",
                           (traceback.format_exc()[-800:], db.now(), row["result_id"]))
            finally:
                self.current = None
                _finish_run_if_done(row["run_id"])

    def process(self, row: dict) -> None:
        run = db.one("SELECT * FROM runs WHERE run_id=?", (row["run_id"],))
        cfg = json.loads(run["model_config"])
        method = cfg["methods"][row["method_id"]]
        case = db.one("SELECT * FROM cases WHERE case_id=?", (row["case_id"],))
        db.execute("UPDATE results SET status='running', started_at=? WHERE result_id=?", (db.now(), row["result_id"]))
        db.execute("UPDATE runs SET status='running' WHERE run_id=? AND status='queued'", (run["run_id"],))
        image = (config.DATA / case["image_path"]).read_bytes()
        if hashlib.sha256(image).hexdigest() != case["image_sha256"]:
            raise RuntimeError("stored image hash mismatch")
        cache_key = config.canonical_hash({"image": case["image_sha256"], "prompt": method["prompt_hash"],
                                           "config": cfg["lock_hash"]})
        lock = cfg["lock"]
        common = (method["prompt_hash"], case["image_sha256"], lock["model_id"], lock.get("model_digest"),
                  lock.get("quantization"), lock.get("runtime_version"), json.dumps(lock["options"]), cache_key)
        hit = run["use_cache"] and db.one(
            "SELECT r.* FROM cache c JOIN results r ON r.result_id=c.result_id WHERE c.cache_key=?", (cache_key,))
        if hit:
            db.execute("UPDATE results SET prompt_hash=?, image_sha256=?, model_id=?, model_digest=?, quantization=?, "
                       "runtime_version=?, options=?, cache_key=?, status=?, raw_output=?, parsed=?, label=?, "
                       "parse_error=?, elapsed_ms=?, origin='cached', provider_meta=?, finished_at=? WHERE result_id=?",
                       (*common, hit["status"], hit["raw_output"], hit["parsed"], hit["label"], hit["parse_error"],
                        hit["elapsed_ms"], json.dumps({"cached_from": hit["result_id"]}), db.now(), row["result_id"]))
            return
        provider = make_provider(lock, cfg["output_schema"])
        res = provider.infer(image, method["prompt"])
        parsed = label = err = None
        if res["status"] == "completed":
            status, parsed, err = parse_output(res["raw"])
            label = parsed["label"] if parsed else None
        else:
            status, err = res["status"], res["meta"].get("error")
        db.execute("UPDATE results SET prompt_hash=?, image_sha256=?, model_id=?, model_digest=?, quantization=?, "
                   "runtime_version=?, options=?, cache_key=?, status=?, raw_output=?, parsed=?, label=?, parse_error=?, "
                   "elapsed_ms=?, origin=?, provider_meta=?, finished_at=? WHERE result_id=?",
                   (*common, status, res["raw"], json.dumps(parsed) if parsed else None, label, err,
                    res["elapsed_ms"], provider.mode, json.dumps(res["meta"]), db.now(), row["result_id"]))
        if provider.mode == "live" and status in ("ok", "invalid_output"):
            db.execute("INSERT OR IGNORE INTO cache(cache_key, result_id, created_at) VALUES (?,?,?)",
                       (cache_key, row["result_id"], db.now()))
