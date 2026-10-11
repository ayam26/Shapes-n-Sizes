#!/usr/bin/env python3
"""Independent artifact verifier. Works only from files committed in the repo:
data/results/*.zip (exported run bundles), data/renders/*.png, experiments/*, state/run.json.

Recomputes image hashes and every reported metric from saved records and checks the
provenance rules of the design. Writes artifacts/verification.json; exit code 1 on any FAIL.

    python3 scripts/verify_artifacts.py
"""
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.evaluate import metrics  # noqa: E402  (pure functions, no server needed)
from app.png import strip_and_validate  # noqa: E402

checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})


def sha(b):
    return hashlib.sha256(b).hexdigest()


state = json.loads((ROOT / "state/run.json").read_text())
manifest_bytes = (ROOT / "experiments/manifest.json").read_bytes()
manifest = json.loads(manifest_bytes)
prompts = json.loads((ROOT / "experiments/prompts.json").read_text())
lock = json.loads((ROOT / "experiments/model-lock.json").read_text())

# --- frozen configuration and manifest ---
check("manifest hash matches state", sha(manifest_bytes) == state["manifest_hash"], sha(manifest_bytes)[:16])
check("prompts frozen", prompts.get("frozen") is True, prompts.get("frozen_at"))
check("model lock frozen", lock.get("frozen") is True, lock.get("frozen_at"))
check("prompts file hash matches state", sha((ROOT / "experiments/prompts.json").read_bytes()) == state.get("prompts_sha256"))
check("model lock hash matches state", sha((ROOT / "experiments/model-lock.json").read_bytes()) == state.get("model_lock_sha256"))
core = [c for c in manifest["cases"] if c["split"] in ("development", "heldout")]
bal = {}
for c in core:
    k = (c["split"], c["spec"]["target"], c["spec"]["appearance"], c["spec"]["lighting"])
    bal[k] = bal.get(k, 0) + 1
check("core manifest is 36 balanced cases", len(core) == 36 and set(bal.values()) == {1} and len(bal) == 36)
check("discovery corpus <= 24", sum(c["split"] == "discovery" for c in manifest["cases"]) <= 24)

# --- renders, marker, textures ---
log = json.loads((ROOT / "experiments/render-log-all.json").read_text())
by_id = {e["case_id"]: e for e in log["cases"]}
spec_of = {c["case_id"]: c["spec"] for c in manifest["cases"]}
check("all manifest cases rendered", set(by_id) == set(spec_of))
check("all renders deterministic (double render)", all(e.get("deterministic") for e in log["cases"]))
mark = {}
for cid, s in spec_of.items():
    key = (s["camera"]["azimuth_deg"], s["camera"]["distance_m"], s["camera"]["height_m"], s["camera"]["fov_deg"])
    mark.setdefault(key, set()).add(json.dumps(by_id[cid]["marker"], sort_keys=True))
check("marker identical across target types for each camera", all(len(v) == 1 for v in mark.values()),
      f"{len(mark)} camera configurations")
bad_png = []
for cid in spec_of:
    clean, _, _ = strip_and_validate((ROOT / "data/renders" / f"{cid}.png").read_bytes(), 2048)
    if sha(clean) != by_id[cid]["server"]["image_sha256"]:
        bad_png.append(cid)
check("local renders hash-match server-ingested images", not bad_png, ",".join(bad_png))
tex_bad = []
for e in log["cases"]:
    if e.get("texture_sha256"):
        asset = ROOT / f"experiments/assets/depiction-{spec_of[e['case_id']]['appearance']}.png"
        if sha(asset.read_bytes()) != e["texture_sha256"]:
            tex_bad.append(e["case_id"])
check("depiction texture assets match recorded hashes", not tex_bad, ",".join(tex_bad))

# --- run bundles ---
runs_summary = []
for z in sorted((ROOT / "data/results").glob("run-*.zip")):
    with zipfile.ZipFile(z) as f:
        run = json.loads(f.read("run.json"))
        recs = [json.loads(line) for line in f.read("results.jsonl").decode().splitlines() if line.strip()]
        saved_metrics = json.loads(f.read("metrics.json"))
        cfg = json.loads(run["model_config"])
        rid = run["run_id"]
        img_bad = []
        for cid in {r["case_id"] for r in recs}:
            case = json.loads(f.read(f"cases/{cid}/case.json"))
            h = sha(f.read(f"cases/{cid}/input.png"))
            if h != case["image_sha256"] or any(r["image_sha256"] != h for r in recs if r["case_id"] == cid):
                img_bad.append(cid)
        check(f"{rid}: image bytes re-hash to recorded sha256 for every result", not img_bad, ",".join(img_bad))
        check(f"{rid}: metrics recomputed from records equal exported metrics.json", metrics(recs) == saved_metrics)
        recount_ok = True
        for split, by in saved_metrics["by_split_method"].items():
            for meth, summ in by.items():
                rs = [r for r in recs if r["split"] == split and r["method_id"] == meth]
                n_correct = sum(1 for r in rs if r["status"] == "ok" and r["label"] == r["truth"])
                n_flat_as_vehicle = sum(1 for r in rs if r["truth"] == "flat_depiction" and r["label"] == "physical_vehicle")
                recount_ok &= (summ["correct_over_attempted"]["n"] == n_correct and summ["attempted"] == len(rs)
                               and summ["false_vehicle_on_depictions"]["n"] == n_flat_as_vehicle)
        check(f"{rid}: independent recount (label == truth) matches reported counts", recount_ok)
        pm = saved_metrics.get("paired_by_split", {})
        check(f"{rid}: paired methods received byte-identical inputs", all(p["hash_mismatch"] == 0 for p in pm.values()))
        live = run["provider_mode"] == "live" and all(r["origin"] == "live" for r in recs)
        check(f"{rid}: all results live local inference (no fixture/cache)", live)
        check(f"{rid}: every result finished", all(r["status"] not in ("queued", "running") for r in recs))
        if any(r["split"] == "heldout" for r in recs):
            after = all(r["queued_at"] > prompts["frozen_at"].replace("Z", "") for r in recs)
            check(f"{rid}: held-out run used frozen prompts+lock and started after freeze",
                  cfg["prompts_frozen"] and cfg["lock_frozen"] and after)
            same = all(cfg["methods"][m]["prompt"] == prompts["methods"][m]["prompt"] for m in cfg["methods"])
            check(f"{rid}: held-out prompts equal the frozen prompts file", same)
        runs_summary.append({"run_id": rid, "run_key": run["run_key"], "label": run["label"], "records": len(recs),
                             "offline_guard": bool(run["offline_guard"]), "provider_mode": run["provider_mode"]})

out = {"verified_at": datetime.now(timezone.utc).isoformat(),
       "passed": sum(c["status"] == "PASS" for c in checks), "failed": sum(c["status"] == "FAIL" for c in checks),
       "checks": checks, "runs": runs_summary}
(ROOT / "artifacts/verification.json").write_text(json.dumps(out, indent=2) + "\n")
for c in checks:
    print(f"{c['status']}  {c['check']}" + (f"  [{c['detail']}]" if c["detail"] else ""))
print(f"\n{out['passed']} passed, {out['failed']} failed")
sys.exit(1 if out["failed"] else 0)
