#!/usr/bin/env python3
"""Enqueue a split against both methods and wait for completion. Restart-safe: the run_key makes
re-invocation resume the same run instead of creating duplicate results.

    python3 scripts/run_batch.py --split development --run-key dev-v1
"""
import argparse
import json
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--api", default="http://127.0.0.1:8765")
ap.add_argument("--split", default=None)
ap.add_argument("--case-ids", default=None, help="comma-separated case ids (alternative to --split)")
ap.add_argument("--run-key", required=True)
ap.add_argument("--label", default=None)
ap.add_argument("--methods", default="rgb_baseline,rgb_structure_prompt")
ap.add_argument("--no-warmup", action="store_true")
ap.add_argument("--model-lock", default="model-lock.json", help="experiments/model-lock*.json (non-default = exploratory)")
a = ap.parse_args()


def call(path, body=None):
    req = urllib.request.Request(a.api + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{path} -> HTTP {e.code}: {e.read().decode()[:400]}")


health = call("/api/health")
print("health:", {k: health.get(k) for k in ("provider_mode", "configured_model", "digest_matches_lock",
                                             "prompts_frozen", "lock_frozen", "offline_guard")})
cases = a.case_ids.split(",") if a.case_ids else [c["case_id"] for c in call(f"/api/cases?split={a.split}")]
print(f"{len(cases)} cases ({a.split or 'explicit list'})")
if not a.no_warmup:
    print("warmup (unrecorded):", call(f"/api/warmup?model_lock={a.model_lock}", {}))
run = call("/api/runs", {"case_ids": cases, "method_ids": a.methods.split(","), "run_key": a.run_key,
                         "label": a.label or f"{a.split} {a.run_key}", "model_lock": a.model_lock})
rid = run["run_id"]
print("run:", run)
while True:
    r = call(f"/api/runs/{rid}")
    done = sum(v for k, v in r["counts"].items() if k not in ("queued", "running"))
    print(f"\r{r['status']} {done}/{r['total']} {r['counts']}", end="", flush=True)
    if r["status"] in ("completed", "cancelled"):
        break
    time.sleep(2)
print()
m = call(f"/api/metrics?run_id={rid}")
for split, by in m["by_split_method"].items():
    for meth, s in by.items():
        print(f"{split:12} {meth:22} correct {s['correct_over_attempted']['n']}/{s['attempted']} abst {s['abstentions']} "
              f"opfail {s['operational_failures']} falseVeh {s['false_vehicle_on_depictions']['n']}/{s['false_vehicle_on_depictions']['d']} "
              f"lat p50 {s['latency_ms']['p50']}")
    p = m["paired_by_split"][split]
    print("paired:", {k: p[k] for k in ("both_correct", "baseline_only", "variant_only", "neither", "unresolved", "pairs", "hash_mismatch")})
print("RUN_ID", rid)
