#!/usr/bin/env python3
"""Measure local-model resource behaviour on the GX10 (run ON the GX10):
cold load time, resident memory, warm latency, all-three-resident switching, and GPU vs CPU-only latency.

Uses the frozen baseline prompt + output schema and a real corpus image (fid-019: high-fidelity oblique truck).
Results are measurements only; they are not part of any accuracy metric.

    ~/perception-lab/.venv/bin/python scripts/measure_models.py > data/model-resources.json
"""
import base64
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLLAMA = "http://127.0.0.1:11434"
MODELS = ["nemotron3:33b", "qwen3.6:35b", "gemma4:26b"]
CPU_MODELS = ["gemma4:26b", "nemotron3:33b"]  # CPU-only runs are slow; two models are enough to show the ratio
prompts = json.loads((ROOT / "experiments/prompts.json").read_text())
PROMPT = prompts["methods"]["rgb_baseline"]["prompt"]
SCHEMA = prompts["output_schema"]
IMAGE = base64.b64encode((ROOT / "data/cases/fid-019/input.png").read_bytes()).decode()
OPTIONS = {"temperature": 0, "seed": 7, "num_predict": 256}


def post(path, body, timeout=900):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get(path):
    with urllib.request.urlopen(OLLAMA + path, timeout=10) as r:
        return json.load(r)


def infer(model, extra_options=None, keep_alive="30m"):
    t = time.perf_counter()
    r = post("/api/chat", {"model": model, "stream": False, "think": False, "keep_alive": keep_alive, "format": SCHEMA,
                           "options": {**OPTIONS, **(extra_options or {})},
                           "messages": [{"role": "user", "content": PROMPT, "images": [IMAGE]}]})
    wall = (time.perf_counter() - t) * 1000
    label = None
    try:
        label = json.loads(r["message"]["content"]).get("label")
    except Exception:
        pass
    return {"wall_ms": wall, "load_ms": r.get("load_duration", 0) / 1e6, "prompt_eval_ms": r.get("prompt_eval_duration", 0) / 1e6,
            "eval_ms": r.get("eval_duration", 0) / 1e6, "eval_count": r.get("eval_count"), "label": label}


def unload_all():
    for m in get("/api/ps").get("models", []):
        post("/api/generate", {"model": m["name"], "keep_alive": 0})
    for _ in range(60):
        if not get("/api/ps").get("models"):
            return
        time.sleep(0.5)


def mem_used_gib():
    out = subprocess.run(["free", "-b"], capture_output=True, text=True).stdout.split("\n")[1].split()
    return {"total_gib": int(out[1]) / 2**30, "used_gib": int(out[2]) / 2**30, "available_gib": int(out[6]) / 2**30}


def resident():
    return [{"model": m["name"], "size_gb": m.get("size", 0) / 1e9, "vram_gb": m.get("size_vram", 0) / 1e9}
            for m in get("/api/ps").get("models", [])]


def pct(v, p):
    s = sorted(v)
    return s[max(0, -(-len(s) * p // 100) - 1)]


report = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "image": "data/cases/fid-019/input.png",
          "prompt": "rgb_baseline (frozen rev 2)", "options": OPTIONS, "per_model": {}}
report["ollama_version"] = get("/api/version")["version"]
unload_all()
report["baseline_memory"] = mem_used_gib()

for m in MODELS:
    unload_all()
    before = mem_used_gib()
    cold = infer(m)
    after = mem_used_gib()
    warm = [infer(m) for _ in range(10)]
    w = [x["wall_ms"] for x in warm]
    report["per_model"][m] = {
        "cold_request_ms": cold["wall_ms"], "cold_load_ms": cold["load_ms"],
        "resident": resident(), "system_mem_delta_gib": after["used_gib"] - before["used_gib"],
        "warm_ms": {"n": len(w), "p50": statistics.median(w), "p95": pct(w, 95), "min": min(w), "max": max(w)},
        "warm_eval_tokens_per_s": statistics.median([x["eval_count"] / (x["eval_ms"] / 1000) for x in warm if x["eval_ms"]]),
        "warm_prompt_eval_ms_p50": statistics.median([x["prompt_eval_ms"] for x in warm]),
        "labels_warm": sorted({str(x["label"]) for x in warm}),
    }
    print(f"[{m}] cold {cold['wall_ms']:.0f} ms, warm p50 {statistics.median(w):.0f} ms", file=sys.stderr)

# all three resident at once + switching cost
unload_all()
for m in MODELS:
    infer(m)
report["all_resident"] = {"resident": resident(), "memory": mem_used_gib(),
                          "total_resident_gb": sum(x["size_gb"] for x in resident())}
switch = []
for _ in range(2):
    for m in MODELS:
        r = infer(m)
        switch.append({"model": m, "wall_ms": r["wall_ms"], "load_ms": r["load_ms"]})
report["switching_all_resident"] = {"requests": switch, "p50_ms": statistics.median([s["wall_ms"] for s in switch]),
                                    "max_load_ms": max(s["load_ms"] for s in switch)}
print(f"[switch] p50 {report['switching_all_resident']['p50_ms']:.0f} ms", file=sys.stderr)

# GPU vs CPU-only (num_gpu=0 forces all layers onto the Arm CPU cores)
report["cpu_vs_gpu"] = {}
for m in CPU_MODELS:
    unload_all()
    cpu_cold = infer(m, {"num_gpu": 0}, keep_alive="5m")
    cpu = [infer(m, {"num_gpu": 0}, keep_alive="5m") for _ in range(2)]
    gpu_p50 = report["per_model"][m]["warm_ms"]["p50"]
    cpu_p50 = statistics.median([c["wall_ms"] for c in cpu])
    report["cpu_vs_gpu"][m] = {"cpu_cold_ms": cpu_cold["wall_ms"], "cpu_warm_ms": [c["wall_ms"] for c in cpu],
                               "cpu_warm_p50_ms": cpu_p50, "gpu_warm_p50_ms": gpu_p50, "speedup_gpu_over_cpu": cpu_p50 / gpu_p50,
                               "cpu_labels": [c["label"] for c in cpu]}
    print(f"[cpu] {m} cpu p50 {cpu_p50:.0f} ms vs gpu {gpu_p50:.0f} ms", file=sys.stderr)
unload_all()
for m in MODELS:  # restore GPU residency for the app
    infer(m)
report["final_resident"] = resident()
print(json.dumps(report, indent=2))
