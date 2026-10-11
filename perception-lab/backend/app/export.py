"""Run exports: JSONL/CSV records, metrics, Markdown summary, portable zip bundle."""
import csv
import io
import json
import zipfile

from . import config, db
from .evaluate import metrics, outcome

RECORD_SQL = """
SELECT r.result_id, r.run_id, r.case_id, r.method_id, r.status, r.label, r.parsed, r.raw_output, r.parse_error,
       r.elapsed_ms, r.origin, r.image_sha256, r.prompt_hash, r.model_id, r.model_digest, r.quantization,
       r.runtime_version, r.options, r.cache_key, r.provider_meta, r.queued_at, r.started_at, r.finished_at,
       c.split, c.family, c.truth, c.observability, c.conditions, c.scene_seed, c.render_status
FROM results r JOIN cases c ON c.case_id = r.case_id WHERE r.run_id = ? ORDER BY r.seq"""

CSV_FIELDS = ["case_id", "split", "family", "truth", "observability", "method_id", "status", "label", "outcome",
              "origin", "elapsed_ms", "image_sha256", "prompt_hash", "model_id", "conditions"]


def records(run_id: str) -> list[dict]:
    out = []
    for r in db.rows(RECORD_SQL, (run_id,)):
        r["outcome"] = outcome(r["truth"], r["status"], r["label"])
        out.append(r)
    return out


def to_csv(recs: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows(recs)
    return buf.getvalue()


def _fmt(fr):
    return "–" if fr["d"] == 0 else f"{fr['n']}/{fr['d']} ({fr['pct']}%)"


def report_md(run: dict, recs: list[dict], m: dict) -> str:
    cfg = json.loads(run["model_config"])
    lock = cfg["lock"]
    lines = [f"# Run {run['run_id']} — {run.get('label') or ''}", "",
             f"- Created: {run['created_at']} · status: {run['status']} · provider mode: **{run['provider_mode']}**",
             f"- Model: `{lock['model_id']}` digest `{(lock.get('model_digest') or '')[:12]}` · {lock.get('quantization')}"
             f" · {lock.get('runtime_version')} · options `{json.dumps(lock['options'])}` · think={lock.get('think')}",
             f"- Prompts frozen: {cfg['prompts_frozen']} · model lock frozen: {cfg['lock_frozen']}"
             f" · offline guard: {bool(run['offline_guard'])} · cache allowed: {bool(run['use_cache'])}",
             f"- Records: {len(recs)} (origins: " + ", ".join(
                 f"{o}={sum(1 for r in recs if str(r['origin']) == o)}" for o in sorted({str(r['origin']) for r in recs})) + ")",
             ""]
    for split, by_m in m["by_split_method"].items():
        lines += [f"## Split: {split}", "",
                  "| Method | Attempted | Valid | Op. failures | Abstain | Correct/attempted | Acc. (answered) | Coverage "
                  "| False vehicle on depictions | Vehicle recall | Control correct | Latency p50/p95 ms (n) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for meth, s in by_m.items():
            lat = s["latency_ms"]
            latstr = "–" if not lat["n"] else f"{lat['p50']:.0f} / {lat['p95']:.0f} ({lat['n']})"
            lines.append(f"| {meth} | {s['attempted']} | {s['valid_responses']} | {s['operational_failures']} | "
                         f"{s['abstentions']} | {_fmt(s['correct_over_attempted'])} | {_fmt(s['accuracy_answered'])} | "
                         f"{_fmt(s['coverage'])} | {_fmt(s['false_vehicle_on_depictions'])} | {_fmt(s['vehicle_recall'])} | "
                         f"{_fmt(s['control_correct'])} | {latstr} |")
        p = m["paired_by_split"][split]
        if p["pairs"]:
            lines += ["", f"Paired (same RGB hash; mismatches: {p['hash_mismatch']}): both correct {p['both_correct']}, "
                          f"baseline-only {p['baseline_only']}, variant-only {p['variant_only']}, neither {p['neither']}, "
                          f"unresolved {p['unresolved']} — of {p['pairs']} pairs."]
        lines.append("")
    lines += ["## All records", "", "| Case | Split | Truth | Method | Status | Label | Outcome | ms | Origin |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        ms = "–" if r["elapsed_ms"] is None else f"{r['elapsed_ms']:.0f}"
        lines.append(f"| {r['case_id']} | {r['split']} | {r['truth']} | {r['method_id']} | {r['status']} | "
                     f"{r['label'] or '–'} | {r['outcome'] or '–'} | {ms} | {r['origin'] or '–'} |")
    return "\n".join(lines) + "\n"


def bundle(run_id: str) -> bytes:
    run = db.one("SELECT * FROM runs WHERE run_id=?", (run_id,))
    recs = records(run_id)
    m = metrics(recs)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("run.json", json.dumps(run, indent=2))
        z.writestr("results.jsonl", "".join(json.dumps(r) + "\n" for r in recs))
        z.writestr("results.csv", to_csv(recs))
        z.writestr("metrics.json", json.dumps(m, indent=2))
        z.writestr("report.md", report_md(run, recs, m))
        for cid in sorted({r["case_id"] for r in recs}):
            case = db.one("SELECT * FROM cases WHERE case_id=?", (cid,))
            z.writestr(f"cases/{cid}/case.json", json.dumps(case, indent=2))
            z.writestr(f"cases/{cid}/scene.json", case["scene_spec"])
            z.write(config.DATA / case["image_path"], f"cases/{cid}/input.png")
    return buf.getvalue()
