"""Outcome classification and metrics. Pure functions over plain dicts so the artifact
verifier can recompute every reported number from exported records."""
import math

OPERATIONAL = ("invalid_output", "timeout", "provider_error")
BASELINE, VARIANT = "rgb_baseline", "rgb_structure_prompt"


def outcome(truth: str, status: str, label):
    if status in OPERATIONAL:
        return status
    if status != "ok":
        return None  # queued / running / cancelled: not yet an outcome
    if label == "uncertain":
        return "abstention"
    if label == truth:
        return "correct"
    if truth == "flat_depiction" and label == "physical_vehicle":
        return "depiction_as_vehicle"
    if truth == "physical_vehicle" and label == "flat_depiction":
        return "vehicle_as_depiction"
    return "wrong_other_label"


def percentile(values, p):
    """Nearest-rank percentile."""
    if not values:
        return None
    v = sorted(values)
    return v[max(0, math.ceil(p / 100 * len(v)) - 1)]


def summarize(rs: list[dict]) -> dict:
    """rs: finished results with keys truth, status, label, origin, elapsed_ms."""
    n = len(rs)
    oc = [outcome(r["truth"], r["status"], r["label"]) for r in rs]
    count = lambda *names: sum(1 for o in oc if o in names)
    valid = sum(1 for r in rs if r["status"] == "ok")
    abst = count("abstention")
    correct = count("correct")
    answered = valid - abst
    flat = [r for r in rs if r["truth"] == "flat_depiction"]
    phys = [r for r in rs if r["truth"] == "physical_vehicle"]
    other = [r for r in rs if r["truth"] == "other"]
    lat = [r["elapsed_ms"] for r in rs if r.get("origin") == "live" and r["status"] in ("ok", "invalid_output")
           and r.get("elapsed_ms") is not None]
    frac = lambda a, b: {"n": a, "d": b, "pct": round(100 * a / b, 1) if b else None}
    return {
        "attempted": n,
        "valid_responses": valid,
        "operational_failures": count(*OPERATIONAL),
        "invalid_output": count("invalid_output"),
        "timeout": count("timeout"),
        "provider_error": count("provider_error"),
        "abstentions": abst,
        "correct_over_attempted": frac(correct, n),
        "accuracy_answered": frac(correct, answered),
        "coverage": frac(answered, n),
        "false_vehicle_on_depictions": frac(
            sum(1 for r in flat if r["status"] == "ok" and r["label"] == "physical_vehicle"), len(flat)),
        "vehicle_recall": frac(
            sum(1 for r in phys if r["status"] == "ok" and r["label"] == "physical_vehicle"), len(phys)),
        "control_correct": frac(sum(1 for r in other if r["status"] == "ok" and r["label"] == "other"), len(other)),
        "outcomes": {k: count(k) for k in ("correct", "abstention", "depiction_as_vehicle", "vehicle_as_depiction",
                                             "wrong_other_label", *OPERATIONAL)},
        "latency_ms": {"n": len(lat), "p50": percentile(lat, 50), "p95": percentile(lat, 95)},
    }


def paired(rs: list[dict], base=BASELINE, var=VARIANT) -> dict:
    by_case: dict = {}
    for r in rs:
        by_case.setdefault(r["case_id"], {})[r["method_id"]] = r
    out = {"both_correct": 0, "baseline_only": 0, "variant_only": 0, "neither": 0, "unresolved": 0,
           "pairs": 0, "hash_mismatch": 0, "cases": {}}
    for cid, m in by_case.items():
        if base not in m or var not in m:
            continue
        b, v = m[base], m[var]
        out["pairs"] += 1
        if b["image_sha256"] != v["image_sha256"]:
            out["hash_mismatch"] += 1
        ob, ov = outcome(b["truth"], b["status"], b["label"]), outcome(v["truth"], v["status"], v["label"])
        if ob in OPERATIONAL or ov in OPERATIONAL or ob is None or ov is None:
            k = "unresolved"
        elif ob == "correct" and ov == "correct":
            k = "both_correct"
        elif ob == "correct":
            k = "baseline_only"
        elif ov == "correct":
            k = "variant_only"
        else:
            k = "neither"
        out[k] += 1
        out["cases"][cid] = k
    return out


def metrics(rs: list[dict]) -> dict:
    """Per split x method summaries, observability breakdown, and paired comparisons."""
    done = [r for r in rs if r["status"] not in ("queued", "running", "cancelled")]
    res = {"by_split_method": {}, "paired_by_split": {}, "by_observability": {}}
    for split in sorted({r["split"] for r in done}):
        srs = [r for r in done if r["split"] == split]
        res["by_split_method"][split] = {m: summarize([r for r in srs if r["method_id"] == m])
                                         for m in sorted({r["method_id"] for r in srs})}
        p = paired(srs)
        res["paired_by_split"][split] = p
        res["by_observability"][split] = {
            obs: {m: summarize([r for r in srs if r["method_id"] == m and r["observability"] == obs])
                  for m in sorted({r["method_id"] for r in srs})}
            for obs in sorted({r["observability"] for r in srs})}
    return res
