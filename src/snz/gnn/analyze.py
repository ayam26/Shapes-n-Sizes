"""G1-G3 and descriptive analyses for the GNN re-scoring arms (spec §2, prereg-v2)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from snz.data.bdd import load_frames, select_frames
from snz.data.manifest import read_manifest
from snz.eval.analyze import SHIFT, SUBSETS, _provenance
from snz.eval.ap import APCalculator
from snz.eval.bootstrap import bootstrap_weights, holm, one_sided_p, percentile_ci
from snz.eval.matching import match_split
from snz.eval.operating import (
    RecallCurve,
    false_detections,
    recall_at,
    threshold_for_false_rate,
)
from snz.eval.predict import read_detections
from snz.gnn import ARMS_G

ALPHA = 0.05
CONFIRMATORY = {"G1": ("GNN", "BASE", "fd_at_R", "less"),
                "G2": ("GNN", "BASE", "recall_at_F", "greater"),
                "G3": ("GNN", "RGB", "fd_at_R", "less")}
DESCRIPTIVE = {"GNN-minus-GNN-SHAPE": ("GNN", "GNN-SHAPE"),
               "GNN-COLOR-minus-GNN": ("GNN-COLOR", "GNN"),
               "RGB-minus-BASE": ("RGB", "BASE"), "GNN-SHAPE-minus-BASE": ("GNN-SHAPE", "BASE")}
RUN = re.compile(r"^(" + "|".join(re.escape(a) for a in ARMS_G) + r")-s0$")


def gnn_analyze(runs_root: Path, val_labels: Path, manifests_dir: Path, n_boot: int = 2000,
                seed: int = 0) -> dict:
    runs_root, manifests_dir = Path(runs_root), Path(manifests_dir)
    arms = [p.name[:-3] for p in sorted(runs_root.iterdir())
            if RUN.match(p.name) and (p / "eval" / "val-all.detections.jsonl").exists()]
    if "BASE" not in arms:
        raise FileNotFoundError(f"BASE-s0 re-scored detections missing under {runs_root}")
    missing = [f"{a}-s0" for a in ARMS_G if a not in arms]
    names = {s: [n for n, _ in read_manifest(manifests_dir / f"{s}.csv")]
             for s in (*SUBSETS, "val-all")}
    shift = [n for s in SHIFT for n in names[s]]
    frames, _ = load_frames(val_labels)
    val_frames = select_frames(frames, names["val-all"])
    clean_frames = select_frames(frames, names["t-clean"])
    m = {}
    for a in arms:
        ev = runs_root / f"{a}-s0" / "eval"
        mv = match_split(val_frames, read_detections(ev / "val-all.detections.jsonl"))
        sub = {s: mv.subset(names[s]) for s in SUBSETS}
        sub["t-shift"], sub["val-all"] = mv.subset(shift), mv
        tex = ev / "t-texture.detections.jsonl"
        if tex.exists():
            sub["t-texture"] = match_split(clean_frames, read_detections(tex))
        m[a] = sub

    base = m["BASE"]["t-shift"]
    r_star = recall_at(base, 0.5)
    f_star = false_detections(base, 0.5)["per_image"]
    point = {}
    for a in arms:
        ms = m[a]["t-shift"]
        op_r = RecallCurve(ms).threshold(r_star)
        fd = false_detections(ms, op_r.threshold)
        op_f = threshold_for_false_rate(ms, f_star)
        aps = {s: APCalculator(mm)() for s, mm in m[a].items()}
        point[a] = {
            "threshold_at_R": vars(op_r), "fd_at_R": fd["per_image"],
            "fd_types_at_R": fd["by_type"], "fd_count_at_R": fd["count"],
            "threshold_at_F": vars(op_f), "recall_at_F": op_f.recall,
            "mAP": {s: v["mAP"] for s, v in aps.items()},
            "AP50": {s: v["AP50"] for s, v in aps.items()},
            "retention": (aps["t-texture"]["mAP"] / aps["t-clean"]["mAP"]
                          if "t-texture" in aps and aps["t-clean"]["mAP"] else None)}

    w = bootstrap_weights(len(shift), n_boot, seed)
    curves = {a: RecallCurve(m[a]["t-shift"]) for a in arms}
    calc = {a: APCalculator(m[a]["t-shift"]) for a in arms}
    boot = {a: {k: np.empty(n_boot) for k in ("fd_at_R", "recall_at_F", "shift_mAP")}
            for a in arms}
    for i in range(n_boot):
        r_b = recall_at(base, 0.5, w[i])
        f_b = false_detections(base, 0.5, w[i])["per_image"]
        for a in arms:
            ms = m[a]["t-shift"]
            th = curves[a].threshold(r_b, w[i]).threshold
            boot[a]["fd_at_R"][i] = false_detections(ms, th, w[i])["per_image"]
            boot[a]["recall_at_F"][i] = threshold_for_false_rate(ms, f_b, w[i]).recall
            boot[a]["shift_mAP"][i] = calc[a](w[i])["mAP"]

    def compare(a: str, b: str, metric: str, direction: str | None) -> dict:
        if a not in point or b not in point:
            return {"status": "missing arm"}
        d = boot[a][metric] - boot[b][metric]
        pt = point[a][metric] if metric != "shift_mAP" else point[a]["mAP"]["t-shift"]
        pb = point[b][metric] if metric != "shift_mAP" else point[b]["mAP"]["t-shift"]
        out = {"status": "ok", "arms": [a, b], "metric": metric, "estimate": pt - pb,
               "ci95": percentile_ci(d)}
        if direction:
            out.update(direction=direction, p=one_sided_p(d, direction))
        return out

    conf = {g: compare(a, b, met, d) for g, (a, b, met, d) in CONFIRMATORY.items()}
    family = {g: (c["p"] if c["status"] == "ok" else 1.0) for g, c in conf.items()}
    for g, p_adj in holm(family).items():
        conf[g]["p_holm"] = p_adj
        conf[g]["supported"] = bool(conf[g]["status"] == "ok" and p_adj < ALPHA)
    desc = {name: {met: compare(a, b, met, None) for met in ("fd_at_R", "recall_at_F",
                                                              "shift_mAP")}
            for name, (a, b) in DESCRIPTIVE.items()}

    ok_rate = {}
    for a in arms:
        prov = runs_root / f"{a}-s0" / "provenance.json"
        if a == "GNN" and prov.exists():
            import json

            from snz.gnn.outlines import load_outlines

            p = json.loads(prov.read_text()).get("val-all", {})
            if p.get("outlines") and Path(p["outlines"]).exists():
                o = load_outlines(Path(p["outlines"]))
                img_ok = {}
                for r in range(len(o.ok)):
                    img_ok.setdefault(o.names[o.image[r]], []).append(bool(o.ok[r]))
                for s in (*SUBSETS, "t-shift"):
                    sel = names[s] if s != "t-shift" else shift
                    vals = [v for n in sel for v in img_ok.get(n, [])]
                    ok_rate[s] = float(np.mean(vals)) if vals else None
    return {"arms": arms, "missing": missing, "n_boot": n_boot, "bootstrap_seed": seed,
            "images": {"t-shift": len(shift), **{s: len(names[s]) for s in names}},
            "R_star": r_star, "F_star": f_star, "point": point, "confirmatory": conf,
            "holm_family_size": len(family), "descriptive": desc, "outline_ok_rate": ok_rate,
            "provenance": _provenance()}


def _n(x, d=3) -> str:
    return "—" if x is None else f"{x:.{d}f}"


def render_gnn_report(r: dict) -> str:
    L = ["# GNN shape verifier — results", "",
         f"Arms: {', '.join(r['arms'])}; not run: {', '.join(r['missing']) or 'none'}. "
         f"Paired bootstrap B = {r['n_boot']} over {r['images']['t-shift']} T-shift images.", "",
         f"R* = {_n(r['R_star'], 4)} (BASE-s0 recall on t-shift at score ≥ 0.5); "
         f"F* = {_n(r['F_star'], 4)} false detections per image (BASE-s0 at score ≥ 0.5).", "",
         "## Confirmatory tests (Holm over G1–G3, α = 0.05)", "",
         "| Test | Comparison | Metric | Estimate | 95% CI | p | p (Holm) | Supported |",
         "|---|---|---|---|---|---|---|---|"]
    for g, (a, b, met, d) in CONFIRMATORY.items():
        c = r["confirmatory"][g]
        if c["status"] != "ok":
            L.append(f"| {g} | {a} vs {b} | {met} | — | — | — | {_n(c['p_holm'])} | "
                     f"{c['status']} |")
            continue
        L.append(f"| {g} | {a} vs {b} ({d}) | {met} | {_n(c['estimate'], 4)} | "
                 f"[{_n(c['ci95'][0], 4)}, {_n(c['ci95'][1], 4)}] | {_n(c['p'], 4)} | "
                 f"{_n(c['p_holm'], 4)} | {'yes' if c['supported'] else 'no'} |")
    L += ["", "## Per arm (t-shift unless stated)", "",
          "| Arm | False det./img at R* | Recall at F* | t-shift mAP | t-clean mAP | "
          "T-texture retention |", "|---|---|---|---|---|---|"]
    for a, p in r["point"].items():
        L.append(f"| {a} | {_n(p['fd_at_R'])} | {_n(p['recall_at_F'])} | "
                 f"{_n(p['mAP']['t-shift'] * 100, 1)} | {_n(p['mAP']['t-clean'] * 100, 1)} | "
                 f"{_n(p['retention'])} |")
    types = next(iter(r["point"].values()))["fd_types_at_R"]
    L += ["", "## False detections per image at R*, by type", "",
          "| Arm | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    for a, p in r["point"].items():
        L.append(f"| {a} | " + " | ".join(_n(v) for v in p["fd_types_at_R"].values()) + " |")
    L += ["", "## mAP by subset (×100)", "", "| Subset | " + " | ".join(r["point"]) + " |",
          "|---|" + "---|" * len(r["point"])]
    for s in next(iter(r["point"].values()))["mAP"]:
        L.append(f"| {s} | " + " | ".join(_n(p["mAP"].get(s, 0) * 100, 1)
                                         for p in r["point"].values()) + " |")
    L += ["", "## Descriptive comparisons (95% CIs, no tests)", "",
          "| Comparison | Metric | Estimate | 95% CI |", "|---|---|---|---|"]
    for name, by_met in r["descriptive"].items():
        for met, c in by_met.items():
            if c["status"] == "ok":
                L.append(f"| {name} | {met} | {_n(c['estimate'], 4)} | "
                         f"[{_n(c['ci95'][0], 4)}, {_n(c['ci95'][1], 4)}] |")
    if r["outline_ok_rate"]:
        L += ["", "Outline success rate (SAM 2 mask usable) by subset: " + ", ".join(
            f"{s} {_n(v, 3)}" for s, v in r["outline_ok_rate"].items()) + "."]
    prov = r.get("provenance", {})
    L += ["", f"Code: {prov.get('git_commit', '?')}."]
    return "\n".join(L) + "\n"


def footage_thresholds(runs_root: Path, val_labels: Path, manifests_dir: Path) -> dict:
    """Per arm, the threshold reaching BASE-s0's full-val recall at score >= 0.5 (spec §6)."""
    runs_root = Path(runs_root)
    names = [n for n, _ in read_manifest(Path(manifests_dir) / "val-all.csv")]
    frames = select_frames(load_frames(val_labels)[0], names)
    arms = [p.name for p in sorted(runs_root.iterdir())
            if RUN.match(p.name) and (p / "eval" / "val-all.detections.jsonl").exists()]
    m = {a: match_split(frames, read_detections(runs_root / a / "eval" /
                                                "val-all.detections.jsonl")) for a in arms}
    r_val = recall_at(m["BASE-s0"], 0.5)
    return {"R_star_val": r_val,
            "footage_thresholds": {a: vars(RecallCurve(mm).threshold(r_val))
                                   for a, mm in m.items()}}
