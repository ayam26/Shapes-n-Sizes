"""All confirmatory and descriptive analyses from per-run detections (spec §3, §7).

Each run directory `<ARM>-s<seed>/eval/` holds `val-all.detections.jsonl` (all BDD val
images; the subsets are slices of it) and `t-texture.detections.jsonl` (stylized copies
of the T-clean images, same names). Matching is done once per run; the bootstrap reuses
it with image weights shared by every run in a replicate.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

from snz.data.bdd import load_frames, select_frames
from snz.data.manifest import read_manifest
from snz.eval.ap import RECALL_POINTS, APCalculator
from snz.eval.bootstrap import bootstrap_weights, holm, one_sided_p, percentile_ci
from snz.eval.guard import PREREG_TAG
from snz.eval.matching import match_split
from snz.eval.operating import RecallCurve, false_detections, recall_at, threshold_for_recall
from snz.eval.predict import read_detections

ARMS = ("BASE", "STY", "SIN", "BOTH")
SHIFT = ("t-night", "t-rain", "t-snow", "t-fog")
SUBSETS = ("t-clean", *SHIFT, "t-dusk", "t-overcast")
REPORTED = (*SUBSETS, "t-shift", "t-texture", "val-all")
METRICS = ("clean_mAP", "texture_mAP", "retention", "shift_mAP", "shift_fd")
CONFIRMATORY = {
    "H1b-STY": ("STY", "retention", "greater"),
    "H1b-SIN": ("SIN", "retention", "greater"),
    "H2": ("STY", "shift_mAP", "greater"),
    "H3": ("SIN", "shift_mAP", "greater"),
    "H4-STY": ("STY", "shift_fd", "less"),
    "H4-SIN": ("SIN", "shift_fd", "less"),
}
ALPHA = 0.05
COCO_TOLERANCE = 0.001  # 0.1 mAP points (spec §7.1)
PR_SUBSETS = ("t-clean", "t-shift", "t-texture", "t-night")
RUN_DIR = re.compile(r"^(BASE|STY|SIN|BOTH)-s(\d+)$")


def _discover(runs_root: Path) -> dict[str, tuple[str, int, Path]]:
    runs = {}
    for p in sorted(Path(runs_root).iterdir()):
        hit = RUN_DIR.match(p.name)
        if hit and (p / "eval" / "val-all.detections.jsonl").exists() \
                and (p / "eval" / "t-texture.detections.jsonl").exists():
            runs[p.name] = (hit[1], int(hit[2]), p)
    return runs


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _summary(per_run: dict[str, float | None]) -> dict:
    vals = np.array([v for v in per_run.values() if v is not None], float)
    if not vals.size:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None, "per_seed": per_run}
    return {"n": int(vals.size), "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "min": float(vals.min()), "max": float(vals.max()), "per_seed": per_run}


def _ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def _provenance() -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    return {"git_commit": git("rev-parse", "HEAD") or "unknown",
            "prereg_tag": git("tag", "--list", PREREG_TAG) == PREREG_TAG}


def analyze(runs_root: Path, val_labels: Path, manifests_dir: Path, n_boot: int = 2000,
            seed: int = 0, seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    runs = _discover(runs_root)
    if "BASE-s0" not in runs:
        raise FileNotFoundError(f"BASE-s0 detections not found under {runs_root}; the "
                                "operating point R* is defined by BASE seed 0")
    missing = sorted(f"{a}-s{s}" for a in ARMS for s in seeds if f"{a}-s{s}" not in runs)
    manifests_dir = Path(manifests_dir)
    names = {s: [n for n, _ in read_manifest(manifests_dir / f"{s}.csv")]
             for s in (*SUBSETS, "val-all")}
    shift_names = [n for s in SHIFT for n in names[s]]
    frames, _ = load_frames(val_labels)
    val_frames = select_frames(frames, names["val-all"])
    clean_frames = select_frames(frames, names["t-clean"])

    matches = {}
    for run, (_, _, path) in runs.items():
        mv = match_split(val_frames, read_detections(path / "eval" / "val-all.detections.jsonl"))
        sub = {s: mv.subset(names[s]) for s in SUBSETS}
        sub["t-shift"] = mv.subset(shift_names)
        sub["val-all"] = mv
        sub["t-texture"] = match_split(
            clean_frames, read_detections(path / "eval" / "t-texture.detections.jsonl"))
        matches[run] = sub

    # point estimates per run
    point = {run: {s: APCalculator(m)() for s, m in sub.items()} for run, sub in matches.items()}
    r_star = recall_at(matches["BASE-s0"]["t-shift"], 0.5)
    ops = {run: threshold_for_recall(sub["t-shift"], r_star) for run, sub in matches.items()}
    fd = {run: false_detections(matches[run]["t-shift"], ops[run].threshold) for run in runs}
    r_val = recall_at(matches["BASE-s0"]["val-all"], 0.5)
    foot = {run: threshold_for_recall(sub["val-all"], r_val) for run, sub in matches.items()}
    run_metric = {run: {
        "clean_mAP": point[run]["t-clean"]["mAP"],
        "texture_mAP": point[run]["t-texture"]["mAP"],
        "retention": _ratio(point[run]["t-texture"]["mAP"], point[run]["t-clean"]["mAP"]),
        "shift_mAP": point[run]["t-shift"]["mAP"],
        "shift_fd": fd[run]["per_image"],
    } for run in runs}

    # paired bootstrap: one weight matrix per image set, shared by all runs. R* and every
    # run's equal-recall threshold are re-estimated inside each replicate.
    w = {"t-clean": bootstrap_weights(len(names["t-clean"]), n_boot, seed),
         "t-shift": bootstrap_weights(len(shift_names), n_boot, seed + 1)}
    for i, sname in enumerate(SUBSETS[1:]):
        w[sname] = bootstrap_weights(len(names[sname]), n_boot, seed + 2 + i)
    base_shift = matches["BASE-s0"]["t-shift"]
    r_star_boot = np.array([recall_at(base_shift, 0.5, w["t-shift"][i]) for i in range(n_boot)])
    boot, boot_sub = {}, {}
    for run, sub in matches.items():
        calc = {s_: APCalculator(sub[s_]) for s_ in (*SUBSETS, "t-shift", "t-texture")}
        curve = RecallCurve(sub["t-shift"])
        b = {k: np.empty(n_boot) for k in METRICS}
        bs = {s_: np.empty(n_boot) for s_ in (*SUBSETS, "t-shift")}
        for i in range(n_boot):
            for s_ in bs:
                bs[s_][i] = calc[s_](w[s_][i])["mAP"]
            b["texture_mAP"][i] = calc["t-texture"](w["t-clean"][i])["mAP"]
            th = curve.threshold(r_star_boot[i], w["t-shift"][i]).threshold
            b["shift_fd"][i] = false_detections(sub["t-shift"], th, w["t-shift"][i])["per_image"]
        b["clean_mAP"], b["shift_mAP"] = bs["t-clean"], bs["t-shift"]
        with np.errstate(divide="ignore", invalid="ignore"):
            b["retention"] = b["texture_mAP"] / b["clean_mAP"]
        boot[run], boot_sub[run] = b, bs

    arm_runs = {a: [r for r, (arm, _, _) in runs.items() if arm == a] for a in ARMS}
    arm_point = {a: {k: float(np.mean([run_metric[r][k] for r in rs])) for k in METRICS}
                 for a, rs in arm_runs.items() if rs}
    arm_boot = {a: {k: np.mean([boot[r][k] for r in rs], axis=0) for k in METRICS}
                for a, rs in arm_runs.items() if rs}

    def compare(arm: str, metric: str, direction: str | None) -> dict:
        if arm not in arm_boot or "BASE" not in arm_boot:
            return {"status": "missing arm"}
        diffs = arm_boot[arm][metric] - arm_boot["BASE"][metric]
        out = {"status": "ok", "arm": arm, "metric": metric,
               "estimate": arm_point[arm][metric] - arm_point["BASE"][metric],
               "ci95": percentile_ci(diffs[np.isfinite(diffs)])}
        if direction:
            out.update(direction=direction, p=one_sided_p(diffs, direction))
        return out

    confirmatory = {h: compare(*spec) for h, spec in CONFIRMATORY.items()}
    # the preregistered family is all six tests; a missing one enters Holm with p = 1
    family = {h: (c["p"] if c["status"] == "ok" else 1.0) for h, c in confirmatory.items()}
    for h, p_adj in holm(family).items():
        if confirmatory[h]["status"] == "ok":
            confirmatory[h].update(p_holm=p_adj, supported=bool(p_adj < ALPHA))

    descriptive = {}
    if "BASE" in arm_boot:
        ret = arm_boot["BASE"]["retention"]
        descriptive["H1a_BASE_retention"] = {"estimate": arm_point["BASE"]["retention"],
                                             "ci95": percentile_ci(ret[np.isfinite(ret)])}
    subset_diff = {}
    for s_ in (*SUBSETS, "t-shift"):
        subset_diff[s_] = {}
        for arm in ("STY", "SIN", "BOTH"):
            if arm not in arm_boot or "BASE" not in arm_boot:
                continue
            arm_b = np.mean([boot_sub[r][s_] for r in arm_runs[arm]], axis=0)
            base_b = np.mean([boot_sub[r][s_] for r in arm_runs["BASE"]], axis=0)
            est = (np.mean([point[r][s_]["mAP"] for r in arm_runs[arm]])
                   - np.mean([point[r][s_]["mAP"] for r in arm_runs["BASE"]]))
            subset_diff[s_][arm] = {"estimate": float(est),
                                    "ci95": percentile_ci(arm_b - base_b)}

    exploratory = {f"{arm}_vs_BASE_{metric}": compare(arm, metric, None)
                   for arm in ("STY", "SIN", "BOTH") for metric in METRICS}
    exploratory["subset_mAP_vs_BASE"] = subset_diff

    coco_check = {}
    for run, (_, _, path) in runs.items():
        for s_ in ("val-all", "t-texture"):
            f = path / "eval" / f"{s_}.metrics.json"
            if f.exists():
                ref = json.loads(f.read_text())["mAP"]
                diff = abs(ref - point[run][s_]["mAP"])
                coco_check.setdefault(run, {})[s_] = {
                    "pycocotools": ref, "ours": point[run][s_]["mAP"], "abs_diff": diff,
                    "agrees": bool(diff <= COCO_TOLERANCE)}
    if all(a in arm_boot for a in ARMS):
        add = (arm_boot["BOTH"]["shift_mAP"] - arm_boot["BASE"]["shift_mAP"]
               - (arm_boot["STY"]["shift_mAP"] - arm_boot["BASE"]["shift_mAP"])
               - (arm_boot["SIN"]["shift_mAP"] - arm_boot["BASE"]["shift_mAP"]))
        est = (arm_point["BOTH"]["shift_mAP"] - arm_point["STY"]["shift_mAP"]
               - arm_point["SIN"]["shift_mAP"] + arm_point["BASE"]["shift_mAP"])
        exploratory["additivity_shift_mAP"] = {"estimate": est, "ci95": percentile_ci(add)}
    else:
        exploratory["additivity_shift_mAP"] = {"status": "missing arm"}

    # precision-recall curves at IoU 0.5 (spec §7.3): mean over classes, then over seeds
    pr_curves = {}
    for s_ in PR_SUBSETS:
        pr_curves[s_] = {}
        for a, rs in arm_runs.items():
            if rs:
                curves = [np.nanmean(APCalculator(matches[r][s_]).precision()[0], axis=1)
                          for r in rs]
                pr_curves[s_][a] = [float(x) for x in np.mean(curves, axis=0)]

    classes = matches["BASE-s0"]["t-clean"].classes
    arms = {}
    for a, rs in arm_runs.items():
        subsets = {}
        for s in REPORTED:
            subsets[s] = {
                "images": len(matches["BASE-s0"][s].image_names),
                "mAP": _summary({r: point[r][s]["mAP"] for r in rs}),
                "AP50": _summary({r: point[r][s]["AP50"] for r in rs}),
                "per_class": {c: _summary({r: _f(point[r][s]["per_class"][c]) for r in rs})
                              for c in classes},
            }
        arms[a] = {
            "runs": rs,
            "subsets": subsets,
            "retention": _summary({r: _f(run_metric[r]["retention"]) for r in rs}),
            "shift_fd_per_image": _summary({r: fd[r]["per_image"] for r in rs}),
            "shift_fd_by_type": {t: _summary({r: fd[r]["by_type"][t] for r in rs})
                                 for t in (fd[rs[0]]["by_type"] if rs else {})},
        }

    t_corrupt = {}
    corrupt_manifest = manifests_dir / "t-corrupt.csv"
    if corrupt_manifest.exists():
        subset = [n for n, _ in read_manifest(corrupt_manifest)]
        sub_frames = select_frames(frames, subset)
        for run, (_, _, path) in runs.items():
            files = sorted((path / "eval").glob("t-corrupt-*.detections.jsonl"))
            if not files:
                continue
            clean = APCalculator(matches[run]["val-all"].subset(subset))()["mAP"]
            maps = {}
            for f in files:
                cs = f.name.removeprefix("t-corrupt-").removesuffix(".detections.jsonl")
                maps[cs] = APCalculator(match_split(sub_frames, read_detections(f)))()["mAP"]
            t_corrupt[run] = {"images": len(subset), "clean_mAP": clean, "mAP": maps,
                              "mPC": float(np.mean(list(maps.values()))),
                              "rPC": _ratio(float(np.mean(list(maps.values()))), clean)}

    return {
        "runs": sorted(runs),
        "missing_runs": missing,
        "n_boot": n_boot,
        "bootstrap_seed": seed,
        "images": {s: len(names[s]) for s in (*SUBSETS, "val-all")} | {
            "t-shift": len(shift_names)},
        "operating_point": {"R_star": r_star, "definition": "BASE-s0 recall on pooled T-shift "
                            "at score >= 0.5, IoU >= 0.5, 9 classes pooled"},
        "thresholds": {"t-shift": {r: vars(o) for r, o in ops.items()}},
        "flagged_runs": sorted(r for r, o in ops.items() if o.flagged),
        "footage_thresholds": {r: vars(o) for r, o in foot.items()},
        "footage_R_star": r_val,
        "confirmatory": confirmatory,
        "holm_family_size": len(family),
        "thresholds_in_bootstrap": "re-estimated per replicate",
        "descriptive": descriptive,
        "coco_check": coco_check,
        "provenance": _provenance(),
        "exploratory": exploratory,
        "arms": arms,
        "t_corrupt": t_corrupt,
        "pr_curves": pr_curves,
        "pr_recall": [float(x) for x in RECALL_POINTS],
    }
