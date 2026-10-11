"""All confirmatory and descriptive analyses from per-run detections (spec §3, §7).

Each run directory `<ARM>-s<seed>/eval/` holds `val-all.detections.jsonl` (all BDD val
images; the subsets are slices of it) and `t-texture.detections.jsonl` (stylized copies
of the T-clean images, same names). Matching is done once per run; the bootstrap reuses
it with image weights shared by every run in a replicate.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from snz.data.bdd import load_frames, select_frames
from snz.data.manifest import read_manifest
from snz.eval.ap import APCalculator
from snz.eval.bootstrap import bootstrap_weights, holm, one_sided_p, percentile_ci
from snz.eval.matching import match_split
from snz.eval.operating import false_detections, recall_at, threshold_for_recall
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


def analyze(runs_root: Path, val_labels: Path, manifests_dir: Path, n_boot: int = 2000,
            seed: int = 0) -> dict:
    runs = _discover(runs_root)
    if "BASE-s0" not in runs:
        raise FileNotFoundError(f"BASE-s0 detections not found under {runs_root}; the "
                                "operating point R* is defined by BASE seed 0")
    seeds = sorted({s for _, s, _ in runs.values()})
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

    # paired bootstrap: one weight matrix per image set, shared by all runs
    w_clean = bootstrap_weights(len(names["t-clean"]), n_boot, seed)
    w_shift = bootstrap_weights(len(shift_names), n_boot, seed + 1)
    boot = {}
    for run, sub in matches.items():
        cc, ct, cs = (APCalculator(sub[s]) for s in ("t-clean", "t-texture", "t-shift"))
        b = {k: np.empty(n_boot) for k in METRICS}
        for i in range(n_boot):
            b["clean_mAP"][i] = cc(w_clean[i])["mAP"]
            b["texture_mAP"][i] = ct(w_clean[i])["mAP"]
            b["shift_mAP"][i] = cs(w_shift[i])["mAP"]
            b["shift_fd"][i] = false_detections(sub["t-shift"], ops[run].threshold,
                                                w_shift[i])["per_image"]
        with np.errstate(divide="ignore", invalid="ignore"):
            b["retention"] = b["texture_mAP"] / b["clean_mAP"]
        boot[run] = b

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
    tested = {h: c["p"] for h, c in confirmatory.items() if c["status"] == "ok"}
    for h, p_adj in holm(tested).items():
        confirmatory[h].update(p_holm=p_adj, supported=bool(p_adj < ALPHA))

    exploratory = {f"{arm}_vs_BASE_{metric}": compare(arm, metric, None)
                   for arm in ("STY", "SIN", "BOTH") for metric in METRICS}
    if all(a in arm_boot for a in ARMS):
        add = (arm_boot["BOTH"]["shift_mAP"] - arm_boot["BASE"]["shift_mAP"]
               - (arm_boot["STY"]["shift_mAP"] - arm_boot["BASE"]["shift_mAP"])
               - (arm_boot["SIN"]["shift_mAP"] - arm_boot["BASE"]["shift_mAP"]))
        est = (arm_point["BOTH"]["shift_mAP"] - arm_point["STY"]["shift_mAP"]
               - arm_point["SIN"]["shift_mAP"] + arm_point["BASE"]["shift_mAP"])
        exploratory["additivity_shift_mAP"] = {"estimate": est, "ci95": percentile_ci(add)}
    else:
        exploratory["additivity_shift_mAP"] = {"status": "missing arm"}

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
        "exploratory": exploratory,
        "arms": arms,
        "t_corrupt": t_corrupt,
    }
