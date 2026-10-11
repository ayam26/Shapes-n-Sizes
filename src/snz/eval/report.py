"""Markdown report from `analyze()` output (spec §7.4: every subset, counts, regressions)."""

from __future__ import annotations

from snz.eval.analyze import ARMS, CONFIRMATORY, REPORTED


def _n(x, digits=3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _ms(s: dict, scale=100.0, digits=1) -> str:
    if not s or s["n"] == 0:
        return "—"
    spread = f" ± {s['sd'] * scale:.{digits}f}" if s["n"] > 1 else ""
    return f"{s['mean'] * scale:.{digits}f}{spread} (n={s['n']})"


def render_report(r: dict) -> str:
    lines = ["# Results", ""]
    lines += [f"Runs: {', '.join(r['runs'])}.",
              f"Missing runs: {', '.join(r['missing_runs']) or 'none'}.",
              f"Paired bootstrap: B = {r['n_boot']} (seed {r['bootstrap_seed']}).", ""]
    lines += ["## Confirmatory tests (Holm-corrected, α = 0.05)", "",
              "| Hypothesis | Arm vs BASE | Metric | Estimate | 95% CI | p | p (Holm) "
              "| Supported |",
              "|---|---|---|---|---|---|---|---|"]
    for h, (arm, metric, direction) in CONFIRMATORY.items():
        c = r["confirmatory"][h]
        if c.get("status") != "ok":
            lines.append(f"| {h} | {arm} | {metric} | — | — | — | — | {c.get('status')} |")
            continue
        lo, hi = c["ci95"]
        lines.append(f"| {h} | {arm} ({direction}) | {metric} | {_n(c['estimate'], 4)} | "
                     f"[{_n(lo, 4)}, {_n(hi, 4)}] | {_n(c['p'], 4)} | {_n(c['p_holm'], 4)} | "
                     f"{'yes' if c['supported'] else 'no'} |")
    lines += ["", "Metrics: retention = mAP(T-texture) / mAP(T-clean); shift_mAP = mAP on pooled "
              "T-shift; shift_fd = false detections per image on T-shift at equal recall. "
              f"Holm family: {r['holm_family_size']} tests (missing ones enter with p = 1). "
              f"Equal-recall thresholds in the bootstrap: {r['thresholds_in_bootstrap']}.", ""]
    h1a = r["descriptive"].get("H1a_BASE_retention")
    if h1a:
        lo, hi = h1a["ci95"]
        lines += [f"H1a (descriptive): BASE retention = {_n(h1a['estimate'], 3)} "
                  f"[{_n(lo, 3)}, {_n(hi, 3)}].", ""]

    op = r["operating_point"]
    lines += ["## Equal-recall operating point", "",
              f"R* = {_n(op['R_star'], 4)} ({op['definition']}).", "",
              "| Run | Threshold | Recall | Flagged |", "|---|---|---|---|"]
    for run, o in r["thresholds"]["t-shift"].items():
        lines.append(f"| {run} | {_n(o['threshold'], 4)} | {_n(o['recall'], 4)} | "
                     f"{'yes' if o['flagged'] else ''} |")
    lines += ["", "## mAP by subset (×100, mean ± SD over seeds)", "",
              "| Subset | Images | " + " | ".join(ARMS) + " |",
              "|---|---|" + "---|" * len(ARMS)]
    for s in REPORTED:
        images = next(iter(r["arms"].values()))["subsets"][s]["images"]
        cells = [_ms(r["arms"][a]["subsets"][s]["mAP"]) if a in r["arms"] else "—" for a in ARMS]
        lines.append(f"| {s} | {images} | " + " | ".join(cells) + " |")
    lines += ["", "## AP50 by subset (×100)", "", "| Subset | " + " | ".join(ARMS) + " |",
              "|---|" + "---|" * len(ARMS)]
    for s in REPORTED:
        cells = [_ms(r["arms"][a]["subsets"][s]["AP50"]) if a in r["arms"] else "—" for a in ARMS]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines += ["", "## Retention and false detections on T-shift", "",
              "| Arm | Retention | False det./image | " + " | ".join(
                  next(iter(r["arms"].values()))["shift_fd_by_type"]) + " |",
              "|---|---|---|" + "---|" * len(next(iter(r["arms"].values()))["shift_fd_by_type"])]
    for a in ARMS:
        if a not in r["arms"]:
            continue
        arm = r["arms"][a]
        types = " | ".join(_ms(v, 1.0, 3) for v in arm["shift_fd_by_type"].values())
        lines.append(f"| {a} | {_ms(arm['retention'], 1.0, 3)} | "
                     f"{_ms(arm['shift_fd_per_image'], 1.0, 3)} | {types} |")
    lines += ["", "## arm − BASE mAP by subset (×100, 95% bootstrap CI; descriptive)", "",
              "| Subset | STY | SIN | BOTH |", "|---|---|---|---|"]
    for s_, by_arm in r["exploratory"]["subset_mAP_vs_BASE"].items():
        cells = []
        for arm in ("STY", "SIN", "BOTH"):
            c = by_arm.get(arm)
            cells.append("—" if not c else f"{c['estimate'] * 100:+.1f} "
                         f"[{c['ci95'][0] * 100:+.1f}, {c['ci95'][1] * 100:+.1f}]")
        lines.append(f"| {s_} | " + " | ".join(cells) + " |")
    lines += ["", "## Exploratory (CIs, no tests)", "", "| Comparison | Estimate | 95% CI |",
              "|---|---|---|"]
    for name, c in r["exploratory"].items():
        if name == "subset_mAP_vs_BASE":
            continue
        if "ci95" not in c:
            lines.append(f"| {name} | — | {c.get('status', '')} |")
            continue
        lo, hi = c["ci95"]
        lines.append(f"| {name} | {_n(c['estimate'], 4)} | [{_n(lo, 4)}, {_n(hi, 4)}] |")
    if r.get("t_corrupt"):
        lines += ["", "## T-corrupt (exploratory, seed-0 runs)", "",
                  "mPC = mean mAP over corruptions × severities; rPC = mPC / clean mAP on the "
                  "same image subset (×100 for mAP).", "",
                  "| Run | Images | Clean mAP | mPC | rPC |", "|---|---|---|---|---|"]
        for run, c in r["t_corrupt"].items():
            lines.append(f"| {run} | {c['images']} | {_n(c['clean_mAP'] * 100, 1)} | "
                         f"{_n(c['mPC'] * 100, 1)} | {_n(c['rPC'], 3)} |")
    if r.get("coco_check"):
        bad = [f"{run}/{s_}" for run, d in r["coco_check"].items() for s_, c in d.items()
               if not c["agrees"]]
        lines += ["", f"pycocotools cross-check: {sum(len(d) for d in r['coco_check'].values())} "
                  f"comparisons, disagreements > 0.1 mAP points: {', '.join(bad) or 'none'}."]
    prov = r.get("provenance", {})
    lines += ["", f"Code: {prov.get('git_commit', '?')}; prereg-v1 tag present: "
              f"{prov.get('prereg_tag')}."]
    lines += ["", "## Footage thresholds (full-val equal recall)", "",
              f"R*_val = {_n(r['footage_R_star'], 4)}.", "", "| Run | Threshold | Flagged |",
              "|---|---|---|"]
    for run, o in r["footage_thresholds"].items():
        lines.append(f"| {run} | {_n(o['threshold'], 4)} | {'yes' if o['flagged'] else ''} |")
    return "\n".join(lines) + "\n"
