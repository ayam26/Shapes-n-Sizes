"""Paper figures from `analyze()` output (static PDF + PNG).

Arms keep one color each everywhere (color follows the arm), plus a second encoding
(line style / marker shape) so identity never rests on color alone. Palette: validated
categorical slots 1-4 (CVD-safe; contrast relief comes from legends, direct labels and the
paper's tables).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARM_STYLE = {
    "BASE": {"color": "#2a78d6", "linestyle": "-", "marker": "o"},
    "STY": {"color": "#eb6834", "linestyle": "--", "marker": "s"},
    "SIN": {"color": "#1baf7a", "linestyle": "-.", "marker": "D"},
    "BOTH": {"color": "#eda100", "linestyle": ":", "marker": "^"},
}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PR_TITLES = {"t-clean": "T-clean (daytime, clear)", "t-shift": "T-shift (night/rain/snow/fog)",
             "t-texture": "T-texture (held-out styles)", "t-night": "Night"}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    paths = []
    for ext in ("pdf", "png"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def pr_figure(r: dict, out_dir: Path) -> list[Path]:
    subsets = [s for s in PR_TITLES if s in r["pr_curves"]]
    fig, axes = plt.subplots(1, len(subsets), figsize=(3.1 * len(subsets), 3.0), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    recall = r["pr_recall"]
    for ax, s in zip(axes, subsets, strict=True):
        _style(ax)
        for arm, curve in r["pr_curves"][s].items():
            st = ARM_STYLE[arm]
            ax.plot(recall, curve, color=st["color"], linestyle=st["linestyle"], linewidth=1.5,
                    label=arm)
        ax.set_title(PR_TITLES[s], fontsize=9, color=INK)
        ax.set_xlabel("Recall (IoU ≥ 0.5)", fontsize=8, color=INK2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Precision (mean over classes)", fontsize=8, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 1.08), labelcolor=INK)
    return _save(fig, out_dir, "pr_curves")


def subset_figure(r: dict, out_dir: Path) -> list[Path]:
    diffs = r["exploratory"]["subset_mAP_vs_BASE"]
    order = ["t-clean", "t-shift", "t-night", "t-rain", "t-snow", "t-fog", "t-dusk", "t-overcast"]
    subsets = [s for s in order if s in diffs]  # pooled T-shift right after T-clean
    arms = [a for a in ("STY", "SIN", "BOTH") if any(a in diffs[s] for s in subsets)]
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(subsets) + 1.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.axvline(0, color=INK2, linewidth=1)
    for j, arm in enumerate(arms):
        st = ARM_STYLE[arm]
        offset = (j - (len(arms) - 1) / 2) * 0.22
        for i, s in enumerate(subsets):
            c = diffs[s].get(arm)
            if not c:
                continue
            y = i + offset
            lo, hi = (100 * v for v in c["ci95"])
            ax.plot([lo, hi], [y, y], color=st["color"], linewidth=1.5)
            ax.plot(100 * c["estimate"], y, marker=st["marker"], markersize=6,
                    color=st["color"], markeredgecolor=SURFACE, markeredgewidth=1,
                    linestyle="none", label=arm if i == 0 else None)
    ax.set_yticks(range(len(subsets)), [f"{s} (n={r['images'].get(s, '?')})" for s in subsets])
    ax.invert_yaxis()
    ax.set_xlabel("mAP difference vs. BASE (points, 95% bootstrap CI)", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=len(arms), labelcolor=INK)
    return _save(fig, out_dir, "subset_diffs")


def make_figures(r: dict, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return pr_figure(r, out_dir) + subset_figure(r, out_dir)
