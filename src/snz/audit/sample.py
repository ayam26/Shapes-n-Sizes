"""Blind sample of footage detections for the precision audit (spec §6).

Per arm, `n_per_arm` detections are drawn uniformly from all detections at or above that
arm's full-val equal-recall threshold. All items are shuffled together; an item shows only
the frame, box and predicted class. The key (item -> arm) is kept separately until
unblinding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from snz.data.bdd import EVAL_CLASSES


def count_detections(records: Sequence[dict], threshold: float) -> int:
    """Detections of the 9 evaluated classes (as on BDD, `train` is left out)."""
    return sum(s >= threshold and c in EVAL_CLASSES
               for r in records for s, c in zip(r["scores"], r["labels"], strict=True))


def sample_audit(
    dets_by_arm: Mapping[str, Sequence[dict]],
    thresholds: Mapping[str, float],
    n_per_arm: int = 150,
    seed: int = 0,
) -> tuple[list[dict], dict[str, dict]]:
    rng = np.random.default_rng(seed)
    pool = []
    for arm in sorted(dets_by_arm):
        cands = [(r, j) for r in dets_by_arm[arm] for j, s in enumerate(r["scores"])
                 if s >= thresholds[arm] and r["labels"][j] in EVAL_CLASSES]
        pick = rng.choice(len(cands), size=min(n_per_arm, len(cands)), replace=False)
        pool += [(arm, *cands[i]) for i in sorted(pick)]
    items, key = [], {}
    for n, idx in enumerate(rng.permutation(len(pool))):
        arm, rec, j = pool[idx]
        item_id = f"a{n:03d}"
        items.append({"id": item_id, "frame": rec["name"], "box": rec["boxes"][j],
                      "class": rec["labels"][j]})
        key[item_id] = {"arm": arm, "frame": rec["name"], "det": int(j)}
    return items, key
