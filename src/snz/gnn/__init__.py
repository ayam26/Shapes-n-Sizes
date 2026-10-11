"""Texture-blind GNN shape verifier (Plan 4, spec 2026-10-11-gnn-shape-verifier-design.md)."""

from snz.data.bdd import EVAL_CLASSES

GNN_CLASSES = (*EVAL_CLASSES, "background")
BACKGROUND = len(EVAL_CLASSES)  # index 9
ARMS_G = ("BASE", "GNN", "RGB", "GNN-SHAPE", "GNN-COLOR")
GNN_PREREG_TAG = "prereg-v2"
