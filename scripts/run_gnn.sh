#!/usr/bin/env bash
# GNN shape verifier stages on the GX10 (Plan 4): scripts/run_gnn.sh <stage>
# Stages: outlines | train | rescore | analyze. Each stage skips outputs that already exist.
# One-time: SAM2_BUILD_CUDA=0 uv pip install sam2 huggingface_hub   (sam2 is sdist-only and
# needs torch to build, so it is installed into the venv rather than locked in uv.lock).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
D="${SNZ_DATA:-$HOME/snz-data}"
L="$D/bdd100k/labels/det_2018"
B="$D/runs/BASE-s0/eval"
G="$D/gnn"
SEG="${SNZ_SEGMENTER:-sam2}"
mkdir -p "$G/outlines" "$G/models" "$G/runs"
case "${1:-}" in
  outlines)
    [ -f "$G/outlines/id-val.npz" ] || uv run snz gnn-outlines --segmenter "$SEG" \
      --detections "$B/id-val.detections.jsonl" --images "$D/bdd100k/images/100k/train" \
      --out "$G/outlines/id-val.npz"
    [ -f "$G/outlines/val-all.npz" ] || uv run snz gnn-outlines --segmenter "$SEG" \
      --detections "$B/val-all.detections.jsonl" --images "$D/bdd100k/images/100k/val" \
      --out "$G/outlines/val-all.npz"
    [ -f "$G/outlines/t-texture.npz" ] || uv run snz gnn-outlines --segmenter "$SEG" \
      --detections "$B/t-texture.detections.jsonl" --images "$D/stylized/t-texture" \
      --stylized-copy 1 --out "$G/outlines/t-texture.npz"
    ;;
  *)
    echo "usage: $0 outlines|train|rescore|analyze" >&2; exit 2 ;;
esac
