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
    # each split in $SHARDS parallel SAM 2 processes, then merged (GPU has headroom)
    SHARDS="${SNZ_SHARDS:-4}"
    outline_split() {  # split detections images [extra args]
      local split=$1 det=$2 img=$3; shift 3
      [ -f "$G/outlines/$split.npz" ] && return
      local parts=()
      for k in $(seq 0 $((SHARDS - 1))); do
        parts+=("$G/outlines/$split.part$k.npz")
        [ -f "$G/outlines/$split.part$k.npz" ] || uv run snz gnn-outlines --segmenter "$SEG" \
          --detections "$det" --images "$img" --shard "$k/$SHARDS" \
          --out "$G/outlines/$split.part$k.npz" "$@" &
      done
      wait
      uv run snz gnn-merge-outlines --parts "${parts[@]}" --detections "$det" \
        --out "$G/outlines/$split.npz"
    }
    outline_split id-val "$B/id-val.detections.jsonl" "$D/bdd100k/images/100k/train"
    outline_split t-texture "$B/t-texture.detections.jsonl" "$D/stylized/t-texture" \
      --stylized-copy 1
    outline_split val-all "$B/val-all.detections.jsonl" "$D/bdd100k/images/100k/val"
    ;;
  train)
    # priority order (spec §3): GNN (G1, G2) -> RGB (G3) -> GNN-SHAPE -> GNN-COLOR
    [ -f "$G/models/gnn-train.json" ] && grep -q '"GNN-COLOR.pt"' "$G/models/gnn-train.json" ||
      uv run snz gnn-train --outlines "$G/outlines/id-val.npz" --labels "$L/det_train.json" \
        --manifest manifests/id-val.csv --images "$D/bdd100k/images/100k/train" \
        --out "$G/models" --arms "${SNZ_GNN_ARMS:-GNN,RGB,GNN-SHAPE,GNN-COLOR}" --fold-dir manifests
    ;;
  *)
    echo "usage: $0 outlines|train|rescore|analyze" >&2; exit 2 ;;
esac
