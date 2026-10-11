#!/usr/bin/env bash
# Time-boxed G5+G6+G7 pipeline (prereg amendment 2026-10-11): one seed per arm, each arm
# trained then scored right away (BASE, STY, SIN first: the confirmatory arms), then the
# analysis, footage predictions and the blind audit page. Safe to rerun: finished steps skip.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
D="${SNZ_DATA:-$HOME/snz-data}"
L="$D/bdd100k/labels/det_2018"
for ARM in BASE STY SIN BOTH; do
  RUN="$D/runs/$ARM-s0"; EV="$RUN/eval"
  echo "=== $ARM-s0 train $(date -Is)"
  [ -f "$RUN/run.json" ] || uv run snz train --arm "$ARM" --seed 0 \
    --labels "$L/det_train.json" --images "$D/bdd100k/images/100k/train" \
    --stylized "$D/stylized/train" \
    --sin-weights "$D/weights/resnet50_train_60_epochs-c8e5653e.pth.tar" --out "$RUN"
  echo "=== $ARM-s0 eval $(date -Is)"
  [ -f "$EV/val-all.detections.jsonl" ] || uv run snz evaluate --split val-all \
    --labels "$L/det_val.json" --manifest manifests/val-all.csv \
    --images "$D/bdd100k/images/100k/val" --checkpoint "$RUN/final.pt" --out "$EV"
  [ -f "$EV/t-texture.detections.jsonl" ] || uv run snz evaluate --split t-texture \
    --labels "$L/det_val.json" --manifest manifests/t-clean.csv \
    --images "$D/stylized/t-texture" --stylized-copy 1 --checkpoint "$RUN/final.pt" --out "$EV"
done
echo "=== analyze $(date -Is)"
uv run snz analyze --runs "$D/runs" --labels "$L/det_val.json" --manifests manifests \
  --out results --seeds 0
for ARM in BASE STY SIN BOTH; do
  RUN="$D/runs/$ARM-s0"
  [ -f "$RUN/eval/footage.detections.jsonl" ] || uv run snz footage-predict \
    --manifest "$D/footage/footage.csv" --frames "$D/footage/frames" \
    --checkpoint "$RUN/final.pt"
done
[ -f "$D/audit_key.json" ] || uv run snz audit-sample --runs "$D/runs" \
  --frames "$D/footage/frames" --out "$D/audit" --key "$D/audit_key.json"
echo "FAST_DONE $(date -Is)"
