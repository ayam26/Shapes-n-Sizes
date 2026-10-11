#!/usr/bin/env bash
# G5: the 12 training runs (4 arms x seeds 0-2) with the frozen recipe.
# Seed-major order keeps partial results balanced across arms. Safe to rerun: finished
# runs (run.json present) are skipped, interrupted runs resume from last.pt.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
D="${SNZ_DATA:-$HOME/snz-data}"
L="$D/bdd100k/labels/det_2018"
for SEED in 0 1 2; do
  for ARM in BASE STY SIN BOTH; do
    OUT="$D/runs/$ARM-s$SEED"
    [ -f "$OUT/run.json" ] && { echo "skip $ARM-s$SEED (finished)"; continue; }
    echo "=== $ARM-s$SEED $(date -Is)"
    uv run snz train --arm "$ARM" --seed "$SEED" --labels "$L/det_train.json" \
      --images "$D/bdd100k/images/100k/train" --stylized "$D/stylized/train" \
      --sin-weights "$D/weights/resnet50_train_60_epochs-c8e5653e.pth.tar" --out "$OUT"
  done
done
echo "G5_DONE $(date -Is)"
