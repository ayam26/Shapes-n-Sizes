#!/usr/bin/env bash
# G6: test-split predictions for every finished run, then the full analysis.
# Blocked by `snz evaluate` until the preregistration is tagged prereg-v1 (G4).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
D="${SNZ_DATA:-$HOME/snz-data}"
L="$D/bdd100k/labels/det_2018"
for RUN in "$D"/runs/{BASE,STY,SIN,BOTH}-s*; do
  [ -f "$RUN/run.json" ] || continue
  EV="$RUN/eval"
  [ -f "$EV/val-all.detections.jsonl" ] || uv run snz evaluate --split val-all \
    --labels "$L/det_val.json" --manifest manifests/val-all.csv \
    --images "$D/bdd100k/images/100k/val" --checkpoint "$RUN/final.pt" --out "$EV"
  [ -f "$EV/t-texture.detections.jsonl" ] || uv run snz evaluate --split t-texture \
    --labels "$L/det_val.json" --manifest manifests/t-clean.csv \
    --images "$D/stylized/t-texture" --stylized-copy 1 --checkpoint "$RUN/final.pt" --out "$EV"
done
# T-corrupt (exploratory): seed-0 runs, 15 corruptions x severities 1/3/5 on a 300-image subset
[ -f manifests/t-corrupt.csv ] || uv run snz corrupt --images "$D/bdd100k/images/100k/val" \
  --out "$D/t-corrupt"
CORRUPTIONS=$(uv run python -c "from snz.corrupt import CORRUPTIONS; print(*CORRUPTIONS)")
for RUN in "$D"/runs/{BASE,STY,SIN,BOTH}-s0; do
  [ -f "$RUN/run.json" ] || continue
  for C in $CORRUPTIONS; do for S in 1 3 5; do
    [ -f "$RUN/eval/t-corrupt-$C-$S.detections.jsonl" ] || uv run snz evaluate \
      --split "t-corrupt-$C-$S" --labels "$L/det_val.json" --manifest manifests/t-corrupt.csv \
      --images "$D/t-corrupt" --corruption "$C" --severity "$S" \
      --checkpoint "$RUN/final.pt" --out "$RUN/eval"
  done; done
done
uv run snz analyze --runs "$D/runs" --labels "$L/det_val.json" --manifests manifests \
  --out results
