# GX10 runbook: gates G1–G3

All commands run on the GX10 from the repository root. Data lives in `$SNZ_DATA`
(default `~/snz-data`) and never enters git.

## G1 — smoke test

```bash
scripts/gx10_setup.sh
scripts/fetch_weights.sh
uv run pytest
uv run snz smoke --sin-weights ~/snz-data/weights/resnet50_train_60_epochs-c8e5653e.pth.tar \
  --work-dir ~/snz-data/smoke
```

Commit `results/g1_smoke.json`.

## G2 — data

1. Read the BDD100K terms and the DTD terms; note the date in the commit message.
2. Download and build. The official BDD hosts were offline in 2026; `fetch_bdd.sh`
   uses the maintainer's Berkeley mirror (`dl.yf.io`) and the **2018 detection labels**
   (det_20 was not obtainable), converted to `labels/det_2018/det_{train,val}.json`.

```bash
BDD_TERMS_ACCEPTED=yes scripts/fetch_bdd.sh
scripts/fetch_styles.sh
D=~/snz-data; W=$D/weights
uv run snz build-splits \
  --train-labels $D/bdd100k/labels/det_2018/det_train.json \
  --val-labels   $D/bdd100k/labels/det_2018/det_val.json \
  --train-images $D/bdd100k/images/100k/train --val-images $D/bdd100k/images/100k/val
uv run snz stylize --set train --images $D/bdd100k/images/100k/train \
  --styles-root $D/dtd/images --out $D/stylized/train \
  --vgg $W/vgg_normalised.pth --decoder $W/decoder.pth
uv run snz stylize --set t-texture --images $D/bdd100k/images/100k/val \
  --styles-root $D/dtd/images --out $D/stylized/t-texture \
  --vgg $W/vgg_normalised.pth --decoder $W/decoder.pth
```

Commit `manifests/`. Both `stylize` runs can be stopped and rerun; finished images are skipped.

## G3 — pilot (BASE, seed 0, id-val only)

```bash
D=~/snz-data
for LR in 0.01 0.02; do
  uv run snz train --pilot --arm BASE --seed 0 --lr $LR --iterations 6000 \
    --labels $D/bdd100k/labels/det_2018/det_train.json --images $D/bdd100k/images/100k/train \
    --out $D/runs/pilot/lr$LR
  uv run snz evaluate --split id-val --labels $D/bdd100k/labels/det_2018/det_train.json \
    --manifest manifests/id-val.csv --images $D/bdd100k/images/100k/train \
    --checkpoint $D/runs/pilot/lr$LR/final.pt --out results/pilot/lr$LR
done
```

Pick the learning rate with the higher id-val mAP. From `iters_per_sec_this_session` in
each `run.json`, choose `iterations` so that 12 runs fit the budget (spec §5.4). If the
budget is too large, reduce the schedule or input scale **identically for all arms**.
Then edit `configs/recipe.toml`, set `frozen = true`, and commit `G3: recipe frozen`
with `results/pilot/*/id-val.metrics.json` (not the detection files).
Only id-val is evaluated here; test splits stay blocked until `prereg-v1`.

A run that stops partway (power, SSH drop) resumes when the same command is run again.

## G4 — preregistration

After G3: fill in `prereg/prereg.md` (frozen recipe hash, split counts), commit, then
`git tag prereg-v1 && git push origin prereg-v1`. Only now do test splits open.

## G5 — the 12 runs

```bash
nohup scripts/run_g5.sh > ~/snz-data/g5.log 2>&1 < /dev/null &
```

Seed 0 of every arm runs first. Rerunning the script skips finished runs and resumes an
interrupted one.

## G6 — evaluation and analysis

```bash
scripts/run_g6.sh   # predictions on val-all and T-texture per run, then `snz analyze`
```

Commit `results/analysis.json` and `results/report.md` (not the detection files).
