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

After G3:

1. Rebuild the manifests with the current code (adds `t-dusk`, `t-overcast`, `val-all`) and
   check that the existing ones did not change:
   ```bash
   uv run snz build-splits --train-labels ... --val-labels ... --train-images ... --val-images ...
   git diff --exit-code manifests/train.csv manifests/id-val.csv manifests/t-clean.csv \
     manifests/t-night.csv manifests/t-rain.csv manifests/t-snow.csv manifests/t-fog.csv
   ```
2. Fill in `prereg/prereg.md` (frozen recipe hash, split counts), commit, then
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

## G7 — dashcam footage (after G6)

Footage is the open comma2k19 dataset (MIT), Chunk_1, every segment (prereg amendment):

```bash
cd ~/snz-data/footage-src
curl -fL -C - -o Chunk_1.zip https://huggingface.co/datasets/commaai/comma2k19/resolve/main/raw_data/Chunk_1.zip
unzip -q Chunk_1.zip
cd ~/Shapes-n-Sizes
uv run snz footage-drives-comma2k19 --root ~/snz-data/footage-src/Chunk_1 \
  --out ~/snz-data/footage-src/drives.csv
```

(Own drives instead: one folder with `drives.csv`: `drive_id,file,time_of_day,weather[,fps]`.)

1. `drives.csv` lists every drive of the collection period.
2. Sample frames (every N seconds, N chosen for ≥ 2,000 frames; nothing hand-picked):
   `uv run snz footage-sample --drives ~/snz-data/footage-src/drives.csv --out ~/snz-data/footage`
3. Predict with the seed-0 run of each arm:
   ```bash
   for ARM in BASE STY SIN BOTH; do uv run snz footage-predict \
     --manifest ~/snz-data/footage/footage.csv --frames ~/snz-data/footage/frames \
     --checkpoint ~/snz-data/runs/$ARM-s0/final.pt; done
   ```
4. Build the blind audit (600 items) and label it in a browser; keep the key closed:
   `uv run snz audit-sample --runs ~/snz-data/runs --frames ~/snz-data/footage/frames --out ~/snz-data/audit --key ~/snz-data/audit_key.json`
   Open `~/snz-data/audit/index.html`, label every item, click "Download CSV".
5. Draw the 100 frames to label (never pick them by hand):
   `uv run snz footage-label-sample --manifest ~/snz-data/footage/footage.csv --out label_frames.txt`
   Label every object of the 9 classes on those frames (CVAT or Label Studio, no model
   pre-annotation) and export COCO.
6. Unblind and score:
   `uv run snz audit-score --labels audit_labels.csv --key ~/snz-data/audit_key.json --runs ~/snz-data/runs --coco hand_labels.json`

## GNN shape verifier (Plan 4, `prereg-v2`)

One-time on the GX10: `SAM2_BUILD_CUDA=0 uv pip install sam2 huggingface_hub` (sam2 is not in
`uv.lock`; it is sdist-only and needs torch to build).

```bash
scripts/run_gnn.sh outlines   # SAM 2 outlines of BASE-s0 detections (id-val, t-texture, val-all; 4 shards)
scripts/run_gnn.sh train      # GNN, RGB, GNN-SHAPE, GNN-COLOR on id-val fold A; fusion on fold B
# fill prereg/prereg-v2.md from ~/snz-data/gnn/models/gnn-train.json, commit, tag prereg-v2
scripts/run_gnn.sh rescore    # val-all and t-texture, every arm (blocked until prereg-v2)
scripts/run_gnn.sh analyze    # results/gnn/analysis.json + report.md
```

Fallback segmenter: `SNZ_SEGMENTER=grabcut scripts/run_gnn.sh outlines`.
