# Handoff — state on 2026-10-11

## Done

| Gate | What | Where |
|---|---|---|
| G1 | GX10 smoke test: both backbones load, 50 iterations at 1280×720 | `results/g1_smoke.json` |
| G2 | BDD100K 100K images + **2018 labels** (official servers offline; maintainer mirror `dl.yf.io`); splits with content-hash disjointness; 22,458 stylized training copies; 1,764 T-texture images | `manifests/`, `prereg/prereg.md` §3 |
| G3 | Pilot (BASE seed 0, id-val only): lr 0.01 → 24.5 mAP, lr 0.02 → 25.9; recipe frozen at lr 0.02, 18,000 iterations, batch 8 (~4.2 h/run) | `configs/recipe.toml`, `results/pilot/` |
| G4 | Preregistration frozen, git tag `prereg-v1` | `prereg/prereg.md` |
| G7 (part) | Open-source footage instead of own drives (prereg amendment): comma2k19 Chunk_1 (MIT), 188 segments, every 5 s → 2,270 frames, 0 dropped; 100 frames drawn for hand labeling (seed 0) | `manifests/footage/` |
| G8 (part) | Paper draft (intro, methods, limitations); figure code (PR curves, subset dot plot) | `paper/paper.md`, `src/snz/eval/figures.py` |

Code: `snz` package, 133 tests (`uv run pytest`); three implementation plans, each with a fresh whole-branch review whose Critical/Important findings were fixed test-first (`docs/superpowers/plans/`).

## Running

**G5** on the GX10 (`asus@100.104.225.90`): `scripts/run_g5.sh`, log `~/snz-data/g5.log`, runs in
`~/snz-data/runs/<ARM>-s<seed>/`. 12 runs × ~4.2 h ≈ 2 days from 2026-10-11 06:45 PDT.
If it stops for any reason, rerun the script: finished runs are skipped, interrupted ones resume.

## To do

1. **Label the 100 footage frames now (person, not a model).** Frames are in `data/label_frames/`
   on the Mac (git-ignored; list in `manifests/footage/label_frames.txt`). Draw a box around every
   object of the 9 classes (pedestrian, rider, car, truck, bus, motorcycle, bicycle, traffic
   light, traffic sign) with CVAT or Label Studio, **no model pre-annotation**, and export COCO
   JSON with these category names.
2. **G6** after G5: `scripts/run_g6.sh` on the GX10 (val-all and T-texture predictions, T-corrupt,
   `snz analyze`), then commit `results/analysis.json`, `results/report.md`, `results/figures/`.
3. **G7** after G6 (runbook steps 3–6): `footage-predict` for each arm's seed-0 run;
   `audit-sample` builds the blind page (600 items) and a key file kept away from the auditor;
   **a person labels the 600 items** in the browser and downloads the CSV; then `audit-score`
   with that CSV and the COCO labels from step 1.
4. **G8**: fill the paper's results sections from `results/report.md` and `results/footage.json`.

Full commands: `docs/runbook.md`. Decisions made along the way: commit messages and the
deviations log in `prereg/prereg.md` §7.
