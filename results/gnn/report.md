# GNN shape verifier — results

Arms: BASE, GNN-COLOR, GNN-SHAPE, GNN, RGB; not run: none. Paired bootstrap B = 2000 over 4752 T-shift images.

R* = 0.5477 (BASE-s0 recall on t-shift at score ≥ 0.5); F* = 10.1854 false detections per image (BASE-s0 at score ≥ 0.5).

## Confirmatory tests (Holm over G1–G3, α = 0.05)

| Test | Comparison | Metric | Estimate | 95% CI | p | p (Holm) | Supported |
|---|---|---|---|---|---|---|---|
| G1 | GNN vs BASE (less) | fd_at_R | -0.3453 | [-0.4289, -0.2363] | 0.0005 | 0.0015 | yes |
| G2 | GNN vs BASE (greater) | recall_at_F | 0.0036 | [0.0026, 0.0045] | 0.0005 | 0.0015 | yes |
| G3 | GNN vs RGB (less) | fd_at_R | 0.3093 | [0.1601, 0.4472] | 1.0000 | 1.0000 | no |

## Per arm (t-shift unless stated)

| Arm | False det./img at R* | Recall at F* | t-shift mAP | t-clean mAP | T-texture retention |
|---|---|---|---|---|---|
| BASE | 10.185 | 0.548 | 13.8 | 18.6 | 0.121 |
| GNN-COLOR | 9.961 | 0.550 | 14.0 | 18.9 | 0.127 |
| GNN-SHAPE | 9.875 | 0.551 | 14.0 | 18.9 | 0.127 |
| GNN | 9.839 | 0.551 | 14.1 | 19.0 | 0.124 |
| RGB | 9.530 | 0.554 | 14.4 | 19.6 | 0.117 |

## False detections per image at R*, by type

| Arm | duplicate | wrong_class | poor_localization | invented | near_other_class |
|---|---|---|---|---|---|
| BASE | 0.578 | 0.282 | 5.209 | 3.886 | 0.230 |
| GNN-COLOR | 0.603 | 0.362 | 5.071 | 3.691 | 0.234 |
| GNN-SHAPE | 0.594 | 0.335 | 5.076 | 3.641 | 0.229 |
| GNN | 0.594 | 0.363 | 5.058 | 3.593 | 0.230 |
| RGB | 0.707 | 0.304 | 5.316 | 3.009 | 0.193 |

## mAP by subset (×100)

| Subset | BASE | GNN-COLOR | GNN-SHAPE | GNN | RGB |
|---|---|---|---|---|---|
| t-clean | 18.6 | 18.9 | 18.9 | 19.0 | 19.6 |
| t-night | 12.4 | 12.6 | 12.7 | 12.7 | 12.8 |
| t-rain | 19.5 | 19.7 | 19.6 | 19.7 | 20.0 |
| t-snow | 17.1 | 17.0 | 17.1 | 17.0 | 18.0 |
| t-fog | 23.2 | 23.2 | 22.1 | 21.9 | 23.9 |
| t-dusk | 16.5 | 16.8 | 16.7 | 16.8 | 17.9 |
| t-overcast | 18.6 | 18.8 | 18.8 | 18.8 | 19.5 |
| t-shift | 13.8 | 14.0 | 14.0 | 14.1 | 14.4 |
| val-all | 16.6 | 16.9 | 16.9 | 16.9 | 17.6 |
| t-texture | 2.3 | 2.4 | 2.4 | 2.3 | 2.3 |

## Descriptive comparisons (95% CIs, no tests)

| Comparison | Metric | Estimate | 95% CI |
|---|---|---|---|
| GNN-minus-GNN-SHAPE | fd_at_R | -0.0356 | [-0.0903, 0.0411] |
| GNN-minus-GNN-SHAPE | recall_at_F | 0.0005 | [-0.0001, 0.0012] |
| GNN-minus-GNN-SHAPE | shift_mAP | 0.0004 | [-0.0008, 0.0019] |
| GNN-COLOR-minus-GNN | fd_at_R | 0.1221 | [0.0261, 0.1905] |
| GNN-COLOR-minus-GNN | recall_at_F | -0.0015 | [-0.0022, -0.0006] |
| GNN-COLOR-minus-GNN | shift_mAP | -0.0008 | [-0.0020, 0.0005] |
| RGB-minus-BASE | fd_at_R | -0.6547 | [-0.7768, -0.4937] |
| RGB-minus-BASE | recall_at_F | 0.0065 | [0.0050, 0.0079] |
| RGB-minus-BASE | shift_mAP | 0.0053 | [0.0029, 0.0079] |
| GNN-SHAPE-minus-BASE | fd_at_R | -0.3098 | [-0.4000, -0.2201] |
| GNN-SHAPE-minus-BASE | recall_at_F | 0.0031 | [0.0021, 0.0039] |
| GNN-SHAPE-minus-BASE | shift_mAP | 0.0019 | [0.0001, 0.0036] |

Outline success rate (SAM 2 mask usable) by subset: t-clean 1.000, t-night 1.000, t-rain 1.000, t-snow 1.000, t-fog 1.000, t-dusk 1.000, t-overcast 1.000, t-shift 1.000.

Code: 490f1a9cf6930f1eed92dfc295571822bd3b06c4.

## Unfiltered dashcam footage — blind audit (descriptive)

comma2k19 Chunk 1, 2,270 frames (every 5 s); thresholds = each arm's full-val equal-recall point
(R*_val = 0.578): BASE 0.500, GNN 0.184, RGB 0.174. 100 detections per arm, arm hidden; 299/300
labeled; unsure excluded; Wilson 95% CIs. Source: `results/gnn/footage.json`.

| Arm | Detections (per frame) | Precision | Invented share | Est. real | Est. invented |
|---|---|---|---|---|---|
| BASE | 26,821 (11.8) | 0.854 [0.766, 0.913] | 0.101 [0.054, 0.181] | 22,903 | 2,712 |
| GNN | 25,864 (11.4) | 0.946 [0.879, 0.977] | 0.033 [0.011, 0.092] | 24,458 | 843 |
| RGB | 25,205 (11.1) | 0.926 [0.854, 0.963] | 0.064 [0.030, 0.132] | 23,328 | 1,609 |
