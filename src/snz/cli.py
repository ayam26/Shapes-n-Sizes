"""Command-line entry point: `snz <command>`."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import tomllib
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="snz", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("build-splits", help="write split manifests from BDD labels (G2)")
    p.add_argument("--train-labels", type=Path, required=True)
    p.add_argument("--val-labels", type=Path, required=True)
    p.add_argument("--train-images", type=Path, required=True)
    p.add_argument("--val-images", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("manifests"))
    p.add_argument("--config", type=Path, default=Path("configs/splits.toml"))
    p.set_defaults(func=_cmd_build_splits)

    p = commands.add_parser("convert-labels-2018",
                            help="BDD 2018 label zip -> det_train.json / det_val.json")
    p.add_argument("--zip", type=Path, required=True, help="bdd100k_labels.zip")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=_cmd_convert_2018)

    p = commands.add_parser("stylize", help="precompute stylized copies (G2)")
    p.add_argument("--set", choices=["train", "t-texture"], required=True)
    p.add_argument("--manifests", type=Path, default=Path("manifests"))
    p.add_argument("--images", type=Path, required=True, help="BDD train (for train) or val dir")
    p.add_argument("--styles-root", type=Path, required=True, help="e.g. dtd/images")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--vgg", type=Path, required=True)
    p.add_argument("--decoder", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/stylize.toml"))
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=_cmd_stylize)

    p = commands.add_parser("corrupt", help="T-corrupt (exploratory): corrupted T-clean subset")
    p.add_argument("--manifests", type=Path, default=Path("manifests"))
    p.add_argument("--images", type=Path, required=True, help="BDD val images")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--severities", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--corruptions", nargs="+")
    p.add_argument("--workers", type=int, default=16)
    p.set_defaults(func=_cmd_corrupt)

    p = commands.add_parser("train", help="train one arm/seed (pilot at G3, full runs at G5)")
    p.add_argument("--arm", choices=["BASE", "STY", "SIN", "BOTH"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--labels", type=Path, required=True, help="det_train.json")
    p.add_argument("--manifest", type=Path, default=Path("manifests/train.csv"))
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--stylized", type=Path, help="stylized/train (STY, BOTH)")
    p.add_argument("--sin-weights", type=Path, help="resnet50_train_60_epochs-c8e5653e.pth.tar")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--recipe", type=Path, default=Path("configs/recipe.toml"))
    p.add_argument("--arms", type=Path, default=Path("configs/arms.toml"))
    p.add_argument("--stylize-config", type=Path, default=Path("configs/stylize.toml"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--pilot", action="store_true", help="G3 pilot: BASE only, overrides allowed")
    p.add_argument("--iterations", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.set_defaults(func=_cmd_train)

    p = commands.add_parser("evaluate", help="predict + mAP on one split")
    p.add_argument("--split", required=True, help="id-val before G4; test splits after")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="<run>/final.pt")
    p.add_argument("--out", type=Path, required=True, help="directory for detections + metrics")
    p.add_argument("--stylized-copy", type=int, help="read <stem>_s<N>.jpg (T-texture: 1)")
    p.add_argument("--corruption", help="T-corrupt: read <stem>__<corruption>_<severity>.jpg")
    p.add_argument("--severity", type=int)
    p.add_argument("--no-guard", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=_cmd_evaluate)

    p = commands.add_parser("footage-sample", help="G7: unfiltered frames from dashcam drives")
    p.add_argument("--drives", type=Path, required=True,
                   help="CSV: drive_id,file,time_of_day,weather (file relative to the CSV)")
    p.add_argument("--out", type=Path, required=True, help="private dir (never in git)")
    p.add_argument("--min-frames", type=int, default=2000)
    p.set_defaults(func=_cmd_footage_sample)

    p = commands.add_parser("footage-drives-comma2k19",
                            help="G7: drives.csv for comma2k19 (every segment is a drive)")
    p.add_argument("--root", type=Path, required=True, help="extracted Chunk_N folder")
    p.add_argument("--out", type=Path, required=True, help="drives.csv (next to the chunk)")
    p.set_defaults(func=_cmd_comma2k19_drives)

    p = commands.add_parser("footage-predict", help="G7: one run's detections on the footage")
    p.add_argument("--manifest", type=Path, required=True, help="footage.csv")
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="<run>/final.pt")
    p.add_argument("--analysis", type=Path, default=Path("results/analysis.json"),
                   help="must exist: footage thresholds are fixed before footage is processed")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-guard", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(func=_cmd_footage_predict)

    p = commands.add_parser("footage-label-sample", help="G7: the 100 frames to hand-label")
    p.add_argument("--manifest", type=Path, required=True, help="footage.csv")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="text file, one frame per line")
    p.set_defaults(func=_cmd_footage_label_sample)

    p = commands.add_parser("audit-sample", help="G7: blind audit items, key and page")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--analysis", type=Path, default=Path("results/analysis.json"))
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="audit page directory")
    p.add_argument("--key", type=Path, required=True, help="arm key; keep away from the auditor")
    p.add_argument("--n", type=int, default=150, help="detections per arm")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_audit_sample)

    p = commands.add_parser("audit-score", help="G7: unblind and score the audit")
    p.add_argument("--labels", type=Path, required=True, help="audit_labels.csv")
    p.add_argument("--labels2", type=Path, help="second auditor (overlap) for Cohen's kappa")
    p.add_argument("--key", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--analysis", type=Path, default=Path("results/analysis.json"))
    p.add_argument("--coco", type=Path, help="hand-labeled frames (COCO export) for recall")
    p.add_argument("--out", type=Path, default=Path("results/footage.json"))
    p.set_defaults(func=_cmd_audit_score)

    p = commands.add_parser("analyze", help="G6: all analyses from per-run test detections")
    p.add_argument("--runs", type=Path, required=True, help="dir of <ARM>-s<seed>/eval/")
    p.add_argument("--labels", type=Path, required=True, help="det_val.json")
    p.add_argument("--manifests", type=Path, default=Path("manifests"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-guard", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(func=_cmd_analyze)

    p = commands.add_parser("smoke", help="G1: load both backbones, short training run")
    p.add_argument("--sin-weights", type=Path, required=True)
    p.add_argument("--recipe", type=Path, default=Path("configs/recipe.toml"))
    p.add_argument("--work-dir", type=Path, required=True, help="scratch dir outside git")
    p.add_argument("--out", type=Path, default=Path("results/g1_smoke.json"))
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=_cmd_smoke)

    return parser


def _cmd_build_splits(args) -> int:
    from snz.data.build import build_and_write_splits

    cfg = tomllib.loads(args.config.read_text())
    summary = build_and_write_splits(
        args.train_labels, args.val_labels, args.train_images, args.val_images,
        args.out, seed=cfg["seed"], idval_fraction=cfg["idval_fraction"],
    )
    print(json.dumps(summary["counts"], indent=2))
    return 0


def _cmd_convert_2018(args) -> int:
    from snz.data.bdd2018 import write_det20_layout

    print(json.dumps(write_det20_layout(args.zip, args.out)))
    return 0


def _cmd_stylize(args) -> int:
    from snz.stylize.adain import Stylizer
    from snz.stylize.build import load_stylize_config, run_stylize

    cfg = load_stylize_config(args.config)
    stylizer = Stylizer.from_weights(args.vgg, args.decoder, args.device, cfg.style_size,
                                     cfg.alpha)
    rows = run_stylize(args.set, args.manifests, args.images, args.styles_root, args.out, cfg,
                       stylizer)
    print(f"{len(rows)} stylized images in {args.out}")
    return 0


def _cmd_corrupt(args) -> int:
    from snz.corrupt import CORRUPTIONS, build_t_corrupt, choose_subset
    from snz.data.manifest import hash_rows, read_manifest, write_manifest

    names = choose_subset([n for n, _ in read_manifest(args.manifests / "t-clean.csv")],
                          args.k, args.seed)
    write_manifest(args.manifests / "t-corrupt.csv", hash_rows(names, args.images))
    rows = build_t_corrupt(names, args.images, args.out, args.corruptions or CORRUPTIONS,
                           args.severities, args.workers)
    print(f"{len(rows)} corrupted images in {args.out}")
    return 0


def _cmd_train(args) -> int:
    from snz.data.bdd import load_frames, select_frames
    from snz.data.manifest import read_manifest
    from snz.stylize.build import load_stylize_config
    from snz.train.config import load_arms, load_recipe
    from snz.train.model import load_backbone_state
    from snz.train.train import train_run

    if args.manifest.name != "train.csv":
        raise SystemExit(f"error: training reads manifests/train.csv only, not {args.manifest}")
    recipe = load_recipe(args.recipe)
    overrides = {k: v for k, v in (("iterations", args.iterations), ("lr", args.lr),
                                   ("batch_size", args.batch_size)) if v is not None}
    if args.pilot:
        if args.arm != "BASE":
            raise SystemExit("error: the pilot trains the BASE arm only (spec §5.4)")
        recipe = dataclasses.replace(recipe, **overrides)
    elif overrides:
        raise SystemExit("error: --iterations/--lr/--batch-size are pilot-only; edit the recipe")
    elif not recipe.frozen:
        raise SystemExit("error: configs/recipe.toml is not frozen (gate G3); use --pilot")
    arm = load_arms(args.arms)[args.arm]
    if arm.stylized and args.stylized is None:
        raise SystemExit(f"error: arm {arm.name} needs --stylized")
    frames, _ = load_frames(args.labels)
    frames = select_frames(frames, [n for n, _ in read_manifest(args.manifest)])
    summary = train_run(
        arm=arm, seed=args.seed, recipe=recipe, frames=frames, image_dir=args.images,
        stylized_dir=args.stylized,
        n_copies=load_stylize_config(args.stylize_config).train_copies,
        backbone_state=load_backbone_state(arm.backbone, args.sin_weights),
        out_dir=args.out, device=args.device, pilot=args.pilot,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_evaluate(args) -> int:
    import torch

    from snz.data.bdd import load_frames, select_frames
    from snz.data.manifest import read_manifest
    from snz.eval.coco_eval import evaluate
    from snz.eval.guard import ensure_split_allowed
    from snz.eval.predict import predict, write_detections
    from snz.train.config import Recipe
    from snz.train.model import build_model

    if not args.no_guard:  # tests only
        ensure_split_allowed(args.split, manifest=args.manifest)
    run_json = args.checkpoint.parent / "run.json"
    if not run_json.exists():
        raise FileNotFoundError(f"{run_json} missing: evaluate only checkpoints of finished runs")
    recipe = Recipe(**json.loads(run_json.read_text())["recipe"])
    frames, _ = load_frames(args.labels)
    frames = select_frames(frames, [n for n, _ in read_manifest(args.manifest)])
    model = build_model(None, trainable_layers=recipe.trainable_layers,
                        min_size=tuple(recipe.min_sizes), max_size=recipe.max_size)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    file_name = None
    if args.stylized_copy is not None:
        from snz.stylize.build import stylized_name

        # partial, not lambda: DataLoader workers must be able to pickle it
        file_name = functools.partial(stylized_name, copy=args.stylized_copy)
    elif args.corruption is not None:
        from snz.filenames import corrupt_name

        file_name = functools.partial(corrupt_name, corruption=args.corruption,
                                      severity=args.severity)
    records = predict(model, frames, args.images, device=args.device, file_name=file_name)
    args.out.mkdir(parents=True, exist_ok=True)
    write_detections(args.out / f"{args.split}.detections.jsonl", records)
    metrics = {"split": args.split, "checkpoint": str(args.checkpoint),
               "min_size": recipe.min_sizes[-1], **evaluate(frames, records)}
    (args.out / f"{args.split}.metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("split", "mAP", "AP50")}, indent=2))
    return 0


def _cmd_footage_sample(args) -> int:
    from snz.data.footage import sample_footage

    args.out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(sample_footage(args.drives, args.out, args.min_frames), indent=2))
    return 0


def _seed0_footage(runs: Path, analysis: Path) -> tuple[dict, dict]:
    from snz.eval.predict import read_detections

    thresholds = json.loads(analysis.read_text())["footage_thresholds"]
    dets, th = {}, {}
    for arm in ("BASE", "STY", "SIN", "BOTH"):
        f = runs / f"{arm}-s0" / "eval" / "footage.detections.jsonl"
        if f.exists():
            dets[arm] = read_detections(f)
            th[arm] = thresholds[f"{arm}-s0"]["threshold"]
    return dets, th


def _cmd_comma2k19_drives(args) -> int:
    from snz.data.footage import comma2k19_drives

    print(f"{len(comma2k19_drives(args.root, args.out))} segments -> {args.out}")
    return 0


def _cmd_footage_predict(args) -> int:
    import torch

    from snz.data.bdd import Frame
    from snz.data.manifest import read_manifest
    from snz.eval.guard import ensure_split_allowed
    from snz.eval.predict import predict, write_detections
    from snz.train.config import Recipe
    from snz.train.model import build_model

    if not args.no_guard:
        ensure_split_allowed("footage", manifest=args.manifest)
    if not args.analysis.exists():
        raise SystemExit(f"error: {args.analysis} missing; run G6 first so the footage "
                         "thresholds are fixed before any footage is processed")
    recipe = Recipe(**json.loads((args.checkpoint.parent / "run.json").read_text())["recipe"])
    frames = [Frame(n, "undefined", "undefined", ()) for n, _ in read_manifest(args.manifest)]
    model = build_model(None, trainable_layers=recipe.trainable_layers,
                        min_size=tuple(recipe.min_sizes), max_size=recipe.max_size)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    records = predict(model, frames, args.frames, device=args.device)
    write_detections(args.checkpoint.parent / "eval" / "footage.detections.jsonl", records)
    print(f"{sum(len(r['scores']) for r in records)} detections on {len(records)} frames")
    return 0


def _cmd_footage_label_sample(args) -> int:
    from snz.data.footage import label_sample

    names = label_sample(args.manifest, args.k, args.seed)
    args.out.write_text("\n".join(names) + "\n")
    print(f"{len(names)} frames to label -> {args.out}")
    return 0


def _cmd_audit_sample(args) -> int:
    from snz.audit.page import build_audit_page
    from snz.audit.sample import sample_audit

    dets, th = _seed0_footage(args.runs, args.analysis)
    items, key = sample_audit(dets, th, n_per_arm=args.n, seed=args.seed)
    page = build_audit_page(items, args.frames, args.out)
    args.key.parent.mkdir(parents=True, exist_ok=True)
    args.key.write_text(json.dumps(key, indent=1))
    print(f"{len(items)} items -> {page}; key -> {args.key} (do not open before scoring)")
    return 0


def _cmd_audit_score(args) -> int:
    from snz.audit.sample import count_detections
    from snz.audit.score import (
        cohen_kappa,
        detection_counts,
        footage_recall,
        load_coco_labels,
        read_labels,
        score_audit,
    )

    dets, th = _seed0_footage(args.runs, args.analysis)
    key = json.loads(args.key.read_text())
    labels = read_labels(args.labels)
    result = {"thresholds": th,
              "frames": {arm: len(r) for arm, r in dets.items()},
              "detections": {arm: detection_counts(r, th[arm]) for arm, r in dets.items()},
              "audit": score_audit(labels, key,
                                   {arm: count_detections(r, th[arm]) for arm, r in dets.items()})}
    if args.labels2:
        result["cohen_kappa"] = cohen_kappa(labels, read_labels(args.labels2))
    if args.coco:
        result["recall_on_labeled_frames"] = footage_recall(load_coco_labels(args.coco), dets, th)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["audit"], indent=2))
    return 0


def _cmd_analyze(args) -> int:
    from snz.eval.analyze import analyze
    from snz.eval.guard import ensure_split_allowed
    from snz.eval.report import render_report

    if not args.no_guard:  # tests only; real analyses read held-out detections
        ensure_split_allowed("t-clean")
    result = analyze(args.runs, args.labels, args.manifests, n_boot=args.bootstrap,
                     seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.json").write_text(json.dumps(result, indent=2))
    (args.out / "report.md").write_text(render_report(result))
    from snz.eval.figures import make_figures

    make_figures(result, args.out / "figures")
    print((args.out / "report.md").read_text())
    return 0


def _cmd_smoke(args) -> int:
    from snz.train.config import load_recipe
    from snz.train.model import BACKBONES, load_backbone_state
    from snz.train.smoke import run_smoke

    states = {kind: load_backbone_state(kind, args.sin_weights) for kind in BACKBONES}
    report = run_smoke(states, load_recipe(args.recipe), args.work_dir, device=args.device,
                       iterations=args.iterations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["backbones"], indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
