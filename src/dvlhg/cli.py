"""Command line entry point - every stage is one subcommand.

    dvlhg prep         build the manifest + image cache
    dvlhg check-vlm    zero-shot sanity check that the backbone really loaded
    dvlhg diffusion    train the diffusion model (stage 1)
    dvlhg synth        generate the synthetic training rows
    dvlhg train        train the vision-language backbone (stage 2)
    dvlhg hypergraph   build the hypergraph and train the HGNN (stage 3)
    dvlhg eval         evaluate, ablate, export the serving bundle (stage 4)
    dvlhg verify-serving  check the API reproduces the evaluated model
    dvlhg serve        run the API + frontend
    dvlhg smoke        end-to-end run on synthetic data (no downloads, ~2 min)

Every command takes --config and repeated --set key.sub=value overrides.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import DotDict, ensure_dirs, load_config
from .constants import LABELS
from .utils import get_logger, pick_device, set_seed

LOG = get_logger()


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="KEY=VALUE", help="override a config value, repeatable",
    )


def _load(args) -> DotDict:
    cfg = load_config(args.config, args.overrides)
    ensure_dirs(cfg)
    get_logger(logfile=Path(cfg.paths.logs) / "dvlhg.log")
    set_seed(int(cfg.seed))
    return cfg


# --------------------------------------------------------------------------- #
def cmd_prep(args) -> None:
    from .data.prepare import prepare

    cfg = _load(args)
    prepare(
        cfg,
        physionet_user=args.user,
        physionet_pass=args.password,
        download=not args.no_download,
        rebuild_cache=not args.keep_cache,
    )


def cmd_check_vlm(args) -> None:
    """Zero-shot probe: does the pretrained backbone actually distinguish the
    findings before any training? If this is at chance, the weights did not
    load and nothing downstream is worth running."""
    import torch

    from .data.dataset import ordered_loader
    from .data.labels import targets_from_manifest
    from .data.prepare import load_image_cache, load_manifest
    from .eval.metrics import _safe_auroc
    from .vlm.encoders import build_vlm

    cfg = _load(args)
    device = pick_device(cfg.get("device", "auto"))
    manifest = load_manifest(cfg)
    subset = manifest[manifest["split"] == "val"]
    if len(subset) == 0:
        subset = manifest
    subset = subset.head(int(args.limit)).reset_index(drop=True)

    vlm, tokenize = build_vlm(cfg)
    vlm = vlm.to(device).eval()

    prompts_positive = [f"chest x-ray showing {name.lower()}" for name in LABELS]
    prompts_negative = [f"chest x-ray with no {name.lower()}" for name in LABELS]
    tokens = tokenize(prompts_positive + prompts_negative)
    with torch.no_grad():
        _, text_embeddings = vlm.encode_text(
            tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
        )
    text_embeddings = torch.nn.functional.normalize(text_embeddings.float(), dim=-1)

    loader = ordered_loader(subset, {"real": load_image_cache(cfg)}, tokenize, cfg, batch_size=32)
    image_embeddings = []
    with torch.no_grad():
        for batch in loader:
            _, pooled = vlm.encode_image(batch["image"].to(device))
            image_embeddings.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu())
    image_embeddings = torch.cat(image_embeddings)

    similarity = image_embeddings @ text_embeddings.cpu().T
    scores = (similarity[:, : len(LABELS)] - similarity[:, len(LABELS) :]).numpy()
    y, mask = targets_from_manifest(subset)

    print(f"\nZero-shot probe of backbone '{vlm.backend}' on {len(subset)} studies")
    print("(no training - this only checks that pretrained weights loaded and the wiring is right)\n")
    aurocs = []
    for c, name in enumerate(LABELS):
        valid = mask[:, c] > 0
        if valid.sum() == 0 or y[valid, c].sum() in (0, valid.sum()):
            print(f"  {name:20s}   n/a (no usable positives or negatives)")
            continue
        auroc = _safe_auroc(y[valid, c], scores[valid, c])
        aurocs.append(auroc)
        print(f"  {name:20s}   AUROC {auroc:.3f}   (n={int(valid.sum())}, pos={int(y[valid, c].sum())})")
    if aurocs:
        mean = float(np.mean(aurocs))
        print(f"\n  macro AUROC {mean:.3f}")
        if vlm.backend == "biomedclip" and mean < 0.55:
            print(
                "\n  WARNING: a loaded BiomedCLIP should sit well above 0.55 here.\n"
                "  Check that open_clip downloaded the weights (not just the config)."
            )
        elif vlm.backend == "dummy":
            print("\n  (dummy backbone - ~0.5 is expected, it has random weights)")


def cmd_diffusion(args) -> None:
    from .data.prepare import load_image_cache, load_manifest
    from .diffusion.trainer import load_diffusion, sample_grid, train_diffusion

    cfg = _load(args)
    manifest = load_manifest(cfg)
    images = load_image_cache(cfg)
    if not args.samples_only:
        train_diffusion(cfg, manifest, images, resume=not args.no_resume)
    diffusion = load_diffusion(cfg)
    sample_grid(diffusion, cfg, Path(cfg.paths.reports) / "figures" / "diffusion_samples.png")


def cmd_synth(args) -> None:
    from .data.prepare import load_manifest
    from .diffusion.augment import generate_synthetic

    cfg = _load(args)
    manifest = load_manifest(cfg)
    generate_synthetic(cfg, manifest)


def cmd_train(args) -> None:
    import torch

    from .data.prepare import load_manifest
    from .diffusion.augment import merge_with_synthetic
    from .train.loop import train_backbone

    cfg = _load(args)
    manifest = load_manifest(cfg)
    merged, images, _ = merge_with_synthetic(manifest, cfg)

    refiner = None
    if "refine" in list(cfg.diffusion.use) and float(cfg.diffusion.refine_prob) > 0:
        try:
            from .diffusion.ddpm import to_image_space, to_model_space
            from .diffusion.trainer import load_diffusion

            diffusion = load_diffusion(cfg, pick_device(cfg.get("device", "auto")))
            size = int(cfg.diffusion.image_size)

            def refiner(batch_images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                target = batch_images.shape[-1]
                small = torch.nn.functional.interpolate(
                    batch_images, size=(size, size), mode="bilinear", align_corners=False
                )
                refined = diffusion.refine(
                    to_model_space(small), labels,
                    strength=float(cfg.diffusion.refine_t),
                    steps=max(int(cfg.diffusion.sample_steps) // 3, 5),
                )
                out = to_image_space(refined)
                return torch.nn.functional.interpolate(
                    out, size=(target, target), mode="bilinear", align_corners=False
                )

            LOG.info("SDEdit refinement enabled (p=%.2f, strength=%.2f)",
                     float(cfg.diffusion.refine_prob), float(cfg.diffusion.refine_t))
        except FileNotFoundError:
            LOG.info("no diffusion checkpoint - training without SDEdit refinement")

    train_backbone(cfg, merged, images, resume=not args.no_resume, diffusion_refiner=refiner)


def cmd_hypergraph(args) -> None:
    from .data.prepare import load_manifest
    from .diffusion.augment import merge_with_synthetic
    from .train.hgnn_trainer import train_hypergraph
    from .train.loop import extract_features, load_backbone

    cfg = _load(args)
    manifest = load_manifest(cfg)
    merged, images, _ = merge_with_synthetic(manifest, cfg)
    device = pick_device(cfg.get("device", "auto"))
    model, tokenize, _ = load_backbone(cfg, device)
    features = extract_features(cfg, model, merged, images, tokenize, device)
    np.savez_compressed(
        Path(cfg.paths.cache) / "features.npz",
        **{k: v for k, v in features.items() if isinstance(v, np.ndarray)},
    )
    train_hypergraph(cfg, merged, features)


def cmd_eval(args) -> None:
    from .eval.runner import run_evaluation

    cfg = _load(args)
    summary = run_evaluation(cfg, retrain_hypergraph=not args.reuse_hypergraph, export=not args.no_export)
    results = summary["results"]
    print("\n=== test split, macro AUROC ===")
    for name, entry in results.items():
        print(f"  {name:20s} {entry['test']['auroc_macro']:.4f}")
    print(f"\nfull report: {Path(cfg.paths.reports) / 'report.md'}")


def cmd_verify_serving(args) -> None:
    """Send real test studies through the *serving* path and diff the result
    against the numbers the evaluation report quotes.

    This is the check that catches train/serve skew: a preprocessing step that
    differs by one resize, a threshold that did not get exported, a text mode
    the API applies differently. All three are silent failures that look like a
    working demo and a report nobody can reproduce.
    """
    import io

    from PIL import Image

    from .data.prepare import load_image_cache, load_manifest
    from .eval.metrics import compute_metrics
    from .serve.inference import Predictor

    cfg = _load(args)
    bundle = args.bundle or str(Path(cfg.paths.export))
    predictions_file = Path(cfg.paths.reports) / "test_predictions.npz"
    if not predictions_file.exists():
        raise FileNotFoundError(
            f"{predictions_file} not found - run `dvlhg eval` first so there is "
            "something to compare against"
        )

    stored = np.load(predictions_file, allow_pickle=True)
    manifest = load_manifest(cfg)
    images = load_image_cache(cfg)
    predictor = Predictor(bundle, device=str(cfg.serve.device), neighbours=0)

    rows = stored["rows"]
    limit = min(int(args.limit), len(rows))
    rng = np.random.default_rng(int(cfg.seed))
    pick = rng.choice(len(rows), size=limit, replace=False)

    served = np.zeros((limit, len(LABELS)), dtype=np.float64)
    for position, index in enumerate(pick):
        row = int(rows[index])
        record = manifest.iloc[row]
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(images[int(record["cache_index"])]), mode="L").save(buffer, format="PNG")
        result = predictor.predict(
            buffer.getvalue(), report=str(record["text"]), explain=False,
            neighbours=0, report_is_prepared=True,
        )
        served[position] = [f["probability"] for f in result["findings"]]

    reference = stored["probabilities"][pick]
    truth = stored["y"][pick]
    mask = stored["mask"][pick]
    thresholds = stored["thresholds"]

    delta = np.abs(served - reference)
    served_metrics = compute_metrics(truth, served, mask, thresholds)
    reference_metrics = compute_metrics(truth, reference, mask, thresholds)

    print()
    print(f"Serving-path verification on {limit} test studies")
    print(f"  variant                 {str(stored['variant'][0])}")
    print(f"  mean |dp|               {delta.mean():.6f}")
    print(f"  max  |dp|               {delta.max():.6f}")
    print(f"  macro AUROC  served     {served_metrics['auroc_macro']:.4f}")
    print(f"  macro AUROC  reported   {reference_metrics['auroc_macro']:.4f}")
    print(f"  macro F1     served     {served_metrics['f1_macro']:.4f}")
    print(f"  macro F1     reported   {reference_metrics['f1_macro']:.4f}")
    print()

    tolerance = float(args.tolerance)
    if delta.mean() <= tolerance:
        print(f"  PASS - the API reproduces the evaluated model (tolerance {tolerance})")
        return

    print(f"  FAIL - mean |dp| {delta.mean():.6f} exceeds {tolerance}.")
    print("  The served model is not the evaluated model. Check preprocessing")
    print("  (data.cache_size vs data.image_size), the exported thresholds and")
    print("  temperatures, and that text.mode matches bundle.json.")
    raise SystemExit(1)


def cmd_serve(args) -> None:
    from .serve.api import run

    cfg = _load(args)
    bundle = args.bundle or cfg.serve.bundle or str(Path(cfg.paths.export))
    run(
        bundle=bundle,
        host=args.host or str(cfg.serve.host),
        port=int(args.port or cfg.serve.port),
        device=str(cfg.serve.device),
        neighbours=int(cfg.serve.neighbours),
    )


def cmd_smoke(args) -> None:
    """Every stage, end to end, on procedural data. No downloads, no GPU."""
    from .data.prepare import load_image_cache, load_manifest, prepare
    from .diffusion.augment import generate_synthetic, merge_with_synthetic
    from .diffusion.trainer import train_diffusion
    from .eval.runner import run_evaluation
    from .train.loop import train_backbone

    overrides = [
        "data.source=synthetic",
        f"data.subset_size={args.n}",
        "data.cache_size=96",
        "data.image_size=224",
        "data.num_workers=0",
        "vlm.backbone=dummy",
        "fusion.dim=128",
        "fusion.layers=1",
        "fusion.heads=4",
        "hypergraph.hidden=128",
        "hypergraph.knn_fused=6",
        "hypergraph.knn_image=4",
        "hypergraph.knn_text=4",
        "hypergraph.proto_clusters=8",
        f"train.epochs={args.epochs}",
        "train.batch_size=16",
        "hgnn_train.epochs=60",
        "hgnn_train.early_stop_patience=30",
        "diffusion.image_size=32",
        "diffusion.base_channels=16",
        "diffusion.channel_mult=[1,2]",
        "diffusion.timesteps=100",
        "diffusion.sample_steps=8",
        "diffusion.train.epochs=1",
        "diffusion.train.batch_size=16",
        "diffusion.synth_per_class=8",
        "diffusion.refine_prob=0.0",
        "eval.bootstrap=50",
        f"paths.root={args.root}",
    ] + list(args.overrides)
    args.overrides = overrides
    cfg = _load(args)

    LOG.info("=== 1/5 prepare ===")
    manifest = prepare(cfg, download=False)
    images_raw = load_image_cache(cfg)

    LOG.info("=== 2/5 diffusion ===")
    train_diffusion(cfg, manifest, images_raw, resume=False)

    LOG.info("=== 3/5 synthesis ===")
    generate_synthetic(cfg, manifest, batch_size=16)

    LOG.info("=== 4/5 backbone ===")
    merged, images, used = merge_with_synthetic(manifest, cfg)
    train_backbone(cfg, merged, images, resume=False)

    LOG.info("=== 5/5 hypergraph + evaluation + export ===")
    summary = run_evaluation(cfg, retrain_hypergraph=True, export=True)

    print("\n================ SMOKE TEST RESULT ================")
    for name, entry in summary["results"].items():
        print(f"  {name:20s} test macro AUROC {entry['test']['auroc_macro']:.4f}")
    check = summary.get("inductive_check")
    if check:
        print(f"  inductive vs transductive mean |dp| = {check['mean_abs_prob_delta']:.4f}")
    print(f"  synthetic rows used: {used}")
    print(f"  bundle: {Path(cfg.paths.export)}")
    print("===================================================\n")


def cmd_info(args) -> None:
    cfg = _load(args)
    print(f"config: {cfg.get('_config_path')}")
    print("paths:")
    for key, value in cfg.paths.items():
        if key.startswith("_"):
            continue
        marker = "ok " if Path(str(value)).exists() else "-- "
        print(f"  {marker}{key:10s} {value}")
    print(f"\ndata.source          {cfg.data.source}")
    print(f"text.mode            {cfg.text.mode}")
    print(f"vlm.backbone         {cfg.vlm.backbone}")
    print(f"hypergraph.enabled   {cfg.hypergraph.enabled}")
    print(f"device               {pick_device(cfg.get('device', 'auto'))}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dvlhg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prep", help="build the manifest and image cache")
    _common(p)
    p.add_argument("--user", default=None, help="PhysioNet username")
    p.add_argument("--password", default=None, help="PhysioNet password")
    p.add_argument("--no-download", action="store_true", help="use files already on disk")
    p.add_argument("--keep-cache", action="store_true", help="do not rebuild the image cache")
    p.set_defaults(func=cmd_prep)

    p = sub.add_parser("check-vlm", help="zero-shot probe of the vision-language backbone")
    _common(p)
    p.add_argument("--limit", type=int, default=500)
    p.set_defaults(func=cmd_check_vlm)

    p = sub.add_parser("diffusion", help="train the diffusion model")
    _common(p)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--samples-only", action="store_true", help="skip training, just draw samples")
    p.set_defaults(func=cmd_diffusion)

    p = sub.add_parser("synth", help="generate synthetic training rows")
    _common(p)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("train", help="train the vision-language backbone")
    _common(p)
    p.add_argument("--no-resume", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("hypergraph", help="build the hypergraph and train the HGNN")
    _common(p)
    p.set_defaults(func=cmd_hypergraph)

    p = sub.add_parser("eval", help="evaluate, ablate and export")
    _common(p)
    p.add_argument("--reuse-hypergraph", action="store_true", help="load the saved HGNN instead of retraining")
    p.add_argument("--no-export", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("verify-serving", help="diff the API path against the evaluation report")
    _common(p)
    p.add_argument("--bundle", default=None)
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--tolerance", type=float, default=0.01)
    p.set_defaults(func=cmd_verify_serving)

    p = sub.add_parser("serve", help="run the API and frontend")
    _common(p)
    p.add_argument("--bundle", default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("smoke", help="end-to-end run on procedural data")
    _common(p)
    p.add_argument("--n", type=int, default=400, help="number of synthetic studies")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--root", default="./runs/smoke")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("info", help="show the resolved configuration")
    _common(p)
    p.set_defaults(func=cmd_info)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        LOG.warning("interrupted")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
