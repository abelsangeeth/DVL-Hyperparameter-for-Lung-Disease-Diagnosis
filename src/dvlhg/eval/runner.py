"""Stage 4: evaluate, ablate, and export a deployable bundle.

Produces, under paths.reports:
    metrics.json    every number, machine readable
    report.md       the tables and caveats, ready to paste into a write-up
    figures/*.png   ROC / PR / calibration / training curves / edge weights

and, under paths.export, everything the API needs to serve one unseen study.

The ablation table is the point of the whole project. It answers, on the same
split with the same thresholds:
    image only / text only / fused / fused + hypergraph / + diffusion rows
so "the hypergraph helps" is a measured claim rather than an architectural one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ..config import DotDict, save_config
from ..constants import DISCLAIMER, EDGE_GROUPS, LABELS
from ..data.labels import targets_from_manifest
from ..data.prepare import load_manifest
from ..diffusion.augment import merge_with_synthetic
from ..hypergraph.construct import build_query_edges
from ..train.hgnn_trainer import make_graph, train_hypergraph
from ..train.loop import extract_features, load_backbone
from ..utils import get_logger, pick_device, save_json, set_seed
from .metrics import bootstrap_ci, compute_metrics, fit_temperature, format_table, tune_thresholds

LOG = get_logger()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _split_index(split: np.ndarray) -> Dict[str, np.ndarray]:
    return {name: np.where(split == name)[0] for name in ("train", "val", "test")}


def run_evaluation(
    cfg: DotDict,
    retrain_hypergraph: bool = True,
    export: bool = True,
) -> Dict[str, object]:
    set_seed(int(cfg.seed))
    device = pick_device(cfg.get("device", "auto"))

    manifest = load_manifest(cfg)
    merged, images, used_synthetic = merge_with_synthetic(manifest, cfg)

    model, tokenize, state = load_backbone(cfg, device)
    features = extract_features(cfg, model, merged, images, tokenize, device)

    split = merged["split"].astype(str).to_numpy()
    index = _split_index(split)
    y_all, mask_all = targets_from_manifest(merged)
    is_real = (merged.get("cache_source", pd.Series(["real"] * len(merged))).astype(str) == "real").to_numpy()

    if not retrain_hypergraph:
        from ..train.hgnn_trainer import load_hypergraph_model

        graph, node_features, keep = make_graph(cfg, merged, features, device)
        hgnn = load_hypergraph_model(cfg, graph, node_features.shape[1], device)
        info = {"graph": graph.summary()}
    else:
        hgnn, graph, info = train_hypergraph(cfg, merged, features, resume=False)
        keep = info["keep"]
        node_features = torch.from_numpy(features["node"][keep]).float().to(device)

    base_logits = torch.from_numpy(features["logits"][keep]).float().to(device)
    with torch.no_grad():
        hgnn_logits_sub = hgnn(node_features, graph, base_logits).float().cpu().numpy()
    hypergraph_logits = np.full_like(features["logits"], np.nan)
    hypergraph_logits[keep] = hgnn_logits_sub

    variants: Dict[str, np.ndarray] = {
        "image_only": features["image_logits"],
        "text_only": features["text_logits"],
        "fusion": features["logits"],
        "fusion_hypergraph": hypergraph_logits,
    }

    # ---- thresholds + temperature, fitted on validation, applied to test ----
    val_real = np.intersect1d(index["val"], np.where(is_real)[0])
    test_real = np.intersect1d(index["test"], np.where(is_real)[0])
    if len(val_real) == 0 or len(test_real) == 0:
        raise RuntimeError("validation or test split is empty after removing synthetic rows")

    results: Dict[str, object] = {}
    tables: List[str] = []
    calibration: Dict[str, object] = {}

    for name, logits in variants.items():
        usable_val = val_real[~np.isnan(logits[val_real]).any(axis=1)]
        usable_test = test_real[~np.isnan(logits[test_real]).any(axis=1)]
        if len(usable_val) == 0 or len(usable_test) == 0:
            LOG.warning("variant '%s' has no usable rows - skipped", name)
            continue

        temperatures = np.ones(len(LABELS))
        if str(cfg.eval.calibrate) == "temperature":
            temperatures = fit_temperature(y_all[usable_val], logits[usable_val], mask_all[usable_val])

        val_probabilities = _sigmoid(logits[usable_val] / temperatures.reshape(1, -1))
        thresholds = tune_thresholds(
            y_all[usable_val], val_probabilities, mask_all[usable_val],
            method=str(cfg.eval.threshold), fixed=float(cfg.eval.fixed_threshold),
        )
        test_probabilities = _sigmoid(logits[usable_test] / temperatures.reshape(1, -1))

        val_metrics = compute_metrics(y_all[usable_val], val_probabilities, mask_all[usable_val], thresholds)
        test_metrics = compute_metrics(
            y_all[usable_test], test_probabilities, mask_all[usable_test], thresholds,
            with_curves=bool(cfg.eval.save_curves),
        )

        groups = merged["subject_id"].astype(str).to_numpy()[usable_test]
        ci = bootstrap_ci(
            y_all[usable_test], test_probabilities, mask_all[usable_test], thresholds,
            n_resamples=int(cfg.eval.bootstrap), seed=int(cfg.seed), groups=groups,
        )

        results[name] = {"val": val_metrics, "test": test_metrics, "ci": ci,
                         "n_test": int(len(usable_test))}
        calibration[name] = {
            "temperatures": temperatures.tolist(),
            "thresholds": thresholds.tolist(),
        }
        tables.append(format_table(test_metrics, title=f"{name} — test split"))
        if name == "fusion_hypergraph" or (name == "fusion" and "fusion_hypergraph" not in variants):
            # Row-level predictions, so `dvlhg verify-serving` can diff the API
            # against the exact numbers this report quotes.
            np.savez_compressed(
                Path(cfg.paths.reports) / "test_predictions.npz",
                rows=usable_test,
                probabilities=test_probabilities,
                y=y_all[usable_test],
                mask=mask_all[usable_test],
                thresholds=thresholds,
                temperatures=temperatures,
                variant=np.array([name]),
            )
        LOG.info(
            "%-18s test AUROC %.4f  F1 %.4f  AUPRC %.4f",
            name, test_metrics["auroc_macro"], test_metrics["f1_macro"], test_metrics["auprc_macro"],
        )

    # ---- inductive vs transductive agreement --------------------------------
    inductive = _inductive_check(
        cfg, hgnn, graph, node_features, base_logits, merged[keep].reset_index(drop=True),
        features, keep, device, n_samples=int(cfg.eval.get("inductive_samples", 64)),
    )
    if inductive:
        LOG.info(
            "inductive vs transductive on %d held-out studies: mean |dp| = %.4f, AUROC %.4f vs %.4f",
            inductive["n"], inductive["mean_abs_prob_delta"],
            inductive["auroc_inductive"], inductive["auroc_transductive"],
        )

    summary = {
        "labels": LABELS,
        "data": {
            "source": str(cfg.data.source),
            "rows": int(len(manifest)),
            "splits": {k: int(v) for k, v in manifest["split"].value_counts().items()},
            "used_synthetic": bool(used_synthetic),
            "synthetic_rows": int((~is_real).sum()),
        },
        "text": {
            "mode": str(cfg.text.mode),
            "leaky": str(cfg.text.mode) in {"findings", "impression", "full"},
        },
        "vlm_backend": state.get("vlm_backend", model.vlm.backend),
        "backbone_epoch": state.get("epoch"),
        "hypergraph": info.get("graph"),
        "hyperedge_group_weights": info.get("group_weights"),
        "mean_logit_correction": info.get("mean_logit_correction"),
        "results": results,
        "calibration": calibration,
        "inductive_check": inductive,
        "disclaimer": DISCLAIMER,
    }

    reports_dir = Path(cfg.paths.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)
    save_json(summary, reports_dir / "metrics.json")
    _write_report(cfg, summary, tables, reports_dir / "report.md")

    try:
        from .figures import write_figures

        write_figures(cfg, summary, features, merged, reports_dir / "figures")
    except Exception as exc:  # noqa: BLE001 - figures must never fail a run
        LOG.warning("figure generation skipped (%s)", exc)

    if export:
        export_bundle(cfg, model, hgnn, graph, features, merged, keep, summary)

    return summary


def _inductive_check(
    cfg: DotDict,
    hgnn,
    graph,
    node_features: torch.Tensor,
    base_logits: torch.Tensor,
    sub_manifest: pd.DataFrame,
    features: Dict[str, np.ndarray],
    keep: np.ndarray,
    device: torch.device,
    n_samples: int = 64,
) -> Optional[Dict[str, object]]:
    """How close is the deployed (frozen-bank) path to the transductive one?

    Both are computed for the same test studies. A small gap means the served
    model behaves like the evaluated one; a large gap means the reported numbers
    do not describe what the API will do, which is worth knowing before anyone
    quotes them.
    """
    split = sub_manifest["split"].astype(str).to_numpy()
    test_rows = np.where(split == "test")[0]
    bank_rows = np.where(np.isin(split, ("train", "val")))[0]
    if len(test_rows) == 0 or len(bank_rows) < 32:
        return None

    rng = np.random.default_rng(int(cfg.seed))
    sample = rng.choice(test_rows, size=min(n_samples, len(test_rows)), replace=False)

    sub_features = {k: v[keep] for k, v in features.items() if isinstance(v, np.ndarray) and v.ndim == 2}
    bank = {
        "fused": torch.from_numpy(sub_features["node"][bank_rows]).float().to(device),
        "image": torch.from_numpy(sub_features["img_pooled"][bank_rows]).float().to(device),
        "text": torch.from_numpy(sub_features["txt_pooled"][bank_rows]).float().to(device),
    }
    bank_states, bank_degrees = hgnn.bank_states(node_features, graph)
    bank_states = [s[bank_rows] for s in bank_states]
    bank_degrees = bank_degrees[bank_rows]

    with torch.no_grad():
        transductive_logits = hgnn(node_features, graph, base_logits).float().cpu().numpy()

    inductive_logits = np.zeros((len(sample), len(LABELS)), dtype=np.float32)
    for position, row in enumerate(sample):
        query_edges = build_query_edges(
            fused_q=node_features[row],
            image_q=bank["image"].new_tensor(sub_features["img_pooled"][row]),
            text_q=bank["text"].new_tensor(sub_features["txt_pooled"][row]),
            bank=bank,
            graph_config=graph.config,
            centroids=graph.centroids.get("proto_label"),
        )
        with torch.no_grad():
            inductive_logits[position] = (
                hgnn.forward_query(
                    node_features[row], base_logits[row], query_edges, bank_states, bank_degrees
                ).float().cpu().numpy()[0]
            )

    y_all, mask_all = targets_from_manifest(sub_manifest)
    p_ind = _sigmoid(inductive_logits)
    p_trans = _sigmoid(transductive_logits[sample])
    metrics_ind = compute_metrics(y_all[sample], p_ind, mask_all[sample])
    metrics_trans = compute_metrics(y_all[sample], p_trans, mask_all[sample])
    return {
        "n": int(len(sample)),
        "mean_abs_prob_delta": float(np.abs(p_ind - p_trans).mean()),
        "max_abs_prob_delta": float(np.abs(p_ind - p_trans).max()),
        "auroc_inductive": float(metrics_ind["auroc_macro"]),
        "auroc_transductive": float(metrics_trans["auroc_macro"]),
    }


def _write_report(cfg: DotDict, summary: Dict[str, object], tables: List[str], path: Path) -> None:
    results = summary["results"]  # type: ignore[index]
    lines: List[str] = []
    lines.append("# DVL-HGN — results\n")
    lines.append(f"> {DISCLAIMER}\n")
    lines.append("## Setup\n")
    data = summary["data"]  # type: ignore[index]
    lines.append(f"- Dataset: **{data['source']}**, {data['rows']} studies {data['splits']}")
    lines.append(f"- Vision-language backbone: **{summary['vlm_backend']}**")
    text = summary["text"]  # type: ignore[index]
    lines.append(f"- Text channel: **{text['mode']}**" + ("  ⚠️ **leaky — upper bound only**" if text["leaky"] else "  (leakage-free)"))
    lines.append(f"- Diffusion rows used in training: **{data['synthetic_rows']}**")
    graph_summary = summary.get("hypergraph") or {}
    if graph_summary:
        groups = ", ".join(
            f"{name} {count}" for name, count in (graph_summary.get("edges_per_group") or {}).items()
        )
        lines.append(
            f"- Hypergraph: **{graph_summary.get('nodes')} nodes**, "
            f"{graph_summary.get('edges')} hyperedges, {graph_summary.get('nnz')} incidences, "
            f"mean size {float(graph_summary.get('mean_edge_size', 0)):.1f} "
            f"(max {graph_summary.get('max_edge_size')})"
        )
        if groups:
            lines.append(f"  - hyperedges per group: {groups}")
    if summary.get("mean_logit_correction") is not None:
        lines.append(
            f"- Mean absolute logit correction applied by the hypergraph: "
            f"**{float(summary['mean_logit_correction']):.3f}**"
        )
    if summary.get("hyperedge_group_weights"):
        weights = ", ".join(f"{k} {v:.2f}" for k, v in summary["hyperedge_group_weights"].items())  # type: ignore[union-attr]
        lines.append(f"- Learned hyperedge group weights: {weights}")
    lines.append("")

    lines.append("## Ablation (test split, thresholds tuned on validation)\n")
    lines.append("| Variant | AUROC | 95% CI | AUPRC | F1 | Accuracy | Exact match |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, entry in results.items():  # type: ignore[union-attr]
        test = entry["test"]
        ci = entry.get("ci", {}).get("auroc_macro")
        ci_text = f"{ci['lo']:.3f}–{ci['hi']:.3f}" if ci else "–"
        lines.append(
            f"| {name} | {test['auroc_macro']:.3f} | {ci_text} | {test['auprc_macro']:.3f} | "
            f"{test['f1_macro']:.3f} | {test['accuracy_macro']:.3f} | {test['exact_match']:.3f} |"
        )
    lines.append("")

    if summary.get("inductive_check"):
        check = summary["inductive_check"]  # type: ignore[index]
        lines.append("## Served path vs evaluated path\n")
        lines.append(
            f"On {check['n']} held-out studies the deployed frozen-bank inference differs from the "
            f"transductive computation by **{check['mean_abs_prob_delta']:.4f}** mean absolute probability "
            f"(max {check['max_abs_prob_delta']:.4f}); macro AUROC "
            f"{check['auroc_inductive']:.3f} vs {check['auroc_transductive']:.3f}.\n"
        )

    lines.append("## Per-finding detail\n")
    lines.extend(f"{table}\n" for table in tables)

    lines.append("## How to read these numbers\n")
    lines.append(
        "- Thresholds and temperatures are fitted on **validation** only; the test split is touched once.\n"
        "- The bootstrap resamples **patients**, not studies, so repeated films from one patient do not "
        "shrink the interval.\n"
        "- Uncertain labels are excluded from the loss and from every metric under the default policy "
        f"(`{cfg.data.uncertain_policy}`); per-finding `n` columns show how many cells actually counted.\n"
    )
    if text["leaky"]:
        lines.append(
            "- ⚠️ The text channel includes the report sections the CheXpert labels were derived from. "
            "These numbers measure how well the model re-reads the radiologist's own sentence, not how "
            "well it reads the film. Run with `--set text.mode=indication` for the honest figure.\n"
        )

    path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("report written to %s", path)


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def export_bundle(
    cfg: DotDict,
    model,
    hgnn,
    graph,
    features: Dict[str, np.ndarray],
    merged: pd.DataFrame,
    keep: np.ndarray,
    summary: Dict[str, object],
) -> Path:
    """Write everything the API needs into paths.export."""
    out_dir = Path(cfg.paths.export)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub_manifest = merged[keep].reset_index(drop=True)
    split = sub_manifest["split"].astype(str).to_numpy()
    # The neighbour bank is train+val only: test films are never shipped, and a
    # query can never retrieve itself.
    bank_rows = np.where(np.isin(split, ("train", "val")))[0]
    cap = int(cfg.hypergraph.bank_size)
    if len(bank_rows) > cap:
        rng = np.random.default_rng(int(cfg.seed))
        bank_rows = np.sort(rng.choice(bank_rows, size=cap, replace=False))
        LOG.info("bank capped at %d nodes", cap)

    device = next(hgnn.parameters()).device
    node_features = torch.from_numpy(features["node"][keep]).float().to(device)
    bank_states, bank_degrees = hgnn.bank_states(node_features, graph)

    np.savez_compressed(
        out_dir / "bank.npz",
        fused=features["node"][keep][bank_rows].astype(np.float32),
        image=features["img_pooled"][keep][bank_rows].astype(np.float32),
        text=features["txt_pooled"][keep][bank_rows].astype(np.float32),
        degrees=bank_degrees[bank_rows].cpu().numpy().astype(np.float32),
        states=np.stack([s[bank_rows].cpu().numpy().astype(np.float32) for s in bank_states]),
        view=sub_manifest["ViewPosition"].astype(str).to_numpy()[bank_rows]
        if "ViewPosition" in sub_manifest.columns else np.array([""] * len(bank_rows)),
        centroids=graph.centroids.get("proto_label", np.zeros((0, features["node"].shape[1]), np.float32)),
    )

    meta_columns = [
        c for c in ("study_id", "subject_id", "split", "ViewPosition", "cache_source", "source")
        if c in sub_manifest.columns
    ] + [f"y_{name}" for name in LABELS]
    sub_manifest.iloc[bank_rows][meta_columns].to_csv(out_dir / "bank_meta.csv", index=False)

    torch.save(
        {"model": model.state_dict(), "config": cfg.to_dict(), "vlm_backend": model.vlm.backend},
        out_dir / "backbone.pt",
    )
    torch.save(
        {"model": hgnn.state_dict(), "config": cfg.to_dict(), "node_dim": int(node_features.shape[1])},
        out_dir / "hypergraph.pt",
    )

    chosen = "fusion_hypergraph" if "fusion_hypergraph" in summary["results"] else "fusion"  # type: ignore[index]
    calibration = summary["calibration"][chosen]  # type: ignore[index]

    # The bundle's metrics go out over the /api/model-card endpoint on every
    # page load, so drop the ROC/PR point arrays and keep the scalars.
    served_metrics = {
        key: value for key, value in summary["results"][chosen]["test"].items()  # type: ignore[index]
        if key != "per_class"
    }
    served_metrics["per_class"] = {
        name: {k: v for k, v in entry.items() if k not in ("roc", "pr")}
        for name, entry in summary["results"][chosen]["test"]["per_class"].items()  # type: ignore[index]
    }
    save_json(
        {
            "labels": LABELS,
            "serving_variant": chosen,
            "thresholds": calibration["thresholds"],
            "temperatures": calibration["temperatures"],
            "text_mode": str(cfg.text.mode),
            "image_size": int(cfg.data.image_size),
            "cache_size": int(cfg.data.cache_size),
            "graph_config": {**graph.config, "proto_attach": int(cfg.hypergraph.get("proto_attach", 16))},
            "edge_groups": EDGE_GROUPS,
            "test_metrics": served_metrics,
            "disclaimer": DISCLAIMER,
            "data_source": str(cfg.data.source),
            "vlm_backend": str(summary["vlm_backend"]),
        },
        out_dir / "bundle.json",
    )
    if bool(cfg.serve.get("include_diffusion", True)):
        from ..diffusion.trainer import export_slim

        if export_slim(cfg, out_dir / "diffusion.pt") is None:
            LOG.info("no diffusion checkpoint to export - the demo's generate/refine "
                     "panel will be disabled")

    save_config(cfg, out_dir / "config.yaml")
    LOG.info("export bundle written to %s (%d bank nodes)", out_dir, len(bank_rows))
    return out_dir
