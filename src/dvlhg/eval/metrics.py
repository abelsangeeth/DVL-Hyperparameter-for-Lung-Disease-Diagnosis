"""Multi-label metrics, thresholds, calibration and confidence intervals.

Everything here ignores masked cells, so an "uncertain" label never silently
counts as a negative and never inflates accuracy.

Reported per finding and as macro averages:
    AUROC, AUPRC, accuracy, precision, recall (sensitivity), specificity, F1.
Plus, across the four findings: exact-match (subset) accuracy and Hamming
accuracy, which are the numbers that actually describe a multi-label system.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..constants import LABELS

EPS = 1e-12


def _valid(mask: np.ndarray, c: int) -> np.ndarray:
    return mask[:, c] > 0.5


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based AUROC with tie handling. Returns nan for a degenerate class."""
    positives = y_true > 0.5
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    # Average ranks within tied groups so identical scores cannot be gamed.
    sorted_scores = y_score[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _pr_points(y_true: np.ndarray, y_score: np.ndarray):
    """Precision/recall at every distinct score threshold, highest score first.

    Thresholds must be collapsed over tied scores: a classifier cannot separate
    two samples it gave the same score, and counting them one at a time
    overstates average precision.
    """
    order = np.argsort(-y_score, kind="mergesort")
    truth = (y_true[order] > 0.5).astype(np.float64)
    scores = y_score[order]
    tp = np.cumsum(truth)
    fp = np.cumsum(1.0 - truth)
    distinct = np.where(np.diff(scores))[0]
    cut = np.r_[distinct, len(truth) - 1]
    tp, fp = tp[cut], fp[cut]
    n_pos = truth.sum()
    if n_pos == 0:
        return None
    precision = tp / np.maximum(tp + fp, EPS)
    recall = tp / n_pos
    return precision, recall


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (step interpolation), matching sklearn's definition."""
    points = _pr_points(y_true, y_score)
    if points is None:
        return float("nan")
    precision, recall = points
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray, max_points: int = 200):
    order = np.argsort(-y_score, kind="mergesort")
    truth = (y_true[order] > 0.5).astype(np.float64)
    tp = np.cumsum(truth)
    fp = np.cumsum(1 - truth)
    n_pos, n_neg = tp[-1] if len(tp) else 0, fp[-1] if len(fp) else 0
    if n_pos == 0 or n_neg == 0:
        return [], []
    tpr = np.concatenate([[0.0], tp / n_pos])
    fpr = np.concatenate([[0.0], fp / n_neg])
    if len(tpr) > max_points:
        idx = np.linspace(0, len(tpr) - 1, max_points).astype(int)
        tpr, fpr = tpr[idx], fpr[idx]
    return fpr.tolist(), tpr.tolist()


def pr_curve_points(y_true: np.ndarray, y_score: np.ndarray, max_points: int = 200):
    points = _pr_points(y_true, y_score)
    if points is None:
        return [], []
    precision, recall = points
    if len(recall) > max_points:
        idx = np.linspace(0, len(recall) - 1, max_points).astype(int)
        precision, recall = precision[idx], recall[idx]
    return recall.tolist(), precision.tolist()


# --------------------------------------------------------------------------- #
# thresholds
# --------------------------------------------------------------------------- #
def tune_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mask: np.ndarray,
    method: str = "f1",
    fixed: float = 0.5,
    grid: int = 199,
) -> np.ndarray:
    """Per-class operating point chosen on the validation split only."""
    thresholds = np.full(y_true.shape[1], float(fixed), dtype=np.float64)
    if method == "fixed":
        return thresholds
    candidates = np.linspace(0.005, 0.995, grid)
    for c in range(y_true.shape[1]):
        valid = _valid(mask, c)
        if valid.sum() == 0 or y_true[valid, c].sum() == 0:
            continue
        truth, score = y_true[valid, c] > 0.5, y_prob[valid, c]
        best_value, best_threshold = -np.inf, float(fixed)
        for threshold in candidates:
            predicted = score >= threshold
            tp = float((predicted & truth).sum())
            fp = float((predicted & ~truth).sum())
            fn = float((~predicted & truth).sum())
            tn = float((~predicted & ~truth).sum())
            if method == "f1":
                value = 2 * tp / max(2 * tp + fp + fn, EPS)
            elif method == "youden":
                sensitivity = tp / max(tp + fn, EPS)
                specificity = tn / max(tn + fp, EPS)
                value = sensitivity + specificity - 1
            else:
                raise ValueError(f"unknown threshold method '{method}'")
            if value > best_value:
                best_value, best_threshold = value, float(threshold)
        thresholds[c] = best_threshold
    return thresholds


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def fit_temperature(
    y_true: np.ndarray, logits: np.ndarray, mask: np.ndarray, grid: Optional[np.ndarray] = None
) -> np.ndarray:
    """Per-class temperature by grid search on validation NLL.

    A grid beats gradient descent here: it needs no optimiser state, cannot
    diverge, and 60 points over [0.25, 4] is finer than the effect size.
    """
    grid = grid if grid is not None else np.exp(np.linspace(np.log(0.25), np.log(4.0), 60))
    temperatures = np.ones(y_true.shape[1], dtype=np.float64)
    for c in range(y_true.shape[1]):
        valid = _valid(mask, c)
        if valid.sum() == 0:
            continue
        truth, logit = y_true[valid, c], logits[valid, c]
        best_nll, best_temperature = np.inf, 1.0
        for temperature in grid:
            probability = 1.0 / (1.0 + np.exp(-logit / temperature))
            probability = np.clip(probability, 1e-7, 1 - 1e-7)
            nll = -np.mean(truth * np.log(probability) + (1 - truth) * np.log(1 - probability))
            if nll < best_nll:
                best_nll, best_temperature = nll, float(temperature)
        temperatures[c] = best_temperature
    return temperatures


def apply_temperature(logits: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits / temperatures.reshape(1, -1)))


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, mask: np.ndarray, bins: int = 10
) -> float:
    """Mean |confidence - accuracy| over equal-width bins, across all cells."""
    valid = mask > 0.5
    if valid.sum() == 0:
        return float("nan")
    probability = y_prob[valid]
    truth = y_true[valid] > 0.5
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for i in range(bins):
        inside = (probability >= edges[i]) & (probability < edges[i + 1] if i < bins - 1 else probability <= 1.0)
        if inside.sum() == 0:
            continue
        error += inside.sum() / len(probability) * abs(probability[inside].mean() - truth[inside].mean())
    return float(error)


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mask: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    labels: Sequence[str] = LABELS,
    with_curves: bool = False,
) -> Dict[str, object]:
    n_classes = y_true.shape[1]
    if thresholds is None:
        thresholds = np.full(n_classes, 0.5)

    per_class: Dict[str, Dict[str, float]] = {}
    aurocs, auprcs, f1s, precisions, recalls, specificities, accuracies = ([] for _ in range(7))

    for c, name in enumerate(labels):
        valid = _valid(mask, c)
        truth = y_true[valid, c] > 0.5
        score = y_prob[valid, c]
        predicted = score >= thresholds[c]

        tp = float((predicted & truth).sum())
        fp = float((predicted & ~truth).sum())
        fn = float((~predicted & truth).sum())
        tn = float((~predicted & ~truth).sum())

        auroc = _safe_auroc(truth.astype(float), score)
        auprc = _safe_auprc(truth.astype(float), score)
        precision = tp / max(tp + fp, EPS)
        recall = tp / max(tp + fn, EPS)
        specificity = tn / max(tn + fp, EPS)
        f1 = 2 * precision * recall / max(precision + recall, EPS)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, EPS)

        entry = {
            "auroc": auroc, "auprc": auprc, "accuracy": accuracy,
            "precision": precision, "recall": recall, "specificity": specificity, "f1": f1,
            "threshold": float(thresholds[c]),
            "n": int(valid.sum()), "n_positive": int(truth.sum()),
            "prevalence": float(truth.mean()) if valid.sum() else float("nan"),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }
        if with_curves:
            fpr, tpr = roc_curve_points(truth.astype(float), score)
            rec, prec = pr_curve_points(truth.astype(float), score)
            entry["roc"] = {"fpr": fpr, "tpr": tpr}
            entry["pr"] = {"recall": rec, "precision": prec}
        per_class[name] = entry

        if not np.isnan(auroc):
            aurocs.append(auroc)
        if not np.isnan(auprc):
            auprcs.append(auprc)
        f1s.append(f1); precisions.append(precision); recalls.append(recall)
        specificities.append(specificity); accuracies.append(accuracy)

    # Multi-label views over rows where every cell is labelled.
    fully_labelled = (mask > 0.5).all(axis=1)
    predictions = (y_prob >= thresholds.reshape(1, -1))
    if fully_labelled.sum():
        exact_match = float((predictions[fully_labelled] == (y_true[fully_labelled] > 0.5)).all(axis=1).mean())
    else:
        exact_match = float("nan")
    cells = mask > 0.5
    hamming = float((predictions[cells] == (y_true[cells] > 0.5)).mean()) if cells.sum() else float("nan")

    return {
        "per_class": per_class,
        "auroc_macro": float(np.mean(aurocs)) if aurocs else float("nan"),
        "auprc_macro": float(np.mean(auprcs)) if auprcs else float("nan"),
        "f1_macro": float(np.mean(f1s)),
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "specificity_macro": float(np.mean(specificities)),
        "accuracy_macro": float(np.mean(accuracies)),
        "exact_match": exact_match,
        "hamming_accuracy": hamming,
        "ece": expected_calibration_error(y_true, y_prob, mask),
        "n_rows": int(y_true.shape[0]),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mask: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    n_resamples: int = 1000,
    metrics: Sequence[str] = ("auroc_macro", "f1_macro", "auprc_macro"),
    seed: int = 1337,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap 95% CI.

    When `groups` (patient ids) is given, whole patients are resampled rather
    than individual studies — otherwise repeated studies from one patient make
    the interval look tighter than it is.
    """
    if n_resamples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    collected: Dict[str, List[float]] = {name: [] for name in metrics}
    per_class_auroc: Dict[str, List[float]] = {name: [] for name in LABELS}

    if groups is not None:
        unique_groups, group_index = np.unique(groups, return_inverse=True)
        buckets = [np.where(group_index == g)[0] for g in range(len(unique_groups))]

    n = y_true.shape[0]
    for _ in range(int(n_resamples)):
        if groups is not None:
            chosen = rng.integers(0, len(buckets), size=len(buckets))
            idx = np.concatenate([buckets[c] for c in chosen])
        else:
            idx = rng.integers(0, n, size=n)
        result = compute_metrics(y_true[idx], y_prob[idx], mask[idx], thresholds)
        for name in metrics:
            value = result.get(name)
            if value is not None and not np.isnan(value):
                collected[name].append(float(value))
        for name in LABELS:
            value = result["per_class"][name]["auroc"]
            if not np.isnan(value):
                per_class_auroc[name].append(float(value))

    out: Dict[str, Dict[str, float]] = {}
    for name, values in collected.items():
        if values:
            out[name] = {
                "lo": float(np.percentile(values, 2.5)),
                "hi": float(np.percentile(values, 97.5)),
                "mean": float(np.mean(values)),
            }
    for name, values in per_class_auroc.items():
        if values:
            out[f"auroc::{name}"] = {
                "lo": float(np.percentile(values, 2.5)),
                "hi": float(np.percentile(values, 97.5)),
                "mean": float(np.mean(values)),
            }
    return out


def format_table(metrics: Dict[str, object], title: str = "") -> str:
    """Markdown table, ready to paste into the report."""
    lines = []
    if title:
        lines.append(f"**{title}**\n")
    lines.append("| Finding | AUROC | AUPRC | Acc | Prec | Recall | F1 | Thr | n (pos) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, entry in metrics["per_class"].items():  # type: ignore[index]
        lines.append(
            f"| {name} | {entry['auroc']:.3f} | {entry['auprc']:.3f} | {entry['accuracy']:.3f} | "
            f"{entry['precision']:.3f} | {entry['recall']:.3f} | {entry['f1']:.3f} | "
            f"{entry['threshold']:.2f} | {entry['n']} ({entry['n_positive']}) |"
        )
    lines.append(
        f"| **Macro** | **{metrics['auroc_macro']:.3f}** | **{metrics['auprc_macro']:.3f}** | "
        f"**{metrics['accuracy_macro']:.3f}** | **{metrics['precision_macro']:.3f}** | "
        f"**{metrics['recall_macro']:.3f}** | **{metrics['f1_macro']:.3f}** | - | - |"
    )
    lines.append("")
    lines.append(
        f"Exact-match accuracy {metrics['exact_match']:.3f} · "
        f"Hamming accuracy {metrics['hamming_accuracy']:.3f} · ECE {metrics['ece']:.3f}"
    )
    return "\n".join(lines)
