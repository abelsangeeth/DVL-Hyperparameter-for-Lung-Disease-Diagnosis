"""Report figures.

Conventions held throughout, so the whole set reads as one system:
  * one measure per axis — never two y-scales on one plot; a second measure gets
    its own stacked panel
  * identity is never colour alone — every multi-series plot carries a legend
    whose entries also state the value, and each series gets its own dash pattern
  * fixed categorical slot order (blue, orange, aqua, yellow), never cycled
  * recessive grid and spines; the data is the darkest thing on the surface
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..constants import EDGE_GROUPS
from ..utils import get_logger, load_json

LOG = get_logger()

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated slot order
DASHES = [(None, None), (5, 2), (1.6, 1.6), (7, 2, 1.6, 2)]


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=9)


def _dash(ax_line, index: int) -> None:
    on_off = DASHES[index % len(DASHES)]
    if on_off[0] is not None:
        ax_line.set_dashes(list(on_off))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    LOG.info("figure -> %s", path)


def write_figures(
    cfg, summary: Dict[str, object], features: Dict[str, np.ndarray], manifest: pd.DataFrame, out_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    results = summary.get("results", {})
    best = "fusion_hypergraph" if "fusion_hypergraph" in results else next(iter(results), None)
    if best is None:
        LOG.warning("no results to plot")
        return

    test = results[best]["test"]
    _roc_figure(plt, test, out_dir / "roc_curves.png", best)
    _pr_figure(plt, test, out_dir / "pr_curves.png", best)
    _ablation_figure(plt, results, out_dir / "ablation.png")
    _edge_weight_figure(plt, summary.get("hyperedge_group_weights"), out_dir / "hyperedge_weights.png")
    _history_figure(plt, cfg, out_dir / "training_history.png")


def _roc_figure(plt, test: Dict[str, object], path: Path, variant: str) -> None:
    per_class = test["per_class"]  # type: ignore[index]
    if not any("roc" in entry for entry in per_class.values()):  # type: ignore[union-attr]
        return
    fig, ax = plt.subplots(figsize=(5.4, 4.6), facecolor=SURFACE)
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, linestyle=":", zorder=1)
    for i, (name, entry) in enumerate(per_class.items()):  # type: ignore[union-attr]
        roc = entry.get("roc")
        if not roc or not roc["fpr"]:
            continue
        (line,) = ax.plot(
            roc["fpr"], roc["tpr"], color=SERIES[i % len(SERIES)], linewidth=2,
            label=f"{name} — AUROC {entry['auroc']:.3f}", zorder=3,
        )
        _dash(line, i)
    _style(ax, f"ROC — {variant}, test split", "False positive rate", "True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    legend = ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    for text in legend.get_texts():
        text.set_color(INK_SOFT)
    _save(fig, path)
    plt.close(fig)


def _pr_figure(plt, test: Dict[str, object], path: Path, variant: str) -> None:
    per_class = test["per_class"]  # type: ignore[index]
    if not any("pr" in entry for entry in per_class.values()):  # type: ignore[union-attr]
        return
    fig, ax = plt.subplots(figsize=(5.4, 4.6), facecolor=SURFACE)
    for i, (name, entry) in enumerate(per_class.items()):  # type: ignore[union-attr]
        pr = entry.get("pr")
        if not pr or not pr["recall"]:
            continue
        (line,) = ax.plot(
            pr["recall"], pr["precision"], color=SERIES[i % len(SERIES)], linewidth=2,
            label=f"{name} — AP {entry['auprc']:.3f} (prev {entry['prevalence']:.2f})", zorder=3,
        )
        _dash(line, i)
    _style(ax, f"Precision–recall — {variant}, test split", "Recall", "Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(INK_SOFT)
    _save(fig, path)
    plt.close(fig)


def _ablation_figure(plt, results: Dict[str, object], path: Path) -> None:
    """Dot-and-interval, not bars.

    These are estimates with uncertainty, not magnitudes with a meaningful zero.
    A bar chart would either start at 0 (wasting the whole plot on the region no
    model occupies) or start partway up, which exaggerates small differences by
    truncating the baseline. A dot with its interval has no baseline obligation,
    and it puts the comparison that matters -- do the intervals overlap? --
    directly in front of the reader.
    """
    names = list(results.keys())
    if not names:
        return

    values, lows, highs = [], [], []
    for name in names:
        value = results[name]["test"]["auroc_macro"]  # type: ignore[index]
        ci = results[name].get("ci", {}).get("auroc_macro")  # type: ignore[union-attr]
        values.append(value)
        lows.append(ci["lo"] if ci else value)
        highs.append(ci["hi"] if ci else value)

    fig, ax = plt.subplots(figsize=(6.6, 0.62 * len(names) + 1.5), facecolor=SURFACE)
    positions = np.arange(len(names))

    span = max(highs) - min(lows) or 0.1
    left = min(min(lows), 0.5) - 0.04 * span
    right = max(highs) + 0.02 * span
    label_x = right + 0.04 * span

    if left <= 0.5 <= right:
        ax.axvline(0.5, color=INK_MUTED, linewidth=1, linestyle=":", zorder=1)
        # Top of the (inverted) axis, clear of the x tick labels.
        ax.text(0.5, -0.5, " chance", fontsize=8, color=INK_MUTED, va="center")

    for position, (value, low, high) in zip(positions, zip(values, lows, highs)):
        ax.plot([low, high], [position, position], color=SERIES[0], linewidth=2,
                solid_capstyle="round", alpha=0.45, zorder=3)
        ax.plot([value], [position], marker="o", markersize=7, color=SERIES[0], zorder=4)
        text = f"{value:.3f}" if low == value else f"{value:.3f}  [{low:.3f}, {high:.3f}]"
        ax.text(label_x, position, text, va="center", fontsize=9, color=INK,
                fontfamily="DejaVu Sans Mono")

    ax.set_yticks(positions)
    ax.set_yticklabels([name.replace("_", " ") for name in names], fontsize=9.5)
    _style(ax, "Macro AUROC by variant (test split, 95% CI over patients)", "Macro AUROC", "")
    ax.set_xlim(left, label_x + 0.42 * span)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    _save(fig, path)
    plt.close(fig)


def _edge_weight_figure(plt, weights: Optional[Dict[str, float]], path: Path) -> None:
    if not weights:
        return
    names = [name for name in EDGE_GROUPS if name in weights]
    values = [weights[name] for name in names]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.2), facecolor=SURFACE)
    positions = np.arange(len(names))
    ax.bar(positions, values, width=0.6, color=SERIES[0], zorder=3)
    ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle=":", zorder=2)
    for position, value in zip(positions, values):
        ax.text(position, value + 0.02, f"{value:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(positions)
    ax.set_xticklabels([name.replace("_", "\n") for name in names], fontsize=8.5)
    _style(ax, "Learned hyperedge group weights (dotted line = initial value)", "", "softplus weight")
    ax.grid(axis="x", visible=False)
    _save(fig, path)
    plt.close(fig)


def _history_figure(plt, cfg, path: Path) -> None:
    logs = Path(cfg.paths.logs)
    history_file = logs / "backbone_history.json"
    if not history_file.exists():
        return
    history = load_json(history_file)
    if not history:
        return
    epochs = [row["epoch"] for row in history]

    # Loss and AUROC are different measures: two panels, never two y-axes.
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 5.0), sharex=True, facecolor=SURFACE)
    (line,) = axes[0].plot(epochs, [row["train_loss"] for row in history], color=SERIES[0], linewidth=2)
    _style(axes[0], "Backbone training loss", "", "loss")
    axes[0].legend([line], ["train loss"], loc="upper right", frameon=False, fontsize=8.5,
                   labelcolor=INK_SOFT)

    for i, key in enumerate(("val_auroc_macro", "val_f1_macro")):
        if key not in history[0]:
            continue
        (line,) = axes[1].plot(
            epochs, [row[key] for row in history], color=SERIES[i % len(SERIES)], linewidth=2,
            marker="o", markersize=4,
            label=f"{key.replace('val_', 'val ').replace('_', ' ')} — best {max(row[key] for row in history):.3f}",
        )
        _dash(line, i)
    _style(axes[1], "Validation metrics", "epoch", "score")
    legend = axes[1].legend(loc="lower right", frameon=False, fontsize=8.5)
    for text in legend.get_texts():
        text.set_color(INK_SOFT)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)
