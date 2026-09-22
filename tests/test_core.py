"""Tests for the pieces that are easy to get subtly wrong.

Run with:  PYTHONPATH=src python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dvlhg.config import DotDict, load_config
from dvlhg.constants import LABELS
from dvlhg.data.labels import apply_policy, pos_weight_from
from dvlhg.data.text import build_text, mask_findings, split_sections
from dvlhg.eval.metrics import (
    _safe_auprc,
    _safe_auroc,
    compute_metrics,
    fit_temperature,
    tune_thresholds,
)
from dvlhg.hypergraph.construct import Hypergraph, QueryEdges, knn
from dvlhg.hypergraph.hgnn import HypergraphEncoder
from dvlhg.train.losses import AsymmetricLoss, MaskedBCE

sklearn = pytest.importorskip("sklearn.metrics", reason="scikit-learn is the metric reference")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("decimals", [1, 2, 3, 6])
def test_auroc_and_ap_match_sklearn_including_ties(decimals):
    rng = np.random.default_rng(decimals)
    for _ in range(8):
        n = int(rng.integers(40, 400))
        truth = (rng.random(n) < rng.uniform(0.1, 0.7)).astype(float)
        if truth.sum() in (0, n):
            continue
        score = np.round(rng.random(n), decimals)   # heavy ties at low decimals
        assert _safe_auroc(truth, score) == pytest.approx(
            sklearn.roc_auc_score(truth, score), abs=1e-9
        )
        assert _safe_auprc(truth, score) == pytest.approx(
            sklearn.average_precision_score(truth, score), abs=1e-9
        )


def test_compute_metrics_matches_sklearn():
    rng = np.random.default_rng(0)
    n, c = 400, len(LABELS)
    y = (rng.random((n, c)) < 0.3).astype(np.float32)
    p = np.clip(y * 0.4 + rng.random((n, c)) * 0.6, 0, 1).astype(np.float32)
    mask = np.ones((n, c), np.float32)

    result = compute_metrics(y, p, mask, thresholds=np.full(c, 0.5))
    predicted = p >= 0.5
    assert result["auroc_macro"] == pytest.approx(sklearn.roc_auc_score(y, p, average="macro"), abs=1e-9)
    assert result["auprc_macro"] == pytest.approx(
        sklearn.average_precision_score(y, p, average="macro"), abs=1e-9
    )
    assert result["f1_macro"] == pytest.approx(
        sklearn.f1_score(y, predicted, average="macro", zero_division=0), abs=1e-9
    )
    assert result["exact_match"] == pytest.approx(sklearn.accuracy_score(y, predicted), abs=1e-9)


def test_masked_cells_are_excluded_from_every_metric():
    rng = np.random.default_rng(1)
    n, c = 300, len(LABELS)
    y = (rng.random((n, c)) < 0.4).astype(np.float32)
    p = np.clip(y * 0.5 + rng.random((n, c)) * 0.5, 0, 1).astype(np.float32)
    mask = np.ones((n, c), np.float32)
    mask[:120, 0] = 0.0

    result = compute_metrics(y, p, mask, thresholds=np.full(c, 0.5))
    entry = result["per_class"][LABELS[0]]
    assert entry["n"] == 180
    assert entry["auroc"] == pytest.approx(sklearn.roc_auc_score(y[120:, 0], p[120:, 0]), abs=1e-9)


def test_degenerate_class_gives_nan_not_a_crash():
    y = np.zeros((50, len(LABELS)), np.float32)
    p = np.random.default_rng(0).random((50, len(LABELS))).astype(np.float32)
    result = compute_metrics(y, p, np.ones_like(y))
    assert np.isnan(result["per_class"][LABELS[0]]["auroc"])


def test_threshold_tuning_beats_a_fixed_half_on_its_own_split():
    rng = np.random.default_rng(3)
    n, c = 500, len(LABELS)
    y = (rng.random((n, c)) < 0.12).astype(np.float32)        # rare positives
    p = np.clip(y * 0.35 + rng.random((n, c)) * 0.5, 0, 1).astype(np.float32)
    mask = np.ones((n, c), np.float32)
    tuned = tune_thresholds(y, p, mask, "f1")
    assert compute_metrics(y, p, mask, tuned)["f1_macro"] >= compute_metrics(
        y, p, mask, np.full(c, 0.5)
    )["f1_macro"]


def test_temperature_fit_reduces_nll_on_miscalibrated_logits():
    rng = np.random.default_rng(4)
    n, c = 600, len(LABELS)
    y = (rng.random((n, c)) < 0.3).astype(np.float32)
    logits = (y * 2 - 1) * rng.uniform(0.5, 1.5, (n, c)) * 6.0   # over-confident
    mask = np.ones((n, c), np.float32)
    temperatures = fit_temperature(y, logits, mask)

    def nll(t):
        p = np.clip(1 / (1 + np.exp(-logits / t)), 1e-7, 1 - 1e-7)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    assert nll(temperatures.reshape(1, -1)) <= nll(np.ones((1, c))) + 1e-9


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
import pandas as pd  # noqa: E402


def _frame(values):
    return pd.DataFrame(np.asarray(values, dtype=float), columns=LABELS)


def test_uncertain_policy_ignore_masks_the_cell():
    frame = _frame([[1.0, 0.0, -1.0, np.nan]])
    y, mask = apply_policy(frame, "ignore", blank_policy="zeros")
    assert y[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert mask[0].tolist() == [1.0, 1.0, 0.0, 1.0]   # uncertain ignored, blank = negative


def test_uncertain_policy_ones_and_zeros():
    frame = _frame([[-1.0, -1.0, -1.0, -1.0]])
    y_ones, mask_ones = apply_policy(frame, "ones")
    assert y_ones[0].tolist() == [1.0] * 4 and mask_ones[0].tolist() == [1.0] * 4
    y_zeros, _ = apply_policy(frame, "zeros")
    assert y_zeros[0].tolist() == [0.0] * 4


def test_per_class_policy_is_applied_per_class():
    frame = _frame([[-1.0, -1.0, -1.0, -1.0]])
    policy = {"Atelectasis": "ones", "Cardiomegaly": "zeros", "Edema": "ones", "Pleural Effusion": "ignore"}
    y, mask = apply_policy(frame, "per_class", policy)
    assert y[0].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert mask[0].tolist() == [1.0, 1.0, 1.0, 0.0]


def test_blank_policy_ignore_masks_unmentioned_findings():
    frame = _frame([[np.nan, 1.0, np.nan, 0.0]])
    _, mask = apply_policy(frame, "ignore", blank_policy="ignore")
    assert mask[0].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_pos_weight_is_negatives_over_positives_and_clamped():
    y = np.zeros((100, 4), np.float32)
    y[:10, 0] = 1.0        # 10 positives, 90 negatives -> 9.0
    y[:1, 1] = 1.0         # 1 positive, 99 negatives  -> clamped to the cap
    weights = pos_weight_from(y, np.ones_like(y), cap=10.0)
    assert weights[0] == pytest.approx(9.0)
    assert weights[1] == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# report text
# --------------------------------------------------------------------------- #
REPORT = """                                 FINAL REPORT
 EXAMINATION:  CHEST (PA AND LAT)

 INDICATION:  ___F with new onset dyspnea  // eval for effusion

 TECHNIQUE:  Chest PA and lateral

 COMPARISON:  None.

 FINDINGS:
 There is a moderate left pleural effusion. The cardiac silhouette is enlarged.

 IMPRESSION:
 Moderate left pleural effusion and cardiomegaly.
"""


def test_sections_are_split_on_headers():
    sections = split_sections(REPORT)
    assert "INDICATION" in sections and "FINDINGS" in sections and "IMPRESSION" in sections
    assert "dyspnea" in sections["INDICATION"]


def test_indication_mode_excludes_the_findings_that_produced_the_labels():
    text = build_text(REPORT, mode="indication")
    assert "dyspnea" in text
    for leaked in ("cardiac silhouette is enlarged", "moderate left pleural effusion"):
        assert leaked not in text
    # the indication legitimately mentions the reason for exam
    assert "eval for effusion" in text


def test_findings_mode_is_leaky_by_construction():
    text = build_text(REPORT, mode="findings")
    assert "pleural effusion" in text and "enlarged" in text


def test_masked_mode_redacts_the_finding_terms():
    text = build_text(REPORT, mode="findings_masked")
    assert "effusion" not in text and "enlarged" not in text
    assert "[finding]" in text


def test_mask_findings_covers_confounders():
    masked = mask_findings("there is consolidation and a pneumothorax with atelectasis")
    assert "consolidation" not in masked and "pneumothorax" not in masked and "atelectasis" not in masked


def test_deidentification_underscores_are_stripped():
    assert "___" not in build_text(REPORT, mode="indication")


def test_unsectioned_report_does_not_leak_into_indication_mode():
    # A single paragraph with no headers could be anything; returning it whole
    # would silently leak the findings, so `indication` must return nothing.
    assert build_text("moderate pleural effusion and cardiomegaly.", mode="indication") == ""


def test_none_mode_is_empty():
    assert build_text(REPORT, mode="none") == ""


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def test_masked_bce_ignores_masked_cells():
    logits = torch.zeros(4, 4)
    targets = torch.ones(4, 4)
    full = MaskedBCE()(logits, targets, torch.ones(4, 4))
    half = MaskedBCE()(logits, targets, torch.cat([torch.ones(4, 2), torch.zeros(4, 2)], dim=1))
    assert full == pytest.approx(float(half), abs=1e-6)   # same per-cell value, same mean


def test_masked_bce_gradient_is_zero_on_masked_cells():
    logits = torch.zeros(2, 4, requires_grad=True)
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    MaskedBCE()(logits, torch.ones(2, 4), mask).backward()
    assert torch.allclose(logits.grad[:, 1], torch.zeros(2))
    assert not torch.allclose(logits.grad[:, 0], torch.zeros(2))


def test_asymmetric_loss_downweights_easy_negatives():
    loss = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    mask = torch.ones(1, 4)
    easy = loss(torch.full((1, 4), -8.0), torch.zeros(1, 4), mask)     # confidently negative
    hard = loss(torch.full((1, 4), 2.0), torch.zeros(1, 4), mask)      # wrong and confident
    assert float(easy) < float(hard)


def test_sample_weight_scales_the_loss():
    loss = MaskedBCE()
    logits, targets, mask = torch.zeros(4, 4), torch.ones(4, 4), torch.ones(4, 4)
    full = loss(logits, targets, mask, torch.ones(4))
    half = loss(logits, targets, mask, torch.full((4,), 0.5))
    assert float(full) == pytest.approx(float(half), abs=1e-6)  # uniform weights cancel in the mean


# --------------------------------------------------------------------------- #
# hypergraph
# --------------------------------------------------------------------------- #
def _toy_graph(n=40, edges_per_node=3, seed=0):
    rng = np.random.default_rng(seed)
    members, groups = [], []
    for i in range(n):
        others = rng.choice([j for j in range(n) if j != i], size=edges_per_node, replace=False)
        members.append(np.concatenate([[i], others]))
        groups.append(0)
    rows = np.concatenate(members)
    cols = np.concatenate([np.full(len(m), e, np.int64) for e, m in enumerate(members)])
    incidence = torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([rows, cols])),
        torch.ones(len(rows)),
        size=(n, len(members)),
    ).coalesce()
    return Hypergraph(
        incidence=incidence,
        edge_group=torch.zeros(len(members), dtype=torch.long),
        num_nodes=n,
        num_edges=len(members),
        edge_sizes=torch.tensor([float(len(m)) for m in members]),
        config={"knn_fused": edges_per_node},
    ), members


def test_hgnn_propagate_matches_the_dense_formula():
    """Dv^-1/2 H W De^-1 H^T Dv^-1/2 X, written out densely as an independent check."""
    graph, _ = _toy_graph(n=30, seed=1)
    encoder = HypergraphEncoder(in_dim=8, hidden=8, layers=1, dropout=0.0, edge_dropout=0.0)
    x = torch.randn(30, 8)
    weights = encoder.edge_weights(graph, training=False)

    dense_h = graph.incidence.to_dense()
    dv = (dense_h * weights.unsqueeze(0)).sum(dim=1).clamp_min(1e-6)
    de = dense_h.sum(dim=0).clamp_min(1.0)
    operator = (
        torch.diag(dv.pow(-0.5)) @ dense_h @ torch.diag(weights) @ torch.diag(de.pow(-1.0))
        @ dense_h.t() @ torch.diag(dv.pow(-0.5))
    )
    assert torch.allclose(encoder.propagate(x, graph, weights), operator @ x, atol=1e-5)


def test_hgnn_operator_preserves_a_constant_signal_shape():
    graph, _ = _toy_graph(n=25, seed=2)
    encoder = HypergraphEncoder(in_dim=4, hidden=4, layers=2, dropout=0.0, edge_dropout=0.0)
    out = encoder(torch.randn(25, 4), graph)
    assert out.shape == (25, 4) and torch.isfinite(out).all()


def test_edge_dropout_only_fires_in_training():
    graph, _ = _toy_graph(n=20, seed=3)
    encoder = HypergraphEncoder(in_dim=4, hidden=4, layers=1, edge_dropout=0.9)
    a = encoder.edge_weights(graph, training=False)
    b = encoder.edge_weights(graph, training=False)
    assert torch.allclose(a, b)                       # deterministic at eval
    assert (encoder.edge_weights(graph, training=True) == 0).any()


def test_inductive_query_matches_an_independent_dense_frozen_bank_computation():
    """The served path is the project's riskiest code. This reimplements the
    frozen-bank formula from scratch and checks `forward_query` against it."""
    torch.manual_seed(0)
    graph, _ = _toy_graph(n=36, seed=4)
    encoder = HypergraphEncoder(in_dim=6, hidden=6, layers=2, dropout=0.0, edge_dropout=0.0)
    encoder.eval()

    x = torch.randn(36, 6)
    _, states = encoder(x, graph, return_states=True)
    bank_states = [s.detach() for s in states[:-1]]
    bank_degrees = encoder.node_degrees(graph)

    query_x = torch.randn(6)
    query_edges = QueryEdges(
        members=[np.array([1, 4, 9, 17]), np.array([2, 3])],
        groups=[0, 0],
        similarities=[np.ones(4, np.float32), np.ones(2, np.float32)],
    )
    produced = encoder.forward_query(query_x, query_edges, bank_states, bank_degrees)

    # --- independent reference ---------------------------------------------
    group_w = torch.nn.functional.softplus(encoder.group_logit)
    edge_w = torch.stack([group_w[g] for g in query_edges.groups])
    dv_q = edge_w.sum().clamp_min(1e-6)
    inv_q = dv_q.pow(-0.5)
    inv_bank = bank_degrees.clamp_min(1e-6).pow(-0.5)

    hidden = query_x.view(1, -1)
    for layer in range(encoder.layers):
        scaled = bank_states[layer] * inv_bank.unsqueeze(1)
        acc = torch.zeros(1, bank_states[layer].shape[1])
        for e, members in enumerate(query_edges.members):
            inner = scaled[torch.as_tensor(members)].sum(dim=0, keepdim=True) + hidden * inv_q
            acc = acc + inner * (edge_w[e] / float(len(members) + 1))
        aggregated = acc * inv_q
        hidden = encoder._apply_layer(layer, aggregated, hidden)

    assert torch.allclose(produced, hidden, atol=1e-6)


def test_knn_excludes_self_and_is_ordered_by_similarity():
    features = torch.randn(50, 12)
    idx, sim = knn(features, k=5)
    assert idx.shape == (50, 5)
    assert all(i not in idx[i] for i in range(50))
    assert (np.diff(sim, axis=1) <= 1e-6).all()


def test_knn_against_a_restricted_candidate_pool():
    features = torch.randn(8, 5)
    pool = torch.randn(30, 5)
    idx, _ = knn(features, k=3, exclude_self=False, candidates=pool)
    assert idx.shape == (8, 3) and idx.max() < 30


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_config_loads_and_overrides_are_typed():
    cfg = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "default.yaml",
        ["train.epochs=3", "train.amp=false", "hypergraph.knn_fused=7", "text.mode=findings"],
    )
    assert cfg.train.epochs == 3 and isinstance(cfg.train.epochs, int)
    assert cfg.train.amp is False
    assert cfg.hypergraph.knn_fused == 7
    assert cfg.text.mode == "findings"
    assert Path(cfg.paths.cache).is_absolute()


def test_dotdict_round_trips():
    cfg = DotDict({"a": {"b": [1, {"c": 2}]}})
    assert cfg.a.b[1].c == 2
    assert cfg.to_dict() == {"a": {"b": [1, {"c": 2}]}}
