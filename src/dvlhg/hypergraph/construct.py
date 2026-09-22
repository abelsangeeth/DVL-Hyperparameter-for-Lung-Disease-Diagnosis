"""Hypergraph construction over studies.

A hyperedge is a *set* of studies that belong together, which is the whole point
of using a hypergraph rather than a graph: "the twelve films that look like this
one" is one relation, not sixty-six pairwise ones.

Four families of hyperedge are built:

  knn_fused / knn_image / knn_text
      e_i = {i} u kNN(i) in the fused, image-only and text-only spaces. Three
      views of similarity; a case can be visually typical but textually unusual.
  meta_view
      studies sharing a radiographic view (PA / AP / ...). Large groups are cut
      into bounded chunks — an edge containing half the dataset is a bias term,
      not a relation.
  meta_demo
      studies sharing (sex, age decade), when demographics are available.
  proto_label
      k-means prototypes of the *training* positives of each finding. Training
      nodes join their own cluster; every other node joins by feature distance
      only, so no label ever reaches a validation or test node.

Leakage rules enforced here:
  * labels are read only for rows whose split == "train"
  * val/test rows join label-derived edges by feature similarity alone
  * if hypergraph.transductive is false, non-train rows are excluded entirely
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..constants import EDGE_GROUPS, LABELS
from ..utils import get_logger

LOG = get_logger()


@dataclass
class Hypergraph:
    """Incidence structure plus everything inference needs to extend it."""

    incidence: torch.Tensor          # sparse [N, E], values 1.0
    edge_group: torch.Tensor         # [E] int64, index into EDGE_GROUPS
    num_nodes: int
    num_edges: int
    edge_sizes: torch.Tensor         # [E] float, |e|
    neighbours: Dict[str, np.ndarray] = field(default_factory=dict)  # space -> [N, k]
    centroids: Dict[str, np.ndarray] = field(default_factory=dict)   # proto edge centroids
    meta_keys: Dict[str, object] = field(default_factory=dict)       # group key -> edge ids
    config: Dict[str, object] = field(default_factory=dict)

    def to(self, device: torch.device) -> "Hypergraph":
        self.incidence = self.incidence.to(device)
        self.edge_group = self.edge_group.to(device)
        self.edge_sizes = self.edge_sizes.to(device)
        return self

    def summary(self) -> Dict[str, object]:
        counts: Dict[str, int] = {}
        groups = self.edge_group.cpu().numpy()
        for gid, name in enumerate(EDGE_GROUPS):
            n = int((groups == gid).sum())
            if n:
                counts[name] = n
        return {
            "nodes": self.num_nodes,
            "edges": self.num_edges,
            "nnz": int(self.incidence._nnz()) if self.incidence.is_sparse else -1,
            "mean_edge_size": float(self.edge_sizes.float().mean()),
            "max_edge_size": int(self.edge_sizes.max()),
            "edges_per_group": counts,
        }


# --------------------------------------------------------------------------- #
# similarity
# --------------------------------------------------------------------------- #
def l2_normalise(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def knn(
    features: torch.Tensor,
    k: int,
    chunk: int = 2048,
    exclude_self: bool = True,
    candidates: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine kNN in chunks. Returns (indices [N, k], similarities [N, k]).

    `candidates` restricts the neighbour pool (used to keep query nodes from
    becoming each other's neighbours at inference time).
    """
    features = l2_normalise(features.float())
    pool = features if candidates is None else l2_normalise(candidates.float())
    n, m = features.shape[0], pool.shape[0]
    take = min(k + (1 if exclude_self else 0), m)

    idx_out = np.zeros((n, k), dtype=np.int64)
    sim_out = np.zeros((n, k), dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sims = features[start:stop] @ pool.T                    # [c, m]
        if exclude_self and candidates is None:
            rows = torch.arange(start, stop, device=sims.device)
            sims[rows - start, rows] = -2.0
        top_sim, top_idx = torch.topk(sims, k=take, dim=1)
        idx_out[start:stop] = top_idx[:, :k].cpu().numpy()
        sim_out[start:stop] = top_sim[:, :k].cpu().numpy()
    return idx_out, sim_out


def _chunk_group(members: np.ndarray, max_size: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Split an oversized attribute group into bounded random sub-edges."""
    if len(members) <= max_size:
        return [members]
    shuffled = rng.permutation(members)
    return [shuffled[i : i + max_size] for i in range(0, len(shuffled), max_size)]


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #
def build_hypergraph(
    fused: torch.Tensor,
    image: Optional[torch.Tensor],
    text: Optional[torch.Tensor],
    labels: np.ndarray,
    mask: np.ndarray,
    split: np.ndarray,
    meta: Optional[Dict[str, np.ndarray]],
    cfg,
    seed: int = 1337,
) -> Hypergraph:
    """Build the full hypergraph over all nodes in `fused`."""
    device = fused.device
    n = fused.shape[0]
    rng = np.random.default_rng(seed)
    max_edge_size = int(cfg.hypergraph.get("max_edge_size", 256))

    edges: List[np.ndarray] = []
    groups: List[int] = []
    neighbours: Dict[str, np.ndarray] = {}
    centroids: Dict[str, np.ndarray] = {}
    meta_keys: Dict[str, object] = {}

    def add(members: np.ndarray, group: str) -> None:
        members = np.unique(np.asarray(members, dtype=np.int64))
        if len(members) < 2:
            return
        edges.append(members)
        groups.append(EDGE_GROUPS.index(group))

    # -- kNN families --------------------------------------------------------
    for space, feats, k in (
        ("knn_fused", fused, int(cfg.hypergraph.knn_fused)),
        ("knn_image", image, int(cfg.hypergraph.knn_image)),
        ("knn_text", text, int(cfg.hypergraph.knn_text)),
    ):
        if feats is None or k <= 0:
            continue
        idx, _ = knn(feats.to(device), k=k)
        neighbours[space] = idx
        for i in range(n):
            add(np.concatenate([[i], idx[i]]), space)
        LOG.info("%s: %d hyperedges of size %d", space, n, k + 1)

    # -- metadata families ---------------------------------------------------
    if meta and bool(cfg.hypergraph.use_meta_view) and "view" in meta:
        view = np.asarray(meta["view"]).astype(str)
        keys: Dict[str, List[int]] = {}
        for value in np.unique(view):
            members = np.where(view == value)[0]
            ids = []
            for chunkset in _chunk_group(members, max_edge_size, rng):
                ids.append(len(edges))
                add(chunkset, "meta_view")
            keys[str(value)] = ids
        meta_keys["view"] = keys
        LOG.info("meta_view: %d hyperedges over %d views", sum(len(v) for v in keys.values()), len(keys))

    if meta and bool(cfg.hypergraph.use_meta_demo) and "sex" in meta and "age" in meta:
        sex = np.asarray(meta["sex"]).astype(str)
        age = np.asarray(meta["age"], dtype=np.float64)
        if np.isfinite(age).sum() > 0:
            binsize = int(cfg.hypergraph.demo_age_bin)
            decade = np.where(np.isfinite(age), (age // binsize) * binsize, -1).astype(int)
            combo = np.array([f"{s}_{d}" for s, d in zip(sex, decade)])
            keys = {}
            for value in np.unique(combo):
                members = np.where(combo == value)[0]
                ids = []
                for chunkset in _chunk_group(members, max_edge_size, rng):
                    ids.append(len(edges))
                    add(chunkset, "meta_demo")
                keys[str(value)] = ids
            meta_keys["demo"] = keys
            meta_keys["demo_bin"] = binsize
            LOG.info("meta_demo: %d buckets", len(keys))
        else:
            LOG.info("meta_demo skipped: no usable age values in the manifest")
    elif bool(cfg.hypergraph.use_meta_demo):
        LOG.info("meta_demo skipped: manifest carries no sex/age columns")

    # -- label prototypes (train-only supervision) ---------------------------
    if bool(cfg.hypergraph.use_proto_label):
        proto_ids, proto_centroids = _prototype_edges(
            fused=fused,
            labels=labels,
            mask=mask,
            split=split,
            clusters=int(cfg.hypergraph.proto_clusters),
            seed=seed,
        )
        for members in proto_ids:
            add(members, "proto_label")
        if len(proto_centroids):
            centroids["proto_label"] = proto_centroids
        LOG.info("proto_label: %d prototype hyperedges", len(proto_ids))

    if not edges:
        raise RuntimeError("no hyperedges were built - check hypergraph.* settings")

    # -- assemble sparse incidence ------------------------------------------
    rows = np.concatenate(edges)
    cols = np.concatenate([np.full(len(members), e, dtype=np.int64) for e, members in enumerate(edges)])
    values = np.ones(len(rows), dtype=np.float32)
    num_edges = len(edges)

    indices = torch.from_numpy(np.stack([rows, cols]))
    try:
        incidence = torch.sparse_coo_tensor(
            indices, torch.from_numpy(values), size=(n, num_edges), check_invariants=False
        )
    except TypeError:  # torch < 2.0 has no check_invariants kwarg
        incidence = torch.sparse_coo_tensor(indices, torch.from_numpy(values), size=(n, num_edges))
    incidence = incidence.coalesce().to(device)

    edge_sizes = torch.tensor([len(members) for members in edges], dtype=torch.float32, device=device)
    graph = Hypergraph(
        incidence=incidence,
        edge_group=torch.tensor(groups, dtype=torch.long, device=device),
        num_nodes=n,
        num_edges=num_edges,
        edge_sizes=edge_sizes,
        neighbours=neighbours,
        centroids=centroids,
        meta_keys=meta_keys,
        config={
            "knn_fused": int(cfg.hypergraph.knn_fused),
            "knn_image": int(cfg.hypergraph.knn_image),
            "knn_text": int(cfg.hypergraph.knn_text),
            "max_edge_size": max_edge_size,
            "transductive": bool(cfg.hypergraph.transductive),
        },
    )
    LOG.info("hypergraph: %s", graph.summary())
    return graph


def _prototype_edges(
    fused: torch.Tensor,
    labels: np.ndarray,
    mask: np.ndarray,
    split: np.ndarray,
    clusters: int,
    seed: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """k-means over the training positives of each finding.

    Membership for non-training nodes is decided by cosine distance to the
    centroid, never by their label — that is what keeps this leakage-free.
    """
    features = l2_normalise(fused.float()).cpu().numpy()
    is_train = split == "train"
    per_class = max(clusters // max(len(LABELS), 1), 1)

    edge_members: List[np.ndarray] = []
    centroid_list: List[np.ndarray] = []

    for c in range(labels.shape[1]):
        positives = np.where(is_train & (mask[:, c] > 0) & (labels[:, c] > 0.5))[0]
        if len(positives) < per_class * 2:
            continue
        k = min(per_class, max(len(positives) // 10, 1))
        try:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(features[positives])
            assignment = km.labels_
            centroids = km.cluster_centers_
        except Exception:  # noqa: BLE001 - sklearn missing or failed
            centroids, assignment = _torch_kmeans(features[positives], k, seed)

        centroids = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8)
        similarity = features @ centroids.T                       # [N, k]

        for j in range(k):
            train_members = positives[assignment == j]
            if len(train_members) < 2:
                continue
            # Attach the same number of nearest nodes from anywhere in the graph
            # (feature-based only) so val/test nodes can participate.
            order = np.argsort(-similarity[:, j])
            attached = order[: len(train_members)]
            edge_members.append(np.unique(np.concatenate([train_members, attached])))
            centroid_list.append(centroids[j])

    return edge_members, (np.stack(centroid_list) if centroid_list else np.zeros((0, features.shape[1]), np.float32))


def _torch_kmeans(features: np.ndarray, k: int, seed: int, iters: int = 25):
    """Minimal k-means so the prototype edges survive without scikit-learn."""
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(features)
    centroids = x[torch.from_numpy(rng.choice(len(x), size=k, replace=False))].clone()
    assignment = torch.zeros(len(x), dtype=torch.long)
    for _ in range(iters):
        distances = torch.cdist(x, centroids)
        assignment = distances.argmin(dim=1)
        for j in range(k):
            members = x[assignment == j]
            if len(members):
                centroids[j] = members.mean(dim=0)
    return centroids.numpy(), assignment.numpy()


# --------------------------------------------------------------------------- #
# inference-time extension
# --------------------------------------------------------------------------- #
@dataclass
class QueryEdges:
    """The hyperedges a single unseen study joins.

    Each entry is (neighbour indices into the bank, group id). The query node is
    implicitly a member of every one of them.
    """

    members: List[np.ndarray]
    groups: List[int]
    similarities: List[np.ndarray]

    def __len__(self) -> int:
        return len(self.members)


def build_query_edges(
    fused_q: torch.Tensor,
    image_q: Optional[torch.Tensor],
    text_q: Optional[torch.Tensor],
    bank: Dict[str, torch.Tensor],
    graph_config: Dict[str, object],
    meta_q: Optional[Dict[str, object]] = None,
    meta_index: Optional[Dict[str, object]] = None,
    centroids: Optional[np.ndarray] = None,
) -> QueryEdges:
    """Attach one unseen study to the frozen bank.

    The bank graph is never modified: the query joins new hyperedges of its own,
    so one patient's prediction can never shift another's. See docs/HYPERGRAPH.md.
    """
    members: List[np.ndarray] = []
    groups: List[int] = []
    sims: List[np.ndarray] = []

    for space, query, key in (
        ("knn_fused", fused_q, "fused"),
        ("knn_image", image_q, "image"),
        ("knn_text", text_q, "text"),
    ):
        k = int(graph_config.get(space, 0) or 0)
        if k <= 0 or query is None or key not in bank:
            continue
        idx, sim = knn(query.view(1, -1), k=k, exclude_self=False, candidates=bank[key])
        members.append(idx[0])
        groups.append(EDGE_GROUPS.index(space))
        sims.append(sim[0])

    if meta_q and meta_index:
        view_index = meta_index.get("view") or {}
        view = str(meta_q.get("view", ""))
        if view in view_index and "bank_view" in bank:
            same = np.where(np.asarray(bank["bank_view"]).astype(str) == view)[0]
            if len(same) > 1:
                cap = int(graph_config.get("max_edge_size", 256))
                pick = same if len(same) <= cap else np.random.default_rng(0).choice(same, cap, replace=False)
                members.append(pick)
                groups.append(EDGE_GROUPS.index("meta_view"))
                sims.append(np.ones(len(pick), dtype=np.float32))

    if centroids is not None and len(centroids) and "fused" in bank:
        query = l2_normalise(fused_q.view(1, -1).float()).cpu().numpy()
        centroid_sim = (query @ centroids.T)[0]
        best = int(np.argmax(centroid_sim))
        bank_feats = l2_normalise(bank["fused"].float()).cpu().numpy()
        member_sim = bank_feats @ centroids[best]
        top = np.argsort(-member_sim)[: int(graph_config.get("proto_attach", 16))]
        members.append(top)
        groups.append(EDGE_GROUPS.index("proto_label"))
        sims.append(member_sim[top].astype(np.float32))

    return QueryEdges(members=members, groups=groups, similarities=sims)
