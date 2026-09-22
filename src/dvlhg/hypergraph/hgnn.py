"""Hypergraph convolution (Feng et al., HGNN) with a matching inductive path.

The layer operator is

    X' = sigma( Dv^-1/2 H W De^-1 H^T Dv^-1/2 X Theta )

computed as two sparse mat-muls, never as an N x N matrix, so a 20k-node graph
costs a few hundred megabytes rather than several gigabytes.

Edge weights are learned per *group* (kNN-fused, kNN-image, meta-view, ...) with
an optional per-edge residual. The group term is what makes inference on an
unseen study well defined: a new node's hyperedges have no learned per-edge
weight, so they use the weight their group learned.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import EDGE_GROUPS
from .construct import Hypergraph, QueryEdges


class HypergraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 384,
        layers: int = 2,
        dropout: float = 0.3,
        residual: bool = True,
        learn_edge_weights: bool = True,
        edge_dropout: float = 0.1,
        num_groups: int = len(EDGE_GROUPS),
    ):
        super().__init__()
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.residual = bool(residual)
        self.edge_dropout = float(edge_dropout)
        self.learn_edge_weights = bool(learn_edge_weights)

        dims = [in_dim] + [hidden] * self.layers
        self.weights = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=True) for i in range(self.layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dims[i + 1]) for i in range(self.layers)])
        self.skips = nn.ModuleList(
            [
                nn.Linear(dims[i], dims[i + 1], bias=False) if dims[i] != dims[i + 1] else nn.Identity()
                for i in range(self.layers)
            ]
        )
        # softplus(0.5413) == 1.0, so every group starts at unit weight.
        self.group_logit = nn.Parameter(torch.full((num_groups,), 0.5413))
        self.edge_delta: Optional[nn.Parameter] = None
        self._edge_delta_size = 0

    # -- edge weights ------------------------------------------------------- #
    def bind_graph(self, graph: Hypergraph) -> None:
        """Allocate the per-edge weight residual for this graph."""
        if self.learn_edge_weights and self._edge_delta_size != graph.num_edges:
            device = self.group_logit.device
            self.edge_delta = nn.Parameter(torch.zeros(graph.num_edges, device=device))
            self._edge_delta_size = graph.num_edges

    def edge_weights(self, graph: Hypergraph, training: bool = False) -> torch.Tensor:
        weights = self.group_logit[graph.edge_group]
        if self.edge_delta is not None and self.edge_delta.numel() == graph.num_edges:
            weights = weights + self.edge_delta
        weights = F.softplus(weights)
        if training and self.edge_dropout > 0:
            keep = (torch.rand_like(weights) >= self.edge_dropout).float()
            weights = weights * keep / max(1.0 - self.edge_dropout, 1e-6)
        return weights

    def group_weights(self) -> torch.Tensor:
        return F.softplus(self.group_logit)

    # -- transductive ------------------------------------------------------- #
    def propagate(
        self, x: torch.Tensor, graph: Hypergraph, weights: torch.Tensor
    ) -> torch.Tensor:
        """Dv^-1/2 H W De^-1 H^T Dv^-1/2 x"""
        incidence = graph.incidence
        node_degree = torch.sparse.mm(incidence, weights.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
        inv_sqrt = node_degree.pow(-0.5).unsqueeze(1)                   # [N, 1]

        scattered = torch.sparse.mm(incidence.t(), x * inv_sqrt)        # [E, d]
        scattered = scattered * (weights / graph.edge_sizes.clamp_min(1.0)).unsqueeze(1)
        gathered = torch.sparse.mm(incidence, scattered)                # [N, d]
        return gathered * inv_sqrt

    def _apply_layer(self, index: int, aggregated: torch.Tensor, residual_in: torch.Tensor) -> torch.Tensor:
        out = self.weights[index](aggregated)
        out = self.norms[index](out)
        out = F.gelu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        if self.residual:
            out = out + self.skips[index](residual_in)
        return out

    def forward(
        self, x: torch.Tensor, graph: Hypergraph, return_states: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        self.bind_graph(graph)
        weights = self.edge_weights(graph, training=self.training)
        states: List[torch.Tensor] = [x]
        hidden = x
        for index in range(self.layers):
            aggregated = self.propagate(hidden, graph, weights)
            hidden = self._apply_layer(index, aggregated, hidden)
            states.append(hidden)
        if return_states:
            return hidden, states
        return hidden

    @torch.no_grad()
    def node_degrees(self, graph: Hypergraph) -> torch.Tensor:
        weights = self.edge_weights(graph, training=False)
        return torch.sparse.mm(graph.incidence, weights.unsqueeze(1)).squeeze(1).clamp_min(1e-6)

    # -- inductive ---------------------------------------------------------- #
    def forward_query(
        self,
        x_query: torch.Tensor,
        query_edges: QueryEdges,
        bank_states: Sequence[torch.Tensor],
        bank_degrees: torch.Tensor,
    ) -> torch.Tensor:
        """Run the encoder for one unseen study against a frozen bank.

        `bank_states[l]` must be the input of layer l for every bank node, and
        `bank_degrees` their node degrees in the frozen graph.

        Protocol (frozen-bank extension): the query joins new hyperedges of its
        own and the bank's edges, sizes and degrees are left untouched. That is
        what makes inference stable — one patient's prediction can never shift
        another's — and this routine computes that quantity exactly. It is not
        identical to rebuilding the whole hypergraph with the query inserted;
        `dvlhg eval` reports the measured gap between the two. See
        docs/HYPERGRAPH.md.
        """
        if len(query_edges) == 0:
            raise ValueError("query joined no hyperedges - check build_query_edges")
        device = x_query.device
        weights = self.group_weights().to(device)

        edge_weight = torch.stack([weights[g] for g in query_edges.groups])       # [Eq]
        query_degree = edge_weight.sum().clamp_min(1e-6)
        query_inv_sqrt = query_degree.pow(-0.5)
        bank_inv_sqrt = bank_degrees.to(device).clamp_min(1e-6).pow(-0.5)

        hidden = x_query.view(1, -1)
        for index in range(self.layers):
            bank_x = bank_states[index].to(device)
            scaled_bank = bank_x * bank_inv_sqrt.unsqueeze(1)
            accumulator = torch.zeros(1, bank_x.shape[1], device=device, dtype=hidden.dtype)
            for e, members in enumerate(query_edges.members):
                idx = torch.as_tensor(members, dtype=torch.long, device=device)
                inner = scaled_bank.index_select(0, idx).sum(dim=0, keepdim=True)
                inner = inner + hidden * query_inv_sqrt
                size = float(len(members) + 1)
                accumulator = accumulator + inner * (edge_weight[e] / size)
            aggregated = accumulator * query_inv_sqrt
            hidden = self._apply_layer(index, aggregated, hidden)
        return hidden


@torch.no_grad()
def collect_bank_states(
    encoder: HypergraphEncoder, x: torch.Tensor, graph: Hypergraph
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Per-layer inputs and node degrees for export - everything the inductive
    path needs to reproduce the transductive computation."""
    was_training = encoder.training
    encoder.eval()
    _, states = encoder(x, graph, return_states=True)
    degrees = encoder.node_degrees(graph)
    if was_training:
        encoder.train()
    # states has L+1 entries; the query path consumes the first L.
    return [state.detach().cpu() for state in states[:-1]], degrees.detach().cpu()
