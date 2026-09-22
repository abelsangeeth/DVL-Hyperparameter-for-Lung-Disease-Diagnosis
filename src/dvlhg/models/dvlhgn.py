"""The two trainable models.

Backbone            film + report -> node feature -> 4 logits   (stage 2)
HypergraphClassifier node features + hypergraph -> 4 logits     (stage 3)

They are trained separately rather than jointly, because the hypergraph is
transductive over the whole dataset: it needs every node's feature at once,
which cannot be held on a T4 while gradients flow through two transformers.
Stage 3 trains in seconds on frozen features, so the hypergraph design can be
iterated on dozens of times in the time a single end-to-end epoch would take.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..constants import NUM_LABELS
from ..fusion.cross_attn import ClassifierHead, CrossModalFusion
from ..hypergraph.construct import Hypergraph, QueryEdges
from ..hypergraph.hgnn import HypergraphEncoder
from ..vlm.encoders import VLMEncoder


class Backbone(nn.Module):
    """Vision-language encoder + cross-modal fusion + linear head."""

    def __init__(self, vlm: VLMEncoder, cfg, num_labels: int = NUM_LABELS):
        super().__init__()
        self.vlm = vlm
        self.fusion = CrossModalFusion(
            vision_dim=vlm.vision_dim,
            text_dim=vlm.text_dim,
            embed_dim=vlm.embed_dim,
            dim=int(cfg.fusion.dim),
            layers=int(cfg.fusion.layers),
            heads=int(cfg.fusion.heads),
            dropout=float(cfg.fusion.dropout),
            token_level=bool(cfg.fusion.token_level),
            max_image_tokens=int(cfg.fusion.max_image_tokens),
            pool=str(cfg.fusion.pool),
        )
        self.head = ClassifierHead(int(cfg.fusion.dim), num_labels, dropout=float(cfg.fusion.dropout))
        # Auxiliary single-modality heads. They cost almost nothing, they give
        # the image-only / text-only ablation for free, and the extra gradient
        # stops one modality from being ignored early in training.
        self.image_head = ClassifierHead(int(cfg.fusion.dim), num_labels, dropout=float(cfg.fusion.dropout))
        self.text_head = ClassifierHead(int(cfg.fusion.dim), num_labels, dropout=float(cfg.fusion.dropout))
        self.node_dim = int(cfg.fusion.dim)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_extras: bool = False,
    ) -> Dict[str, torch.Tensor]:
        out = self.vlm(images, input_ids, attention_mask)
        node, extras = self.fusion(
            img_tokens=out.img_tokens,
            img_pooled=out.img_pooled,
            txt_tokens=out.txt_tokens,
            txt_pooled=out.txt_pooled,
            txt_mask=out.txt_mask,
            return_attention=return_extras,
        )
        result = {
            "logits": self.head(node),
            "node": node,
            "image_logits": self.image_head(extras["h_image"]),
            "text_logits": self.text_head(extras["h_text"]),
            "img_pooled": out.img_pooled,
            "txt_pooled": out.txt_pooled,
            "gate": extras["gate"],
        }
        if return_extras:
            result["extras"] = extras
        return result

    def trainable_parameter_groups(self, lr: float, encoder_lr_scale: float, weight_decay: float):
        """Backbone blocks get a smaller learning rate than the new modules, and
        norms/biases are excluded from weight decay."""
        backbone_decay, backbone_plain, head_decay, head_plain = [], [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_backbone = name.startswith("vlm.")
            no_decay = param.ndim <= 1 or name.endswith(".bias")
            if is_backbone:
                (backbone_plain if no_decay else backbone_decay).append(param)
            else:
                (head_plain if no_decay else head_decay).append(param)
        groups = []
        encoder_lr = lr * float(encoder_lr_scale)
        if backbone_decay:
            groups.append({"params": backbone_decay, "lr": encoder_lr, "weight_decay": weight_decay})
        if backbone_plain:
            groups.append({"params": backbone_plain, "lr": encoder_lr, "weight_decay": 0.0})
        if head_decay:
            groups.append({"params": head_decay, "lr": lr, "weight_decay": weight_decay})
        if head_plain:
            groups.append({"params": head_plain, "lr": lr, "weight_decay": 0.0})
        return groups


class HypergraphClassifier(nn.Module):
    """HGNN over frozen node features, predicting a *correction* to the
    backbone's logits rather than a fresh decision.

        final_logits = backbone_logits + head([x ; hgnn(x)])

    with the head's last layer zero-initialised, so training starts exactly at
    the backbone's behaviour. This matters more than it looks: the HGNN sees
    only a few thousand training nodes, and a head learned from scratch on that
    reliably lands *below* the backbone it was meant to improve. As a residual,
    the graph can only spend capacity on what it actually adds, and the ablation
    "fusion vs fusion+hypergraph" becomes a clean measurement of that.
    """

    def __init__(self, node_dim: int, cfg, num_labels: int = NUM_LABELS):
        super().__init__()
        hidden = int(cfg.hypergraph.hidden)
        self.encoder = HypergraphEncoder(
            in_dim=node_dim,
            hidden=hidden,
            layers=int(cfg.hypergraph.layers),
            dropout=float(cfg.hypergraph.dropout),
            residual=bool(cfg.hypergraph.residual),
            learn_edge_weights=bool(cfg.hypergraph.learn_edge_weights),
            edge_dropout=float(cfg.hypergraph.get("edge_dropout", 0.1)),
        )
        self.input_norm = nn.LayerNorm(node_dim)
        self.head = ClassifierHead(
            node_dim + hidden, num_labels, dropout=float(cfg.hypergraph.dropout),
            hidden=hidden, zero_init=True,
        )
        # Scales the correction. Starts at 1; the trainer can watch it to see
        # how hard the graph is actually pushing.
        self.delta_scale = nn.Parameter(torch.ones(1))
        self.node_dim = node_dim
        self.hidden = hidden

    def forward(
        self,
        x: torch.Tensor,
        graph: Hypergraph,
        base_logits: torch.Tensor,
        return_states: bool = False,
    ):
        x = self.input_norm(x)
        if return_states:
            hidden, states = self.encoder(x, graph, return_states=True)
            delta = self.head(torch.cat([x, hidden], dim=-1)) * self.delta_scale
            return base_logits + delta, states
        hidden = self.encoder(x, graph)
        delta = self.head(torch.cat([x, hidden], dim=-1)) * self.delta_scale
        return base_logits + delta

    def forward_query(
        self,
        x_query: torch.Tensor,
        base_logits: torch.Tensor,
        query_edges: QueryEdges,
        bank_states,
        bank_degrees: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_norm(x_query.view(1, -1))
        hidden = self.encoder.forward_query(x, query_edges, bank_states, bank_degrees)
        delta = self.head(torch.cat([x, hidden], dim=-1)) * self.delta_scale
        return base_logits.view(1, -1) + delta

    @torch.no_grad()
    def bank_states(self, x: torch.Tensor, graph: Hypergraph):
        """Per-layer bank inputs + degrees, in the normalised space the query
        path expects (input_norm is applied here, once)."""
        was_training = self.training
        self.eval()
        normalised = self.input_norm(x)
        _, states = self.encoder(normalised, graph, return_states=True)
        degrees = self.encoder.node_degrees(graph)
        if was_training:
            self.train()
        return [s.detach() for s in states[:-1]], degrees.detach()
