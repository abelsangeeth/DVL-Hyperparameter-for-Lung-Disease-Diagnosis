"""Single-study inference against an exported bundle.

Loads the backbone, the hypergraph read-out and the frozen neighbour bank, then
answers one (film, report) pair at a time. The hypergraph step uses the
frozen-bank protocol from docs/HYPERGRAPH.md: the query joins hyperedges of its
own and nothing in the bank moves, so two people uploading at the same moment
cannot influence each other's result.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..config import DotDict, load_config
from ..constants import DISCLAIMER, LABELS
from ..data.text import build_text, clean as clean_text, describe_mode
from ..eval.explain import grad_cam, neighbour_records, overlay_heatmap, text_importance, to_png_base64
from ..hypergraph.construct import build_query_edges
from ..models.dvlhgn import HypergraphClassifier
from ..utils import get_logger, load_json, pick_device
from ..vlm.encoders import build_vlm

LOG = get_logger()


def preprocess_pil(image, image_size: int, cache_size: int) -> torch.Tensor:
    """PIL image -> [1, 1, image_size, image_size] float in [0, 1].

    This reproduces the *training* pipeline exactly, in both of its steps:

        1. `data.prepare.preprocess_image` writes the cache at `cache_size`
           (grayscale, shorter-side resize, centre crop)
        2. `data.dataset._center_crop_resize` takes the cached film down to
           `image_size` for the encoder

    Going straight from the original file to `image_size` skips step 1 and hands
    the model a sharper image than anything it saw in training. The difference
    looks small and is not: it flattens the outputs, because the encoder is
    being fed off-distribution inputs.
    """
    from ..data.prepare import preprocess_image_pil

    cached = preprocess_image_pil(image, cache_size)              # uint8 [cache, cache]
    # ascontiguousarray: the cache is a read-only memmap and torch will not
    # take a non-writable buffer without complaining.
    tensor = torch.from_numpy(np.ascontiguousarray(cached)).float().div_(255.0).unsqueeze(0)
    if cache_size != image_size:
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=(image_size, image_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
    return tensor.unsqueeze(0)


class Predictor:
    def __init__(self, bundle_dir: str | Path, device: str = "auto", neighbours: int = 6):
        self.bundle_dir = Path(bundle_dir)
        if not (self.bundle_dir / "bundle.json").exists():
            raise FileNotFoundError(
                f"{self.bundle_dir}/bundle.json not found - run `dvlhg eval --export` first"
            )
        self.meta = load_json(self.bundle_dir / "bundle.json")
        self.device = pick_device(device)
        self.neighbour_count = int(neighbours)

        cfg = load_config(self.bundle_dir / "config.yaml") if (self.bundle_dir / "config.yaml").exists() else None
        backbone_state = torch.load(self.bundle_dir / "backbone.pt", map_location=self.device, weights_only=False)
        self.cfg: DotDict = DotDict(backbone_state.get("config", cfg.to_dict() if cfg else {}))

        from ..models.dvlhgn import Backbone

        vlm, self.tokenize = build_vlm(self.cfg)
        self.backbone = Backbone(vlm, self.cfg).to(self.device)
        self.backbone.load_state_dict(backbone_state["model"])
        self.backbone.eval()

        hypergraph_state = torch.load(
            self.bundle_dir / "hypergraph.pt", map_location=self.device, weights_only=False
        )
        node_dim = int(hypergraph_state.get("node_dim", self.cfg.fusion.dim))
        self.hgnn = HypergraphClassifier(node_dim=node_dim, cfg=self.cfg).to(self.device)
        missing, unexpected = self.hgnn.load_state_dict(hypergraph_state["model"], strict=False)
        expected_unexpected = {"encoder.edge_delta"}
        if missing or (set(unexpected) - expected_unexpected):
            LOG.warning("hypergraph weights: missing=%s unexpected=%s", missing, unexpected)
        self.hgnn.eval()

        bank = np.load(self.bundle_dir / "bank.npz", allow_pickle=True)
        self.bank = {
            "fused": torch.from_numpy(bank["fused"]).float().to(self.device),
            "image": torch.from_numpy(bank["image"]).float().to(self.device),
            "text": torch.from_numpy(bank["text"]).float().to(self.device),
        }
        self.bank["bank_view"] = bank["view"]
        self.bank_states = [torch.from_numpy(s).float().to(self.device) for s in bank["states"]]
        self.bank_degrees = torch.from_numpy(bank["degrees"]).float().to(self.device)
        self.centroids = bank["centroids"] if bank["centroids"].size else None

        meta_path = self.bundle_dir / "bank_meta.csv"
        self.bank_meta: List[dict] = (
            pd.read_csv(meta_path).to_dict("records") if meta_path.exists() else []
        )

        self.thresholds = np.asarray(self.meta["thresholds"], dtype=np.float64)
        self.temperatures = np.asarray(self.meta["temperatures"], dtype=np.float64)
        self.image_size = int(self.meta.get("image_size", 224))
        self.cache_size = int(self.meta.get("cache_size", self.image_size))
        self.text_mode = str(self.meta.get("text_mode", "indication"))
        self.graph_config = dict(self.meta.get("graph_config", {}))
        self._warm_up()
        LOG.info(
            "predictor ready: %d bank nodes, backbone=%s, text mode=%s",
            self.bank["fused"].shape[0], self.meta.get("vlm_backend"), self.text_mode,
        )

    def _warm_up(self) -> None:
        """One throwaway prediction so the first real request is not 20x slower."""
        try:
            from PIL import Image

            blank = Image.new("L", (self.image_size, self.image_size), color=32)
            buffer = io.BytesIO()
            blank.save(buffer, format="PNG")
            started = time.perf_counter()
            self.predict(buffer.getvalue(), report="", explain=True, neighbours=1)
            LOG.info("warm-up pass: %.0f ms", (time.perf_counter() - started) * 1000)
        except Exception as exc:  # noqa: BLE001 - warm-up must never block startup
            LOG.warning("warm-up skipped (%s)", exc)

    # -- helpers ------------------------------------------------------------ #
    def _decode_image(self, data: bytes):
        from PIL import Image

        return Image.open(io.BytesIO(data))

    def _tokenise(
        self, report: str, prepared_already: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, str, List[str]]:
        if prepared_already:
            prepared = clean_text(report or "", lowercase=True)
        else:
            prepared = build_text(report or "", mode=self.text_mode, lowercase=True)
            if not prepared and (report or "").strip():
                # The user typed free text with no section headers. Dropping it
                # silently would make the report box look broken, so take it as
                # written -- most people paste the indication line by itself.
                prepared = clean_text(report, lowercase=True)
        if not prepared:
            prepared = str(self.cfg.text.empty_placeholder)
        tokens = self.tokenize([prepared])
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        readable = self._token_strings(tokens["input_ids"][0], prepared)
        return input_ids, attention_mask, prepared, readable

    def _token_strings(self, ids: torch.Tensor, prepared: str = "") -> List[str]:
        if self.backbone.vlm.backend == "dummy":
            # The hashing tokenizer emits ["[cls]"] + words, so the words are
            # exactly the right labels even though the ids cannot be inverted.
            return ["[CLS]"] + prepared.split()
        hf = getattr(getattr(self.backbone.vlm.text, "transformer", None), "config", None)
        try:  # best effort; a backend may not expose a HuggingFace vocabulary
            tokenizer = getattr(self, "_hf_tokenizer", None)
            if tokenizer is None and hf is not None:
                name = getattr(hf, "_name_or_path", "")
                if name:
                    from transformers import AutoTokenizer as AT

                    tokenizer = AT.from_pretrained(name)
                    self._hf_tokenizer = tokenizer
            if tokenizer is not None:
                return tokenizer.convert_ids_to_tokens(ids.tolist())
        except Exception:  # noqa: BLE001
            pass
        return [str(int(i)) for i in ids.tolist()]

    # -- main entry point --------------------------------------------------- #
    @torch.no_grad()
    def _backbone_pass(self, image: torch.Tensor, input_ids, attention_mask):
        return self.backbone(image, input_ids, attention_mask)

    def predict(
        self,
        image_bytes: bytes,
        report: str = "",
        explain: bool = True,
        neighbours: Optional[int] = None,
        report_is_prepared: bool = False,
    ) -> Dict[str, object]:
        started = time.perf_counter()
        pil = self._decode_image(image_bytes)
        image = preprocess_pil(pil, self.image_size, self.cache_size).to(self.device)
        input_ids, attention_mask, prepared_text, token_strings = self._tokenise(
            report, prepared_already=report_is_prepared
        )

        out = self._backbone_pass(image, input_ids, attention_mask)
        node = out["node"][0]
        backbone_logits = out["logits"][0].float().cpu().numpy()

        query_edges = build_query_edges(
            fused_q=node,
            image_q=out["img_pooled"][0],
            text_q=out["txt_pooled"][0],
            bank=self.bank,
            graph_config=self.graph_config,
            meta_q=None,
            meta_index=None,
            centroids=self.centroids,
        )
        with torch.no_grad():
            hypergraph_logits = (
                self.hgnn.forward_query(
                    node, out["logits"][0], query_edges, self.bank_states, self.bank_degrees
                ).float().cpu().numpy()[0]
            )

        logits = hypergraph_logits if self.meta.get("serving_variant") == "fusion_hypergraph" else backbone_logits
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits / self.temperatures, -30, 30)))
        backbone_probabilities = 1.0 / (1.0 + np.exp(-np.clip(backbone_logits / self.temperatures, -30, 30)))

        findings = []
        for c, name in enumerate(LABELS):
            probability = float(probabilities[c])
            threshold = float(self.thresholds[c])
            findings.append(
                {
                    "label": name,
                    "probability": round(probability, 4),
                    "threshold": round(threshold, 3),
                    "flagged": bool(probability >= threshold),
                    "margin": round(probability - threshold, 4),
                    "without_hypergraph": round(float(backbone_probabilities[c]), 4),
                }
            )

        response: Dict[str, object] = {
            "findings": findings,
            "text_used": prepared_text,
            "text_mode": self.text_mode,
            "text_mode_description": describe_mode(self.text_mode),
            "modality_gate": round(float(out["gate"][0]), 4),
            "serving_variant": self.meta.get("serving_variant"),
            "disclaimer": DISCLAIMER,
            "latency_ms": None,
        }

        limit = self.neighbour_count if neighbours is None else int(neighbours)
        if limit > 0 and query_edges.members and self.bank_meta:
            response["neighbours"] = neighbour_records(
                query_edges.members[0], query_edges.similarities[0], self.bank_meta, limit=limit
            )

        if explain:
            top = int(np.argmax(probabilities))
            heat, extras = grad_cam(self.backbone, image, input_ids, attention_mask, top)
            if heat is not None:
                base = image[0, 0].detach().cpu().numpy()
                response["saliency"] = {
                    "label": LABELS[top],
                    "overlay_png": to_png_base64(overlay_heatmap(base, heat)),
                }
            important = text_importance(extras, token_strings)
            if important:
                response["text_attention"] = important

        response["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return response

    def model_card(self) -> Dict[str, object]:
        return {
            "labels": LABELS,
            "serving_variant": self.meta.get("serving_variant"),
            "data_source": self.meta.get("data_source"),
            "vlm_backend": self.meta.get("vlm_backend"),
            "text_mode": self.text_mode,
            "text_mode_description": describe_mode(self.text_mode),
            "thresholds": {name: round(float(t), 3) for name, t in zip(LABELS, self.thresholds)},
            "bank_nodes": int(self.bank["fused"].shape[0]),
            "edge_groups": self.meta.get("edge_groups"),
            "test_metrics": self.meta.get("test_metrics"),
            "disclaimer": DISCLAIMER,
            "intended_use": (
                "Research and teaching demonstration of a diffusion-augmented "
                "vision-language hypergraph model. It has not been validated on any "
                "prospective cohort, is not registered as a medical device, and must "
                "not inform care for any patient."
            ),
            "known_limitations": [
                "Trained on a subset of a single-centre dataset; performance on films from "
                "other scanners, populations or positioning is unknown.",
                "Labels come from an NLP labeler applied to radiology reports, not from "
                "expert image review, so the ceiling is the labeler's accuracy.",
                "Saliency maps show where the model looked, not whether that was the right "
                "place to look.",
                "The four findings are far from the full space of chest pathology; anything "
                "outside them is invisible to this model.",
            ],
        }
