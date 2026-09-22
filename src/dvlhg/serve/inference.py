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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from ..config import DotDict, load_config
from ..constants import DISCLAIMER, LABELS
from ..data.text import build_text, clean as clean_text, describe_mode
from ..eval.explain import (
    grad_cam_multi,
    neighbour_records,
    overlay_heatmap,
    text_importance,
    to_png_base64,
)
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
        self._diffusion = None          # lazily loaded on first /api/generate
        self._diffusion_missing = False
        self._diffusion_probe: Optional[Dict[str, object]] = None
        self._zero_shot_cache: Optional[torch.Tensor] = None
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

    # -- vision-language panel --------------------------------------------- #
    ZERO_SHOT_POSITIVE = "chest x-ray showing {}"
    ZERO_SHOT_NEGATIVE = "chest x-ray with no {}"

    def _zero_shot_embeddings(self) -> torch.Tensor:
        """Normalised text embeddings for the 4 positive + 4 negative prompts.

        Computed once. This is the *fine-tuned* backbone's space, not stock
        BiomedCLIP's: after training the two towers have drifted, so these
        similarities describe this model rather than the pretrained checkpoint.
        """
        if self._zero_shot_cache is not None:
            return self._zero_shot_cache
        prompts = (
            [self.ZERO_SHOT_POSITIVE.format(name.lower()) for name in LABELS]
            + [self.ZERO_SHOT_NEGATIVE.format(name.lower()) for name in LABELS]
        )
        tokens = self.tokenize(prompts)
        with torch.no_grad():
            _, pooled = self.backbone.vlm.encode_text(
                tokens["input_ids"].to(self.device), tokens["attention_mask"].to(self.device)
            )
        self._zero_shot_cache = torch.nn.functional.normalize(pooled.float(), dim=-1)
        return self._zero_shot_cache

    def _zero_shot_scores(self, img_pooled: torch.Tensor) -> List[Dict[str, float]]:
        text = self._zero_shot_embeddings()
        image = torch.nn.functional.normalize(img_pooled.view(1, -1).float(), dim=-1)
        similarity = (image @ text.T)[0].cpu().numpy()
        n = len(LABELS)
        return [
            {
                "label": name,
                "positive": round(float(similarity[i]), 4),
                "negative": round(float(similarity[i + n]), 4),
                "margin": round(float(similarity[i] - similarity[i + n]), 4),
            }
            for i, name in enumerate(LABELS)
        ]

    def _full_pass(self, image: torch.Tensor, input_ids, attention_mask):
        """Backbone + hypergraph for one (film, text) pair."""
        out = self._backbone_pass(image, input_ids, attention_mask)
        node = out["node"][0]
        query_edges = build_query_edges(
            fused_q=node,
            image_q=out["img_pooled"][0],
            text_q=out["txt_pooled"][0],
            bank=self.bank,
            graph_config=self.graph_config,
            centroids=self.centroids,
        )
        with torch.no_grad():
            hypergraph_logits = (
                self.hgnn.forward_query(
                    node, out["logits"][0], query_edges, self.bank_states, self.bank_degrees
                ).float().cpu().numpy()[0]
            )
        backbone_logits = out["logits"][0].float().cpu().numpy()
        use_hypergraph = self.meta.get("serving_variant") == "fusion_hypergraph"
        logits = hypergraph_logits if use_hypergraph else backbone_logits
        return out, query_edges, logits, backbone_logits

    def _probabilities(self, logits) -> "np.ndarray":
        return 1.0 / (1.0 + np.exp(-np.clip(logits / self.temperatures, -30, 30)))

    # -- diffusion panel ---------------------------------------------------- #
    def diffusion_available(self) -> bool:
        return (self.bundle_dir / "diffusion.pt").exists()

    def _load_diffusion(self):
        if self._diffusion is not None:
            return self._diffusion
        if self._diffusion_missing:
            return None
        path = self.bundle_dir / "diffusion.pt"
        if not path.exists():
            self._diffusion_missing = True
            return None
        from ..diffusion.trainer import load_diffusion_from

        self._diffusion = load_diffusion_from(path, self.device)
        LOG.info("diffusion model loaded for the demo panel")
        return self._diffusion

    def diffusion_info(self) -> Dict[str, object]:
        """Whether the diffusion panel will actually do anything.

        A freshly initialised UNet emits exactly zero (every residual branch and
        the output projection are zero-init by design), so an undertrained model
        returns the *same* film for every label vector. That looks like a broken
        demo rather than an untrained one, so we measure it and say so:
        `conditioning_strength` is the mean absolute difference between the
        model's prediction under two different label vectors.
        """
        path = self.bundle_dir / "diffusion.pt"
        if not path.exists():
            return {"available": False}
        if self._diffusion_probe is not None:
            return self._diffusion_probe

        diffusion = self._load_diffusion()
        if diffusion is None:
            return {"available": False}

        size = int(self.cfg.diffusion.image_size)
        generator = torch.Generator(device="cpu").manual_seed(0)
        noise = torch.randn(1, 1, size, size, generator=generator).to(self.device)
        step = torch.tensor([diffusion.timesteps // 2], device=self.device)
        with torch.no_grad():
            none_on = diffusion.model(noise, step, torch.zeros(1, len(LABELS), device=self.device))
            all_on = diffusion.model(noise, step, torch.ones(1, len(LABELS), device=self.device))
            strength = float((none_on - all_on).abs().mean())
            magnitude = float(all_on.abs().mean())

        state = torch.load(path, map_location="cpu", weights_only=False)
        self._diffusion_probe = {
            "available": True,
            "image_size": size,
            "epochs_trained": state.get("epoch"),
            "from_ema": bool(state.get("from_ema", False)),
            "conditioning_strength": round(strength, 6),
            "output_magnitude": round(magnitude, 6),
            # Below this the label vector moves the prediction less than the
            # sampler's own rounding, so every sample comes back identical.
            "usable": bool(strength > 1e-4),
        }
        return self._diffusion_probe

    def generate(
        self,
        labels: Sequence[float],
        guidance_scale: Optional[float] = None,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, object]:
        """Sample one synthetic film conditioned on a label vector."""
        diffusion = self._load_diffusion()
        if diffusion is None:
            raise FileNotFoundError(
                "no diffusion.pt in this bundle - train one with `dvlhg diffusion`, "
                "then re-run `dvlhg eval` with serve.include_diffusion: true"
            )
        from ..diffusion.ddpm import to_image_space

        started = time.perf_counter()
        if seed is not None:
            torch.manual_seed(int(seed))
        size = int(self.cfg.diffusion.image_size)
        resolved_steps = int(steps or self.cfg.diffusion.sample_steps)
        resolved_guidance = float(
            guidance_scale if guidance_scale is not None else self.cfg.diffusion.guidance_scale
        )
        vector = torch.tensor([[float(v) for v in labels]], dtype=torch.float32, device=self.device)
        sample = diffusion.ddim_sample(
            shape=(1, 1, size, size),
            labels=vector,
            steps=resolved_steps,
            guidance_scale=resolved_guidance,
            device=self.device,
        )
        array = to_image_space(sample)[0, 0].cpu().numpy()
        return {
            "png": to_png_base64(array),
            "labels": {name: bool(float(v) > 0.5) for name, v in zip(LABELS, labels)},
            "size": size,
            "steps": resolved_steps,
            "guidance_scale": resolved_guidance,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "synthetic": True,
            "conditioning": self.diffusion_info(),
            "disclaimer": "Generated image. Not a real patient film.",
        }

    def refine(
        self,
        image_bytes: bytes,
        strength: Optional[float] = None,
        steps: int = 20,
        labels: Optional[Sequence[float]] = None,
    ) -> Dict[str, object]:
        """SDEdit: noise the uploaded film partway down the schedule, denoise back."""
        diffusion = self._load_diffusion()
        if diffusion is None:
            raise FileNotFoundError("no diffusion.pt in this bundle")
        from ..diffusion.ddpm import to_image_space, to_model_space

        started = time.perf_counter()
        size = int(self.cfg.diffusion.image_size)
        resolved_strength = float(
            strength if strength is not None else self.cfg.diffusion.refine_t
        )
        pil = self._decode_image(image_bytes)
        small = preprocess_pil(pil, size, size).to(self.device)

        vector = None
        if labels is not None:
            vector = torch.tensor([[float(v) for v in labels]], dtype=torch.float32, device=self.device)
        refined = diffusion.refine(
            to_model_space(small), vector,
            strength=resolved_strength,
            steps=int(steps),
            guidance_scale=1.0 if vector is None else float(self.cfg.diffusion.guidance_scale),
        )
        original = small[0, 0].cpu().numpy()
        output = to_image_space(refined)[0, 0].cpu().numpy()
        return {
            "original_png": to_png_base64(original),
            "refined_png": to_png_base64(output),
            "difference_png": to_png_base64(np.clip(np.abs(output - original) * 3.0, 0, 1)),
            "strength": resolved_strength,
            "steps": int(steps),
            "size": size,
            "mean_abs_change": round(float(np.abs(output - original).mean()), 4),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    def predict(
        self,
        image_bytes: bytes,
        report: str = "",
        explain: bool = True,
        neighbours: Optional[int] = None,
        report_is_prepared: bool = False,
        vlm_panel: bool = True,
    ) -> Dict[str, object]:
        started = time.perf_counter()
        pil = self._decode_image(image_bytes)
        image = preprocess_pil(pil, self.image_size, self.cache_size).to(self.device)
        input_ids, attention_mask, prepared_text, token_strings = self._tokenise(
            report, prepared_already=report_is_prepared
        )

        out, query_edges, logits, backbone_logits = self._full_pass(image, input_ids, attention_mask)
        probabilities = self._probabilities(logits)
        backbone_probabilities = self._probabilities(backbone_logits)

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
            "diffusion_available": self.diffusion_available(),
            "disclaimer": DISCLAIMER,
            "latency_ms": None,
        }

        limit = self.neighbour_count if neighbours is None else int(neighbours)
        if limit > 0 and query_edges.members and self.bank_meta:
            response["neighbours"] = neighbour_records(
                query_edges.members[0], query_edges.similarities[0], self.bank_meta, limit=limit
            )

        # ---- Grad-CAM for every finding, from one forward pass --------------
        if explain:
            maps, extras = grad_cam_multi(
                self.backbone, image, input_ids, attention_mask, range(len(LABELS))
            )
            if maps:
                base = image[0, 0].detach().cpu().numpy()
                response["saliency"] = {
                    LABELS[index]: to_png_base64(overlay_heatmap(base, heat))
                    for index, heat in maps.items()
                }
                response["saliency_default"] = LABELS[int(np.argmax(probabilities))]
                response["film_png"] = to_png_base64(base)
            important = text_importance(extras, token_strings)
            if important:
                response["text_attention"] = important

        # ---- vision-language panel ------------------------------------------
        if vlm_panel:
            panel: Dict[str, object] = {
                "backend": self.meta.get("vlm_backend"),
                "zero_shot": self._zero_shot_scores(out["img_pooled"][0]),
                "gate": round(float(out["gate"][0]), 4),
                "text_used": prepared_text,
            }
            # What does the report actually contribute? Re-run the whole path
            # with the text blanked -- the same substitution training used for
            # modality dropout -- and report the difference per finding.
            blank_ids, blank_mask, _, _ = self._tokenise("", prepared_already=False)
            _, _, blank_logits, _ = self._full_pass(image, blank_ids, blank_mask)
            blank_probabilities = self._probabilities(blank_logits)
            panel["image_only"] = [
                {
                    "label": name,
                    "probability": round(float(blank_probabilities[c]), 4),
                    "delta": round(float(probabilities[c] - blank_probabilities[c]), 4),
                }
                for c, name in enumerate(LABELS)
            ]
            panel["mean_abs_text_effect"] = round(
                float(np.abs(probabilities - blank_probabilities).mean()), 4
            )
            response["vlm"] = panel

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
            "diffusion_available": self.diffusion_available(),
            "image_size": self.image_size,
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
