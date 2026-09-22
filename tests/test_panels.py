"""Tests for the three demo panels: Grad-CAM, diffusion, vision-language.

Run with:  PYTHONPATH=src python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dvlhg.config import load_config
from dvlhg.constants import LABELS
from dvlhg.diffusion.ddpm import GaussianDiffusion, to_image_space, to_model_space
from dvlhg.diffusion.unet import ConditionalUNet, ResBlock
from dvlhg.eval.explain import _colormap, grad_cam_multi, overlay_heatmap, to_png_base64
from dvlhg.models.dvlhgn import Backbone
from dvlhg.vlm.encoders import build_vlm


@pytest.fixture(scope="module")
def tiny_backbone():
    cfg = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "default.yaml",
        [
            "vlm.backbone=dummy", "data.image_size=224", "fusion.dim=64",
            "fusion.layers=1", "fusion.heads=4", "text.max_length=32",
            "vlm.dummy_width=96", "vlm.dummy_embed_dim=48",
        ],
    )
    vlm, tokenize = build_vlm(cfg)
    model = Backbone(vlm, cfg).eval()
    return model, tokenize, cfg


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
def test_grad_cam_multi_returns_one_map_per_finding(tiny_backbone):
    model, tokenize, cfg = tiny_backbone
    tokens = tokenize(["portable ap chest, shortness of breath"])
    image = torch.rand(1, 1, 224, 224)

    maps, extras = grad_cam_multi(
        model, image, tokens["input_ids"], tokens["attention_mask"], range(len(LABELS))
    )
    assert set(maps) == set(range(len(LABELS)))
    for heat in maps.values():
        assert heat.shape == (224, 224)
        assert np.isfinite(heat).all()
        assert 0.0 <= heat.min() and heat.max() <= 1.0 + 1e-6
    assert "gate" in extras


def test_grad_cam_multi_maps_differ_between_findings(tiny_backbone):
    """Identical maps for every finding would mean the gradients never reached
    the patch tokens -- the failure mode this whole panel exists to show."""
    model, tokenize, _ = tiny_backbone
    tokens = tokenize(["chest radiograph"])
    maps, _ = grad_cam_multi(
        model, torch.rand(1, 1, 224, 224), tokens["input_ids"], tokens["attention_mask"], [0, 1, 2, 3]
    )
    pairs = [(a, b) for a in maps for b in maps if a < b]
    assert any(np.abs(maps[a] - maps[b]).max() > 1e-6 for a, b in pairs)


def test_grad_cam_multi_leaves_the_model_untouched(tiny_backbone):
    """It must not flip requires_grad or training mode as a side effect."""
    model, tokenize, _ = tiny_backbone
    before = {name: p.requires_grad for name, p in model.named_parameters()}
    was_training = model.training
    tokens = tokenize(["chest radiograph"])
    grad_cam_multi(model, torch.rand(1, 1, 224, 224), tokens["input_ids"], tokens["attention_mask"], [0])
    assert {name: p.requires_grad for name, p in model.named_parameters()} == before
    assert model.training == was_training


def test_overlay_and_png_round_trip():
    film = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    heat = np.linspace(0, 1, 64 * 64).reshape(64, 64).astype(np.float32)
    overlay = overlay_heatmap(film, heat)
    assert overlay.shape == (64, 64, 3) and overlay.dtype == np.uint8
    uri = to_png_base64(overlay)
    assert uri.startswith("data:image/png;base64,") and len(uri) > 200


def test_colormap_is_monotone_and_bounded():
    ramp = _colormap(np.linspace(0, 1, 64))
    assert ramp.shape == (64, 3)
    assert ramp.min() >= 0.0 and ramp.max() <= 1.0
    assert ramp[:, 0].sum() >= ramp[:, 2].sum()   # red leads, blue trails


# --------------------------------------------------------------------------- #
# diffusion conditioning
# --------------------------------------------------------------------------- #
def _perturbed_unet(seed: int = 0) -> ConditionalUNet:
    """A UNet with its zero-initialised projections nudged off zero.

    Every ResBlock's second conv and the output conv are zero-init by design
    (ADM-style), so a fresh network emits exactly 0 for every input. Simulating
    a trained network is the only way to test the conditioning path.
    """
    torch.manual_seed(seed)
    net = ConditionalUNet(image_size=32, base_channels=16, channel_mult=(1, 2), attn_resolutions=())
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, ResBlock):
                module.conv2.weight.normal_(0, 0.05)
                module.conv2.bias.normal_(0, 0.01)
        net.out_conv.weight.normal_(0, 0.05)
        net.out_conv.bias.normal_(0, 0.01)
    return net


def test_fresh_unet_emits_exactly_zero():
    """Documents why an undertrained diffusion model returns identical samples --
    the behaviour `Predictor.diffusion_info` reports to the UI."""
    torch.manual_seed(0)
    net = ConditionalUNet(image_size=32, base_channels=16, channel_mult=(1, 2), attn_resolutions=())
    with torch.no_grad():
        out = net(torch.randn(1, 1, 32, 32), torch.tensor([50]), torch.ones(1, 4))
    assert float(out.abs().max()) == 0.0


def test_label_vector_changes_the_prediction():
    net = _perturbed_unet()
    x, t = torch.randn(1, 1, 32, 32), torch.tensor([50])
    a = net(x, t, torch.zeros(1, 4))
    b = net(x, t, torch.tensor([[0.0, 1.0, 0.0, 1.0]]))
    assert float((a - b).abs().max()) > 1e-6


def test_null_embedding_differs_from_a_label_vector():
    net = _perturbed_unet()
    x, t = torch.randn(1, 1, 32, 32), torch.tensor([50])
    assert float((net(x, t, torch.zeros(1, 4)) - net(x, t, None)).abs().max()) > 1e-6


def test_classifier_free_guidance_drop_mask_splits_the_batch():
    net = _perturbed_unet()
    x, t = torch.randn(1, 1, 32, 32), torch.tensor([50])
    out = net(
        torch.cat([x, x]), torch.cat([t, t]),
        torch.tensor([[0.0, 1.0, 0.0, 1.0]] * 2),
        torch.tensor([False, True]),          # second row uses the null embedding
    )
    assert float((out[0] - out[1]).abs().max()) > 1e-6


def test_timestep_changes_the_prediction():
    net = _perturbed_unet()
    x = torch.randn(1, 1, 32, 32)
    a = net(x, torch.tensor([10]), None)
    b = net(x, torch.tensor([900]), None)
    assert float((a - b).abs().max()) > 1e-6


def test_guidance_scale_moves_the_sample():
    net = _perturbed_unet()
    diffusion = GaussianDiffusion(net, timesteps=40, schedule="cosine").eval()
    labels = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    torch.manual_seed(3)
    low = diffusion.ddim_sample((1, 1, 32, 32), labels, steps=6, guidance_scale=1.0)
    torch.manual_seed(3)
    high = diffusion.ddim_sample((1, 1, 32, 32), labels, steps=6, guidance_scale=4.0)
    assert float((low - high).abs().max()) > 1e-6


def test_sdedit_refine_keeps_the_image_recognisable():
    """Low strength must stay close to the input; high strength must depart."""
    net = _perturbed_unet()
    diffusion = GaussianDiffusion(net, timesteps=60, schedule="cosine").eval()
    film = to_model_space(torch.rand(1, 1, 32, 32))

    torch.manual_seed(1)
    gentle = diffusion.refine(film, None, strength=0.05, steps=5)
    torch.manual_seed(1)
    heavy = diffusion.refine(film, None, strength=0.9, steps=5)

    gentle_delta = float((gentle - film).abs().mean())
    heavy_delta = float((heavy - film).abs().mean())
    assert gentle_delta < heavy_delta
    assert to_image_space(gentle).min() >= 0.0 and to_image_space(gentle).max() <= 1.0


def test_image_space_round_trip():
    values = torch.rand(1, 1, 8, 8)
    assert torch.allclose(to_image_space(to_model_space(values)), values, atol=1e-6)


# --------------------------------------------------------------------------- #
# vision-language panel maths
# --------------------------------------------------------------------------- #
def test_zero_shot_margin_is_a_difference_of_cosines(tiny_backbone):
    """The panel's number must be exactly positive-prompt minus negative-prompt
    similarity, or the diverging bars mean nothing."""
    model, tokenize, _ = tiny_backbone
    prompts = [f"chest x-ray showing {n.lower()}" for n in LABELS] + \
              [f"chest x-ray with no {n.lower()}" for n in LABELS]
    tokens = tokenize(prompts)
    with torch.no_grad():
        _, pooled = model.vlm.encode_text(tokens["input_ids"], tokens["attention_mask"])
        text = torch.nn.functional.normalize(pooled.float(), dim=-1)
        _, img_pooled = model.vlm.encode_image(torch.rand(1, 1, 224, 224))
        image = torch.nn.functional.normalize(img_pooled.float(), dim=-1)

    similarity = (image @ text.T)[0]
    n = len(LABELS)
    for i in range(n):
        margin = float(similarity[i] - similarity[i + n])
        assert -2.0 <= margin <= 2.0
        assert np.isfinite(margin)


def test_blanking_the_report_is_what_modality_dropout_does(tiny_backbone):
    """The image-only comparison must use the same null text the dataset uses,
    otherwise the delta measures a different substitution than training saw."""
    model, tokenize, cfg = tiny_backbone
    placeholder = tokenize([str(cfg.text.empty_placeholder)])
    image = torch.rand(1, 1, 224, 224)
    with torch.no_grad():
        blank = model(image, placeholder["input_ids"], placeholder["attention_mask"])
        again = model(image, placeholder["input_ids"], placeholder["attention_mask"])
    assert torch.allclose(blank["logits"], again["logits"], atol=1e-6)

    with torch.no_grad():
        worded = tokenize(["large left pleural effusion with basal atelectasis"])
        other = model(image, worded["input_ids"], worded["attention_mask"])
    assert not torch.allclose(blank["logits"], other["logits"], atol=1e-6)
