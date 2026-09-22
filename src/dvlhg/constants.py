"""Label space and dataset-level constants shared by every stage."""

from __future__ import annotations

# The four target findings, in a fixed order. Every tensor of shape [B, 4]
# in this project follows this order. Never reorder without re-exporting.
LABELS = ["Atelectasis", "Cardiomegaly", "Edema", "Pleural Effusion"]
NUM_LABELS = len(LABELS)
LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS)}

# Short codes used in filenames / logs.
LABEL_CODES = ["ATEL", "CMEG", "EDEM", "PEFF"]

# CheXpert-style raw cell values found in the MIMIC-CXR-JPG label CSVs.
POSITIVE = 1.0
NEGATIVE = 0.0
UNCERTAIN = -1.0
# "not mentioned" is an empty cell -> NaN after pandas parsing.

# How the hypergraph groups its hyperedges. Used for logging, for the
# per-group learnable weight, and for the "why" panel in the frontend.
EDGE_GROUPS = [
    "knn_fused",   # k nearest neighbours in the fused representation
    "knn_image",   # k nearest neighbours in image space only
    "knn_text",    # k nearest neighbours in report space only
    "meta_view",   # same radiographic view (PA / AP / LATERAL)
    "meta_demo",   # same (sex, age-decade) bucket
    "proto_label", # train-only prototype edges from warm-up pseudo-labels
]

# Imaging views kept by default. Lateral films are dropped because the four
# target findings are scored by the CheXpert labeler from the frontal study.
FRONTAL_VIEWS = ("PA", "AP", "AP AXIAL", "AP LLD", "AP RLD", "PA LLD", "PA RLD")

# Radiology-report section headers recognised by the parser, longest first so
# that "FINAL REPORT" does not shadow "FINAL".
REPORT_SECTIONS = [
    "EXAMINATION",
    "INDICATION",
    "HISTORY",
    "CLINICAL HISTORY",
    "CLINICAL INFORMATION",
    "REASON FOR EXAMINATION",
    "REASON FOR EXAM",
    "TECHNIQUE",
    "COMPARISON",
    "COMPARISONS",
    "FINDINGS",
    "IMPRESSION",
    "CONCLUSION",
    "RECOMMENDATION",
    "RECOMMENDATIONS",
    "NOTIFICATION",
    "WET READ",
    "FINAL REPORT",
]

# Sections that are written *before* the radiologist reads the film. Using only
# these keeps the text channel free of the diagnosis that the labels were
# derived from. See docs/LEAKAGE.md.
PRE_READ_SECTIONS = [
    "EXAMINATION",
    "INDICATION",
    "HISTORY",
    "CLINICAL HISTORY",
    "CLINICAL INFORMATION",
    "REASON FOR EXAMINATION",
    "REASON FOR EXAM",
    "TECHNIQUE",
    "COMPARISON",
    "COMPARISONS",
]

# Surface forms of the four findings (plus close confounders). Used by the
# `*_masked` text modes to redact the label from the report body.
DISEASE_TERMS = [
    # Atelectasis
    r"atelecta\w*", r"collapse\w*", r"volume\s+loss",
    # Cardiomegaly
    r"cardiomegal\w*", r"enlarged\s+card\w*", r"cardiac\s+enlargement",
    r"enlargement\s+of\s+the\s+cardiac\s+silhouette", r"cardiac\s+silhouette\s+is\s+enlarged",
    r"heart\s+size\s+is\s+(?:mildly\s+|moderately\s+|severely\s+)?enlarged",
    # Edema
    r"edema", r"oedema", r"vascular\s+congestion", r"pulmonary\s+congestion",
    r"interstitial\s+markings", r"kerley",
    # Pleural effusion
    r"effusion\w*", r"pleural\s+fluid", r"blunting\s+of\s+the\s+costophrenic",
    r"costophrenic\s+angle\s+blunting", r"hydrothorax",
    # Frequent co-mentions that would otherwise leak by association
    r"consolidat\w*", r"pneumoni\w*", r"opacit\w*", r"opacificat\w*",
    r"pneumothora\w*", r"infiltrat\w*",
]

MASK_TOKEN = "[finding]"

# Shown on every prediction surface. This is a research prototype.
DISCLAIMER = (
    "Research prototype. Not a medical device. Not validated for clinical use. "
    "Outputs must not be used to diagnose, treat, or make decisions about any patient."
)
